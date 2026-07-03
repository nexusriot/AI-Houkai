"""Vector-backed memory store for an AI agent.

Uses ChromaDB as the persistence + similarity layer. Each memory is a
small document with structured metadata so the agent can filter as
well as semantically search.

New in this version:
  • Memory linking — typed directed edges between memories
  • Conflict / contradiction detection — detect duplicates and contradictions
  • Hybrid retrieval — cosine + BM25 + recency + importance scoring
"""

from __future__ import annotations

import functools
import json
import logging
import math
import gzip
import os
import re
import time
import uuid
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Literal

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from .journal import Journal, JournalEntry

MemoryType = Literal["episodic", "semantic", "procedural", "feedback"]

# Canonical enum vocabularies. The store validates against these at runtime
# so every surface (CLI, HTTP, MCP, TUI) rejects typos in ONE place instead
# of silently degrading (e.g. mode="hybird" used to run semantic search, an
# unknown on_conflict used to scan and then discard the conflicts).
MEMORY_TYPES:      tuple[str, ...] = ("episodic", "semantic", "procedural", "feedback")
LINK_RELS:         tuple[str, ...] = ("related", "refines", "derived_from",
                                      "example_of", "contradicts", "supersedes")
RECALL_MODES:      tuple[str, ...] = ("semantic", "hybrid")
FUSIONS:           tuple[str, ...] = ("weighted", "rrf")
CONFLICT_POLICIES: tuple[str, ...] = ("ignore", "warn", "supersede", "raise")
IMPORT_POLICIES:   tuple[str, ...] = ("skip", "overwrite", "rename", "error")
DIRECTIONS:        tuple[str, ...] = ("out", "in", "both")


def _validate_choice(value: object, allowed: tuple[str, ...], param: str) -> None:
    """Raise ValueError naming the parameter and the allowed vocabulary."""
    if value not in allowed:
        raise ValueError(
            f"{param} must be one of: {', '.join(allowed)} — got {value!r}"
        )


def _validate_tags(tags: list[str]) -> None:
    """Tags are stored comma-joined in Chroma metadata, so a comma inside a
    tag would silently split it into two on the next read."""
    for t in tags:
        if "," in t:
            raise ValueError(
                f"tags must not contain commas — got {t!r} "
                f"(tags are stored as a comma-joined string)"
            )


@functools.lru_cache(maxsize=None)
def _get_embed_fn(model_name: str):
    """Return a cached embedding function for *model_name*.

    Loaded once per process per model name — subsequent calls return the
    same instance, avoiding redundant disk reads and RAM allocation.

    Side-effects applied before the first load:
      • HF_HUB_DISABLE_PROGRESS_BARS=1  — suppress "Loading weights" bars
      • huggingface_hub loggers → ERROR  — suppress unauthenticated-request
        warnings (the model lives in the local cache; no auth is needed)
    """
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    for _logger_name in (
        "huggingface_hub",
        "huggingface_hub.utils",
        "huggingface_hub.utils._http",
        "huggingface_hub.utils._headers",
    ):
        logging.getLogger(_logger_name).setLevel(logging.ERROR)

    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name
    )
ConflictFn = Callable[["Memory", "Memory"], bool]

@dataclass
class Link:
    """A directed, typed edge from one memory to another."""
    to:  str
    rel: str  # "supersedes" | "refines" | "derived_from" | "example_of" | "contradicts" | "related"


@dataclass
class Graph:
    """Subgraph of memories and their links."""
    nodes: dict[str, "Memory"]
    edges: list[tuple[str, str, str]]  # (src_id, dst_id, rel)


@dataclass(frozen=True)
class HybridWeights:
    """Blend weights for hybrid recall scoring."""
    cosine:         float = 0.55
    lexical:        float = 0.20
    recency:        float = 0.15
    importance:     float = 0.10
    decay_rate:     float = 0.10   # λ — shared with DecayEngine
    polarity_weight: float = 0.05  # additive bonus: +0.05 for polarity=+1, -0.05 for -1
    # Which timestamp the recency term measures. "created" (default) scores by
    # how recently the fact was *learned* — stable across recalls. "accessed"
    # scores by how recently it was *retrieved*, which `_touch` moves on every
    # recall (self-reinforcing); kept as an opt-in for the old behaviour.
    recency_basis:  Literal["created", "accessed"] = "created"

    def __post_init__(self) -> None:
        total = self.cosine + self.lexical + self.recency + self.importance
        if total <= 0:
            raise ValueError("HybridWeights: at least one weight must be > 0")


@dataclass(frozen=True)
class ExpandSpec:
    """Controls graph-walk expansion after recall."""
    rels:  tuple[str, ...] = ("refines", "example_of")
    depth: int   = 1
    cap:   int   = 5
    score: float = 0.70
    # Per-hop score multiplier beyond the first hop: a hop-h neighbour is scored
    # ``score * decay**(h-1)``. 1.0 (default) keeps every expanded node at
    # ``score`` regardless of distance (backward-compatible).
    decay: float = 1.0


@dataclass
class PackedMemory:
    """A memory selected for a context pack, with its token cost."""
    memory: "Memory"
    score:  float
    tokens: int


@dataclass
class CompressedGroup:
    """Several similar low-ranked memories folded into one summary line."""
    memories: "list[Memory]"
    text:     str   # full formatted line: "- (compressed) [×N] ..."
    tokens:   int


@dataclass
class PackResult:
    """Result of recall_pack — memories that fit a token budget plus a
    ready-to-inject context block."""
    items:             list[PackedMemory]
    text:              str        # formatted block (empty when no items fit)
    used_tokens:       int
    budget:            int
    truncated:         bool       # True if ranked candidates were dropped to fit
    compressed_groups: "list[CompressedGroup]" = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.items)

    def ids(self) -> list[str]:
        return [p.memory.id for p in self.items]


@dataclass
class Conflict:
    a:          "Memory"
    b:          "Memory"
    similarity: float
    kind:       Literal["duplicate", "contradiction"]
    reason:     str   # "negation_diff" | "custom_fn" | "similarity"


class ConflictError(Exception):
    def __init__(self, conflicts: list[Conflict]) -> None:
        super().__init__(f"{len(conflicts)} conflict(s) detected")
        self.conflicts = conflicts


class _ImportConflict(Exception):
    """Internal — used to short-circuit a single import row."""


class ImportConflictError(Exception):
    def __init__(self, msg: str, *, collisions: list[tuple[str, str]]) -> None:
        super().__init__(msg)
        self.collisions = collisions


@dataclass
class ExportSummary:
    path:    Path
    count:   int
    bytes:   int
    elapsed: float


@dataclass
class ImportSummary:
    imported:    int = 0
    skipped:     int = 0
    overwritten: int = 0
    renamed:     int = 0
    errors:      list[tuple[str, str]] = field(default_factory=list)
    vectors_regenerated: bool = False


@dataclass
class Memory:
    id:            str
    text:          str
    type:          MemoryType
    tags:          list[str]  = field(default_factory=list)
    importance:    float      = 0.5          # 0..1
    created_at:    float      = field(default_factory=time.time)
    last_accessed: float      = field(default_factory=time.time)
    access_count:  int        = 0
    source:        str | None = None
    # — linking —
    links:         list[Link] = field(default_factory=list)
    # — conflict management —
    superseded_by: str  = ""    # id of the superseding memory, "" if active
    superseded_at: float = 0.0
    polarity:      int  = 0     # -1 / 0 / +1

    def to_metadata(self) -> dict[str, Any]:
        return {
            "type":          self.type,
            "tags":          ",".join(self.tags),
            "importance":    self.importance,
            "created_at":    self.created_at,
            "last_accessed": self.last_accessed,
            "access_count":  self.access_count,
            "source":        self.source or "",
            "links":         json.dumps([{"to": l.to, "rel": l.rel} for l in self.links]),
            "superseded_by": self.superseded_by,
            "superseded_at": self.superseded_at,
            "polarity":      self.polarity,
        }

    def to_dict(self) -> dict[str, Any]:
        """Full self-contained snapshot — used by the journal and export."""
        return {
            "id":            self.id,
            "text":          self.text,
            "type":          self.type,
            "tags":          list(self.tags),
            "importance":    self.importance,
            "created_at":    self.created_at,
            "last_accessed": self.last_accessed,
            "access_count":  self.access_count,
            "source":        self.source,
            "links":         [{"to": l.to, "rel": l.rel} for l in self.links],
            "superseded_by": self.superseded_by,
            "superseded_at": self.superseded_at,
            "polarity":      self.polarity,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Memory":
        return cls(
            id=d["id"],
            text=d["text"],
            type=d.get("type", "semantic"),
            tags=list(d.get("tags") or []),
            importance=float(d.get("importance", 0.5)),
            created_at=float(d.get("created_at", time.time())),
            last_accessed=float(d.get("last_accessed", time.time())),
            access_count=int(d.get("access_count", 0)),
            source=d.get("source"),
            links=[Link(to=l["to"], rel=l["rel"]) for l in (d.get("links") or [])],
            superseded_by=str(d.get("superseded_by") or ""),
            superseded_at=float(d.get("superseded_at") or 0.0),
            polarity=int(d.get("polarity") or 0),
        )

    @classmethod
    def from_record(cls, mid: str, text: str, meta: dict[str, Any]) -> "Memory":
        raw_links = meta.get("links") or "[]"
        try:
            links = [Link(to=l["to"], rel=l["rel"]) for l in json.loads(raw_links)]
        except (json.JSONDecodeError, KeyError, TypeError):
            links = []
        return cls(
            id=mid,
            text=text,
            type=meta.get("type", "semantic"),
            tags=[t for t in meta.get("tags", "").split(",") if t],
            importance=float(meta.get("importance", 0.5)),
            created_at=float(meta.get("created_at", time.time())),
            last_accessed=float(meta.get("last_accessed", time.time())),
            access_count=int(meta.get("access_count", 0)),
            source=meta.get("source") or None,
            links=links,
            superseded_by=str(meta.get("superseded_by", "")),
            superseded_at=float(meta.get("superseded_at", 0.0)),
            polarity=int(meta.get("polarity", 0)),
        )


_BM25_K1 = 1.5
_BM25_B  = 0.75


# CJK / Korean ranges: these scripts are written without spaces, so a run of
# them collapses into a single \w+ token. We additionally emit character bigrams
# of each run so lexical (BM25/Jaccard) matching works for non-Latin queries —
# a standard, dependency-free IR technique for CJK.
_CJK_RE = re.compile(
    r"[぀-ヿ㐀-䶿一-鿿豈-﫿가-힣]"
)


def _tokenize(text: str) -> list[str]:
    # Strip apostrophes so "don't" → "dont", "won't" → "wont", matching _NEG
    normalized = text.lower().replace("’", "").replace("'", "")
    tokens = re.findall(r"\b\w+\b", normalized)
    if not _CJK_RE.search(normalized):
        return tokens
    extra: list[str] = []
    for tok in tokens:
        chars = [c for c in tok if _CJK_RE.match(c)]
        # Bigrams of multi-char CJK runs; a single CJK char is already its own
        # \w+ token, so we don't re-emit it (avoids double-counting in tf).
        if len(chars) >= 2:
            extra.extend(chars[i] + chars[i + 1] for i in range(len(chars) - 1))
    return tokens + extra


def _bm25_score_pool(
    query: str,
    docs: list[str],
    k1: float = _BM25_K1,
    b: float  = _BM25_B,
) -> list[float]:
    """BM25 scores for each doc relative to query; normalised to [0, 1]."""
    if not docs:
        return []
    query_tokens = set(_tokenize(query))
    tokenized = [_tokenize(d) for d in docs]
    n = len(docs)
    avgdl = sum(len(t) for t in tokenized) / n

    df: dict[str, int] = {}
    for tokens in tokenized:
        for t in set(tokens):
            df[t] = df.get(t, 0) + 1

    raw: list[float] = []
    for tokens in tokenized:
        tf: dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        dl = len(tokens)
        score = 0.0
        for t in query_tokens:
            if t not in df:
                continue
            idf = math.log((n - df[t] + 0.5) / (df[t] + 0.5) + 1.0)
            freq = tf.get(t, 0)
            score += idf * (freq * (k1 + 1)) / (
                freq + k1 * (1.0 - b + b * dl / max(avgdl, 1.0))
            )
        raw.append(score)

    mx = max(raw) if raw else 0.0
    return [s / mx if mx > 0 else 0.0 for s in raw]


_NEG: frozenset[str] = frozenset({
    "not", "never", "no", "dont", "don't", "doesnt", "doesn't",
    "wont", "won't", "shouldnt", "shouldn't", "cant", "can't",
    "without", "avoid", "neither", "nor", "nothing", "nobody",
    "nowhere", "none",
})


def _count_neg(text: str) -> int:
    return sum(1 for t in _tokenize(text) if t in _NEG)


def _negation_diff(s1: str, s2: str) -> bool:
    """True when one text has an odd more negation words than the other."""
    return (_count_neg(s1) % 2) != (_count_neg(s2) % 2)


def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _estimate_tokens(text: str) -> int:
    """Cheap, tokenizer-free token estimate (~4 chars/token).

    A soft heuristic so token_budget stays a dependency-free ceiling.
    Callers wanting exact counts pass their own token_counter.
    """
    return max(1, round(len(text) / 4))


_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must",
    "to", "for", "of", "in", "on", "at", "by", "from", "with", "about",
    "and", "or", "but", "if", "then", "so", "as", "that", "this", "these",
    "those", "it", "its", "i", "my", "we", "our", "you", "your", "they",
    "their", "what", "how", "when", "where", "who", "which", "why", "not",
    "just", "now", "also", "than", "only", "any", "all", "each",
})


def extract_key_phrases(task: str, max_phrases: int = 3) -> list[str]:
    """Extract up to *max_phrases* key phrases from *task* without any NLP library.

    Filters stop words, then prefers bigrams (more specific) over single words.
    Used by :meth:`MemoryStore.auto_context_pack` to fan out recall over multiple
    query angles derived from the same task description.
    """
    words = [w for w in _tokenize(task) if w not in _STOP_WORDS and len(w) > 2]
    phrases: list[str] = []
    for i in range(len(words) - 1):
        phrases.append(f"{words[i]} {words[i + 1]}")
    phrases.extend(words)
    seen: set[str] = set()
    unique: list[str] = []
    for p in phrases:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique[:max_phrases]


def _jaccard_sim(a: str, b: str) -> float:
    ta = set(_tokenize(a))
    tb = set(_tokenize(b))
    union = ta | tb
    return len(ta & tb) / len(union) if union else 0.0


def _cluster_by_jaccard(
    candidates: list[tuple["Memory", float]],
    threshold: float,
    min_size: int,
) -> "list[list[Memory]]":
    """Greedy single-linkage clustering of truncated candidates by token-Jaccard."""
    n = len(candidates)
    used = [False] * n
    clusters: list[list[Memory]] = []
    for i in range(n):
        if used[i]:
            continue
        mem_i = candidates[i][0]
        cluster: list[Memory] = [mem_i]
        used[i] = True
        for j in range(i + 1, n):
            if used[j]:
                continue
            if _jaccard_sim(mem_i.text, candidates[j][0].text) >= threshold:
                cluster.append(candidates[j][0])
                used[j] = True
        if len(cluster) >= min_size:
            clusters.append(cluster)
    return clusters


def _compress_group(memories: "list[Memory]") -> str:
    """Extract first sentence of each memory (most-important first), join with ' | '."""
    ordered = sorted(memories, key=lambda m: m.importance, reverse=True)
    snippets = []
    for m in ordered[:3]:
        snippet = m.text.split(".")[0].strip()
        if snippet:
            snippets.append(snippet)
    summary = " | ".join(snippets)
    return f"[×{len(memories)} similar] {summary}"


def _build_where(
    type: str | None = None,
    min_importance: float | None = None,
    source: str | None = None,
    since: float | None = None,
    until: float | None = None,
) -> dict[str, Any] | None:
    """Assemble a ChromaDB ``where`` clause from metadata filters.

    Chroma rejects a multi-key clause (``{"a": 1, "b": 2}``) and a multi-
    operator expression (``{"created_at": {"$gte": x, "$lte": y}}``); every
    leaf must hold exactly one operator, and conjunctions must be expressed
    with an explicit ``$and``.  This helper produces a single-condition clause
    when only one filter is set and an ``$and`` of single-operator leaves
    otherwise — including the ``since``/``until`` range, which becomes two
    separate ``created_at`` leaves.
    """
    conditions: list[dict[str, Any]] = []
    if type:
        conditions.append({"type": type})
    if min_importance is not None:
        conditions.append({"importance": {"$gte": min_importance}})
    if source:
        conditions.append({"source": source})
    if since is not None:
        conditions.append({"created_at": {"$gte": since}})
    if until is not None:
        conditions.append({"created_at": {"$lte": until}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


class MemoryStore:
    """Long-term memory backed by ChromaDB with linking, conflict detection,
    and hybrid retrieval."""

    def __init__(
        self,
        path: str = "./.chroma",
        collection: str = "ai_houkai",
        embedding_model: str = "all-MiniLM-L6-v2",
        *,
        conflict_policy: Literal["ignore", "warn", "supersede", "raise"] = "ignore",
        conflict_threshold: float = 0.80,
        contradiction_fn: ConflictFn | None = None,
        hybrid_weights: HybridWeights | None = None,
        importance_fn: "Callable[[str, str, list[str]], float] | None" = None,
        actor: str = "lib",
        journal_enabled: bool = True,
        journal_path: str | None = None,
        journal_rotate_mb: int = 64,
        journal_keep_days: int = 90,
    ) -> None:
        _validate_choice(conflict_policy, CONFLICT_POLICIES, "conflict_policy")
        self.path = path
        self.collection_name = collection
        self.embedding_model = embedding_model
        self.client = chromadb.PersistentClient(
            path=path, settings=Settings(anonymized_telemetry=False)
        )
        embed_fn = _get_embed_fn(embedding_model)
        self.collection = self.client.get_or_create_collection(
            name=collection,
            embedding_function=embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
        self._conflict_policy    = conflict_policy
        self._conflict_threshold = conflict_threshold
        self._contradiction_fn   = contradiction_fn
        self._hybrid_weights     = hybrid_weights
        self._importance_fn      = importance_fn

        self._actor           = actor
        self._journal_enabled = journal_enabled
        jp = Path(journal_path) if journal_path else Path(path).parent / "journal.log"
        self.journal = Journal(
            jp, rotate_mb=journal_rotate_mb, keep_days=journal_keep_days,
        )

    @contextmanager
    def as_actor(self, name: str) -> Iterator[None]:
        """Temporarily attribute journal entries to *name* (e.g. 'reflection')."""
        previous = self._actor
        self._actor = name
        try:
            yield
        finally:
            self._actor = previous

    def _journal(
        self,
        op: str,
        mid: str,
        *,
        before: dict[str, Any] | None = None,
        after:  dict[str, Any] | None = None,
        meta:   dict[str, Any] | None = None,
    ) -> None:
        if not self._journal_enabled:
            return
        self.journal.append(JournalEntry(
            ts=time.time(), op=op, actor=self._actor, id=mid,
            before=before, after=after, meta=meta or {},
        ))

    def remember(
        self,
        text: str,
        type: MemoryType = "semantic",
        tags: Iterable[str] = (),
        importance: float | None = None,
        source: str | None = None,
        *,
        polarity: int = 0,
        on_conflict: Literal["ignore", "warn", "supersede", "raise"] | None = None,
        contradiction_fn: ConflictFn | None = None,
    ) -> Memory:
        _validate_choice(type, MEMORY_TYPES, "type")
        if polarity not in (-1, 0, 1):
            raise ValueError(f"polarity must be -1, 0, or +1 — got {polarity!r}")
        if on_conflict is not None:
            _validate_choice(on_conflict, CONFLICT_POLICIES, "on_conflict")
        tags = list(tags)
        _validate_tags(tags)
        # importance=None → auto-score when an importance_fn is configured,
        # else keep the historical 0.5 default.
        if importance is None:
            if self._importance_fn is not None:
                importance = self._importance_fn(text, type, list(tags))
            else:
                importance = 0.5
        mem = Memory(
            id=str(uuid.uuid4()),
            text=text.strip(),
            type=type,
            tags=list(tags),
            importance=max(0.0, min(1.0, importance)),
            source=source,
            polarity=polarity,
        )
        self.collection.add(
            ids=[mem.id],
            documents=[mem.text],
            metadatas=[mem.to_metadata()],
        )
        self._journal("remember", mem.id, after=mem.to_dict())

        policy = on_conflict if on_conflict is not None else self._conflict_policy
        if policy != "ignore":
            cfn = contradiction_fn or self._contradiction_fn
            conflicts = self._check_conflicts(mem, self._conflict_threshold, cfn)
            if conflicts:
                if policy == "warn":
                    warnings.warn(
                        f"remember(): {len(conflicts)} conflict(s) detected "
                        f"for memory {mem.id!r}: "
                        + ", ".join(f"{c.kind}({c.b.id!r})" for c in conflicts),
                        UserWarning,
                        stacklevel=2,
                    )
                elif policy == "supersede":
                    for c in conflicts:
                        self.supersede(old_id=c.b.id, new_id=mem.id)
                elif policy == "raise":
                    # Roll back the just-added doc — the caller is told the
                    # memory was not stored, so it must not linger in Chroma.
                    self.forget(mem.id)
                    raise ConflictError(conflicts)

        return mem

    def forget(self, memory_id: str) -> bool:
        before = self._get_by_id(memory_id)
        if before is None:
            return False
        self.collection.delete(ids=[memory_id])
        self._journal("forget", memory_id, before=before.to_dict())
        return True

    _UNSET: Any = object()  # sentinel: "leave field unchanged" where None is meaningful

    def edit(
        self,
        memory_id: str,
        *,
        text: str | None = None,
        type: MemoryType | None = None,
        tags: Iterable[str] | None = None,
        importance: float | None = None,
        polarity: int | None = None,
        source: str | None = _UNSET,
    ) -> Memory:
        """Update fields of an existing memory in place, keeping its id.

        Every ``None`` (or omitted) field is left unchanged; ``source`` uses a
        sentinel so ``source=None`` explicitly clears it. Text changes re-embed
        the document. Links, superseded_by, created_at, and access tracking are
        preserved — unlike a forget()+remember() round-trip.

        The change is journaled (op ``edit``, with before/after snapshots) and
        reversible via :meth:`undo`. A call that changes nothing is a no-op:
        no write, no journal entry.

        Raises KeyError if *memory_id* does not exist, ValueError on a bad
        type / polarity / importance.
        """
        mem = self._get_by_id(memory_id)
        if mem is None:
            raise KeyError(f"memory_id {memory_id!r} not found")
        before = mem.to_dict()

        if type is not None:
            _validate_choice(type, MEMORY_TYPES, "type")
            mem.type = type
        if polarity is not None:
            if polarity not in (-1, 0, 1):
                raise ValueError(f"polarity must be -1, 0, or +1 — got {polarity!r}")
            mem.polarity = polarity
        if importance is not None:
            mem.importance = max(0.0, min(1.0, float(importance)))
        if tags is not None:
            new_tags = list(tags)
            _validate_tags(new_tags)
            mem.tags = new_tags
        if source is not self._UNSET:
            mem.source = source
        text_changed = False
        if text is not None:
            new_text = text.strip()
            if not new_text:
                raise ValueError("text must be non-empty")
            text_changed = new_text != mem.text
            mem.text = new_text

        after = mem.to_dict()
        if after == before:
            return mem  # nothing to do — keep the journal quiet

        if text_changed:
            # Passing documents re-embeds via the collection's embedding fn.
            self.collection.update(
                ids=[memory_id],
                documents=[mem.text],
                metadatas=[mem.to_metadata()],
            )
        else:
            self.collection.update(ids=[memory_id], metadatas=[mem.to_metadata()])
        self._journal("edit", memory_id, before=before, after=after)
        return mem

    def nuke(self) -> int:
        """Delete every memory in the current collection. Returns the count deleted."""
        result = self.collection.get(include=[])
        ids = result["ids"]
        if not ids:
            return 0
        self.collection.delete(ids=ids)
        self._journal("nuke", "*", meta={"count": len(ids)})
        return len(ids)

    def recall(
        self,
        query: str,
        k: int = 5,
        type: MemoryType | None = None,
        tag: str | None = None,
        min_importance: float | None = None,
        *,
        source: str | None = None,
        since: float | None = None,
        until: float | None = None,
        mode: Literal["semantic", "hybrid"] = "semantic",
        weights: HybridWeights | None = None,
        fusion: Literal["weighted", "rrf"] = "weighted",
        diversity: float | None = None,
        dedup_threshold: float | None = None,
        min_cosine: float | None = None,
        overfetch: int = 4,
        expand: ExpandSpec | None = None,
        include_superseded: bool = False,
        touch: bool = True,
        explain: bool = False,
    ) -> list[tuple[Memory, float]] | list[tuple[Memory, float, dict[str, Any]]]:
        """Semantic (or hybrid) search with optional metadata filters.

        ``type``/``min_importance``/``source`` match exactly (``source`` is the
        provenance string set at :meth:`remember` time); ``since``/``until`` are
        Unix timestamps bounding ``created_at`` (inclusive). ``tag`` is matched
        post-query against the memory's tag list.

        Ranking controls (hybrid mode unless noted):
          ``fusion="rrf"`` blends signals by Reciprocal Rank Fusion (scale-free)
          instead of the default weighted sum.
          ``diversity`` (0..1) re-selects results with Maximal Marginal Relevance:
          higher → more relevance, lower → more novelty. ``dedup_threshold``
          (e.g. 0.92) hard-drops a candidate whose cosine to an already-selected
          result exceeds it. Both also apply in semantic mode.
          ``min_cosine`` drops any candidate below an absolute cosine floor —
          a relevance gate that lets callers receive *nothing* rather than weak hits.

        ``touch=False`` skips the access-count / last_accessed bump (read-only
        recall — e.g. for evaluation). ``explain=True`` returns
        ``(Memory, score, breakdown)`` triples with per-signal contributions.
        """
        _validate_choice(mode, RECALL_MODES, "mode")
        _validate_choice(fusion, FUSIONS, "fusion")
        if type is not None:
            _validate_choice(type, MEMORY_TYPES, "type")
        if diversity is not None and not (0.0 <= diversity <= 1.0):
            raise ValueError("diversity must be in [0, 1]")
        if dedup_threshold is not None and not (0.0 <= dedup_threshold <= 1.0):
            raise ValueError("dedup_threshold must be in [0, 1]")
        if min_cosine is not None and not (-1.0 <= min_cosine <= 1.0):
            raise ValueError("min_cosine must be in [-1, 1]")

        count = self.collection.count()
        if k <= 0 or count == 0:
            return []

        need_emb = diversity is not None or dedup_threshold is not None

        # The fast path (fetch exactly k) is only safe when no post-query
        # filtering or re-selection can drop/reorder rows.
        no_post_filter = (
            mode == "semantic" and include_superseded and tag is None
            and not need_emb and min_cosine is None
        )
        n_fetch = k if no_post_filter else min(k * overfetch, count)
        n_fetch = max(n_fetch, k)

        where = _build_where(type, min_importance, source, since, until)

        include = ["documents", "metadatas", "distances"]
        if need_emb:
            include = include + ["embeddings"]
        res = self.collection.query(
            query_texts=[query],
            n_results=n_fetch,
            where=where,
            include=include,
        )

        w = weights or self._hybrid_weights or HybridWeights()
        expl: dict[str, dict] | None = {} if explain else None
        if mode == "hybrid":
            if fusion == "rrf":
                scored = self._rrf_score(
                    res, query, tag, include_superseded, weights, min_cosine, expl)
            else:
                scored = self._hybrid_score(
                    res, query, tag, include_superseded, weights, min_cosine, expl)
        else:
            scored = self._semantic_filter(
                res, tag, include_superseded, w.polarity_weight, min_cosine, expl)

        if need_emb:
            out = self._mmr_select(
                scored, k, self._emb_map(res), diversity, dedup_threshold)
        else:
            out = scored[:k]

        if touch and out:
            self._touch_many([mem for mem, _ in out])

        # Expand via outgoing links
        if expand is not None and expand.cap > 0:
            out = self._expand_links(out, expand, include_superseded, expl)

        if explain:
            return [(mem, score, (expl.get(mem.id, {}) if expl else {}))
                    for mem, score in out]
        return out

    def recall_pack(
        self,
        query: str,
        *,
        token_budget: int = 800,
        type: MemoryType | None = None,
        tag: str | None = None,
        min_importance: float | None = None,
        source: str | None = None,
        since: float | None = None,
        until: float | None = None,
        mode: Literal["semantic", "hybrid"] = "hybrid",
        weights: HybridWeights | None = None,
        fusion: Literal["weighted", "rrf"] = "weighted",
        diversity: float | None = None,
        dedup_threshold: float | None = None,
        min_cosine: float | None = None,
        expand: ExpandSpec | None = None,
        include_superseded: bool = False,
        max_items: int = 50,
        touch: bool = True,
        token_counter: Callable[[str], int] | None = None,
        header: str = "## Relevant memory",
        compress: bool = False,
        compress_threshold: float = 0.30,
        compress_min_group: int = 2,
    ) -> PackResult:
        """Assemble the highest-ranked memories that fit a token budget into a
        ready-to-inject context block.

        Ranks candidates with :meth:`recall` (hybrid by default), then greedily
        packs them in rank order while the running token estimate stays within
        ``token_budget``. ``token_budget`` covers the rendered memory lines
        only — the ``header`` is not counted against it.

        ``token_counter`` overrides the default ~4-chars/token estimate; the
        budget is therefore a soft ceiling unless an exact counter is supplied.

        ``compress=True`` enables query-time compression: candidates that could
        not be packed individually are clustered by Jaccard similarity; clusters
        of ≥ ``compress_min_group`` members are folded into a single summary
        line, which is packed if it fits the remaining budget.

        ``fusion``/``diversity``/``dedup_threshold``/``min_cosine`` are passed
        through to :meth:`recall` — e.g. ``diversity`` stops the budget being
        spent on near-duplicate memories; ``min_cosine`` keeps weak hits out.
        """
        count_fn = token_counter or _estimate_tokens

        ranked = self.recall(
            query=query,
            k=max_items,
            type=type,
            tag=tag,
            min_importance=min_importance,
            source=source,
            since=since,
            until=until,
            mode=mode,
            weights=weights,
            fusion=fusion,
            diversity=diversity,
            dedup_threshold=dedup_threshold,
            min_cosine=min_cosine,
            expand=expand,
            include_superseded=include_superseded,
            touch=touch,
        )
        return self._pack_ranked(
            ranked, token_budget=token_budget, count_fn=count_fn, header=header,
            compress=compress, compress_threshold=compress_threshold,
            compress_min_group=compress_min_group,
        )

    def _pack_ranked(
        self,
        ranked: list[tuple[Memory, float]],
        *,
        token_budget: int,
        count_fn: Callable[[str], int],
        header: str,
        compress: bool = False,
        compress_threshold: float = 0.30,
        compress_min_group: int = 2,
    ) -> PackResult:
        """Greedily pack already-ranked ``(memory, score)`` pairs to a token
        budget (shared by recall_pack and auto_context_pack)."""
        items: list[PackedMemory] = []
        dropped: list[tuple[Memory, float]] = []
        used = 0
        truncated = False
        for mem, score in ranked:
            line = self._pack_line(mem)
            cost = count_fn(line)
            if used + cost > token_budget:
                truncated = True
                dropped.append((mem, score))
                continue  # a smaller, lower-ranked item may still fit
            items.append(PackedMemory(memory=mem, score=score, tokens=cost))
            used += cost

        compressed_groups: list[CompressedGroup] = []
        if compress and dropped:
            for group_mems in _cluster_by_jaccard(dropped, compress_threshold, compress_min_group):
                summary = _compress_group(group_mems)
                line = f"- (compressed) {summary}"
                cost = count_fn(line)
                if used + cost <= token_budget:
                    compressed_groups.append(CompressedGroup(
                        memories=group_mems, text=line, tokens=cost,
                    ))
                    used += cost

        if items or compressed_groups:
            parts = [self._pack_line(p.memory) for p in items]
            parts.extend(cg.text for cg in compressed_groups)
            body = "\n".join(parts)
            text = f"{header}\n{body}" if header else body
        else:
            text = ""

        return PackResult(
            items=items,
            text=text,
            used_tokens=used,
            budget=token_budget,
            truncated=truncated,
            compressed_groups=compressed_groups,
        )

    def auto_context_pack(
        self,
        task: str,
        *,
        token_budget: int = 800,
        max_phrases: int = 3,
        mode: Literal["semantic", "hybrid"] = "hybrid",
        min_cosine: float | None = None,
        header: str = "## Relevant memory",
        token_counter: Callable[[str], int] | None = None,
        compress: bool = False,
        compress_threshold: float = 0.30,
        compress_min_group: int = 2,
    ) -> PackResult:
        """Fan-out recall over *task* and extracted key phrases, deduplicate, and pack.

        More thorough than a single :meth:`recall_pack` call: extracts up to
        *max_phrases* bigram/keyword angles from the task description, runs
        :meth:`recall` for each, deduplicates by ID (keeping the highest score
        seen), then packs greedily within *token_budget* via the shared packer
        (so ``compress`` works here too).

        ``min_cosine`` applies an absolute relevance floor to every fan-out
        query, so an off-topic task injects nothing rather than padding context
        with weak hits.

        Note: this uses the default *weighted* fusion. ``fusion="rrf"`` is not
        offered here because RRF scores are rank-relative to each query's own
        candidate pool, so they cannot be compared across the fan-out queries.
        """
        count_fn = token_counter or _estimate_tokens
        queries = [task] + extract_key_phrases(task, max_phrases)

        best: dict[str, tuple[Memory, float]] = {}
        for q in queries:
            for mem, score in self.recall(
                query=q, k=10, mode=mode, min_cosine=min_cosine,
            ):
                if mem.id not in best or score > best[mem.id][1]:
                    best[mem.id] = (mem, score)

        ranked = sorted(best.values(), key=lambda x: x[1], reverse=True)
        return self._pack_ranked(
            ranked, token_budget=token_budget, count_fn=count_fn, header=header,
            compress=compress, compress_threshold=compress_threshold,
            compress_min_group=compress_min_group,
        )

    @staticmethod
    def _pack_line(mem: Memory) -> str:
        return f"- ({mem.type}) {mem.text}"

    def list_recent(
        self,
        limit: int = 20,
        *,
        include_superseded: bool = False,
    ) -> list[Memory]:
        if self.collection.count() == 0:
            return []
        res = self.collection.get(include=["documents", "metadatas"])
        memories = [
            Memory.from_record(i, d, m)
            for i, d, m in zip(res["ids"], res["documents"], res["metadatas"])
        ]
        if not include_superseded:
            memories = [m for m in memories if not m.superseded_by]
        memories.sort(key=lambda m: m.created_at, reverse=True)
        return memories[:limit]

    def count(self) -> int:
        return self.collection.count()

    def find_conflicts(
        self,
        memory_id: str | None = None,
        *,
        threshold: float | None = None,
    ) -> list[Conflict]:
        """Find duplicate / contradiction pairs.

        If *memory_id* is given, check that specific memory against all
        others.  Otherwise scan pairwise over the whole store (requires
        embeddings — O(n²) but fine at agent scale).
        """
        thr = threshold if threshold is not None else self._conflict_threshold
        if self.collection.count() == 0:
            return []

        if memory_id is not None:
            mem = self._get_by_id(memory_id)
            if mem is None:
                return []
            return self._check_conflicts(mem, thr)

        # Global pairwise scan
        res = self.collection.get(include=["embeddings", "documents", "metadatas"])
        ids   = res.get("ids") or []
        docs  = res.get("documents") or []
        metas = res.get("metadatas") or []
        raw   = res.get("embeddings")
        embs  = [] if raw is None else [list(e) for e in raw]

        memories = [Memory.from_record(i, d, m) for i, d, m in zip(ids, docs, metas)]
        seen: set[frozenset] = set()
        conflicts: list[Conflict] = []

        for i, mem_a in enumerate(memories):
            if mem_a.superseded_by or i >= len(embs):
                continue
            for j in range(i + 1, len(memories)):
                mem_b = memories[j]
                if mem_b.superseded_by or j >= len(embs):
                    continue
                if mem_a.type != mem_b.type:
                    continue
                pair = frozenset([mem_a.id, mem_b.id])
                if pair in seen:
                    continue
                sim = _cosine_sim(embs[i], embs[j])
                if sim < thr:
                    continue
                # Tag overlap guard
                if (mem_a.tags and mem_b.tags
                        and not (set(mem_a.tags) & set(mem_b.tags))):
                    continue
                seen.add(pair)
                if (mem_a.polarity != 0 and mem_b.polarity != 0
                        and mem_a.polarity != mem_b.polarity):
                    kind, reason = "contradiction", "polarity_diff"
                elif _negation_diff(mem_a.text, mem_b.text):
                    kind, reason = "contradiction", "negation_diff"
                elif self._contradiction_fn and self._contradiction_fn(mem_a, mem_b):
                    kind, reason = "contradiction", "custom_fn"
                else:
                    kind, reason = "duplicate", "similarity"
                conflicts.append(Conflict(
                    a=mem_a, b=mem_b,
                    similarity=round(sim, 4),
                    kind=kind, reason=reason,
                ))

        return conflicts

    def supersede(self, old_id: str, new_id: str) -> None:
        """Mark *old_id* as superseded by *new_id* and add a 'supersedes' link."""
        if old_id == new_id:
            raise ValueError("Cannot supersede a memory with itself")
        old = self._get_by_id(old_id)
        if old is None:
            raise KeyError(f"old_id {old_id!r} not found")
        new = self._get_by_id(new_id)
        if new is None:
            raise KeyError(f"new_id {new_id!r} not found")
        if new.superseded_by == old_id:
            raise ValueError(f"Cycle: {new_id!r} is already superseded by {old_id!r}")
        if old.superseded_by == new_id:
            return  # idempotent

        before = old.to_dict()
        old.superseded_by = new_id
        old.superseded_at = time.time()
        self.collection.update(ids=[old_id], metadatas=[old.to_metadata()])
        self._link_raw(new_id, old_id, "supersedes")
        self._journal(
            "supersede", old_id,
            before=before, after=old.to_dict(),
            meta={"old_id": old_id, "new_id": new_id},
        )

    def restore(self, memory_id: str) -> bool:
        """Clear the superseded_by marker on a memory (undo a supersede)."""
        mem = self._get_by_id(memory_id)
        if mem is None or not mem.superseded_by:
            return False
        before = mem.to_dict()
        superseder_id = mem.superseded_by
        mem.superseded_by = ""
        mem.superseded_at = 0.0
        self.collection.update(ids=[memory_id], metadatas=[mem.to_metadata()])
        self._unlink_raw(superseder_id, memory_id, "supersedes")
        self._journal(
            "restore", memory_id,
            before=before, after=mem.to_dict(),
            meta={"superseder_id": superseder_id},
        )
        return True

    def link(self, src_id: str, dst_id: str, rel: str = "related") -> None:
        """Add a directed link src → dst with the given relation (idempotent)."""
        added = self._link_raw(src_id, dst_id, rel)
        if added:
            self._journal(
                "link", src_id,
                meta={"src_id": src_id, "dst_id": dst_id, "rel": rel},
            )

    def _link_raw(self, src_id: str, dst_id: str, rel: str) -> bool:
        """Add link without journaling. Returns True iff a new edge was inserted."""
        _validate_choice(rel, LINK_RELS, "rel")
        if src_id == dst_id:
            raise ValueError("Cannot link a memory to itself")
        src = self._get_by_id(src_id)
        if src is None:
            raise KeyError(f"src_id {src_id!r} not found")
        if self._get_by_id(dst_id) is None:
            # Refuse dangling edges — graph walkers skip targets that don't
            # resolve, so the link would be stored but unreachable.
            raise KeyError(f"dst_id {dst_id!r} not found")
        for existing in src.links:
            if existing.to == dst_id and existing.rel == rel:
                return False
        src.links.append(Link(to=dst_id, rel=rel))
        self.collection.update(ids=[src_id], metadatas=[src.to_metadata()])
        return True

    def unlink(self, src_id: str, dst_id: str, rel: str | None = None) -> int:
        """Remove link(s) from src to dst. *rel=None* removes all rels.
        Returns number of links removed."""
        removed_rels = self._unlink_raw(src_id, dst_id, rel)
        if removed_rels:
            # removed_rels is what makes the entry undoable: a rel=None
            # unlink may drop several differently-typed edges at once.
            self._journal(
                "unlink", src_id,
                meta={"src_id": src_id, "dst_id": dst_id, "rel": rel,
                      "removed": len(removed_rels),
                      "removed_rels": removed_rels},
            )
        return len(removed_rels)

    def _unlink_raw(self, src_id: str, dst_id: str, rel: str | None) -> list[str]:
        """Remove matching links without journaling. Returns the removed rels."""
        if rel is not None:
            _validate_choice(rel, LINK_RELS, "rel")
        src = self._get_by_id(src_id)
        if src is None:
            return []
        removed = [l.rel for l in src.links
                   if l.to == dst_id and (rel is None or l.rel == rel)]
        if removed:
            src.links = [l for l in src.links
                         if not (l.to == dst_id and (rel is None or l.rel == rel))]
            self.collection.update(ids=[src_id], metadatas=[src.to_metadata()])
        return removed

    def neighbors(
        self,
        memory_id: str,
        *,
        rel: str | None = None,
        direction: Literal["out", "in", "both"] = "both",
        depth: int = 1,
    ) -> list[tuple[Memory, str]]:
        """Return (memory, rel) pairs reachable from *memory_id* via links.

        Outgoing traversal is O(links per node); incoming requires a full
        store scan per hop — fine at agent scale.
        """
        _validate_choice(direction, DIRECTIONS, "direction")
        if rel is not None:
            _validate_choice(rel, LINK_RELS, "rel")
        visited: set[str] = {memory_id}
        frontier: list[str] = [memory_id]
        result:   list[tuple[Memory, str]] = []

        for _ in range(depth):
            next_frontier: list[str] = []
            for mid in frontier:
                # outgoing
                if direction in ("out", "both"):
                    mem = self._get_by_id(mid)
                    if mem is not None:
                        for lnk in mem.links:
                            if rel is not None and lnk.rel != rel:
                                continue
                            if lnk.to in visited:
                                continue
                            target = self._get_by_id(lnk.to)
                            if target is None:
                                continue
                            # Two memories may be joined by several
                            # differently-typed edges — report each rel,
                            # but visit/expand the node only once.
                            for parallel in mem.links:
                                if parallel.to == lnk.to and (
                                        rel is None or parallel.rel == rel):
                                    result.append((target, parallel.rel))
                            visited.add(lnk.to)
                            next_frontier.append(lnk.to)
                if direction in ("in", "both"):
                    for candidate in self._get_all_memories():
                        if candidate.id in visited:
                            continue
                        matched_rels = [
                            lnk.rel for lnk in candidate.links
                            if lnk.to == mid and (rel is None or lnk.rel == rel)
                        ]
                        if matched_rels:
                            for r in matched_rels:
                                result.append((candidate, r))
                            visited.add(candidate.id)
                            next_frontier.append(candidate.id)
            frontier = next_frontier
            if not frontier:
                break

        return result

    def subgraph(
        self,
        memory_ids: Iterable[str],
        *,
        depth: int = 1,
    ) -> Graph:
        """Return a Graph of memories reachable from *memory_ids* within *depth* hops."""
        nodes: dict[str, Memory] = {}
        edges: list[tuple[str, str, str]] = []
        # Best remaining budget seen per node — a plain visited set would
        # truncate diamonds: a node first reached with 0 remaining hops was
        # never expanded again when a shorter path reached it with budget
        # to spare (e.g. a→b→c, a→c, c→d at depth 2 lost c→d and d).
        best_remaining: dict[str, int] = {}

        def _visit(mid: str, remaining: int) -> None:
            prev = best_remaining.get(mid)
            if prev is not None and prev >= remaining:
                return
            best_remaining[mid] = remaining
            mem = nodes.get(mid)
            if mem is None:
                fetched = self._get_by_id(mid)
                if fetched is None:
                    return
                nodes[mid] = mem = fetched
            if remaining > 0:
                for lnk in mem.links:
                    edge = (mid, lnk.to, lnk.rel)
                    if edge not in edges:
                        edges.append(edge)
                    _visit(lnk.to, remaining - 1)

        for mid in memory_ids:
            _visit(mid, depth)

        return Graph(nodes=nodes, edges=edges)


    def export(
        self,
        path: str | Path,
        *,
        include_vectors:    bool = True,
        include_superseded: bool = False,
        types: Iterable[MemoryType] | None = None,
        tags:  Iterable[str]        | None = None,
        since: float | None         = None,
    ) -> "ExportSummary":
        """Stream the collection to a gzipped JSONL `.ahkai` file.

        See PROPOSALS.md §2 for the format. Memories are written in
        ``created_at`` ascending order so two exports of the same store
        produce byte-identical files modulo the header timestamp.
        """
        types_set = set(types) if types else None
        tags_set  = set(tags)  if tags  else None
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        total = self.collection.count()
        include = ["documents", "metadatas"]
        if include_vectors:
            include = ["documents", "metadatas", "embeddings"]
        res = self.collection.get(include=include) if total else {
            "ids": [], "documents": [], "metadatas": [], "embeddings": None,
        }

        ids   = res.get("ids") or []
        docs  = res.get("documents") or []
        metas = res.get("metadatas") or []
        raw   = res.get("embeddings")
        embs  = [] if raw is None else [list(e) for e in raw]

        memories = [Memory.from_record(i, d, m) for i, d, m in zip(ids, docs, metas)]
        idx = sorted(range(len(memories)), key=lambda k: memories[k].created_at)

        # Filter
        kept: list[int] = []
        for k in idx:
            m = memories[k]
            if not include_superseded and m.superseded_by:
                continue
            if types_set is not None and m.type not in types_set:
                continue
            if tags_set is not None and not (set(m.tags) & tags_set):
                continue
            if since is not None and m.created_at < since:
                continue
            kept.append(k)

        header = {
            "format":      "ai-houkai/export",
            "version":     1,
            "exported_at": t0,
            "source": {
                "collection":      self.collection_name,
                "embedding_model": self.embedding_model,
                "embedding_dim":   (len(embs[0]) if embs else 0),
                "count":           len(kept),
            },
            "options": {
                "include_vectors":    include_vectors,
                "include_superseded": include_superseded,
                "types": sorted(types_set) if types_set else None,
                "tags":  sorted(tags_set)  if tags_set  else None,
                "since": since,
            },
        }

        with gzip.open(out_path, "wt", encoding="utf-8") as f:
            f.write(json.dumps(header, separators=(",", ":")) + "\n")
            for k in kept:
                m = memories[k]
                row: dict[str, Any] = {
                    "id":   m.id,
                    "text": m.text,
                    "meta": m.to_dict(),
                }
                if include_vectors and k < len(embs):
                    row["vector"] = embs[k]
                f.write(json.dumps(row, separators=(",", ":")) + "\n")

        size = out_path.stat().st_size
        self._journal("export", "", meta={
            "path": str(out_path), "count": len(kept), "bytes": size,
        })
        return ExportSummary(
            path=out_path, count=len(kept), bytes=size,
            elapsed=time.time() - t0,
        )

    def import_(
        self,
        path: str | Path,
        *,
        on_conflict: Literal["skip", "overwrite", "rename", "error"] = "skip",
        regenerate_vectors: bool = False,
        dry_run: bool = False,
    ) -> "ImportSummary":
        """Load memories from an ``.ahkai`` file (gzipped JSONL).

        On embedding-model mismatch, raises ``ImportError`` unless
        *regenerate_vectors=True* (re-embeds text on the way in).
        """
        _validate_choice(on_conflict, IMPORT_POLICIES, "on_conflict")
        in_path = Path(path)
        if not in_path.exists():
            raise FileNotFoundError(in_path)

        summary = ImportSummary()
        collisions: list[tuple[str, str]] = []

        with gzip.open(in_path, "rt", encoding="utf-8") as f:
            header_line = f.readline()
            if not header_line:
                raise ImportError(f"{in_path}: empty file")
            try:
                header = json.loads(header_line)
            except json.JSONDecodeError as e:
                raise ImportError(f"{in_path}: not an ai-houkai export ({e})")
            if header.get("format") != "ai-houkai/export":
                raise ImportError(f"{in_path}: missing/bad format header")
            if int(header.get("version", 0)) > 1:
                raise ImportError(
                    f"{in_path}: written by ai-houkai with version "
                    f"{header['version']} > reader version 1"
                )
            src = header.get("source") or {}
            src_model = src.get("embedding_model")
            src_dim   = int(src.get("embedding_dim") or 0)

            model_mismatch = src_model and src_model != self.embedding_model
            if model_mismatch and not regenerate_vectors:
                raise ImportError(
                    f"embedding model mismatch (file: {src_model!r}, "
                    f"store: {self.embedding_model!r}) — pass "
                    f"regenerate_vectors=True to re-embed on import"
                )
            summary.vectors_regenerated = bool(model_mismatch)

            with self.as_actor("import"):
                for lineno, line in enumerate(f, 2):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as e:
                        summary.errors.append((f"line:{lineno}", f"bad json: {e}"))
                        continue
                    try:
                        self._import_one(
                            row, on_conflict=on_conflict,
                            use_vectors=not model_mismatch and src_dim > 0,
                            dry_run=dry_run, summary=summary,
                            collisions=collisions,
                        )
                    except _ImportConflict:
                        # Collected; raised below
                        pass
                    except Exception as e:    # pragma: no cover — defensive
                        summary.errors.append((row.get("id", "?"), str(e)))

        if on_conflict == "error" and collisions:
            head = collisions[:10]
            raise ImportConflictError(
                f"{len(collisions)} id collision(s) on import; "
                f"first: {head}", collisions=collisions,
            )
        return summary

    def _import_one(
        self,
        row: dict[str, Any],
        *,
        on_conflict: str,
        use_vectors: bool,
        dry_run: bool,
        summary: "ImportSummary",
        collisions: list[tuple[str, str]],
    ) -> None:
        meta = row.get("meta") or {}
        # The export wrote `meta` as Memory.to_dict(); rebuild a Memory.
        mem = Memory.from_dict({
            **meta,
            "id":   row.get("id") or meta.get("id"),
            "text": row.get("text") or meta.get("text", ""),
        })
        vector = row.get("vector") if use_vectors else None
        existing = self._get_by_id(mem.id)

        if existing is not None:
            if on_conflict == "skip":
                summary.skipped += 1
                return
            if on_conflict == "error":
                collisions.append((mem.id, existing.text[:80]))
                summary.skipped += 1
                return
            if on_conflict == "rename":
                mem.id = str(uuid.uuid4())
                if not dry_run:
                    self._add_imported(mem, vector)
                summary.renamed += 1
                return
            if on_conflict == "overwrite":
                if not dry_run:
                    self.collection.delete(ids=[mem.id])
                    self._add_imported(mem, vector)
                summary.overwritten += 1
                return

        if not dry_run:
            self._add_imported(mem, vector)
        summary.imported += 1

    def _add_imported(self, mem: Memory, vector: list[float] | None) -> None:
        kwargs: dict[str, Any] = {
            "ids":       [mem.id],
            "documents": [mem.text],
            "metadatas": [mem.to_metadata()],
        }
        if vector is not None:
            kwargs["embeddings"] = [vector]
        self.collection.add(**kwargs)
        self._journal("import", mem.id, after=mem.to_dict(),
                      meta={"vectors_preserved": vector is not None})

    def undo(self, entry: JournalEntry) -> bool:
        """Reverse a single journal entry where possible.

        Returns True on success, False if the entry is not undoable or the
        target state has already diverged in a way that makes undo unsafe.
        """
        if entry.op == "remember":
            ok = self.forget(entry.id)
            if ok:
                self._journal("undo", entry.id,
                              meta={"of": entry.ts, "of_op": entry.op})
            return ok

        if entry.op == "forget":
            if entry.before is None:
                return False
            if self._get_by_id(entry.id) is not None:
                return False  # id already exists, refuse to clobber
            mem = Memory.from_dict(entry.before)
            self.collection.add(
                ids=[mem.id],
                documents=[mem.text],
                metadatas=[mem.to_metadata()],
            )
            self._journal("undo", mem.id, after=mem.to_dict(),
                          meta={"of": entry.ts, "of_op": entry.op})
            return True

        if entry.op == "edit":
            if entry.before is None:
                return False
            current = self._get_by_id(entry.id)
            if current is None:
                return False  # memory has since been forgotten
            restored = Memory.from_dict(entry.before)
            # Always pass documents: the edit may have changed the text, and
            # re-embedding unchanged text is harmless.
            self.collection.update(
                ids=[entry.id],
                documents=[restored.text],
                metadatas=[restored.to_metadata()],
            )
            self._journal("undo", entry.id,
                          before=current.to_dict(), after=restored.to_dict(),
                          meta={"of": entry.ts, "of_op": entry.op})
            return True

        if entry.op == "supersede":
            ok = self.restore(entry.id)
            if ok:
                self._journal("undo", entry.id,
                              meta={"of": entry.ts, "of_op": entry.op})
            return ok

        if entry.op == "restore":
            superseder = entry.meta.get("superseder_id")
            if not superseder or self._get_by_id(superseder) is None:
                return False
            self.supersede(old_id=entry.id, new_id=superseder)
            self._journal("undo", entry.id,
                          meta={"of": entry.ts, "of_op": entry.op})
            return True

        if entry.op == "link":
            removed = self._unlink_raw(
                entry.meta.get("src_id", entry.id),
                entry.meta.get("dst_id", ""),
                entry.meta.get("rel"),
            )
            if removed:
                self._journal("undo", entry.id,
                              meta={"of": entry.ts, "of_op": entry.op})
            return bool(removed)

        if entry.op == "unlink":
            src = entry.meta.get("src_id", entry.id)
            dst = entry.meta.get("dst_id", "")
            # A rel=None unlink may have removed several differently-typed
            # edges; meta["removed_rels"] records them. Entries written
            # before that field existed fall back to the single rel.
            rels = entry.meta.get("removed_rels") \
                or [entry.meta.get("rel") or "related"]
            added_any = False
            try:
                for r in rels:
                    if self._link_raw(src, dst, r):
                        added_any = True
            except KeyError:
                return False  # an endpoint has since been forgotten
            if added_any:
                self._journal("undo", entry.id,
                              meta={"of": entry.ts, "of_op": entry.op})
            return added_any

        # reflect / decay / import / export / undo themselves: not undoable
        return False

    def _get_by_id(self, memory_id: str) -> Memory | None:
        res = self.collection.get(
            ids=[memory_id],
            include=["documents", "metadatas"],
        )
        if not res["ids"]:
            return None
        return Memory.from_record(res["ids"][0], res["documents"][0], res["metadatas"][0])

    def _get_all_memories(self) -> list[Memory]:
        if self.collection.count() == 0:
            return []
        res = self.collection.get(include=["documents", "metadatas"])
        return [
            Memory.from_record(i, d, m)
            for i, d, m in zip(res["ids"], res["documents"], res["metadatas"])
        ]

    def _touch(self, mem: Memory) -> None:
        mem.last_accessed = time.time()
        mem.access_count += 1
        self.collection.update(ids=[mem.id], metadatas=[mem.to_metadata()])

    def _touch_many(self, mems: list[Memory]) -> None:
        """Bump access tracking for several memories in ONE Chroma write.

        A recall hitting ``k`` memories used to issue ``k`` separate updates
        (and, under the HTTP server's single lock, serialise every read behind
        them). Batching collapses that to a single read-modify-write.
        """
        if not mems:
            return
        now = time.time()
        ids: list[str] = []
        metas: list[dict[str, Any]] = []
        for mem in mems:
            mem.last_accessed = now
            mem.access_count += 1
            ids.append(mem.id)
            metas.append(mem.to_metadata())
        self.collection.update(ids=ids, metadatas=metas)

    @staticmethod
    def _emb_map(res: dict) -> dict[str, list[float]]:
        """Build {id: embedding} from a Chroma query result (when embeddings
        were requested)."""
        raw = res.get("embeddings")
        if raw is None:
            return {}
        ids = res["ids"][0]
        embs = raw[0]
        return {ids[i]: list(embs[i]) for i in range(len(ids)) if i < len(embs)}

    def _mmr_select(
        self,
        scored: list[tuple[Memory, float]],
        k: int,
        emb_by_id: dict[str, list[float]],
        diversity: float | None,
        dedup_threshold: float | None,
    ) -> list[tuple[Memory, float]]:
        """Re-select up to *k* results balancing relevance and novelty (MMR) and/or
        hard-dropping near-duplicates of already-selected results.

        ``diversity`` (λ): score = λ·relevance − (1−λ)·max_cosine_to_selected.
        ``dedup_threshold``: skip a candidate whose cosine to any selected
        result is ≥ the threshold. With neither set this is a plain top-k.
        """
        lam = diversity
        # Min-max normalize relevance to [0,1] so the MMR trade-off is on the
        # same scale as the cosine novelty penalty. Critical for RRF scores
        # (which live around 1/rrf_k ≈ 0.016, so without this the novelty term
        # would dominate and scramble the order); a harmless uniform rescale for
        # the already-[0,1] weighted scores.
        rel_vals = [s for _, s in scored]
        lo = min(rel_vals) if rel_vals else 0.0
        hi = max(rel_vals) if rel_vals else 1.0
        span = hi - lo

        def _rel(s: float) -> float:
            return (s - lo) / span if span > 1e-12 else 1.0

        selected: list[tuple[Memory, float]] = []
        sel_embs: list[list[float]] = []
        pool = list(scored)
        while pool and len(selected) < k:
            best_idx = -1
            best_val: float | None = None
            for idx, (mem, score) in enumerate(pool):
                emb = emb_by_id.get(mem.id)
                max_sim = (
                    max((_cosine_sim(emb, e) for e in sel_embs), default=0.0)
                    if emb is not None else 0.0
                )
                if (dedup_threshold is not None and sel_embs and emb is not None
                        and max_sim >= dedup_threshold):
                    continue
                if lam is not None:
                    val = lam * _rel(score) - (1.0 - lam) * max_sim
                else:
                    val = score
                if best_val is None or val > best_val:
                    best_val = val
                    best_idx = idx
            if best_idx < 0:
                break  # every remaining candidate was a near-duplicate
            mem, score = pool.pop(best_idx)
            selected.append((mem, score))
            emb = emb_by_id.get(mem.id)
            if emb is not None:
                sel_embs.append(emb)
        return selected

    def _check_conflicts(
        self,
        mem: Memory,
        threshold: float,
        contradiction_fn: ConflictFn | None = None,
    ) -> list[Conflict]:
        count = self.collection.count()
        if count <= 1:
            return []
        n_query = min(count, 12)
        res = self.collection.query(
            query_texts=[mem.text],
            n_results=n_query,
            include=["documents", "metadatas", "distances"],
        )
        conflicts: list[Conflict] = []
        for mid, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0],
            res["metadatas"][0], res["distances"][0],
        ):
            if mid == mem.id:
                continue
            candidate = Memory.from_record(mid, doc, meta)
            if candidate.superseded_by:
                continue
            if candidate.type != mem.type:
                continue
            sim = 1.0 - dist
            if sim < threshold:
                continue
            # Tag overlap guard
            if mem.tags and candidate.tags and not (set(mem.tags) & set(candidate.tags)):
                continue
            cfn = contradiction_fn or self._contradiction_fn
            if (mem.polarity != 0 and candidate.polarity != 0
                    and mem.polarity != candidate.polarity):
                kind, reason = "contradiction", "polarity_diff"
            elif _negation_diff(mem.text, candidate.text):
                kind, reason = "contradiction", "negation_diff"
            elif cfn and cfn(mem, candidate):
                kind, reason = "contradiction", "custom_fn"
            else:
                kind, reason = "duplicate", "similarity"
            conflicts.append(Conflict(
                a=mem, b=candidate,
                similarity=round(sim, 4),
                kind=kind, reason=reason,
            ))
        return conflicts

    def _recency(self, mem: Memory, w: HybridWeights, now: float) -> float:
        basis = mem.created_at if w.recency_basis == "created" else mem.last_accessed
        age_d = max(0.0, (now - basis) / 86_400.0)
        return math.exp(-w.decay_rate * age_d)

    def _semantic_filter(
        self,
        res: dict,
        tag: str | None,
        include_superseded: bool,
        polarity_weight: float = 0.0,
        min_cosine: float | None = None,
        explanations: dict | None = None,
    ) -> list[tuple[Memory, float]]:
        out: list[tuple[Memory, float]] = []
        for mid, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0],
            res["metadatas"][0], res["distances"][0],
        ):
            mem = Memory.from_record(mid, doc, meta)
            if tag and tag not in mem.tags:
                continue
            if not include_superseded and mem.superseded_by:
                continue
            cosine = 1.0 - dist
            if min_cosine is not None and cosine < min_cosine:
                continue
            score = cosine + polarity_weight * mem.polarity
            if explanations is not None:
                explanations[mem.id] = {
                    "mode": "semantic", "cosine": round(cosine, 4),
                    "polarity": mem.polarity, "score": round(score, 4),
                }
            out.append((mem, score))
        if polarity_weight:
            out.sort(key=lambda x: x[1], reverse=True)
        return out

    def _hybrid_score(
        self,
        res: dict,
        query: str,
        tag: str | None,
        include_superseded: bool,
        weights: HybridWeights | None,
        min_cosine: float | None = None,
        explanations: dict | None = None,
    ) -> list[tuple[Memory, float]]:
        w = weights or self._hybrid_weights or HybridWeights()
        docs = res["documents"][0]
        bm25 = _bm25_score_pool(query, docs)
        now  = time.time()

        # When the query produces no lexical signal at all (e.g. a non-Latin or
        # all-stopword query that matches no document term), drop the lexical
        # weight and renormalise the remaining core weights so scores are not
        # artificially depressed and stay comparable.
        cw, lw, rw, iw = w.cosine, w.lexical, w.recency, w.importance
        if lw > 0 and not any(s > 0 for s in bm25):
            core = cw + lw + rw + iw
            # Only renormalise when other signals exist to absorb the lexical
            # weight; for a lexical-only config there is nothing to scale into,
            # so leave the weights untouched rather than zeroing them all.
            if core > lw:
                scale = core / (core - lw)
                cw, rw, iw, lw = cw * scale, rw * scale, iw * scale, 0.0

        candidates: list[tuple[Memory, float]] = []
        for i, (mid, doc, meta, dist) in enumerate(zip(
            res["ids"][0], res["documents"][0],
            res["metadatas"][0], res["distances"][0],
        )):
            mem = Memory.from_record(mid, doc, meta)
            if tag and tag not in mem.tags:
                continue
            if not include_superseded and mem.superseded_by:
                continue
            cosine  = 1.0 - dist
            if min_cosine is not None and cosine < min_cosine:
                continue
            lexical = bm25[i] if i < len(bm25) else 0.0
            recency = self._recency(mem, w, now)
            score   = (cw * cosine + lw * lexical
                       + rw * recency + iw * mem.importance
                       + w.polarity_weight * mem.polarity)
            if explanations is not None:
                explanations[mem.id] = {
                    "mode": "hybrid", "fusion": "weighted",
                    "cosine": round(cosine, 4), "lexical": round(lexical, 4),
                    "recency": round(recency, 4),
                    "importance": round(mem.importance, 4), "polarity": mem.polarity,
                    "weights": {"cosine": round(cw, 4), "lexical": round(lw, 4),
                                "recency": round(rw, 4), "importance": round(iw, 4),
                                "polarity": w.polarity_weight},
                    "score": round(score, 4),
                }
            candidates.append((mem, score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates

    def _rrf_score(
        self,
        res: dict,
        query: str,
        tag: str | None,
        include_superseded: bool,
        weights: HybridWeights | None,
        min_cosine: float | None = None,
        explanations: dict | None = None,
        rrf_k: int = 60,
    ) -> list[tuple[Memory, float]]:
        """Reciprocal Rank Fusion of the hybrid signals.

        Each signal (cosine, lexical, recency, importance) ranks the candidate
        pool independently; the fused score is ``Σ weight_s / (rrf_k + rank_s)``.
        Because it consumes only ordinal ranks it is scale-free — immune to the
        BM25 pool-normalisation magnitude and the cosine-vs-lexical scale
        mismatch that the weighted blend mixes. Polarity stays a tiny additive
        nudge. (Ranks are pool-relative, so scores are comparable across queries
        only for identically-pooled fan-outs, not arbitrary recalls.)
        """
        w = weights or self._hybrid_weights or HybridWeights()
        ids   = res["ids"][0]
        docs  = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        bm25  = _bm25_score_pool(query, docs)
        now   = time.time()

        rows: list[dict[str, Any]] = []
        for i in range(len(ids)):
            mem = Memory.from_record(ids[i], docs[i], metas[i])
            if tag and tag not in mem.tags:
                continue
            if not include_superseded and mem.superseded_by:
                continue
            cosine = 1.0 - dists[i]
            if min_cosine is not None and cosine < min_cosine:
                continue
            rows.append({
                "mem": mem, "cosine": cosine,
                "lexical": bm25[i] if i < len(bm25) else 0.0,
                "recency": self._recency(mem, w, now),
                "importance": mem.importance,
            })
        if not rows:
            return []

        n = len(rows)
        signals = [
            ("cosine", w.cosine), ("lexical", w.lexical),
            ("recency", w.recency), ("importance", w.importance),
        ]
        ranks: dict[str, list[int]] = {}
        for name, wt in signals:
            if wt <= 0:
                continue
            order = sorted(range(n), key=lambda j: rows[j][name], reverse=True)
            r = [0] * n
            for pos, j in enumerate(order):
                r[j] = pos
            ranks[name] = r

        out: list[tuple[Memory, float]] = []
        for i, row in enumerate(rows):
            mem = row["mem"]
            score = 0.0
            contrib: dict[str, Any] = {}
            for name, wt in signals:
                if wt <= 0:
                    continue
                part = wt / (rrf_k + ranks[name][i])
                score += part
                contrib[name] = {"rank": ranks[name][i], "contribution": round(part, 6)}
            score += w.polarity_weight * mem.polarity / (rrf_k + 1)
            if explanations is not None:
                explanations[mem.id] = {
                    "mode": "hybrid", "fusion": "rrf", "rrf_k": rrf_k,
                    "polarity": mem.polarity, "signals": contrib,
                    "score": round(score, 6),
                }
            out.append((mem, score))
        out.sort(key=lambda x: x[1], reverse=True)
        return out

    def _expand_links(
        self,
        out: list[tuple[Memory, float]],
        spec: ExpandSpec,
        include_superseded: bool,
        explanations: dict | None = None,
    ) -> list[tuple[Memory, float]]:
        """Breadth-first graph expansion honouring ``spec.depth`` (multi-hop) with
        a per-hop ``spec.decay`` applied to the assigned ``spec.score``."""
        seen = {m.id for m, _ in out}
        extra: list[tuple[Memory, float]] = []
        added = 0
        frontier = [m.id for m, _ in out]
        for hop in range(1, spec.depth + 1):
            if added >= spec.cap or not frontier:
                break
            hop_score = spec.score * (spec.decay ** (hop - 1))
            next_frontier: list[str] = []
            for mid in frontier:
                if added >= spec.cap:
                    break
                src = self._get_by_id(mid)
                if src is None:
                    continue
                for lnk in src.links:
                    if lnk.rel not in spec.rels:
                        continue
                    if lnk.to in seen:
                        continue
                    nb = self._get_by_id(lnk.to)
                    if nb is None:
                        continue
                    if not include_superseded and nb.superseded_by:
                        continue
                    extra.append((nb, hop_score))
                    if explanations is not None:
                        explanations[nb.id] = {
                            "source": "graph_expansion", "rel": lnk.rel,
                            "hop": hop, "score": round(hop_score, 4),
                        }
                    seen.add(lnk.to)
                    next_frontier.append(lnk.to)
                    added += 1
                    if added >= spec.cap:
                        break
            frontier = next_frontier
        return out + extra
