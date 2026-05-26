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
    cosine:     float = 0.55
    lexical:    float = 0.20
    recency:    float = 0.15
    importance: float = 0.10
    decay_rate: float = 0.10   # λ — shared with DecayEngine

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


def _tokenize(text: str) -> list[str]:
    # Strip apostrophes so "don't" → "dont", "won't" → "wont", matching _NEG
    normalized = text.lower().replace("’", "").replace("'", "")
    return re.findall(r"\b\w+\b", normalized)


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
        actor: str = "lib",
        journal_enabled: bool = True,
        journal_path: str | None = None,
        journal_rotate_mb: int = 64,
        journal_keep_days: int = 90,
    ) -> None:
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
        importance: float = 0.5,
        source: str | None = None,
        *,
        polarity: int = 0,
        on_conflict: Literal["ignore", "warn", "supersede", "raise"] | None = None,
        contradiction_fn: ConflictFn | None = None,
    ) -> Memory:
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

    def recall(
        self,
        query: str,
        k: int = 5,
        type: MemoryType | None = None,
        tag: str | None = None,
        min_importance: float | None = None,
        *,
        mode: Literal["semantic", "hybrid"] = "semantic",
        weights: HybridWeights | None = None,
        overfetch: int = 4,
        expand: ExpandSpec | None = None,
        include_superseded: bool = False,
    ) -> list[tuple[Memory, float]]:
        count = self.collection.count()
        if k <= 0 or count == 0:
            return []

        n_fetch = k if (mode == "semantic" and include_superseded) else min(k * overfetch, count)
        n_fetch = max(n_fetch, k)

        where: dict[str, Any] = {}
        if type:
            where["type"] = type
        if min_importance is not None:
            where["importance"] = {"$gte": min_importance}

        res = self.collection.query(
            query_texts=[query],
            n_results=n_fetch,
            where=where or None,
        )

        if mode == "hybrid":
            out = self._hybrid_score(res, query, k, tag, include_superseded, weights)
        else:
            out = self._semantic_filter(res, k, tag, include_superseded)

        for mem, _ in out:
            self._touch(mem)

        # Expand via outgoing links
        if expand is not None and expand.cap > 0:
            out = self._expand_links(out, expand, include_superseded)

        return out

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
                if _negation_diff(mem_a.text, mem_b.text):
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
        if src_id == dst_id:
            raise ValueError("Cannot link a memory to itself")
        src = self._get_by_id(src_id)
        if src is None:
            raise KeyError(f"src_id {src_id!r} not found")
        for existing in src.links:
            if existing.to == dst_id and existing.rel == rel:
                return False
        src.links.append(Link(to=dst_id, rel=rel))
        self.collection.update(ids=[src_id], metadatas=[src.to_metadata()])
        return True

    def unlink(self, src_id: str, dst_id: str, rel: str | None = None) -> int:
        """Remove link(s) from src to dst. *rel=None* removes all rels.
        Returns number of links removed."""
        removed = self._unlink_raw(src_id, dst_id, rel)
        if removed > 0:
            self._journal(
                "unlink", src_id,
                meta={"src_id": src_id, "dst_id": dst_id,
                      "rel": rel, "removed": removed},
            )
        return removed

    def _unlink_raw(self, src_id: str, dst_id: str, rel: str | None) -> int:
        src = self._get_by_id(src_id)
        if src is None:
            return 0
        before = len(src.links)
        if rel is None:
            src.links = [l for l in src.links if l.to != dst_id]
        else:
            src.links = [l for l in src.links if not (l.to == dst_id and l.rel == rel)]
        removed = before - len(src.links)
        if removed > 0:
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
        visited: set[str] = {memory_id}
        frontier: list[str] = [memory_id]
        result:   list[tuple[Memory, str]] = []

        for _ in range(depth):
            next_frontier: list[str] = []
            for mid in frontier:
                # ── outgoing ──
                if direction in ("out", "both"):
                    mem = self._get_by_id(mid)
                    if mem is not None:
                        for lnk in mem.links:
                            if rel is not None and lnk.rel != rel:
                                continue
                            if lnk.to in visited:
                                continue
                            target = self._get_by_id(lnk.to)
                            if target is not None:
                                result.append((target, lnk.rel))
                                visited.add(lnk.to)
                                next_frontier.append(lnk.to)
                if direction in ("in", "both"):
                    for candidate in self._get_all_memories():
                        if candidate.id in visited:
                            continue
                        for lnk in candidate.links:
                            if lnk.to == mid:
                                if rel is not None and lnk.rel != rel:
                                    continue
                                result.append((candidate, lnk.rel))
                                visited.add(candidate.id)
                                next_frontier.append(candidate.id)
                                break
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
        visited: set[str] = set()

        def _visit(mid: str, remaining: int) -> None:
            if mid in visited:
                return
            visited.add(mid)
            mem = self._get_by_id(mid)
            if mem is None:
                return
            nodes[mid] = mem
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
            added = self._link_raw(
                entry.meta.get("src_id", entry.id),
                entry.meta.get("dst_id", ""),
                entry.meta.get("rel") or "related",
            )
            if added:
                self._journal("undo", entry.id,
                              meta={"of": entry.ts, "of_op": entry.op})
            return added

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
            if _negation_diff(mem.text, candidate.text):
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

    def _semantic_filter(
        self,
        res: dict,
        k: int,
        tag: str | None,
        include_superseded: bool,
    ) -> list[tuple[Memory, float]]:
        out: list[tuple[Memory, float]] = []
        for mid, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0],
            res["metadatas"][0], res["distances"][0],
        ):
            if len(out) >= k:
                break
            mem = Memory.from_record(mid, doc, meta)
            if tag and tag not in mem.tags:
                continue
            if not include_superseded and mem.superseded_by:
                continue
            out.append((mem, 1.0 - dist))
        return out

    def _hybrid_score(
        self,
        res: dict,
        query: str,
        k: int,
        tag: str | None,
        include_superseded: bool,
        weights: HybridWeights | None,
    ) -> list[tuple[Memory, float]]:
        w = weights or self._hybrid_weights or HybridWeights()
        docs = res["documents"][0]
        bm25 = _bm25_score_pool(query, docs)
        now  = time.time()

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
            lexical = bm25[i] if i < len(bm25) else 0.0
            age_d   = max(0.0, (now - mem.last_accessed) / 86_400.0)
            recency = math.exp(-w.decay_rate * age_d)
            score   = (w.cosine * cosine + w.lexical * lexical
                       + w.recency * recency + w.importance * mem.importance)
            candidates.append((mem, score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:k]

    def _expand_links(
        self,
        out: list[tuple[Memory, float]],
        spec: ExpandSpec,
        include_superseded: bool,
    ) -> list[tuple[Memory, float]]:
        added = 0
        seen  = {m.id for m, _ in out}
        extra: list[tuple[Memory, float]] = []
        for base_mem, _ in out:
            if added >= spec.cap:
                break
            src = self._get_by_id(base_mem.id)
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
                extra.append((nb, spec.score))
                seen.add(lnk.to)
                added += 1
                if added >= spec.cap:
                    break
        return out + extra
