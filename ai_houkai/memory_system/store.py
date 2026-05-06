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

import json
import math
import re
import time
import uuid
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

MemoryType = Literal["episodic", "semantic", "procedural", "feedback"]
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
    ) -> None:
        self.client = chromadb.PersistentClient(
            path=path, settings=Settings(anonymized_telemetry=False)
        )
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        self.collection = self.client.get_or_create_collection(
            name=collection,
            embedding_function=embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
        self._conflict_policy    = conflict_policy
        self._conflict_threshold = conflict_threshold
        self._contradiction_fn   = contradiction_fn
        self._hybrid_weights     = hybrid_weights

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
                    raise ConflictError(conflicts)

        return mem

    def forget(self, memory_id: str) -> bool:
        existing = self.collection.get(ids=[memory_id])
        if not existing["ids"]:
            return False
        self.collection.delete(ids=[memory_id])
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

        old.superseded_by = new_id
        old.superseded_at = time.time()
        self.collection.update(ids=[old_id], metadatas=[old.to_metadata()])
        self.link(src_id=new_id, dst_id=old_id, rel="supersedes")

    def restore(self, memory_id: str) -> bool:
        """Clear the superseded_by marker on a memory (undo a supersede)."""
        mem = self._get_by_id(memory_id)
        if mem is None or not mem.superseded_by:
            return False
        superseder_id = mem.superseded_by
        mem.superseded_by = ""
        mem.superseded_at = 0.0
        self.collection.update(ids=[memory_id], metadatas=[mem.to_metadata()])
        # Remove the "supersedes" edge from the superseder
        self.unlink(src_id=superseder_id, dst_id=memory_id, rel="supersedes")
        return True

    def link(self, src_id: str, dst_id: str, rel: str = "related") -> None:
        """Add a directed link src → dst with the given relation (idempotent)."""
        if src_id == dst_id:
            raise ValueError("Cannot link a memory to itself")
        src = self._get_by_id(src_id)
        if src is None:
            raise KeyError(f"src_id {src_id!r} not found")
        for existing in src.links:
            if existing.to == dst_id and existing.rel == rel:
                return  # already exists
        src.links.append(Link(to=dst_id, rel=rel))
        self.collection.update(ids=[src_id], metadatas=[src.to_metadata()])

    def unlink(self, src_id: str, dst_id: str, rel: str | None = None) -> int:
        """Remove link(s) from src to dst. *rel=None* removes all rels.
        Returns number of links removed."""
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
