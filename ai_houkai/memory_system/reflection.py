"""Reflection engine — cluster episodic memories and synthesise semantic ones.

Algorithm
1. Fetch all episodic memories together with their stored HNSW embeddings.
2. Compute pairwise cosine similarity.
3. Greedy single-linkage clustering: seed on the highest-importance memory,
   absorb neighbours above `similarity_threshold`, repeat until exhausted.
4. For each cluster with ≥ min_cluster_size members, call `summarizer()` and
   store a new semantic memory tagged with "reflection".
5. If `consolidate=True`, delete the source episodic memories afterwards.

Summarizer
By default an extractive summarizer is used (no LLM required): it orders
memories by importance, joins their texts, and trims to 512 chars.

You can inject any callable that takes ``list[Memory] → str``:

    def my_llm_summarizer(memories: list[Memory]) -> str:
        prompt = "\n".join(m.text for m in memories)
        return call_llm(f"Summarise: {prompt}")

    engine = ReflectionEngine(store, summarizer=my_llm_summarizer)

Usage
    from memory_system import MemoryStore, ReflectionEngine

    store  = MemoryStore(...)
    engine = ReflectionEngine(store)

    plan    = engine.reflect(dry_run=True)    # preview without writing
    created = engine.reflect(consolidate=True) # reflect + remove sources
"""

from __future__ import annotations

import math
import uuid
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .store import Memory, MemoryStore


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _default_summarizer(memories: "list[Memory]") -> str:
    """Extractive: most-important first, concatenated, capped at 512 chars."""
    ordered = sorted(memories, key=lambda m: m.importance, reverse=True)
    body = " | ".join(m.text for m in ordered)
    prefix = f"[Reflection ×{len(memories)}] "
    return (prefix + body)[: 512]


class ReflectionEngine:
    """Cluster episodic memories and condense them into semantic ones."""

    def __init__(
        self,
        store: "MemoryStore",
        similarity_threshold: float = 0.75,
        min_cluster_size: int = 2,
        summarizer: Callable[["list[Memory]"], str] | None = None,
    ) -> None:
        """
        Parameters
        store
            The MemoryStore to operate on.
        similarity_threshold
            Cosine similarity ≥ this value groups two memories together.
            0.75 works well for thematically similar short sentences.
        min_cluster_size
            Clusters smaller than this are ignored.
        summarizer
            Callable(list[Memory]) → str used to produce the reflection text.
            Defaults to the built-in extractive summarizer.
        """
        self.store = store
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size
        self.summarizer = summarizer or _default_summarizer


    def reflect(
        self,
        dry_run: bool = False,
        consolidate: bool = False,
    ) -> "list[Memory]":
        """
        Cluster episodic memories and create semantic reflections.

        Parameters
        dry_run
            If True, return candidate Memory objects without writing anything.
        consolidate
            If True (and not dry_run), delete source episodic memories after
            creating the reflection.  Use carefully — data is lost.

        Returns
        Newly created (or candidate) semantic Memory objects, one per cluster.
        """
        mems, embeddings = self._fetch_episodic()
        if len(mems) < self.min_cluster_size:
            return []

        clusters = self._cluster(mems, embeddings)
        created: "list[Memory]" = []

        for idx_list in clusters:
            group = [mems[i] for i in idx_list]
            text  = self.summarizer(group)

            # Merge tags: "reflection" first, then unique tags from sources
            all_tags: list[str] = ["reflection"]
            seen: set[str] = {"reflection"}
            for m in group:
                for tag in m.tags:
                    if tag not in seen:
                        all_tags.append(tag)
                        seen.add(tag)

            importance = round(
                sum(m.importance for m in group) / len(group), 3
            )

            if dry_run:
                from .store import Memory as _Memory
                candidate = _Memory(
                    id=str(uuid.uuid4()),
                    text=text,
                    type="semantic",
                    tags=all_tags,
                    importance=importance,
                    source="reflection/dry-run",
                )
                created.append(candidate)
            else:
                new_mem = self.store.remember(
                    text=text,
                    type="semantic",
                    tags=all_tags,
                    importance=importance,
                    source="reflection",
                )
                created.append(new_mem)
                if consolidate:
                    for m in group:
                        self.store.forget(m.id)

        return created

    def clusters(self) -> "list[list[Memory]]":
        """Return detected clusters without writing anything (for inspection)."""
        mems, embeddings = self._fetch_episodic()
        if len(mems) < self.min_cluster_size:
            return []
        return [
            [mems[i] for i in idx_list]
            for idx_list in self._cluster(mems, embeddings)
        ]

    def _fetch_episodic(
        self,
    ) -> "tuple[list[Memory], list[list[float]]]":
        from .store import Memory as _Memory

        res = self.store.collection.get(
            where={"type": "episodic"},
            include=["embeddings", "documents", "metadatas"],
        )
        ids   = res.get("ids") or []
        docs  = res.get("documents") or []
        metas = res.get("metadatas") or []
        raw   = res.get("embeddings")
        embs  = [] if raw is None else raw  # numpy arrays are truthy-ambiguous

        mems = [
            _Memory.from_record(i, d, m)
            for i, d, m in zip(ids, docs, metas)
        ]
        return mems, [list(e) for e in embs]

    def _cluster(
        self,
        mems: "list[Memory]",
        embeddings: list[list[float]],
    ) -> list[list[int]]:
        """Greedy single-linkage: seed on highest importance, absorb neighbours."""
        n = len(mems)
        used  = [False] * n
        order = sorted(range(n), key=lambda i: mems[i].importance, reverse=True)
        result: list[list[int]] = []

        for seed in order:
            if used[seed]:
                continue
            cluster = [seed]
            used[seed] = True
            for j in range(n):
                if used[j]:
                    continue
                if _cosine(embeddings[seed], embeddings[j]) >= self.similarity_threshold:
                    cluster.append(j)
                    used[j] = True
            if len(cluster) >= self.min_cluster_size:
                result.append(cluster)

        return result
