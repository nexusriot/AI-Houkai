"""Reflection engine — cluster episodic memories and synthesise semantic ones.

Algorithm
1. Fetch all candidate memories (``types``, default episodic) with their
   stored HNSW embeddings.
2. Compute pairwise cosine similarity.
3. Greedy single-linkage clustering: seed on the highest-importance memory,
   absorb neighbours above `similarity_threshold`, repeat until exhausted.
4. For each cluster with ≥ min_cluster_size members, call `summarizer()` and
   store a new semantic memory tagged with "reflection".
5. consolidate behaviour (default False):
     False        — leave source episodics untouched
     True         — soft-delete: mark sources as superseded_by the summary
                    and add a "derived_from" link from summary → source
     "hard"       — hard-delete: remove source episodics (old behaviour)

Summarizer
By default an extractive summarizer is used (no LLM required): it orders
memories by importance, joins their texts, and trims to 512 chars.

    def my_llm_summarizer(memories: list[Memory]) -> str:
        prompt = "\\n".join(m.text for m in memories)
        return call_llm(f"Summarise: {prompt}")

    engine = ReflectionEngine(store, summarizer=my_llm_summarizer)

Usage
    from ai_houkai.memory_system import MemoryStore, ReflectionEngine

    store  = MemoryStore(...)
    engine = ReflectionEngine(store)

    plan    = engine.reflect(dry_run=True)        # preview without writing
    created = engine.reflect(consolidate=True)    # reflect + soft-delete sources
    created = engine.reflect(consolidate="hard")  # reflect + hard-delete sources
"""

from __future__ import annotations

import math
import uuid
from typing import Callable, Literal

from contextlib import nullcontext

from .store import Memory, MemoryStore


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _level_of(mem: "Memory") -> int:
    """Reflection tier of a memory: 0 for a raw one, N for a ``level:N`` tag.

    Encoded as a tag rather than a metadata field so it round-trips through
    export/import and old archives read as level 0.
    """
    for tag in mem.tags:
        if tag.startswith("level:"):
            try:
                return int(tag.split(":", 1)[1])
            except ValueError:
                continue
    return 0


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
        types: tuple[str, ...] = ("episodic",),
        max_level: int = 1,
    ) -> None:
        """
        types
            Which memory types to cluster. Historically hard-coded to
            ``("episodic",)``, which meant semantic memories — including
            reflections themselves — never consolidated, so a long-lived store
            accumulated summaries without bound, and ``feedback`` /
            ``procedural`` never benefited at all.
        max_level
            How many tiers of reflection-of-reflections to allow. Each summary
            is tagged ``level:N``; a cluster whose members are already at
            ``max_level`` is skipped. 1 (default) reproduces the old behaviour
            — summaries are produced but never re-summarised. Raise it to build
            a hierarchy; the cap is what stops runaway re-summarisation from
            eating the store.
        """
        self.store = store
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size
        self.summarizer = summarizer or _default_summarizer
        self.types = tuple(types)
        self.max_level = max(1, max_level)

    def reflect(
        self,
        dry_run: bool = False,
        consolidate: bool | Literal["hard"] = False,
    ) -> "list[Memory]":
        """Cluster episodic memories and create semantic reflections.

        Parameters
        dry_run
            If True, return candidate Memory objects without writing anything.
        consolidate
            False  — leave sources untouched (default).
            True   — soft-delete: supersede sources and link summary → source
                     with rel="derived_from". Sources remain in the store but
                     are hidden from default recall/list_recent queries.
            "hard" — hard-delete: permanently remove source episodics (old
                     behaviour, data is lost).

        Returns
        Newly created (or candidate) semantic Memory objects, one per cluster.
        """
        mems, embeddings = self._fetch_candidates()
        if len(mems) < self.min_cluster_size:
            return []

        clusters = self._cluster(mems, embeddings)
        created: "list[Memory]" = []

        ctx = self.store.as_actor("reflection") if not dry_run else nullcontext()
        with ctx:
            for idx_list in clusters:
                group = [mems[i] for i in idx_list]
                text  = self.summarizer(group)

                # The new summary sits one tier above its deepest member, so a
                # hierarchy can be walked (and re-reflected) by level.
                level = max((_level_of(m) for m in group), default=0) + 1
                level_tag = f"level:{level}"
                all_tags: list[str] = ["reflection", level_tag]
                seen: set[str] = {"reflection", level_tag}
                for m in group:
                    for tag in m.tags:
                        # Do not inherit the members' level tags: the summary
                        # has its own tier.
                        if tag.startswith("level:") or tag in seen:
                            continue
                        all_tags.append(tag)
                        seen.add(tag)

                importance = round(
                    sum(m.importance for m in group) / len(group), 3
                )

                if dry_run:
                    candidate = Memory(
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

                    if consolidate == "hard":
                        for m in group:
                            self.store.forget(m.id)
                    elif consolidate:
                        for m in group:
                            # Soft-delete: supersede + derived_from link
                            self.store.supersede(old_id=m.id, new_id=new_mem.id)
                            self.store.link(
                                src_id=new_mem.id,
                                dst_id=m.id,
                                rel="derived_from",
                            )

        return created

    def clusters(self) -> "list[list[Memory]]":
        """Return detected clusters without writing anything (for inspection)."""
        mems, embeddings = self._fetch_candidates()
        if len(mems) < self.min_cluster_size:
            return []
        return [
            [mems[i] for i in idx_list]
            for idx_list in self._cluster(mems, embeddings)
        ]

    def _fetch_candidates(
        self,
    ) -> "tuple[list[Memory], list[list[float]]]":
        # A single-element $in is rejected by some Chroma versions, and the
        # common case is one type, so build the narrowest clause that works.
        where: dict = ({"type": self.types[0]} if len(self.types) == 1
                       else {"type": {"$in": list(self.types)}})
        res = self.store.collection.get(
            where=where,
            include=["embeddings", "documents", "metadatas"],
        )
        ids   = res.get("ids") or []
        docs  = res.get("documents") or []
        metas = res.get("metadatas") or []
        raw   = res.get("embeddings")
        embs  = [] if raw is None else raw  # numpy arrays are truthy-ambiguous

        # Skip episodics that were already consolidated (superseded by an
        # earlier reflection). Otherwise every run re-clusters the same
        # sources: with consolidate=False it emits duplicate summaries, and
        # with consolidate=soft it re-supersedes them under a fresh summary.
        mems: list[Memory] = []
        kept_embs: list[list[float]] = []
        for i, d, m, e in zip(ids, docs, metas, embs):
            mem = Memory.from_record(i, d, m)
            if mem.superseded_by:
                continue
            # A memory already at the deepest allowed tier must not be folded
            # into yet another summary — that is the runaway case max_level
            # exists to stop.
            if _level_of(mem) >= self.max_level:
                continue
            mems.append(mem)
            kept_embs.append(list(e))
        return mems, kept_embs

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
            # The cluster's effective polarity: the first non-zero polarity
            # absorbed. Comparing against the seed alone would let a neutral
            # seed bridge a +1 and a -1 into one blended summary.
            cluster_polarity = mems[seed].polarity
            for j in range(n):
                if used[j]:
                    continue
                # Never merge memories with explicitly opposite polarities:
                # a positive and a negative memory describe contradictory states.
                j_polarity = mems[j].polarity
                if cluster_polarity != 0 and j_polarity != 0 and cluster_polarity != j_polarity:
                    continue
                if _cosine(embeddings[seed], embeddings[j]) >= self.similarity_threshold:
                    cluster.append(j)
                    used[j] = True
                    if cluster_polarity == 0:
                        cluster_polarity = j_polarity
            if len(cluster) >= self.min_cluster_size:
                result.append(cluster)

        return result
