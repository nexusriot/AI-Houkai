"""Decay engine — prune memories that have become old and unimportant.

Score formula

    score(m) = importance × exp(-decay_rate × days_since_last_access)

                importance  decay_rate  days    score
                0.9         0.1          7 d    0.45   ← kept
                0.9         0.1         30 d    0.04   ← pruned
                0.5         0.1          1 d    0.45   ← kept
                0.1         0.1          1 d    0.09   ← borderline

Default parameters give a half-life of ~7 days for a 0.5-importance memory.

Usage
    from ai_houkai.memory_system import MemoryStore, DecayEngine

    store  = MemoryStore(...)
    engine = DecayEngine(store)

    candidates = engine.prune(dry_run=True)   # see what would be removed
    removed    = engine.prune()               # actually remove them
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING
from contextlib import nullcontext

from .store import Memory, MemoryStore


class DecayEngine:
    """Scores and prunes memories using an exponential decay formula."""

    def __init__(
        self,
        store: "MemoryStore",
        decay_rate: float = 0.1,
        min_score: float = 0.05,
        protect_types: tuple[str, ...] = ("procedural",),
        frequency_weight: float = 0.0,
    ) -> None:
        """
        Parameters
        store
            The MemoryStore to operate on.
        decay_rate
            λ in exp(-λ × days).  Higher values → faster forgetting.
            0.1  ≈ half-life 7 days for importance=0.5
            0.05 ≈ half-life 14 days
            0.01 ≈ half-life 69 days
        min_score
            Memories with score < min_score are candidates for pruning.
        protect_types
            Memory types never pruned regardless of score.
            Defaults to ("procedural",) — runbooks should not be forgotten.
        frequency_weight
            Reinforcement: how strongly a memory's recall count resists decay.
            The score is multiplied by ``1 + frequency_weight × ln(1 + access_count)``,
            so a frequently-recalled memory ages out more slowly than an
            untouched one of equal importance and age. ``0.0`` (the default)
            disables reinforcement — scores match the recency-only behaviour.
            ``0.3`` roughly doubles the effective score of a memory recalled
            ~20 times.
        """
        self.store = store
        self.decay_rate = decay_rate
        self.min_score = min_score
        self.protect_types = protect_types
        self.frequency_weight = frequency_weight


    def score(self, memory: "Memory", now: float | None = None) -> float:
        """Return the current decay score for a single memory.

        ``importance × exp(-decay_rate × days) × reinforcement`` where the
        reinforcement factor is ``1 + frequency_weight × ln(1 + access_count)``
        (``1.0`` when ``frequency_weight == 0``). With reinforcement enabled the
        score can exceed ``importance``; ``min_score`` is interpreted against
        the reinforced value.
        """
        t = now if now is not None else time.time()
        days = max(0.0, (t - memory.last_accessed) / 86_400.0)
        base = memory.importance * math.exp(-self.decay_rate * days)
        if self.frequency_weight:
            base *= 1.0 + self.frequency_weight * math.log1p(max(0, memory.access_count))
        return base

    def score_all(
        self, now: float | None = None
    ) -> list[tuple["Memory", float]]:
        """Return (memory, score) for every memory, sorted score descending."""
        t = now if now is not None else time.time()
        # include_superseded=True so soft-deleted memories also age out —
        # otherwise they linger in the store forever.
        pairs = [
            (m, self.score(m, t))
            for m in self.store.list_recent(limit=100_000, include_superseded=True)
        ]
        pairs.sort(key=lambda p: p[1], reverse=True)
        return pairs

    def prune(
        self,
        dry_run: bool = False,
        now: float | None = None,
    ) -> list["Memory"]:
        """
        Remove memories whose score has dropped below min_score.

        Parameters
        dry_run
            If True, return candidates without deleting anything.
        now
            Override current time (useful in tests / simulations).

        Returns
        List of Memory objects that were (or would be) pruned.
        """
        t = now if now is not None else time.time()
        pruned: list["Memory"] = []

        ctx = self.store.as_actor("decay") if not dry_run else nullcontext()
        with ctx:
            for mem, score in self.score_all(t):
                if mem.type in self.protect_types:
                    continue
                if score < self.min_score:
                    if not dry_run:
                        self.store.forget(mem.id)
                    pruned.append(mem)

        return pruned
