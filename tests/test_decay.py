"""Unit tests for memory_system.DecayEngine."""

from __future__ import annotations

import math
import time

import pytest

from memory_system import DecayEngine, Memory, MemoryStore


def _days_ago(days: float) -> float:
    """Return a timestamp `days` days in the past."""
    return time.time() - days * 86_400


def _store_aged(
    store: MemoryStore,
    text: str,
    importance: float,
    days_old: float,
    type: str = "semantic",
    tags: list[str] | None = None,
) -> Memory:
    """Store a memory and backdate its last_accessed timestamp."""
    mem = store.remember(text, type=type, tags=tags or [], importance=importance)
    mem.last_accessed = _days_ago(days_old)
    mem.created_at    = _days_ago(days_old)
    store.collection.update(ids=[mem.id], metadatas=[mem.to_metadata()])
    return mem


class TestScore:
    def test_fresh_memory_scores_near_importance(self, store: MemoryStore):
        engine = DecayEngine(store, decay_rate=0.1)
        mem = store.remember("fresh", importance=0.8)
        score = engine.score(mem)
        # last_accessed is ~now, so days≈0 → score ≈ importance
        assert score == pytest.approx(0.8, rel=0.01)

    def test_score_decreases_with_age(self, store: MemoryStore):
        engine = DecayEngine(store, decay_rate=0.1)
        mem7  = Memory(id="a", text="x", type="semantic", importance=0.9, last_accessed=_days_ago(7))
        mem30 = Memory(id="b", text="x", type="semantic", importance=0.9, last_accessed=_days_ago(30))
        assert engine.score(mem7) > engine.score(mem30)

    def test_score_formula_exact(self, store: MemoryStore):
        engine = DecayEngine(store, decay_rate=0.1)
        now = time.time()
        mem = Memory(id="x", text="t", type="semantic",
                     importance=0.9, last_accessed=now - 10 * 86_400)
        expected = 0.9 * math.exp(-0.1 * 10)
        assert engine.score(mem, now=now) == pytest.approx(expected, rel=1e-6)

    def test_score_never_exceeds_importance(self, store: MemoryStore):
        engine = DecayEngine(store, decay_rate=0.1)
        mem = store.remember("x", importance=0.7)
        assert engine.score(mem) <= 0.7 + 1e-9

    def test_custom_now_parameter(self, store: MemoryStore):
        """score() must use the supplied now rather than time.time()."""
        engine = DecayEngine(store, decay_rate=0.1)
        fixed_now = 1_000_000.0
        mem = Memory(id="x", text="t", type="semantic",
                     importance=1.0, last_accessed=fixed_now - 7 * 86_400)
        expected = math.exp(-0.1 * 7)
        assert engine.score(mem, now=fixed_now) == pytest.approx(expected, rel=1e-6)



class TestScoreAll:
    def test_returns_all_memories(self, store: MemoryStore):
        store.remember("a", importance=0.5)
        store.remember("b", importance=0.5)
        engine = DecayEngine(store)
        pairs = engine.score_all()
        assert len(pairs) == 2

    def test_sorted_descending(self, store: MemoryStore):
        _store_aged(store, "old low",  importance=0.1, days_old=30)
        _store_aged(store, "fresh high", importance=0.9, days_old=0)
        engine = DecayEngine(store)
        pairs = engine.score_all()
        scores = [s for _, s in pairs]
        assert scores == sorted(scores, reverse=True)


class TestPrune:
    def test_dry_run_does_not_delete(self, store: MemoryStore):
        _store_aged(store, "very old", importance=0.1, days_old=60)
        engine = DecayEngine(store, min_score=0.05)
        candidates = engine.prune(dry_run=True)
        assert len(candidates) > 0
        assert store.count() == 1   # nothing deleted

    def test_prune_removes_stale_memories(self, store: MemoryStore):
        _store_aged(store, "recent important", importance=0.9, days_old=1)
        _store_aged(store, "old unimportant",  importance=0.1, days_old=60)
        engine = DecayEngine(store, decay_rate=0.1, min_score=0.05)
        removed = engine.prune()
        assert len(removed) == 1
        assert removed[0].text == "old unimportant"
        assert store.count() == 1

    def test_prune_keeps_important_recent_memories(self, store: MemoryStore):
        _store_aged(store, "important recent", importance=0.9, days_old=1)
        engine = DecayEngine(store, min_score=0.05)
        removed = engine.prune()
        assert removed == []
        assert store.count() == 1

    def test_prune_returns_list_of_memory_objects(self, store: MemoryStore):
        _store_aged(store, "doomed", importance=0.05, days_old=90)
        engine = DecayEngine(store, min_score=0.05)
        removed = engine.prune()
        assert all(isinstance(m, Memory) for m in removed)

    def test_protect_types_never_pruned(self, store: MemoryStore):
        _store_aged(store, "old runbook", importance=0.1, days_old=90,
                    type="procedural")
        engine = DecayEngine(store, min_score=0.05,
                             protect_types=("procedural",))
        removed = engine.prune()
        assert removed == []
        assert store.count() == 1

    def test_prune_respects_custom_now(self, store: MemoryStore):
        """Memories accessed at t=0 should be pruned when 'now' is far future."""
        mem = store.remember("test", importance=0.5)
        # Set last_accessed to t=0 (Unix epoch)
        mem.last_accessed = 0.0
        store.collection.update(ids=[mem.id], metadatas=[mem.to_metadata()])

        engine = DecayEngine(store, decay_rate=0.1, min_score=0.05)
        # Simulate "now" = 100 days from epoch → score ≈ 0.5 × exp(-10) ≈ 0
        far_future = 100 * 86_400.0
        removed = engine.prune(now=far_future)
        assert len(removed) == 1

    def test_multiple_pruned_in_one_pass(self, store: MemoryStore):
        for i in range(5):
            _store_aged(store, f"stale {i}", importance=0.1, days_old=60)
        _store_aged(store, "keeper", importance=0.9, days_old=1)
        engine = DecayEngine(store, decay_rate=0.1, min_score=0.05)
        removed = engine.prune()
        assert len(removed) == 5
        assert store.count() == 1

    def test_empty_store_returns_empty(self, store: MemoryStore):
        engine = DecayEngine(store)
        assert engine.prune() == []
