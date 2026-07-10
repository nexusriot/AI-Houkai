"""Tests for store runtime metrics (Feature 5, store layer)."""

from __future__ import annotations

from ai_houkai.memory_system import MemoryStore


class TestMetrics:
    def test_fresh_store_has_zeroed_counters(self, store: MemoryStore):
        m = store.metrics()
        assert m["calls"] == {"remember": 0, "recall": 0, "forget": 0,
                              "edit": 0, "supersede": 0}
        assert m["recall_latency_ms"]["count"] == 0
        assert m["count"] == 0
        assert m["uptime_seconds"] >= 0

    def test_counters_track_operations(self, store: MemoryStore):
        a = store.remember("first")
        b = store.remember("second")
        store.recall("first", k=2)
        store.recall("second", k=2)
        store.recall("third", k=2)
        store.edit(a.id, importance=0.9)
        store.supersede(old_id=a.id, new_id=b.id)
        store.forget(b.id)

        calls = store.metrics()["calls"]
        assert calls["remember"] == 2
        assert calls["recall"] == 3
        assert calls["edit"] == 1
        assert calls["supersede"] == 1
        assert calls["forget"] == 1

    def test_recall_latency_recorded(self, store: MemoryStore):
        store.remember("something to find")
        store.recall("something", k=1)
        store.recall("something", k=1)
        lat = store.metrics()["recall_latency_ms"]
        assert lat["count"] == 2
        assert lat["avg"] >= 0.0
        assert lat["max"] >= lat["avg"]

    def test_empty_recall_still_counts(self, store: MemoryStore):
        # recall on an empty store returns [] early but must still be counted.
        store.recall("nothing here", k=3)
        assert store.metrics()["calls"]["recall"] == 1
        assert store.metrics()["recall_latency_ms"]["count"] == 1

    def test_count_reflects_store_size(self, store: MemoryStore):
        store.remember("a")
        store.remember("b")
        assert store.metrics()["count"] == 2
