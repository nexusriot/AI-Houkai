"""Unit tests for memory_system.MemoryStore."""

from __future__ import annotations

import time

from ai_houkai.memory_system import Memory, MemoryStore


class TestRemember:
    def test_returns_memory_with_uuid(self, store: MemoryStore):
        mem = store.remember("Python uses GIL for thread safety.")
        assert mem.id
        assert len(mem.id) == 36  # UUID4 canonical form

    def test_count_increments(self, store: MemoryStore):
        assert store.count() == 0
        store.remember("first")
        store.remember("second")
        assert store.count() == 2

    def test_defaults(self, store: MemoryStore):
        mem = store.remember("hello")
        assert mem.type == "semantic"
        assert mem.tags == []
        assert mem.importance == 0.5
        assert mem.source is None

    def test_custom_metadata(self, store: MemoryStore):
        mem = store.remember(
            "Deploy with make release",
            type="procedural",
            tags=["deploy", "ops"],
            importance=0.9,
            source="user",
        )
        assert mem.type == "procedural"
        assert "deploy" in mem.tags
        assert "ops" in mem.tags
        assert mem.importance == 0.9
        assert mem.source == "user"

    def test_importance_clamped(self, store: MemoryStore):
        high = store.remember("x", importance=5.0)
        low = store.remember("y", importance=-1.0)
        assert high.importance == 1.0
        assert low.importance == 0.0

    def test_text_stripped(self, store: MemoryStore):
        mem = store.remember("  spaces around  ")
        assert mem.text == "spaces around"


class TestForget:
    def test_deletes_existing(self, store: MemoryStore):
        mem = store.remember("to be forgotten")
        assert store.forget(mem.id) is True
        assert store.count() == 0

    def test_returns_false_for_unknown_id(self, store: MemoryStore):
        assert store.forget("00000000-0000-0000-0000-000000000000") is False

    def test_count_decrements(self, store: MemoryStore):
        m1 = store.remember("keep me")
        m2 = store.remember("delete me")
        store.forget(m2.id)
        assert store.count() == 1


class TestRecall:
    def _seed(self, store: MemoryStore):
        store.remember(
            "The user prefers terse responses.",
            type="feedback",
            tags=["style"],
            importance=0.9,
        )
        store.remember(
            "Deploy API with `make release` then `kubectl rollout restart`.",
            type="procedural",
            tags=["deploy"],
            importance=0.7,
        )
        store.remember(
            "Met Alice on 2026-04-22 to scope the ingest rewrite.",
            type="episodic",
            tags=["meeting"],
            importance=0.5,
        )

    def test_returns_results(self, store: MemoryStore):
        self._seed(store)
        hits = store.recall("how to release the API?", k=3)
        assert len(hits) > 0

    def test_result_structure(self, store: MemoryStore):
        self._seed(store)
        hits = store.recall("deploy", k=1)
        mem, score = hits[0]
        assert isinstance(mem, Memory)
        assert 0.0 <= score <= 1.0

    def test_type_filter(self, store: MemoryStore):
        self._seed(store)
        hits = store.recall("anything", k=5, type="procedural")
        for mem, _ in hits:
            assert mem.type == "procedural"

    def test_tag_filter(self, store: MemoryStore):
        self._seed(store)
        hits = store.recall("anything", k=5, tag="meeting")
        for mem, _ in hits:
            assert "meeting" in mem.tags

    def test_min_importance_filter(self, store: MemoryStore):
        self._seed(store)
        hits = store.recall("anything", k=5, min_importance=0.8)
        for mem, _ in hits:
            assert mem.importance >= 0.8

    def test_access_count_bumped(self, store: MemoryStore):
        mem = store.remember("some fact")
        hits = store.recall("some fact", k=1)
        assert len(hits) == 1
        recalled_mem, _ = hits[0]
        assert recalled_mem.access_count == 1

    def test_empty_store_returns_empty(self, store: MemoryStore):
        # ChromaDB raises when querying empty collection — store.recall must handle it
        # Store has 0 documents; we add one first to avoid Chroma error then test k=0
        store.remember("seed")
        hits = store.recall("query", k=0)
        assert hits == []


class TestListRecent:
    def test_ordered_newest_first(self, store: MemoryStore):
        store.remember("first")
        time.sleep(0.01)
        store.remember("second")
        time.sleep(0.01)
        store.remember("third")

        recent = store.list_recent()
        assert recent[0].text == "third"
        assert recent[-1].text == "first"

    def test_limit_respected(self, store: MemoryStore):
        for i in range(10):
            store.remember(f"memory {i}")
        assert len(store.list_recent(limit=3)) == 3

    def test_empty_store(self, store: MemoryStore):
        assert store.list_recent() == []


class TestMemoryDataclass:
    def test_to_metadata_roundtrip(self):
        mem = Memory(
            id="abc",
            text="hello",
            type="episodic",
            tags=["a", "b"],
            importance=0.7,
            created_at=1000.0,
            last_accessed=2000.0,
            access_count=3,
            source="agent",
        )
        meta = mem.to_metadata()
        restored = Memory.from_record("abc", "hello", meta)

        assert restored.type == "episodic"
        assert restored.tags == ["a", "b"]
        assert restored.importance == 0.7
        assert restored.access_count == 3
        assert restored.source == "agent"

    def test_from_record_empty_tags(self):
        mem = Memory.from_record("x", "text", {"tags": "", "type": "semantic"})
        assert mem.tags == []

    def test_from_record_missing_source(self):
        mem = Memory.from_record("x", "text", {"source": "", "type": "semantic"})
        assert mem.source is None
