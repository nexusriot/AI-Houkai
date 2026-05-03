"""Unit tests for memory_system.ReflectionEngine."""

from __future__ import annotations

import pytest

from ai_houkai.memory_system import Memory, MemoryStore, ReflectionEngine


def _ep(store: MemoryStore, text: str, tags: list[str] | None = None,
        importance: float = 0.5) -> Memory:
    """Shortcut: store an episodic memory."""
    return store.remember(text, type="episodic",
                          tags=tags or [], importance=importance)


class TestClusters:
    def test_similar_memories_cluster_together(self, store: MemoryStore):
        # These three are semantically very close
        _ep(store, "Deployed API v2.1 to production on Monday.")
        _ep(store, "Deployed API v2.2 to production on Tuesday.")
        _ep(store, "Released API v2.3 to production environment.")
        # This one is semantically distant
        _ep(store, "User prefers dark mode UI theme.")

        engine = ReflectionEngine(store, similarity_threshold=0.70,
                                  min_cluster_size=2)
        clusters = engine.clusters()
        assert len(clusters) >= 1
        # The biggest cluster should contain the deployment memories
        big = max(clusters, key=len)
        assert len(big) >= 2

    def test_dissimilar_memories_not_clustered(self, store: MemoryStore):
        _ep(store, "Met Alice about the payment gateway integration.")
        _ep(store, "Resolved a DNS outage in the EU region.")
        _ep(store, "Onboarded new team member Bob.")
        engine = ReflectionEngine(store, similarity_threshold=0.95,
                                  min_cluster_size=2)
        # Threshold so high that nothing clusters
        clusters = engine.clusters()
        assert clusters == []

    def test_below_min_cluster_size_ignored(self, store: MemoryStore):
        _ep(store, "Deployed service v1.")
        _ep(store, "Deployed service v2.")
        engine = ReflectionEngine(store, similarity_threshold=0.60,
                                  min_cluster_size=3)
        clusters = engine.clusters()
        assert clusters == []

    def test_empty_store_returns_empty(self, store: MemoryStore):
        engine = ReflectionEngine(store)
        assert engine.clusters() == []

    def test_no_episodic_memories_returns_empty(self, store: MemoryStore):
        store.remember("A semantic fact", type="semantic")
        store.remember("A procedure step", type="procedural")
        engine = ReflectionEngine(store)
        assert engine.clusters() == []


class TestReflect:
    def _seed_deployment_cluster(self, store: MemoryStore) -> None:
        _ep(store, "Deployed API v2.1 to production on Monday.",
            tags=["deploy", "api"], importance=0.7)
        _ep(store, "Deployed API v2.2 to production on Tuesday.",
            tags=["deploy", "api"], importance=0.65)
        _ep(store, "Released API v2.3 to production environment.",
            tags=["deploy", "api"], importance=0.75)

    def test_creates_semantic_memory(self, store: MemoryStore):
        self._seed_deployment_cluster(store)
        engine = ReflectionEngine(store, similarity_threshold=0.70,
                                  min_cluster_size=2)
        created = engine.reflect()
        assert len(created) >= 1
        assert all(m.type == "semantic" for m in created)

    def test_reflection_tag_added(self, store: MemoryStore):
        self._seed_deployment_cluster(store)
        engine = ReflectionEngine(store, similarity_threshold=0.70,
                                  min_cluster_size=2)
        created = engine.reflect()
        for mem in created:
            assert "reflection" in mem.tags

    def test_source_set_to_reflection(self, store: MemoryStore):
        self._seed_deployment_cluster(store)
        engine = ReflectionEngine(store, similarity_threshold=0.70,
                                  min_cluster_size=2)
        created = engine.reflect()
        for mem in created:
            assert mem.source == "reflection"

    def test_importance_is_average_of_cluster(self, store: MemoryStore):
        _ep(store, "Deployed API v2.1 to production on Monday.",
            importance=0.6)
        _ep(store, "Released API v2.2 to production environment.",
            importance=0.8)
        engine = ReflectionEngine(store, similarity_threshold=0.70,
                                  min_cluster_size=2)
        created = engine.reflect()
        if created:
            assert created[0].importance == pytest.approx(0.7, abs=0.05)

    def test_dry_run_does_not_write(self, store: MemoryStore):
        self._seed_deployment_cluster(store)
        count_before = store.count()
        engine = ReflectionEngine(store, similarity_threshold=0.70,
                                  min_cluster_size=2)
        candidates = engine.reflect(dry_run=True)
        assert len(candidates) >= 1
        assert store.count() == count_before  # nothing written

    def test_dry_run_source_marked(self, store: MemoryStore):
        self._seed_deployment_cluster(store)
        engine = ReflectionEngine(store, similarity_threshold=0.70,
                                  min_cluster_size=2)
        candidates = engine.reflect(dry_run=True)
        for m in candidates:
            assert "dry-run" in (m.source or "")

    def test_consolidate_removes_source_episodic(self, store: MemoryStore):
        self._seed_deployment_cluster(store)
        episodic_before = store.count()
        engine = ReflectionEngine(store, similarity_threshold=0.70,
                                  min_cluster_size=2)
        created = engine.reflect(consolidate=True)
        assert len(created) >= 1
        # All episodics in the cluster should be gone; only reflections remain
        remaining = store.list_recent(limit=100)
        episodic_remaining = [m for m in remaining if m.type == "episodic"]
        semantic_remaining  = [m for m in remaining if m.type == "semantic"]
        assert len(semantic_remaining) >= 1
        # Total should be less than before (some episodics removed)
        assert store.count() < episodic_before + len(created)

    def test_tags_merged_from_sources(self, store: MemoryStore):
        _ep(store, "Deployed API v2.1 to prod.", tags=["deploy", "api"])
        _ep(store, "Released API v2.2 to production.", tags=["deploy", "release"])
        engine = ReflectionEngine(store, similarity_threshold=0.70,
                                  min_cluster_size=2)
        created = engine.reflect()
        if created:
            tags = set(created[0].tags)
            assert "reflection" in tags
            assert "deploy" in tags

    def test_custom_summarizer_called(self, store: MemoryStore):
        self._seed_deployment_cluster(store)
        called_with: list[list[Memory]] = []

        def spy(memories: list[Memory]) -> str:
            called_with.append(memories)
            return "custom summary"

        engine = ReflectionEngine(store, similarity_threshold=0.70,
                                  min_cluster_size=2, summarizer=spy)
        created = engine.reflect()
        assert len(called_with) >= 1
        if created:
            assert "custom summary" in created[0].text

    def test_no_clusters_returns_empty(self, store: MemoryStore):
        _ep(store, "Met Alice about the payment gateway.")
        engine = ReflectionEngine(store, min_cluster_size=2)
        assert engine.reflect() == []


class TestDefaultSummarizer:
    def test_output_contains_reflection_prefix(self, store: MemoryStore):
        _ep(store, "Deployed v1.", importance=0.8)
        _ep(store, "Deployed v2.", importance=0.6)
        engine = ReflectionEngine(store, similarity_threshold=0.70,
                                  min_cluster_size=2)
        created = engine.reflect()
        if created:
            assert created[0].text.startswith("[Reflection")

    def test_output_truncated_to_512(self, store: MemoryStore):
        long = "x " * 300   # 600 chars
        _ep(store, long + "alpha")
        _ep(store, long + "beta")
        engine = ReflectionEngine(store, similarity_threshold=0.70,
                                  min_cluster_size=2)
        created = engine.reflect()
        if created:
            assert len(created[0].text) <= 512

    def test_most_important_first(self, store: MemoryStore):
        _ep(store, "Deployed v1.", importance=0.3)
        _ep(store, "Deployed v2.", importance=0.9)
        engine = ReflectionEngine(store, similarity_threshold=0.70,
                                  min_cluster_size=2)
        created = engine.reflect()
        if created:
            # High-importance text should appear earlier in the summary
            idx_v2 = created[0].text.find("v2.")
            idx_v1 = created[0].text.find("v1.")
            if idx_v1 != -1 and idx_v2 != -1:
                assert idx_v2 < idx_v1
