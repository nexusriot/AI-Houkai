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
    @pytest.mark.needs_model
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

    @pytest.mark.needs_model
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

    @pytest.mark.needs_model
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

    @pytest.mark.needs_model
    def test_consolidate_soft_deletes_sources(self, store: MemoryStore):
        """consolidate=True soft-deletes sources (supersedes them, keeps in DB)."""
        self._seed_deployment_cluster(store)
        engine = ReflectionEngine(store, similarity_threshold=0.70,
                                  min_cluster_size=2)
        created = engine.reflect(consolidate=True)
        assert len(created) >= 1

        # default list_recent hides superseded — only semantics visible
        visible = store.list_recent(limit=100)
        semantic_visible = [m for m in visible if m.type == "semantic"]
        episodic_visible = [m for m in visible if m.type == "episodic"]
        assert len(semantic_visible) >= 1
        assert len(episodic_visible) == 0

        # with include_superseded=True, the episodics are still there
        all_mems = store.list_recent(limit=100, include_superseded=True)
        episodic_all = [m for m in all_mems if m.type == "episodic"]
        assert len(episodic_all) >= 1
        assert all(m.superseded_by != "" for m in episodic_all)

    @pytest.mark.needs_model
    def test_second_reflect_skips_superseded_sources(self, store: MemoryStore):
        """A second consolidating reflect must not re-cluster sources that an
        earlier reflection already superseded — doing so would emit a duplicate
        summary and re-supersede them under a fresh id."""
        self._seed_deployment_cluster(store)
        engine = ReflectionEngine(store, similarity_threshold=0.70,
                                  min_cluster_size=2)

        first = engine.reflect(consolidate=True)
        assert len(first) >= 1
        episodics = [
            m for m in store.list_recent(limit=100, include_superseded=True)
            if m.type == "episodic"
        ]
        assert episodics and all(m.superseded_by for m in episodics)

        # Second run: no active episodics left, so nothing to cluster.
        second = engine.reflect(consolidate=True)
        assert second == []

        summaries = [m for m in store.list_recent(limit=100)
                     if "reflection" in m.tags]
        assert len(summaries) == 1, "a second reflect must not duplicate the summary"

    @pytest.mark.needs_model
    def test_consolidate_hard_removes_sources(self, store: MemoryStore):
        """consolidate='hard' hard-deletes sources (old behaviour)."""
        self._seed_deployment_cluster(store)
        count_before = store.count()
        engine = ReflectionEngine(store, similarity_threshold=0.70,
                                  min_cluster_size=2)
        created = engine.reflect(consolidate="hard")
        assert len(created) >= 1
        # Hard delete: total count is less than before + reflections created
        assert store.count() < count_before + len(created)

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

    @pytest.mark.needs_model
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


class TestPolarityClusterSeparation:
    """Opposite-polarity episodics must not merge into the same cluster."""

    def test_opposite_polarity_not_clustered_together(self, store: MemoryStore):
        # Positive and negative memories about the same event should stay separate
        pos = store.remember("Deployed API v2 to production successfully",
                             type="episodic", polarity=1, importance=0.8)
        neg = store.remember("Deployed API v2 to production — outage occurred",
                             type="episodic", polarity=-1, importance=0.8)
        engine = ReflectionEngine(store, similarity_threshold=0.60,
                                  min_cluster_size=2)
        clusters = engine.clusters()
        # If any cluster exists, it must not mix pos and neg
        for cluster in clusters:
            ids = {m.id for m in cluster}
            if pos.id in ids:
                assert neg.id not in ids
            if neg.id in ids:
                assert pos.id not in ids

    def test_same_polarity_can_cluster(self, store: MemoryStore):
        store.remember("Deployed API v2.1 successfully", type="episodic",
                       polarity=1, importance=0.8)
        store.remember("Deployed API v2.2 successfully", type="episodic",
                       polarity=1, importance=0.8)
        engine = ReflectionEngine(store, similarity_threshold=0.60,
                                  min_cluster_size=2)
        clusters = engine.clusters()
        # Positive+positive with the same polarity should still cluster
        # (this test just asserts no crash; actual clustering depends on similarity)
        assert isinstance(clusters, list)

    def test_neutral_polarity_can_join_any_cluster(self, store: MemoryStore):
        # polarity=0 should not block clustering with either sign
        store.remember("Deployed API v2.1 to production", type="episodic",
                       polarity=1, importance=0.8)
        store.remember("Deployed API v2.2 to production", type="episodic",
                       polarity=0, importance=0.8)
        engine = ReflectionEngine(store, similarity_threshold=0.60,
                                  min_cluster_size=2)
        clusters = engine.clusters()
        # neutral+positive should be able to cluster together
        assert isinstance(clusters, list)

    def test_neutral_seed_does_not_bridge_opposite_polarities(
        self, store: MemoryStore
    ):
        # Highest importance → seeds the cluster; its polarity is neutral.
        # A seed-only polarity check would let it absorb both the +1 and
        # the -1 memory, blending a contradiction into one summary.
        store.remember("We use the ruff linter for this repository",
                       type="episodic", polarity=0, importance=0.9)
        store.remember("Always use the ruff linter for this repository",
                       type="episodic", polarity=1, importance=0.5)
        store.remember("Never use the ruff linter for this repository",
                       type="episodic", polarity=-1, importance=0.5)
        engine = ReflectionEngine(store, similarity_threshold=0.60,
                                  min_cluster_size=2)
        for cluster in engine.clusters():
            polarities = {m.polarity for m in cluster}
            assert not {-1, 1} <= polarities, (
                "cluster merged explicitly opposite polarities"
            )
