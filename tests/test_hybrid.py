"""Unit tests for hybrid retrieval (BM25 + cosine + recency + importance)."""

from __future__ import annotations

import time

import pytest

from ai_houkai.memory_system import ExpandSpec, HybridWeights, MemoryStore
from ai_houkai.memory_system.store import _bm25_score_pool, _tokenize
from ai_houkai.memory_system import Memory

class TestTokenize:
    def test_lowercase_split(self):
        assert _tokenize("Hello World") == ["hello", "world"]

    def test_strips_punctuation(self):
        tokens = _tokenize("foo, bar. baz!")
        assert "foo" in tokens and "bar" in tokens and "baz" in tokens

    def test_empty_string(self):
        assert _tokenize("") == []

    def test_numbers(self):
        assert "42" in _tokenize("version 42")


class TestBM25ScorePool:
    def test_exact_match_scores_highest(self):
        # Use tokens that survive normalization unchanged
        query = "python gil"
        docs = [
            "python gil blocks cpu parallelism",
            "javascript is single-threaded",
            "rust has no gil",
        ]
        scores = _bm25_score_pool(query, docs)
        # doc[0] has both "python" and "gil"; doc[2] has only "gil"
        assert scores[0] > scores[1]   # doc[0] beats unrelated doc
        assert scores[0] > scores[2]   # doc[0] beats single-token match
        assert max(scores) == pytest.approx(1.0)  # normalised max = 1

    def test_no_match_scores_zero(self):
        scores = _bm25_score_pool("unrelated query", ["foo bar", "baz qux"])
        assert all(s == 0.0 for s in scores)

    def test_empty_docs_returns_empty(self):
        assert _bm25_score_pool("query", []) == []

    def test_output_length_matches_docs(self):
        docs = ["a", "b", "c"]
        assert len(_bm25_score_pool("a", docs)) == len(docs)

    def test_scores_normalised_to_one(self):
        scores = _bm25_score_pool("python testing", [
            "Python unit testing with pytest",
            "JavaScript end-to-end testing",
            "Cooking recipe for pasta",
        ])
        assert max(scores) == pytest.approx(1.0)
        assert all(0.0 <= s <= 1.0 for s in scores)


class TestHybridWeights:
    def test_defaults_sum_to_one(self):
        w = HybridWeights()
        total = w.cosine + w.lexical + w.recency + w.importance
        assert total == pytest.approx(1.0)

    def test_all_zero_raises(self):
        with pytest.raises(ValueError):
            HybridWeights(cosine=0, lexical=0, recency=0, importance=0)

    def test_custom_weights(self):
        w = HybridWeights(cosine=1.0, lexical=0.0, recency=0.0, importance=0.0)
        assert w.cosine == 1.0


class TestHybridRecall:
    def _seed(self, store: MemoryStore) -> None:
        store.remember("pytest tmp_path fixture for test isolation",
                       type="procedural", tags=["testing"], importance=0.9)
        store.remember("Never use EphemeralClient in tests",
                       type="procedural", tags=["testing"], importance=0.95)
        store.remember("Deploy API with make release",
                       type="procedural", tags=["deploy"], importance=0.7)
        store.remember("API versioned at /api/v1/",
                       type="semantic", tags=["api"], importance=0.8)

    def test_hybrid_returns_same_shape_as_semantic(self, store: MemoryStore):
        self._seed(store)
        sem = store.recall("pytest testing", k=3, mode="semantic")
        hyb = store.recall("pytest testing", k=3, mode="hybrid")
        assert len(hyb) == len(sem)
        for mem, score in hyb:
            assert 0.0 <= score <= 1.0 + 0.01  # small tolerance for weights > 1

    def test_hybrid_returns_memories(self, store: MemoryStore):
        self._seed(store)
        hits = store.recall("pytest", k=2, mode="hybrid")
        assert len(hits) > 0
        assert all(isinstance(m, Memory) for m, _ in hits)

    def test_lexical_boost_prefers_exact_token(self, store: MemoryStore):
        # "PR #441" should score higher in hybrid than a near-synonym paraphrase
        store.remember("Fixed bug in PR #441", type="episodic", importance=0.5)
        store.remember("Resolved a regression issue", type="episodic", importance=0.5)
        hits = store.recall("PR #441", k=2, mode="hybrid")
        ids  = [m.id for m, _ in hits]
        # The exact-match memory should appear in results
        pr_mem = next(
            (m for m, _ in store.recall("PR #441", k=5, mode="semantic")
             if "PR #441" in m.text), None
        )
        if pr_mem:
            assert pr_mem.id in ids

    def test_custom_weights_accepted(self, store: MemoryStore):
        self._seed(store)
        w = HybridWeights(cosine=0.8, lexical=0.1, recency=0.05, importance=0.05)
        hits = store.recall("testing", k=3, mode="hybrid", weights=w)
        assert len(hits) > 0

    def test_hybrid_mode_default_is_semantic(self, store: MemoryStore):
        self._seed(store)
        default = store.recall("testing", k=3)
        semantic = store.recall("testing", k=3, mode="semantic")
        # Same result set (ordering may differ slightly from touch timing)
        default_ids  = {m.id for m, _ in default}
        semantic_ids = {m.id for m, _ in semantic}
        assert default_ids == semantic_ids

    def test_hybrid_skips_superseded(self, store: MemoryStore):
        self._seed(store)
        old = store.remember("Old testing rule", type="procedural", tags=["testing"])
        new = store.remember("New testing rule", type="procedural", tags=["testing"])
        store.supersede(old_id=old.id, new_id=new.id)
        hits = store.recall("testing rule", k=10, mode="hybrid")
        ids  = [m.id for m, _ in hits]
        assert old.id not in ids

    def test_hybrid_include_superseded(self, store: MemoryStore):
        old = store.remember("Old rule", type="procedural")
        new = store.remember("New rule", type="procedural")
        store.supersede(old_id=old.id, new_id=new.id)
        hits = store.recall("rule", k=10, mode="hybrid", include_superseded=True)
        ids  = [m.id for m, _ in hits]
        assert old.id in ids

    def test_overfetch_param_accepted(self, store: MemoryStore):
        self._seed(store)
        hits = store.recall("deploy", k=2, mode="hybrid", overfetch=8)
        assert isinstance(hits, list)


class TestExpandLinks:
    def test_expand_follows_outgoing_rel(self, store: MemoryStore):
        rule    = store.remember("Always isolate tests with tmp_path",
                                  type="procedural", tags=["testing"])
        example = store.remember("def fixture(tmp_path): return MemoryStore(tmp_path)",
                                  type="episodic", tags=["testing"])
        store.link(rule.id, example.id, rel="example_of")

        hits = store.recall(
            "test isolation", k=3,
            expand=ExpandSpec(rels=("example_of",), cap=3, score=0.65),
        )
        ids = [m.id for m, _ in hits]
        # rule should be in semantic recall; example should be expanded
        assert rule.id in ids or example.id in ids

    def test_expand_cap_limits_results(self, store: MemoryStore):
        root = store.remember("root concept", type="semantic")
        for i in range(5):
            child = store.remember(f"child {i}", type="semantic")
            store.link(root.id, child.id, rel="refines")

        hits_no_expand  = store.recall("root concept", k=1)
        hits_with_expand = store.recall(
            "root concept", k=1,
            expand=ExpandSpec(rels=("refines",), cap=2, score=0.5),
        )
        # cap=2 means at most 2 extra nodes added
        assert len(hits_with_expand) <= len(hits_no_expand) + 2

    def test_expand_assigned_score(self, store: MemoryStore):
        parent = store.remember("parent rule", type="procedural")
        child  = store.remember("child example", type="episodic")
        store.link(parent.id, child.id, rel="example_of")

        spec = ExpandSpec(rels=("example_of",), cap=5, score=0.55)
        hits = store.recall("parent rule", k=1, expand=spec)
        expanded = [(m, s) for m, s in hits if m.id == child.id]
        if expanded:
            _, score = expanded[0]
            assert score == pytest.approx(0.55)

    def test_expand_ignores_superseded(self, store: MemoryStore):
        parent = store.remember("parent", type="semantic")
        child  = store.remember("old child", type="semantic")
        new_c  = store.remember("new child", type="semantic")
        store.link(parent.id, child.id, rel="refines")
        store.supersede(old_id=child.id, new_id=new_c.id)

        hits = store.recall(
            "parent", k=1,
            expand=ExpandSpec(rels=("refines",), cap=5, score=0.5),
        )
        ids = [m.id for m, _ in hits]
        assert child.id not in ids


class TestPolarityScoring:
    """Polarity (+1/-1) should shift hybrid and semantic scores."""

    def test_positive_polarity_scores_higher_than_neutral(self, store: MemoryStore):
        neutral  = store.remember("Use pytest for testing", type="procedural",
                                  importance=0.5, polarity=0)
        positive = store.remember("Use pytest for testing", type="procedural",
                                  importance=0.5, polarity=1)
        hits = {m.id: s for m, s in store.recall("pytest testing", k=5, mode="hybrid")}
        assert hits.get(positive.id, 0) > hits.get(neutral.id, 0)

    def test_negative_polarity_scores_lower_than_neutral(self, store: MemoryStore):
        neutral  = store.remember("Run ruff linter", type="procedural",
                                  importance=0.5, polarity=0)
        negative = store.remember("Run ruff linter", type="procedural",
                                  importance=0.5, polarity=-1)
        hits = {m.id: s for m, s in store.recall("ruff linter", k=5, mode="hybrid")}
        assert hits.get(neutral.id, 0) > hits.get(negative.id, 0)

    def test_polarity_ordering_semantic_mode(self, store: MemoryStore):
        neg = store.remember("Deploy to production", type="procedural",
                             importance=0.5, polarity=-1)
        pos = store.remember("Deploy to production", type="procedural",
                             importance=0.5, polarity=1)
        hits = store.recall("deploy production", k=5, mode="semantic")
        ids = [m.id for m, _ in hits]
        # positive polarity should outrank negative when text is identical
        assert ids.index(pos.id) < ids.index(neg.id)

    def test_zero_polarity_no_change(self, store: MemoryStore):
        # Two identical memories with polarity=0 should score the same
        m1 = store.remember("Use docker compose", type="procedural",
                             importance=0.5, polarity=0)
        m2 = store.remember("Use docker compose", type="procedural",
                             importance=0.5, polarity=0)
        hits = {m.id: s for m, s in store.recall("docker compose", k=5, mode="hybrid")}
        if m1.id in hits and m2.id in hits:
            assert abs(hits[m1.id] - hits[m2.id]) < 0.05  # negligible diff

    def test_polarity_weight_zero_disables_boost(self, store: MemoryStore):
        # Explicitly passing polarity_weight=0 should remove the polarity effect
        neg = store.remember("Use pytest for testing", type="procedural",
                             importance=0.5, polarity=-1)
        pos = store.remember("Use pytest for testing", type="procedural",
                             importance=0.5, polarity=1)
        w = HybridWeights(cosine=0.55, lexical=0.25, recency=0.1, importance=0.1,
                          polarity_weight=0.0)
        hits = {m.id: s for m, s in store.recall(
            "pytest testing", k=5, mode="hybrid", weights=w)}
        if pos.id in hits and neg.id in hits:
            # with polarity_weight=0 both identical texts should score the same
            assert abs(hits[pos.id] - hits[neg.id]) < 0.05

    def test_store_level_hybrid_weights_accepted(self, tmp_path):
        # MemoryStore constructor accepts hybrid_weights and applies them store-wide
        w = HybridWeights(cosine=0.6, lexical=0.2, recency=0.1, importance=0.1,
                          polarity_weight=0.0)
        store = MemoryStore(str(tmp_path), hybrid_weights=w)
        m = store.remember("pytest testing isolates tests", type="procedural")
        hits = store.recall("pytest", k=5, mode="hybrid")
        assert any(x.id == m.id for x, _ in hits)


class TestRecencyBasis:
    def test_default_is_created(self):
        assert HybridWeights().recency_basis == "created"

    def test_created_vs_accessed_basis_differ_for_aged_memory(self, store: MemoryStore):
        m = store.remember("zenith unique recency token", type="semantic", importance=0.5)
        # Backdate created_at by 60 days but keep last_accessed recent.
        rec = store.collection.get(ids=[m.id], include=["metadatas"])
        meta = dict(rec["metadatas"][0])
        meta["created_at"] = time.time() - 60 * 86_400.0
        store.collection.update(ids=[m.id], metadatas=[meta])

        c = store.recall("zenith unique recency token", k=1, mode="hybrid",
                         weights=HybridWeights(recency_basis="created"), touch=False)
        a = store.recall("zenith unique recency token", k=1, mode="hybrid",
                         weights=HybridWeights(recency_basis="accessed"), touch=False)
        # 'accessed' sees a fresh last_accessed (recency≈1); 'created' sees a
        # 60-day-old fact (recency≈0) → accessed scores strictly higher.
        assert a[0][1] > c[0][1]


class TestRRFFusion:
    def _seed(self, store: MemoryStore):
        store.remember("python gil blocks cpu parallelism", type="semantic", importance=0.8)
        store.remember("javascript event loop concurrency", type="semantic", importance=0.5)
        store.remember("rust has no garbage collector", type="semantic", importance=0.6)

    def test_rrf_returns_ranked_results(self, store: MemoryStore):
        self._seed(store)
        hits = store.recall("python gil", k=3, mode="hybrid", fusion="rrf")
        assert len(hits) > 0
        scores = [s for _, s in hits]
        assert scores == sorted(scores, reverse=True)
        assert all(s > 0 for s in scores)

    @pytest.mark.needs_model
    def test_rrf_ranks_exact_match_top(self, store: MemoryStore):
        self._seed(store)
        hits = store.recall("python gil parallelism", k=3, mode="hybrid", fusion="rrf")
        assert "python gil" in hits[0][0].text

    def test_rrf_single_doc_pool_does_not_crash(self, store: MemoryStore):
        store.remember("solitary memory", type="semantic")
        hits = store.recall("solitary", k=5, mode="hybrid", fusion="rrf")
        assert len(hits) == 1


class TestDiversityAndDedup:
    def test_dedup_collapses_identical_memories(self, store: MemoryStore):
        for _ in range(3):
            store.remember("alpha duplicate fact", type="semantic")
        hits = store.recall("alpha duplicate fact", k=5, dedup_threshold=0.95)
        # Three identical texts → identical embeddings → only one survives.
        assert len(hits) == 1

    def test_no_dedup_keeps_duplicates(self, store: MemoryStore):
        for _ in range(3):
            store.remember("beta duplicate fact", type="semantic")
        hits = store.recall("beta duplicate fact", k=5)
        assert len(hits) == 3

    @pytest.mark.needs_model
    def test_diversity_first_pick_is_most_relevant(self, store: MemoryStore):
        # First MMR pick has no novelty penalty → always the most relevant.
        best = store.remember("python testing with pytest fixtures", type="semantic")
        store.remember("python docker deployment pipeline", type="semantic")
        store.remember("banana bread baking recipe", type="semantic")
        hits = store.recall("python testing pytest fixtures", k=3, diversity=0.5)
        assert hits[0][0].id == best.id

    @pytest.mark.needs_model
    def test_rrf_diversity_does_not_promote_irrelevant(self, store: MemoryStore):
        # Regression for the RRF/MMR scale mismatch: with diversity high
        # (relevance-dominant) an off-topic memory must not be promoted into the
        # top results over relevant near-duplicates.
        for _ in range(3):
            store.remember("python concurrency with asyncio event loop", type="semantic")
        banana = store.remember("strawberry banana smoothie recipe", type="semantic")
        hits = store.recall("python asyncio concurrency", k=2,
                            mode="hybrid", fusion="rrf", diversity=0.9)
        ids = [m.id for m, _ in hits]
        assert banana.id not in ids


class TestRangeValidation:
    def test_diversity_out_of_range_raises(self, store: MemoryStore):
        store.remember("x", type="semantic")
        with pytest.raises(ValueError):
            store.recall("x", diversity=1.5)

    def test_dedup_threshold_out_of_range_raises(self, store: MemoryStore):
        store.remember("x", type="semantic")
        with pytest.raises(ValueError):
            store.recall("x", dedup_threshold=2.0)

    def test_min_cosine_out_of_range_raises(self, store: MemoryStore):
        store.remember("x", type="semantic")
        with pytest.raises(ValueError):
            store.recall("x", min_cosine=5.0)


class TestMinCosineGate:
    def test_unrelated_query_returns_nothing(self, store: MemoryStore):
        store.remember("quantum chromodynamics lattice gauge theory", type="semantic")
        hits = store.recall("banana smoothie brunch recipe", k=5, min_cosine=0.9)
        assert hits == []

    def test_matching_query_passes_floor(self, store: MemoryStore):
        m = store.remember("kubernetes ingress controller setup", type="semantic")
        hits = store.recall("kubernetes ingress controller setup", k=5, min_cosine=0.5)
        assert any(x.id == m.id for x, _ in hits)


class TestExplain:
    def test_hybrid_explain_breakdown(self, store: MemoryStore):
        store.remember("explainable hybrid scoring memory", type="semantic", importance=0.7)
        hits = store.recall("hybrid scoring", k=1, mode="hybrid", explain=True)
        mem, score, breakdown = hits[0]
        assert isinstance(mem, Memory)
        assert breakdown["mode"] == "hybrid" and breakdown["fusion"] == "weighted"
        assert {"cosine", "lexical", "recency", "importance", "weights"} <= set(breakdown)

    def test_rrf_explain_has_signal_ranks(self, store: MemoryStore):
        store.remember("explainable rrf scoring memory", type="semantic")
        hits = store.recall("rrf scoring", k=1, mode="hybrid", fusion="rrf", explain=True)
        _, _, breakdown = hits[0]
        assert breakdown["fusion"] == "rrf"
        assert "signals" in breakdown and "cosine" in breakdown["signals"]

    def test_semantic_explain(self, store: MemoryStore):
        store.remember("semantic explain memory", type="semantic")
        hits = store.recall("semantic explain", k=1, mode="semantic", explain=True)
        _, _, breakdown = hits[0]
        assert breakdown["mode"] == "semantic" and "cosine" in breakdown

    def test_explain_false_returns_two_tuples(self, store: MemoryStore):
        store.remember("plain memory", type="semantic")
        hits = store.recall("plain", k=1)
        assert len(hits[0]) == 2


class TestExpansionDepthDecay:
    def _chain(self, store: MemoryStore):
        a = store.remember("root alpha concept", type="semantic")
        b = store.remember("child beta detail", type="semantic")
        c = store.remember("grandchild gamma detail", type="semantic")
        store.link(a.id, b.id, rel="refines")
        store.link(b.id, c.id, rel="refines")
        return a, b, c

    def test_depth_one_stops_at_first_hop(self, store: MemoryStore):
        a, b, c = self._chain(store)
        hits = store.recall("root alpha concept", k=1,
                            expand=ExpandSpec(rels=("refines",), depth=1, cap=5, score=0.6))
        ids = [m.id for m, _ in hits]
        assert b.id in ids and c.id not in ids

    def test_depth_two_reaches_second_hop(self, store: MemoryStore):
        a, b, c = self._chain(store)
        hits = store.recall("root alpha concept", k=1,
                            expand=ExpandSpec(rels=("refines",), depth=2, cap=5, score=0.6))
        ids = [m.id for m, _ in hits]
        assert b.id in ids and c.id in ids

    def test_per_hop_decay(self, store: MemoryStore):
        a, b, c = self._chain(store)
        hits = store.recall("root alpha concept", k=1,
                            expand=ExpandSpec(rels=("refines",), depth=2, cap=5,
                                              score=0.6, decay=0.5))
        smap = {m.id: s for m, s in hits}
        assert smap[b.id] == pytest.approx(0.6)        # hop 1
        assert smap[c.id] == pytest.approx(0.3)        # hop 2 = 0.6 * 0.5

    def test_decay_default_keeps_flat_score(self, store: MemoryStore):
        a, b, c = self._chain(store)
        hits = store.recall("root alpha concept", k=1,
                            expand=ExpandSpec(rels=("refines",), depth=2, cap=5, score=0.55))
        smap = {m.id: s for m, s in hits}
        assert smap[b.id] == pytest.approx(0.55)
        assert smap[c.id] == pytest.approx(0.55)       # decay defaults to 1.0


class TestCJKTokenization:
    def test_ascii_unchanged(self):
        assert _tokenize("Hello World") == ["hello", "world"]

    def test_cjk_emits_bigrams(self):
        toks = _tokenize("日本語")
        assert "日本" in toks and "本語" in toks

    def test_cjk_query_recall(self, store: MemoryStore):
        store.remember("日本語のテストメモリ", type="semantic")
        hits = store.recall("日本語", k=3, mode="hybrid")
        assert any("日本語" in m.text for m, _ in hits)
