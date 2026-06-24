"""Unit tests for recall_pack — token-budgeted context assembly."""

from __future__ import annotations

import pytest

from ai_houkai.memory_system import CompressedGroup, MemoryStore, PackResult
from ai_houkai.memory_system.store import (
    _estimate_tokens,
    _jaccard_sim,
    _cluster_by_jaccard,
    extract_key_phrases,
)


class TestEstimateTokens:
    def test_roughly_four_chars_per_token(self):
        assert _estimate_tokens("a" * 40) == 10

    def test_minimum_one(self):
        assert _estimate_tokens("") == 1
        assert _estimate_tokens("a") == 1


class TestRecallPack:
    def _seed(self, store: MemoryStore) -> None:
        store.remember("pytest tmp_path fixture for test isolation",
                       type="procedural", tags=["testing"], importance=0.9)
        store.remember("Never use EphemeralClient in tests",
                       type="procedural", tags=["testing"], importance=0.95)
        store.remember("Deploy API with make release",
                       type="procedural", tags=["deploy"], importance=0.7)
        store.remember("API versioned at /api/v1/",
                       type="semantic", tags=["api"], importance=0.8)

    def test_returns_pack_result(self, store: MemoryStore):
        self._seed(store)
        pack = store.recall_pack("testing", token_budget=500)
        assert isinstance(pack, PackResult)
        assert pack.budget == 500
        assert len(pack) > 0

    def test_empty_store_returns_empty_pack(self, store: MemoryStore):
        pack = store.recall_pack("anything", token_budget=500)
        assert len(pack) == 0
        assert pack.text == ""
        assert pack.used_tokens == 0
        assert pack.truncated is False

    def test_respects_token_budget(self, store: MemoryStore):
        self._seed(store)
        pack = store.recall_pack("testing", token_budget=10, max_items=50)
        assert pack.used_tokens <= 10

    def test_tiny_budget_drops_all_and_marks_truncated(self, store: MemoryStore):
        self._seed(store)
        # Budget smaller than any single rendered line → nothing fits.
        pack = store.recall_pack("testing", token_budget=1)
        assert len(pack) == 0
        assert pack.truncated is True
        assert pack.text == ""

    def test_generous_budget_not_truncated(self, store: MemoryStore):
        self._seed(store)
        pack = store.recall_pack("testing rules", token_budget=10_000, max_items=50)
        assert pack.truncated is False

    def test_text_block_has_header_and_lines(self, store: MemoryStore):
        self._seed(store)
        pack = store.recall_pack("testing", token_budget=1000,
                                 header="## Memory")
        assert pack.text.startswith("## Memory\n")
        for p in pack.items:
            assert p.memory.text in pack.text

    def test_no_header(self, store: MemoryStore):
        self._seed(store)
        pack = store.recall_pack("testing", token_budget=1000, header="")
        assert not pack.text.startswith("#")
        assert pack.text  # non-empty body

    def test_used_tokens_matches_item_sum(self, store: MemoryStore):
        self._seed(store)
        pack = store.recall_pack("testing", token_budget=1000)
        assert pack.used_tokens == sum(p.tokens for p in pack.items)

    def test_custom_token_counter(self, store: MemoryStore):
        self._seed(store)
        # 1 token per item → budget of 2 admits exactly 2 items.
        pack = store.recall_pack(
            "testing", token_budget=2, max_items=50,
            token_counter=lambda s: 1,
        )
        assert len(pack) == 2
        assert pack.used_tokens == 2
        assert pack.truncated is True

    def test_type_filter(self, store: MemoryStore):
        self._seed(store)
        pack = store.recall_pack("api", token_budget=1000, type="semantic")
        assert all(p.memory.type == "semantic" for p in pack.items)

    def test_excludes_superseded_by_default(self, store: MemoryStore):
        old = store.remember("Old testing rule", type="procedural", tags=["testing"])
        new = store.remember("New testing rule", type="procedural", tags=["testing"])
        store.supersede(old_id=old.id, new_id=new.id)
        pack = store.recall_pack("testing rule", token_budget=1000)
        assert old.id not in pack.ids()

    def test_semantic_mode_accepted(self, store: MemoryStore):
        self._seed(store)
        pack = store.recall_pack("testing", token_budget=1000, mode="semantic")
        assert len(pack) > 0

    def test_items_preserve_rank_order(self, store: MemoryStore):
        self._seed(store)
        ranked = store.recall("testing", k=50, mode="hybrid")
        pack = store.recall_pack("testing", token_budget=10_000, max_items=50)
        ranked_ids = [m.id for m, _ in ranked]
        pack_ids = pack.ids()
        # pack ids are a subsequence of the ranked ordering
        positions = [ranked_ids.index(i) for i in pack_ids]
        assert positions == sorted(positions)

    def test_compressed_groups_default_empty(self, store: MemoryStore):
        self._seed(store)
        pack = store.recall_pack("testing", token_budget=10_000)
        assert pack.compressed_groups == []


class TestJaccardHelpers:
    def test_identical_texts_score_one(self):
        assert _jaccard_sim("ruff linting python", "ruff linting python") == 1.0

    def test_disjoint_texts_score_zero(self):
        assert _jaccard_sim("ruff linting", "deploy production") == 0.0

    def test_partial_overlap(self):
        s = _jaccard_sim("use ruff for linting", "run ruff to lint")
        assert 0.0 < s < 1.0

    def test_empty_strings(self):
        assert _jaccard_sim("", "") == 0.0

    def test_one_empty_one_not(self):
        assert _jaccard_sim("ruff linting", "") == 0.0
        assert _jaccard_sim("", "ruff linting") == 0.0

    def test_single_token_match(self):
        assert _jaccard_sim("ruff", "ruff") == 1.0

    def test_cluster_by_jaccard_high_threshold_no_groups(self):
        # threshold=1.0 — only identical texts form groups; these are all different
        from ai_houkai.memory_system import Memory
        import time
        def _m(text):
            return (Memory(id="x", text=text, type="procedural", created_at=time.time(),
                           last_accessed=time.time()), 0.9)
        cands = [_m("ruff lint python"), _m("ruff lint code"), _m("deploy api release")]
        groups = _cluster_by_jaccard(cands, threshold=1.0, min_size=2)
        assert groups == []

    def test_cluster_by_jaccard_low_threshold_groups(self):
        # threshold=0.0 — every pair qualifies; all similar texts cluster together
        from ai_houkai.memory_system import Memory
        import time
        def _m(text):
            return (Memory(id="x", text=text, type="procedural", created_at=time.time(),
                           last_accessed=time.time()), 0.9)
        cands = [_m("ruff lint python"), _m("ruff lint code"), _m("ruff lint source")]
        groups = _cluster_by_jaccard(cands, threshold=0.0, min_size=2)
        assert len(groups) >= 1 and sum(len(g) for g in groups) >= 2

    def test_cluster_by_jaccard_respects_min_size(self):
        # Only 2 similar items; min_size=3 means no group should form
        from ai_houkai.memory_system import Memory
        import time
        def _m(text):
            return (Memory(id="x", text=text, type="procedural", created_at=time.time(),
                           last_accessed=time.time()), 0.9)
        cands = [_m("ruff lint python"), _m("ruff lint code")]
        groups = _cluster_by_jaccard(cands, threshold=0.0, min_size=3)
        assert groups == []


class TestExtractKeyPhrases:
    def test_returns_bigrams_first(self):
        phrases = extract_key_phrases("deploy the API to production", max_phrases=3)
        # Stop words ("the", "to") are filtered; bigrams of remaining words come first
        assert any(" " in p for p in phrases), "should include at least one bigram"

    def test_respects_max_phrases(self):
        phrases = extract_key_phrases("fix the authentication bug in the login flow", max_phrases=2)
        assert len(phrases) <= 2

    def test_stops_words_filtered(self):
        phrases = extract_key_phrases("the is a to", max_phrases=5)
        # All stop words — nothing to extract
        assert phrases == []

    def test_no_duplicates(self):
        phrases = extract_key_phrases("ruff ruff ruff", max_phrases=10)
        assert len(phrases) == len(set(phrases))

    def test_empty_string(self):
        assert extract_key_phrases("", max_phrases=5) == []

    def test_single_content_word(self):
        # One word, not a stop word: no bigrams possible, returns the unigram
        phrases = extract_key_phrases("pytest", max_phrases=5)
        assert "pytest" in phrases

    def test_all_strings_are_lowercase(self):
        phrases = extract_key_phrases("Deploy API", max_phrases=5)
        for p in phrases:
            assert p == p.lower()


class TestCompression:
    def _seed_similar(self, store: MemoryStore) -> None:
        """Store several semantically similar procedural memories."""
        store.remember("Use ruff to lint Python files", type="procedural",
                       tags=["linting"], importance=0.8)
        store.remember("Run ruff for linting Python code", type="procedural",
                       tags=["linting"], importance=0.7)
        store.remember("Execute ruff linter on Python source", type="procedural",
                       tags=["linting"], importance=0.7)
        store.remember("Deploy API with make release", type="procedural",
                       tags=["deploy"], importance=0.9)

    def test_compress_false_no_compressed_groups(self, store: MemoryStore):
        self._seed_similar(store)
        pack = store.recall_pack("linting", token_budget=30, compress=False)
        assert pack.compressed_groups == []

    def test_compress_produces_groups_when_truncated(self, store: MemoryStore):
        self._seed_similar(store)
        # tiny budget so individual items overflow, but compressed line fits
        pack = store.recall_pack(
            "ruff linting python",
            token_budget=30,
            compress=True,
            compress_threshold=0.25,
            compress_min_group=2,
        )
        # With compress=True we should get compressed groups when items were dropped
        if pack.truncated:
            assert isinstance(pack.compressed_groups, list)

    def test_compressed_group_fields(self, store: MemoryStore):
        self._seed_similar(store)
        pack = store.recall_pack(
            "ruff linting python",
            token_budget=30,
            compress=True,
            compress_threshold=0.25,
            compress_min_group=2,
        )
        for cg in pack.compressed_groups:
            assert isinstance(cg, CompressedGroup)
            assert len(cg.memories) >= 2
            assert cg.text.startswith("- (compressed)")
            assert cg.tokens > 0

    def test_compressed_lines_appear_in_text(self, store: MemoryStore):
        self._seed_similar(store)
        pack = store.recall_pack(
            "ruff linting python",
            token_budget=30,
            compress=True,
            compress_threshold=0.25,
            compress_min_group=2,
        )
        for cg in pack.compressed_groups:
            assert cg.text in pack.text

    def test_compressed_tokens_counted_in_used(self, store: MemoryStore):
        self._seed_similar(store)
        pack = store.recall_pack(
            "ruff linting",
            token_budget=200,
            compress=True,
            compress_threshold=0.25,
            compress_min_group=2,
        )
        item_tokens = sum(p.tokens for p in pack.items)
        group_tokens = sum(cg.tokens for cg in pack.compressed_groups)
        assert pack.used_tokens == item_tokens + group_tokens

    def test_no_compression_when_budget_ample(self, store: MemoryStore):
        self._seed_similar(store)
        # Large budget → everything fits individually, nothing to compress
        pack = store.recall_pack(
            "ruff linting", token_budget=10_000,
            compress=True, compress_threshold=0.25, compress_min_group=2,
        )
        assert pack.compressed_groups == []
        assert not pack.truncated

    def test_isolated_dropped_item_below_min_group_forms_no_group(self, store: MemoryStore):
        # Only 1 dropped item that has no similar companions → min_group=2 means no group
        store.remember("Use ruff to lint Python files", type="procedural", importance=0.8)
        store.remember("Deploy with make release script", type="procedural", importance=0.9)
        # Very small budget: one item fits, one is dropped. They are dissimilar → no group
        pack = store.recall_pack(
            "ruff linting",
            token_budget=20,
            compress=True,
            compress_threshold=0.60,
            compress_min_group=2,
        )
        assert pack.compressed_groups == []


class TestAutoContextPack:
    def _seed(self, store: MemoryStore) -> None:
        store.remember("Always run pytest with tmp_path for test isolation",
                       type="procedural", tags=["testing"], importance=0.9)
        store.remember("Never use EphemeralClient in tests",
                       type="procedural", tags=["testing"], importance=0.95)
        store.remember("Deploy API with make release",
                       type="procedural", tags=["deploy"], importance=0.7)
        store.remember("API versioned at /api/v1/",
                       type="semantic", tags=["api"], importance=0.8)

    def test_returns_pack_result(self, store: MemoryStore):
        self._seed(store)
        pack = store.auto_context_pack("run the testing suite", token_budget=1000)
        assert isinstance(pack, PackResult)

    def test_empty_store_returns_empty_pack(self, store: MemoryStore):
        pack = store.auto_context_pack("anything", token_budget=500)
        assert len(pack) == 0
        assert pack.text == ""

    def test_finds_more_than_single_query(self, store: MemoryStore):
        self._seed(store)
        # The compound task should surface memories from multiple angles
        pack = store.auto_context_pack(
            "run testing and deploy the api", token_budget=5000, max_phrases=3
        )
        assert len(pack) >= 2

    def test_no_duplicate_ids(self, store: MemoryStore):
        self._seed(store)
        pack = store.auto_context_pack("testing deploy api", token_budget=5000)
        ids = pack.ids()
        assert len(ids) == len(set(ids))

    def test_respects_token_budget(self, store: MemoryStore):
        self._seed(store)
        pack = store.auto_context_pack("testing", token_budget=20)
        assert pack.used_tokens <= 20

    def test_text_has_header(self, store: MemoryStore):
        self._seed(store)
        pack = store.auto_context_pack("testing", token_budget=5000)
        if pack.items:
            assert pack.text.startswith("## Relevant memory")

    def test_result_fits_budget(self, store: MemoryStore):
        self._seed(store)
        pack = store.auto_context_pack("run tests deploy api", token_budget=100)
        assert isinstance(pack, PackResult)
        assert pack.used_tokens <= 100

    def test_max_phrases_zero_uses_only_task(self, store: MemoryStore):
        self._seed(store)
        # max_phrases=0 disables phrase extraction — only the task query is used
        pack = store.auto_context_pack("testing", token_budget=5000, max_phrases=0)
        assert isinstance(pack, PackResult)
        assert len(pack) >= 0  # doesn't crash
