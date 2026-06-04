"""Unit tests for recall_pack — token-budgeted context assembly."""

from __future__ import annotations

import pytest

from ai_houkai.memory_system import MemoryStore, PackResult
from ai_houkai.memory_system.store import _estimate_tokens


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
