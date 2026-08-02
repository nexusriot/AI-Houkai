"""Tests for pluggable cross-encoder reranking (Feature 1)."""

from __future__ import annotations

import pytest

from ai_houkai.memory_system import MemoryStore


def _seed(store: MemoryStore) -> list:
    # Distinct docs that all match the query "topic" lexically/semantically.
    return [store.remember(f"topic entry number {i}") for i in range(6)]


class TestReranker:
    def test_reranker_reorders_results(self, store: MemoryStore):
        mems = _seed(store)
        wanted = mems[-1].id  # a specific memory we force to the top

        def rr(query, cands):
            # Score 1.0 for the wanted memory, 0.0 for everyone else.
            return [1.0 if m.id == wanted else 0.0 for m in cands]

        hits = store.recall("topic entry", k=3, reranker=rr)
        assert hits[0][0].id == wanted
        assert hits[0][1] == 1.0  # reranker score replaces the blended score

    def test_reranker_can_promote_candidate_outside_first_stage_topk(
        self, store: MemoryStore
    ):
        _seed(store)
        # Big overfetch so the pool holds the whole first-stage ranking.
        first_stage = store.recall("topic entry", k=5, overfetch=20)
        assert len(first_stage) >= 3
        first_top = first_stage[0][0].id
        target = first_stage[-1][0].id       # first-stage LAST → force to top
        assert target != first_top

        def rr(query, cands):
            return [1.0 if m.id == target else 0.0 for m in cands]

        # k=1: without rerank we'd get first_top; the reranker promotes the
        # pool's lowest-ranked candidate into the single returned slot.
        top = store.recall("topic entry", k=1, overfetch=20, reranker=rr)[0][0].id
        assert top == target

    def test_explain_records_rerank_block(self, store: MemoryStore):
        _seed(store)

        def rr(query, cands):
            return list(range(len(cands)))  # ascending → last wins

        hits = store.recall("topic entry", k=3, reranker=rr, explain=True)
        top_expl = hits[0][2]
        assert "rerank" in top_expl
        rr_info = top_expl["rerank"]
        assert rr_info["rank"] == 0                      # new top
        assert "first_stage_rank" in rr_info
        assert "first_stage_score" in rr_info
        assert "score" in rr_info

    @pytest.mark.needs_model
    def test_per_store_default_reranker_applies(self, tmp_path):
        wanted = {}

        def rr(query, cands):
            return [1.0 if m.id == wanted.get("id") else 0.0 for m in cands]

        store = MemoryStore(path=str(tmp_path / "chroma"),
                            collection="rr_default", reranker=rr)
        try:
            mems = _seed(store)
            wanted["id"] = mems[2].id
            hits = store.recall("topic entry", k=1)  # no per-call reranker
            assert hits[0][0].id == mems[2].id
        finally:
            store.client.close()

    def test_per_call_reranker_overrides_store_default(self, tmp_path):
        def store_rr(query, cands):
            return [1.0] * len(cands)  # everyone tied — order preserved

        store = MemoryStore(path=str(tmp_path / "chroma"),
                            collection="rr_override", reranker=store_rr)
        try:
            mems = _seed(store)
            call_target = mems[4].id

            def call_rr(query, cands):
                return [1.0 if m.id == call_target else 0.0 for m in cands]

            hits = store.recall("topic entry", k=1, reranker=call_rr)
            assert hits[0][0].id == call_target
        finally:
            store.client.close()

    def test_reranker_wrong_length_raises(self, store: MemoryStore):
        _seed(store)

        def bad_rr(query, cands):
            return [1.0]  # too few

        with pytest.raises(ValueError):
            store.recall("topic entry", k=3, reranker=bad_rr)

    def test_reranker_receives_query_and_memories(self, store: MemoryStore):
        _seed(store)
        seen = {}

        def rr(query, cands):
            seen["query"] = query
            seen["n"] = len(cands)
            return [0.0] * len(cands)

        store.recall("topic entry hello", k=2, reranker=rr)
        assert seen["query"] == "topic entry hello"
        assert seen["n"] >= 2  # got the overfetch pool, not just k
