"""Tests for metadata-filtered recall: source, since/until, and the
`_build_where` clause builder that keeps Chroma's single-operator rule."""

from __future__ import annotations

import time

import pytest

from ai_houkai.memory_system.store import _build_where


class TestBuildWhere:
    def test_empty_is_none(self):
        assert _build_where() is None

    def test_single_condition_is_flat(self):
        assert _build_where(type="semantic") == {"type": "semantic"}
        assert _build_where(source="repo") == {"source": "repo"}

    def test_min_importance_single(self):
        assert _build_where(min_importance=0.5) == {"importance": {"$gte": 0.5}}

    def test_multiple_conditions_use_and(self):
        clause = _build_where(type="semantic", min_importance=0.5)
        assert clause == {"$and": [{"type": "semantic"},
                                   {"importance": {"$gte": 0.5}}]}

    def test_since_until_are_separate_leaves(self):
        # Chroma rejects {"created_at": {"$gte": x, "$lte": y}} — must split.
        clause = _build_where(since=100.0, until=200.0)
        assert clause == {"$and": [{"created_at": {"$gte": 100.0}},
                                   {"created_at": {"$lte": 200.0}}]}

    def test_every_leaf_has_one_operator(self):
        clause = _build_where(
            type="semantic", min_importance=0.3, source="x",
            since=1.0, until=2.0,
        )
        assert "$and" in clause
        for leaf in clause["$and"]:
            assert len(leaf) == 1
            (val,) = leaf.values()
            if isinstance(val, dict):
                assert len(val) == 1  # exactly one $-operator


class TestRecallFilters:
    def _seed(self, store):
        old = store.remember("alpha auth login", source="repo-a")
        new = store.remember("beta auth logout", source="repo-b")
        # Backdate one record so since/until can separate them.
        store.collection.update(ids=[old.id], metadatas=[{
            **store._get_by_id(old.id).to_metadata(), "created_at": 1000.0,
        }])
        store.collection.update(ids=[new.id], metadatas=[{
            **store._get_by_id(new.id).to_metadata(), "created_at": 9_000_000_000.0,
        }])
        return old, new

    def test_source_filter(self, store):
        self._seed(store)
        hits = store.recall("auth", k=10, source="repo-a")
        ids = {m.source for m, _ in hits}
        assert ids == {"repo-a"}

    def test_since_filter_drops_old(self, store):
        old, new = self._seed(store)
        hits = store.recall("auth", k=10, since=2000.0)
        got = {m.id for m, _ in hits}
        assert new.id in got and old.id not in got

    def test_until_filter_drops_new(self, store):
        old, new = self._seed(store)
        hits = store.recall("auth", k=10, until=2000.0)
        got = {m.id for m, _ in hits}
        assert old.id in got and new.id not in got

    def test_type_and_min_importance_combine(self, store):
        # This combination used to build a rejected multi-key where clause.
        store.remember("gamma important", type="semantic", importance=0.9)
        store.remember("delta trivial", type="episodic", importance=0.1)
        hits = store.recall("gamma delta", k=10,
                            type="semantic", min_importance=0.5)
        assert len(hits) == 1
        assert hits[0][0].type == "semantic"

    def test_semantic_tag_with_include_superseded_not_underfetched(self, store):
        """Regression: semantic recall took a fetch-exactly-k fast path when
        include_superseded=True, ignoring that the `tag` filter still runs
        post-query — so higher-ranked untagged hits crowded out the tagged
        ones and recall returned fewer than k matches."""
        # Six untagged near-exact matches outrank the two tagged memories.
        for _ in range(6):
            store.remember("alpha beta gamma", type="semantic")
        keep1 = store.remember("alpha beta", type="semantic", tags=["keep"])
        keep2 = store.remember("alpha beta delta", type="semantic", tags=["keep"])

        hits = store.recall(
            "alpha beta gamma", k=2, tag="keep",
            mode="semantic", include_superseded=True,
        )
        got = {m.id for m, _ in hits}
        assert got == {keep1.id, keep2.id}

    def test_pack_respects_source(self, store):
        store.remember("epsilon note", source="repo-a")
        store.remember("zeta note", source="repo-b")
        res = store.recall_pack("note", source="repo-a", token_budget=200)
        assert all(p.memory.source == "repo-a" for p in res.items)
        assert res.items
