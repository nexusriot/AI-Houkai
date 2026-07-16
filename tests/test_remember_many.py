"""Tests for MemoryStore.remember_many — batched bulk writes."""

from __future__ import annotations

import asyncio
from unittest import mock

import pytest

from ai_houkai.memory_system import AsyncMemoryStore, MemoryStore, RememberItem


def _run(coro):
    # A private loop (not asyncio.get_event_loop()) so this file never clears
    # or replaces the loop that tests/test_async_store.py relies on.
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestRememberManyBasics:
    def test_stores_all_in_input_order(self, store):
        out = store.remember_many(
            ["alpha fact", RememberItem(text="beta fact"), {"text": "gamma fact"}]
        )
        assert [m.text for m in out] == ["alpha fact", "beta fact", "gamma fact"]
        assert store.count() == 3

    def test_empty_returns_empty_no_write(self, store):
        assert store.remember_many([]) == []
        assert store.count() == 0

    def test_field_mapping_across_forms(self, store):
        out = store.remember_many([
            RememberItem(text="b", type="procedural", tags=("x", "y"), importance=0.9),
            {"text": "c", "type": "feedback", "source": "unit", "polarity": 1},
        ])
        assert out[0].type == "procedural" and out[0].tags == ["x", "y"]
        assert abs(out[0].importance - 0.9) < 1e-9
        assert out[1].type == "feedback" and out[1].source == "unit" and out[1].polarity == 1

    def test_bare_string_uses_defaults(self, store):
        (mem,) = store.remember_many(["just text"])
        assert mem.type == "semantic" and mem.tags == [] and abs(mem.importance - 0.5) < 1e-9


class TestRememberManyBatching:
    def test_ceil_add_calls(self, store):
        # 10 items at batch_size=4 → 3 collection.add calls (4, 4, 2), i.e. the
        # embedding is batched into ceil(N / batch_size) encode passes, not N.
        with mock.patch.object(store.collection, "add", wraps=store.collection.add) as add:
            out = store.remember_many([f"chunk number {i}" for i in range(10)], batch_size=4)
        assert len(out) == 10 and store.count() == 10
        assert add.call_count == 3
        assert [len(c.kwargs["ids"]) for c in add.call_args_list] == [4, 4, 2]

    def test_batch_size_must_be_positive(self, store):
        with pytest.raises(ValueError):
            store.remember_many(["a"], batch_size=0)


class TestRememberManyJournal:
    def test_one_entry_per_id(self, store):
        out = store.remember_many(["one", "two", "three"])
        for m in out:
            assert len(list(store.journal.read(op="remember", memory_id=m.id))) == 1

    def test_undo_is_per_id(self, store):
        out = store.remember_many(["keep me", "undo me"])
        entry = list(store.journal.read(memory_id=out[1].id))[-1]
        assert store.undo(entry) is True
        assert store.count() == 1
        assert store._get_by_id(out[1].id) is None
        assert store._get_by_id(out[0].id) is not None


class TestRememberManyValidation:
    def test_bad_item_aborts_before_any_write(self, store):
        with pytest.raises(ValueError):
            store.remember_many(["ok one", {"text": "bad", "type": "not-a-type"}])
        assert store.count() == 0

    def test_unknown_field_raises(self, store):
        with pytest.raises(ValueError):
            store.remember_many([{"text": "x", "bogus": 1}])

    def test_missing_text_raises(self, store):
        with pytest.raises(ValueError):
            store.remember_many([{"type": "semantic"}])

    def test_non_item_type_raises(self, store):
        with pytest.raises(TypeError):
            store.remember_many([123])

    def test_ttl_seconds_sets_expires_at(self, store):
        (mem,) = store.remember_many([{"text": "temp", "ttl_seconds": 3600}])
        assert mem.expires_at > 0

    def test_ttl_and_expires_at_mutually_exclusive(self, store):
        with pytest.raises(ValueError):
            store.remember_many([{"text": "x", "ttl_seconds": 60, "expires_at": 1}])


class TestRememberManyConflicts:
    def test_ignore_stores_all_without_scan(self, store):
        out = store.remember_many(
            ["Use ruff for linting", "Use ruff for linting please"],
            on_conflict="ignore",
        )
        assert store.count() == 2
        assert all(store._get_by_id(m.id).superseded_by == "" for m in out)

    def test_warn_stores_all_and_warns_once(self, store):
        with pytest.warns(UserWarning, match=r"remember_many\(\)"):
            store.remember_many(
                ["Use ruff for linting", "Use ruff for linting please"],
                on_conflict="warn",
            )
        assert store.count() == 2

    def test_supersede_earlier_wins_no_cycle(self, store):
        first, second = store.remember_many(
            ["Use ruff for linting", "Use ruff for linting please"],
            on_conflict="supersede",
        )
        assert store._get_by_id(second.id).superseded_by == first.id
        assert store._get_by_id(first.id).superseded_by == ""

    def test_raise_policy_rejected(self, store):
        with pytest.raises(ValueError, match="raise"):
            store.remember_many(["x"], on_conflict="raise")


class TestRememberManyImportance:
    def test_autoscores_when_store_fn_configured(self, tmp_path):
        s = MemoryStore(
            path=str(tmp_path / "chroma"),
            collection="imp_test",
            importance_fn=lambda text, type, tags: 0.99,
        )
        try:
            (mem,) = s.remember_many(["auto scored"])
            assert abs(mem.importance - 0.99) < 1e-9
            # explicit importance still wins over the auto-scorer
            (mem2,) = s.remember_many([{"text": "explicit", "importance": 0.1}])
            assert abs(mem2.importance - 0.1) < 1e-9
        finally:
            s.client.close()


class TestAsyncRememberMany:
    def test_async_stores_all(self, tmp_path):
        s = AsyncMemoryStore(path=str(tmp_path / "chroma"), collection="async_rm")
        try:
            out = _run(s.remember_many(["a fact", RememberItem(text="b fact")]))
            assert [m.text for m in out] == ["a fact", "b fact"]
            assert _run(s.count()) == 2
        finally:
            s.close()
