"""Tests for MemoryStore.remember_many — batched bulk writes."""

from __future__ import annotations

import asyncio
import inspect
import json
import threading
from http.server import ThreadingHTTPServer
from unittest import mock
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

import ai_houkai.mcp_server.server as mcp_srv
from ai_houkai.http_server.server import build_handler
from ai_houkai.memory_system import AsyncMemoryStore, MemoryStore, RememberItem
from ai_houkai.testing import FakeEmbedder


@pytest.fixture()
def mcp_store(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_HOUKAI_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("AI_HOUKAI_COLLECTION", "batch_mcp")
    monkeypatch.setattr(mcp_srv, "_store", None)
    yield
    if mcp_srv._store is not None:
        mcp_srv._store.client.close()
        mcp_srv._store = None


@pytest.fixture()
def http_client(tmp_path):
    """A live threaded HTTP server over a throwaway store."""
    store = MemoryStore(path=str(tmp_path / "chroma"), collection="batch_http",
                        embedding_function=FakeEmbedder(dim=16))
    server = ThreadingHTTPServer(("127.0.0.1", 0), build_handler(store))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def call(method, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = Request(f"{base}{path}", data=data, method=method,
                      headers={"Content-Type": "application/json"})
        try:
            with urlopen(req) as resp:
                return resp.status, json.loads(resp.read() or b"{}")
        except HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    yield call, store
    server.shutdown()
    server.server_close()
    store.client.close()


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

    @pytest.mark.needs_model
    def test_warn_stores_all_and_warns_once(self, store):
        with pytest.warns(UserWarning, match=r"remember_many\(\)"):
            store.remember_many(
                ["Use ruff for linting", "Use ruff for linting please"],
                on_conflict="warn",
            )
        assert store.count() == 2

    @pytest.mark.needs_model
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


class TestBatchIdempotentSurfaces:
    """`idempotent` has to reach every surface that offers bulk write.

    A store parameter that only the library can pass is the same class of gap
    `parity.json` guards for MCP/HTTP: the feature exists but the callers who
    need it most (an agent replaying a batch every session) cannot reach it.
    """

    def test_store_signature(self):
        assert "idempotent" in inspect.signature(
            MemoryStore.remember_many).parameters

    def test_async_signature(self):
        assert "idempotent" in inspect.signature(
            AsyncMemoryStore.remember_many).parameters

    def test_mcp_tool_signature(self):
        assert "idempotent" in inspect.signature(
            mcp_srv.remember_many).parameters

    def test_mcp_tool_dedupes(self, mcp_store):
        out = mcp_srv.remember_many(
            items=[{"text": "Use ruff for linting"},
                   {"text": "use  ruff for linting"}],
            idempotent=True)
        # Two inputs, one row: every input still gets an id, and `stored`
        # counts rows created — see TestBatchStoredCountsOnlyNewRows.
        assert len(out["ids"]) == 2
        assert len(set(out["ids"])) == 1, "normalised duplicates must collapse"
        assert out["stored"] == 1
        assert mcp_srv.stats()["count"] == 1

    def test_http_batch_dedupes(self, http_client):
        call, store = http_client
        status, body = call("POST", "/memories/batch", {
            "items": [{"text": "shared assertion"}, {"text": "shared  assertion"}],
            "idempotent": True,
        })
        assert status == 201
        assert len({m["id"] for m in body["memories"]}) == 1
        assert store.count() == 1

    def test_http_batch_default_still_duplicates(self, http_client):
        call, store = http_client
        call("POST", "/memories/batch",
             {"items": [{"text": "twice over"}, {"text": "twice over"}]})
        assert store.count() == 2, "idempotency must stay opt-in"


class TestBatchStoredCountsOnlyNewRows:
    """`stored` is how many rows the batch created.

    With `idempotent`, a replayed batch creates nothing — so reporting
    `stored: len(items)` told a client it had written N rows when it had written
    none, the same mis-report the single-write path had. Intra-batch duplicates
    collapse to one row, so the count is over distinct ids.
    """

    def test_a_replayed_batch_reports_no_new_rows(self, http_client):
        http_client, _store = http_client
        items = [{"text": "batch fact one"}, {"text": "batch fact two"}]
        status, first = http_client("POST", "/memories/batch",
                                    {"items": items, "idempotent": True})
        assert status == 201, first
        assert first["stored"] == 2

        status, again = http_client("POST", "/memories/batch",
                                   {"items": items, "idempotent": True})
        assert status == 200, again
        assert again["stored"] == 0
        assert len(again["memories"]) == 2, "the existing rows still come back"

    def test_a_partly_known_batch_counts_only_the_new_one(self, http_client):
        http_client, _store = http_client
        http_client("POST", "/memories", {"text": "already known",
                                          "idempotent": True})
        status, out = http_client(
            "POST", "/memories/batch",
            {"items": [{"text": "already known"}, {"text": "brand new"}],
             "idempotent": True})
        assert status == 201, out
        assert out["stored"] == 1

    def test_intra_batch_duplicates_count_once(self, http_client):
        http_client, _store = http_client
        status, out = http_client(
            "POST", "/memories/batch",
            {"items": [{"text": "same text"}, {"text": "same text"}],
             "idempotent": True})
        assert status == 201, out
        assert out["stored"] == 1

    def test_without_the_flag_every_item_is_a_new_row(self, http_client):
        http_client, _store = http_client
        items = [{"text": "dup text"}, {"text": "dup text"}]
        _, out = http_client("POST", "/memories/batch", {"items": items})
        assert out["stored"] == 2
