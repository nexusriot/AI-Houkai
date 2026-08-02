"""MemoryStore.get() is public API (A2).

It was `_get_by_id` — a private called from the CLI, the HTTP server and
out-of-tree consumers. The alias stays one release so those keep working.
"""

from __future__ import annotations

import asyncio
import time

import pytest

import ai_houkai.mcp_server.server as srv
from ai_houkai.memory_system import AsyncMemoryStore, MemoryStore


@pytest.fixture()
def store(tmp_path):
    s = MemoryStore(path=str(tmp_path / "chroma"), collection="get_test")
    yield s
    s.client.close()


@pytest.fixture()
def mcp_store(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_HOUKAI_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("AI_HOUKAI_COLLECTION", "get_mcp")
    monkeypatch.setattr(srv, "_store", None)
    yield
    if srv._store is not None:
        srv._store.client.close()
        srv._store = None


class TestStoreGet:
    def test_returns_the_memory(self, store):
        mem = store.remember("a fact worth fetching", tags=["x"])
        got = store.get(mem.id)
        assert got is not None
        assert got.id == mem.id
        assert got.text == "a fact worth fetching"
        assert got.tags == ["x"]

    def test_missing_id_is_none(self, store):
        assert store.get("no-such-id") is None

    def test_is_a_plain_read(self, store):
        """No access-count bump and no journal entry — unlike recall."""
        mem = store.remember("plain read subject")
        before = len(list(store.journal.read()))
        store.get(mem.id)
        store.get(mem.id)
        assert store.get(mem.id).access_count == 0
        assert len(list(store.journal.read())) == before

    def test_returns_superseded_and_expired(self, store):
        old = store.remember("the old fact")
        new = store.remember("the new fact")
        store.supersede(old_id=old.id, new_id=new.id)
        assert store.get(old.id).superseded_by == new.id

        gone = store.remember("already expired", expires_at=time.time() - 1)
        assert store.recall("already expired", k=5) == [] or all(
            m.id != gone.id for m, _ in store.recall("already expired", k=5))
        assert store.get(gone.id) is not None

    def test_private_alias_still_works(self, store):
        mem = store.remember("legacy caller subject")
        assert store._get_by_id(mem.id).id == mem.id
        assert MemoryStore._get_by_id is MemoryStore.get


class TestAsyncGet:
    def test_async_wrapper(self, tmp_path):
        async def go():
            store = AsyncMemoryStore(
                path=str(tmp_path / "chroma"), collection="get_async")
            try:
                mem = await store.remember("async fetch subject")
                got = await store.get(mem.id)
                assert got is not None and got.id == mem.id
                assert await store.get("nope") is None
            finally:
                await store.aclose()

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(go())
        finally:
            loop.close()


class TestMcpGetTool:
    def test_found(self, mcp_store):
        created = srv.remember(text="mcp get subject", tags=["a"], importance=0.7)
        out = srv.get(memory_id=created["id"])
        assert out["found"] is True
        assert out["text"] == "mcp get subject"
        assert out["tags"] == ["a"]
        assert out["importance"] == 0.7
        assert out["access_count"] == 0
        assert out["links"] == []

    def test_not_found(self, mcp_store):
        out = srv.get(memory_id="missing")
        assert out == {"found": False, "id": "missing"}

    def test_reports_links_and_supersede_state(self, mcp_store):
        a = srv.remember(text="mcp get link source")
        b = srv.remember(text="mcp get link target")
        srv.link(src_id=a["id"], dst_id=b["id"], rel="refines")
        srv.supersede(old_id=b["id"], new_id=a["id"])
        out_a = srv.get(memory_id=a["id"])
        assert {"to": b["id"], "rel": "refines"} in out_a["links"]
        assert srv.get(memory_id=b["id"])["superseded_by"] == a["id"]
