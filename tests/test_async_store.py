"""Tests for AsyncMemoryStore."""

from __future__ import annotations

import asyncio
import time

import pytest
import pytest_asyncio

from ai_houkai.memory_system import AsyncMemoryStore


@pytest.fixture()
def astore(tmp_path):
    s = AsyncMemoryStore(
        path=str(tmp_path / "chroma"), collection="test_async"
    )
    yield s
    s.close()


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestAsyncRemember:
    def test_returns_memory(self, astore):
        mem = _run(astore.remember("Python uses GIL for thread safety."))
        assert mem.id
        assert len(mem.id) == 36

    def test_count_increments(self, astore):
        assert _run(astore.count()) == 0
        _run(astore.remember("first"))
        _run(astore.remember("second"))
        assert _run(astore.count()) == 2

    def test_custom_metadata(self, astore):
        mem = _run(astore.remember(
            "Deploy with make release",
            type="procedural",
            tags=["deploy"],
            importance=0.9,
            source="user",
        ))
        assert mem.type == "procedural"
        assert "deploy" in mem.tags
        assert mem.importance == 0.9

    def test_text_stripped(self, astore):
        mem = _run(astore.remember("  spaces  "))
        assert mem.text == "spaces"


class TestAsyncForget:
    def test_deletes(self, astore):
        mem = _run(astore.remember("to forget"))
        assert _run(astore.forget(mem.id)) is True
        assert _run(astore.count()) == 0

    def test_unknown_id_returns_false(self, astore):
        assert _run(astore.forget("00000000-0000-0000-0000-000000000000")) is False


class TestAsyncRecall:
    def test_basic_search(self, astore):
        _run(astore.remember("The sky is blue", type="semantic", importance=0.8))
        _run(astore.remember("Python is great", type="semantic", importance=0.7))
        hits = _run(astore.recall("sky colour", k=3))
        assert hits
        texts = [m.text for m, _ in hits]
        assert any("sky" in t for t in texts)

    @pytest.mark.needs_model
    def test_scores_in_range(self, astore):
        _run(astore.remember("memory alpha"))
        hits = _run(astore.recall("alpha", k=5))
        for _, score in hits:
            assert 0.0 <= score <= 1.0

    def test_type_filter(self, astore):
        _run(astore.remember("episodic one", type="episodic"))
        _run(astore.remember("semantic one", type="semantic"))
        hits = _run(astore.recall("one", k=10, type="episodic"))
        assert all(m.type == "episodic" for m, _ in hits)

    def test_access_count_increments(self, astore):
        _run(astore.remember("recall me", type="semantic"))
        _run(astore.recall("recall me", k=1))
        _run(astore.recall("recall me", k=1))
        recent = _run(astore.list_recent(limit=5))
        assert recent[0].access_count == 2


class TestAsyncRecallPack:
    def test_returns_pack_result(self, astore):
        _run(astore.remember("alpha beta gamma"))
        result = _run(astore.recall_pack("alpha", token_budget=200))
        assert result.used_tokens <= result.budget
        assert isinstance(result.text, str)


class TestAsyncLinks:
    def test_link_and_neighbors(self, astore):
        a = _run(astore.remember("concept A"))
        b = _run(astore.remember("concept B"))
        _run(astore.link(a.id, b.id, "related"))
        nb = _run(astore.neighbors(a.id, direction="out"))
        assert any(m.id == b.id for m, _ in nb)

    def test_unlink(self, astore):
        a = _run(astore.remember("src"))
        b = _run(astore.remember("dst"))
        _run(astore.link(a.id, b.id, "related"))
        removed = _run(astore.unlink(a.id, b.id, "related"))
        assert removed == 1

    def test_subgraph(self, astore):
        a = _run(astore.remember("node A"))
        b = _run(astore.remember("node B"))
        _run(astore.link(a.id, b.id, "refines"))
        g = _run(astore.subgraph([a.id], depth=1))
        assert a.id in g.nodes
        assert b.id in g.nodes


class TestAsyncConflictSupersede:
    def test_supersede_and_restore(self, astore):
        old = _run(astore.remember("old fact"))
        new = _run(astore.remember("new fact"))
        _run(astore.supersede(old.id, new.id))
        recent = _run(astore.list_recent(limit=10, include_superseded=True))
        old_mem = next(m for m in recent if m.id == old.id)
        assert old_mem.superseded_by == new.id
        assert _run(astore.restore(old.id)) is True


class TestAsyncContextManager:
    def test_async_with(self, tmp_path):
        async def _inner():
            async with AsyncMemoryStore(
                path=str(tmp_path / "chroma"), collection="ctx_test"
            ) as store:
                mem = await store.remember("context manager test")
                assert mem.id
                return await store.count()

        count = asyncio.get_event_loop().run_until_complete(_inner())
        assert count == 1


class TestAsyncRunHelper:
    def test_run_passthrough(self, astore):
        _run(astore.remember("hello"))
        count = _run(astore.run(astore.sync.count))
        assert count == 1


class TestAsyncConcurrency:
    def test_parallel_remembers(self, astore):
        async def _inner():
            tasks = [
                astore.remember(f"memory {i}", type="semantic")
                for i in range(10)
            ]
            return await asyncio.gather(*tasks)

        mems = asyncio.get_event_loop().run_until_complete(_inner())
        assert len(mems) == 10
        assert _run(astore.count()) == 10

    def test_interleaved_write_read(self, astore):
        async def _inner():
            await astore.remember("shared memory one")
            await astore.remember("shared memory two")
            hits = await astore.recall("shared memory", k=5)
            return hits

        hits = asyncio.get_event_loop().run_until_complete(_inner())
        assert len(hits) >= 1


class TestCloseOrdering:
    def test_close_drains_executor_before_closing_client(self, tmp_path):
        """Regression: close() used to close the Chroma client first, so a
        queued job could run against a closed connection."""
        astore = AsyncMemoryStore(
            path=str(tmp_path / "chroma"), collection="close_order")
        outcome = {}

        def slow_count():
            time.sleep(0.3)                    # still queued when close() starts
            outcome["count"] = astore.sync.count()

        astore._executor.submit(slow_count)
        astore.close()                          # must wait, then close client
        assert outcome.get("count") == 0        # job ran against a live client


class TestAsyncAutoContext:
    def test_auto_context_pack_wrapper(self, astore):
        async def _inner():
            await astore.remember(
                "The deploy pipeline runs through GitHub Actions.",
                type="procedural")
            return await astore.auto_context_pack(
                "deploy the api to production", token_budget=500)

        pack = asyncio.get_event_loop().run_until_complete(_inner())
        assert pack.budget == 500
        assert pack.items
        assert "## Relevant memory" in pack.text

    def test_auto_context_pack_touch_false(self, astore):
        async def _inner():
            mem = await astore.remember(
                "The deploy pipeline runs through GitHub Actions.",
                type="procedural")
            await astore.auto_context_pack(
                "deploy the api", token_budget=500, touch=False)
            return await astore.run(astore.sync._get_by_id, mem.id)

        after = asyncio.get_event_loop().run_until_complete(_inner())
        assert after.access_count == 0


class TestAsyncDiagnostics:
    """The async wrapper must expose the diagnostics surface at parity with
    the sync store (probe_embedding / readiness)."""

    def test_probe_embedding(self, astore):
        probe = _run(astore.probe_embedding())
        assert probe["ok"] is True
        assert probe["dim"] > 0

    def test_readiness_ok(self, astore):
        r = _run(astore.readiness())
        assert r["ready"] is True
        assert r["checks"]["store"]["ok"] is True
        assert r["checks"]["embedder"]["ok"] is True
