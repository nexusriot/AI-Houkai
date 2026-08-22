"""Tests for AsyncMemoryStore."""

from __future__ import annotations

import asyncio
import inspect
import time

import pytest
import pytest_asyncio

from ai_houkai.memory_system import AsyncMemoryStore, MemoryStore


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


class TestAsyncSurfaceCompleteness:
    """AsyncMemoryStore is documented public API, so a missing wrapper is a
    functional gap, not an omission — an async caller simply cannot reach the
    feature. This is the drift `parity.json` catches for MCP/HTTP; the async
    wrapper had no equivalent guard.
    """

    CURATION = (
        "merge", "versions", "list_tags", "rename_tag", "merge_tags",
        "delete_tag", "find_path",
    )
    TRASH = (
        "trash", "trash_list", "trash_restore", "trash_purge",
        "trash_purge_expired",
    )

    @pytest.mark.parametrize("name", CURATION + TRASH)
    def test_method_is_wrapped(self, name):
        assert hasattr(AsyncMemoryStore, name), (
            f"AsyncMemoryStore is missing {name}()")

    @pytest.mark.parametrize("flag", ["pinned", "trust", "idempotent"])
    def test_remember_exposes_write_flag(self, flag):
        params = inspect.signature(AsyncMemoryStore.remember).parameters
        assert flag in params, f"async remember() cannot set {flag}"

    @pytest.mark.parametrize("flag", ["pinned", "trust"])
    def test_edit_exposes_write_flag(self, flag):
        params = inspect.signature(AsyncMemoryStore.edit).parameters
        assert flag in params, f"async edit() cannot set {flag}"

    def test_recall_exposes_min_trust_and_lexical_index(self):
        params = inspect.signature(AsyncMemoryStore.recall).parameters
        for knob in ("min_trust", "lexical_index"):
            assert knob in params, f"async recall() is missing {knob}"

    def test_every_public_store_method_has_a_wrapper(self):
        """Catches the next omission automatically rather than by review."""
        skip = {
            # Sync-only plumbing, or deliberately not part of the async surface.
            "as_actor", "journal", "collection", "client", "index",
            "trash_path", "embedder_name",
        }
        missing = []
        for name in dir(MemoryStore):
            if name.startswith("_") or name in skip:
                continue
            if not callable(getattr(MemoryStore, name, None)):
                continue
            if not hasattr(AsyncMemoryStore, name):
                missing.append(name)
        assert not missing, f"AsyncMemoryStore is missing: {sorted(missing)}"


class TestAsyncNewMethodsWork:
    def test_trash_roundtrip(self, tmp_path):
        async def go():
            store = AsyncMemoryStore(path=str(tmp_path / "c"),
                                     collection="async_trash")
            try:
                mem = await store.remember("async trash subject")
                assert await store.trash(mem.id) is True
                assert [e.memory_id for e in await store.trash_list()] == [mem.id]
                assert (await store.trash_restore(mem.id)).id == mem.id
                assert await store.trash_list() == []
            finally:
                await store.aclose()

        _run(go())

    def test_curation_roundtrip(self, tmp_path):
        async def go():
            store = AsyncMemoryStore(path=str(tmp_path / "c"),
                                     collection="async_curation")
            try:
                a = await store.remember("async merge target", tags=["ops"])
                b = await store.remember("async merge source", tags=["ops"])
                merged = await store.merge(a.id, b.id)
                assert "async merge source" in merged.text
                assert await store.list_tags() == [("ops", 1)]
                assert (await store.rename_tag("ops", "operations")).changed == 1
            finally:
                await store.aclose()

        _run(go())

    def test_write_flags_reach_the_store(self, tmp_path):
        async def go():
            store = AsyncMemoryStore(path=str(tmp_path / "c"),
                                     collection="async_flags")
            try:
                mem = await store.remember("async standing instruction",
                                           pinned=True, trust="reported")
                assert mem.pinned is True and mem.trust == "reported"
                again = await store.remember("async standing instruction",
                                             idempotent=True)
                assert again.id == mem.id
            finally:
                await store.aclose()

        _run(go())


class TestAsyncContentHashLookup:
    def test_finds_a_live_row(self, astore):
        mem = _run(astore.remember("an async fact worth repeating"))
        found = _run(astore.find_by_content_hash(
            "an async fact worth repeating"))
        assert found is not None and found.id == mem.id

    def test_misses_when_nothing_matches(self, astore):
        _run(astore.remember("something else"))
        assert _run(astore.find_by_content_hash("never written")) is None


class TestAsyncSurfaceParity:
    """The async store is a thin wrapper, so it drifts silently.

    Every knob added to a sync method has to be repeated by hand in the
    wrapper, and forgetting one is invisible: the parameter simply is not
    accepted, and only a caller that reaches for it finds out. This compares
    the two signatures mechanically instead of trusting review.

    ``as_actor`` is deliberately exempt — it is a synchronous context manager
    for scoping the journal actor, not an awaitable operation.
    """

    EXEMPT = {"as_actor"}

    @staticmethod
    def _sync_methods() -> dict[str, object]:
        """Public callables on ``MemoryStore`` *and* everything it inherits.

        Walking the MRO is the whole point: twelve curation methods (``trash``,
        ``merge``, the tag ops, …) reach ``MemoryStore`` from ``CurationMixin``,
        and ``vars(MemoryStore)`` sees none of them — so a sweep built on the
        class ``__dict__`` would silently exempt a quarter of the surface.
        """
        found: dict[str, object] = {}
        for klass in reversed(MemoryStore.__mro__):
            if klass is object:
                continue
            for name, attr in vars(klass).items():
                if not name.startswith("_") and callable(attr):
                    found[name] = attr
        return found

    def test_the_sweep_reaches_inherited_methods(self):
        """Guards the guard.

        If ``_sync_methods`` ever narrows back to ``MemoryStore.__dict__``, the
        two checks below keep passing while covering nothing from the mixin.
        This fails loudly instead.
        """
        found = self._sync_methods()
        assert "recall" in found, "own methods missing from the sweep"
        assert {"trash", "merge", "list_tags"} <= set(found), \
            "inherited CurationMixin methods are not being inspected"

    @staticmethod
    def _named_params(fn) -> set[str] | None:
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):  # pragma: no cover — C-level callables
            return None
        return {name for name, p in sig.parameters.items()
                if name != "self" and p.kind is not p.VAR_KEYWORD}

    def test_every_sync_method_has_a_wrapper(self):
        missing = [
            name for name in self._sync_methods()
            if name not in self.EXEMPT
            and getattr(AsyncMemoryStore, name, None) is None
        ]
        assert not missing, f"AsyncMemoryStore is missing: {sorted(missing)}"

    def test_no_wrapper_drops_a_parameter(self):
        gaps = []
        for name, sync_fn in self._sync_methods().items():
            async_fn = getattr(AsyncMemoryStore, name, None)
            if async_fn is None or name in self.EXEMPT:
                continue
            sync_params = self._named_params(sync_fn)
            async_params = self._named_params(async_fn)
            if sync_params is None or async_params is None:
                continue
            dropped = sync_params - async_params
            if dropped:
                gaps.append(f"{name}: missing {sorted(dropped)}")
        assert not gaps, "async wrappers dropped parameters:\n  " + \
            "\n  ".join(sorted(gaps))
