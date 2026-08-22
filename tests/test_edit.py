"""Tests for the store-level edit() API: in-place updates, re-embedding,
journaling, undo, the async wrapper, and the CLI/HTTP surfaces."""

from __future__ import annotations

import asyncio
import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from typer.testing import CliRunner

from ai_houkai.cli.main import app
from ai_houkai.http_server import make_server
from ai_houkai.memory_system import AsyncMemoryStore, MemoryStore


class TestEditStore:
    @pytest.mark.needs_model
    def test_edit_text_reembeds(self, store: MemoryStore):
        m = store.remember("today the weather in the mountains was rainy",
                           tags=["log"])
        # Competitor deliberately CLOSER to the query than the stale text:
        # if edit failed to re-embed, the stale "weather" embedding loses to
        # the competitor and this test fails.
        store.remember("Paris is a lovely European city")
        store.edit(m.id, text="the capital of France is Paris")

        got = store._get_by_id(m.id)
        assert got.text == "the capital of France is Paris"
        hits = store.recall("what is the capital of France?", k=1)
        assert hits and hits[0][0].id == m.id

    def test_edit_preserves_identity_and_graph(self, store: MemoryStore):
        m = store.remember("original", tags=["keep"])
        other = store.remember("other")
        store.link(m.id, other.id, "refines")
        before = store._get_by_id(m.id)

        store.edit(m.id, text="rewritten")
        after = store._get_by_id(m.id)

        assert after.id == m.id
        assert after.created_at == before.created_at
        assert after.tags == ["keep"]
        assert [(l.to, l.rel) for l in after.links] == [(other.id, "refines")]

    def test_edit_fields_individually(self, store: MemoryStore):
        m = store.remember("x" * 30, type="episodic", importance=0.3,
                           source="orig", polarity=0)
        store.edit(m.id, type="semantic", importance=0.9, polarity=1,
                   tags=["a", "b"])
        got = store._get_by_id(m.id)
        assert got.type == "semantic"
        assert got.importance == 0.9
        assert got.polarity == 1
        assert got.tags == ["a", "b"]
        assert got.source == "orig"          # untouched

    def test_edit_source_none_clears_omitted_keeps(self, store: MemoryStore):
        m = store.remember("with source", source="cli")
        store.edit(m.id, importance=0.7)     # source omitted
        assert store._get_by_id(m.id).source == "cli"
        store.edit(m.id, source=None)        # explicit clear
        assert store._get_by_id(m.id).source is None

    def test_edit_importance_clamped(self, store: MemoryStore):
        m = store.remember("x" * 30)
        store.edit(m.id, importance=7.5)
        assert store._get_by_id(m.id).importance == 1.0

    def test_edit_missing_id_raises(self, store: MemoryStore):
        with pytest.raises(KeyError):
            store.edit("00000000-0000-4000-8000-000000000000", text="nope")

    def test_edit_validates_type_and_polarity(self, store: MemoryStore):
        m = store.remember("x" * 30)
        with pytest.raises(ValueError, match="type must be one of"):
            store.edit(m.id, type="sematic")
        with pytest.raises(ValueError, match="polarity"):
            store.edit(m.id, polarity=3)
        with pytest.raises(ValueError, match="non-empty"):
            store.edit(m.id, text="   ")

    def test_edit_is_journaled(self, store: MemoryStore):
        m = store.remember("before text here")
        store.edit(m.id, text="after text here")
        entries = list(store.journal.read(op="edit"))
        assert len(entries) == 1
        e = entries[0]
        assert e.id == m.id
        assert e.before["text"] == "before text here"
        assert e.after["text"] == "after text here"

    def test_noop_edit_writes_nothing(self, store: MemoryStore):
        m = store.remember("stable text", importance=0.5)
        store.edit(m.id, text="stable text", importance=0.5)
        assert list(store.journal.read(op="edit")) == []

    def test_undo_edit_restores_previous_state(self, store: MemoryStore):
        m = store.remember("version one", importance=0.4, tags=["t1"])
        store.edit(m.id, text="version two", importance=0.9, tags=["t2"])
        entry = list(store.journal.read(op="edit"))[-1]

        assert store.undo(entry) is True
        got = store._get_by_id(m.id)
        assert got.text == "version one"
        assert got.importance == 0.4
        assert got.tags == ["t1"]

    def test_undo_edit_after_forget_returns_false(self, store: MemoryStore):
        m = store.remember("some text here")
        store.edit(m.id, text="new text here")
        entry = list(store.journal.read(op="edit"))[-1]
        store.forget(m.id)
        assert store.undo(entry) is False

    def test_edit_journal_summary_line(self, store: MemoryStore):
        m = store.remember("aaa bbb ccc")
        store.edit(m.id, text="ddd eee fff")
        entry = list(store.journal.read(op="edit"))[-1]
        assert entry.summary().startswith("edit ")
        assert "ddd" in entry.summary()


class TestEditAsync:
    def test_async_edit(self, tmp_path):
        async def scenario():
            astore = AsyncMemoryStore(
                path=str(tmp_path / "chroma"), collection="async_edit")
            try:
                m = await astore.remember("async original text")
                got = await astore.edit(m.id, text="async edited text",
                                        importance=0.8)
                assert got.text == "async edited text"
                assert got.importance == 0.8
            finally:
                await astore.aclose()

        # Private loop, not asyncio.run(): run() clears the thread's current
        # loop on exit, which would break test_async_store's
        # get_event_loop()-based helper when it runs later in the process.
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(scenario())
        finally:
            loop.close()


class TestEditCli:
    @pytest.fixture()
    def cli_store(self, tmp_path):
        s = MemoryStore(path=str(tmp_path / "chroma"), collection="cli_edit")
        yield s, str(tmp_path / "chroma")
        s.client.close()

    def _run(self, store_path, args):
        runner = CliRunner()
        return runner.invoke(
            app, ["--store", store_path, "--collection", "cli_edit"] + args)

    def test_bump_journals_and_updates(self, cli_store):
        store, path = cli_store
        m = store.remember("bump me please", importance=0.5)
        res = self._run(path, ["bump", m.id, "+0.2"])
        assert res.exit_code == 0, res.output
        assert store._get_by_id(m.id).importance == pytest.approx(0.7)
        assert any(e.id == m.id for e in store.journal.read(op="edit"))

    def test_bump_bad_delta_clean_error(self, cli_store):
        store, path = cli_store
        m = store.remember("bump me please")
        res = self._run(path, ["bump", m.id, "+abc"])
        assert res.exit_code == 1
        assert "Traceback" not in res.output
        assert "delta" in res.output

    def test_tag_journals(self, cli_store):
        store, path = cli_store
        m = store.remember("tag me please", tags=["old"])
        res = self._run(path, ["tag", m.id, "--add", "new", "--remove", "old"])
        assert res.exit_code == 0, res.output
        assert store._get_by_id(m.id).tags == ["new"]
        assert any(e.id == m.id for e in store.journal.read(op="edit"))


class TestEditHttp:
    @pytest.fixture()
    def server(self, store):
        httpd = make_server(host="127.0.0.1", port=0, store=store)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        host, port = httpd.server_address
        yield f"http://{host}:{port}"
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)

    def _req(self, base, method, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = Request(base + path, data=data, method=method)
        try:
            with urlopen(req) as resp:
                return resp.status, json.loads(resp.read() or b"{}")
        except HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    def test_patch_edits_fields(self, server, store):
        m = store.remember("http editable", importance=0.5)
        status, body = self._req(server, "PATCH", f"/memories/{m.id}",
                                 {"text": "http edited", "importance": 0.9})
        assert status == 200
        assert body["text"] == "http edited"
        assert body["importance"] == 0.9
        assert store._get_by_id(m.id).text == "http edited"

    def test_patch_missing_id_404(self, server):
        status, _ = self._req(server, "PATCH",
                              "/memories/00000000-0000-4000-8000-000000000000",
                              {"text": "x"})
        assert status == 404

    def test_patch_bad_type_400(self, server, store):
        m = store.remember("http editable")
        status, body = self._req(server, "PATCH", f"/memories/{m.id}",
                                 {"type": "sematic"})
        assert status == 400
        assert "type" in body["error"]

    def test_patch_empty_body_400(self, server, store):
        m = store.remember("http editable")
        status, _ = self._req(server, "PATCH", f"/memories/{m.id}", {})
        assert status == 400

    def test_patch_null_field_means_unchanged(self, server, store):
        """null = leave unchanged (was: polarity null silently reset to 0)."""
        m = store.remember("null semantics", polarity=1, importance=0.8)
        status, body = self._req(server, "PATCH", f"/memories/{m.id}",
                                 {"polarity": None, "importance": 0.9})
        assert status == 200
        assert body["polarity"] == 1          # untouched
        assert body["importance"] == 0.9
        # a body containing ONLY nulls is an explicit 400, not a silent no-op
        status, _ = self._req(server, "PATCH", f"/memories/{m.id}",
                              {"polarity": None})
        assert status == 400
