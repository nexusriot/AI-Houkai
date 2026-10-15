"""Store capabilities that previously had no remote surface (A3).

`undo` was the sharpest gap: the append-only journal is the project's
differentiator and undo was reachable only from a local shell.
"""

from __future__ import annotations

import gzip
import json
import time
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from typer.testing import CliRunner

import ai_houkai.mcp_server.server as srv
from ai_houkai.cli.main import app
from ai_houkai.http_server.server import build_handler
from ai_houkai.memory_system import MemoryStore


@pytest.fixture()
def mcp_store(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_HOUKAI_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("AI_HOUKAI_COLLECTION", "surface")
    monkeypatch.setattr(srv, "_store", None)
    yield
    if srv._store is not None:
        srv._store.client.close()
        srv._store = None


@pytest.fixture()
def http(tmp_path):
    """A live threaded HTTP server over a throwaway store."""
    store = MemoryStore(path=str(tmp_path / "chroma"), collection="surface_http")
    server = ThreadingHTTPServer(("127.0.0.1", 0), build_handler(store))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
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


@pytest.fixture()
def http_tokened(tmp_path):
    """Like ``http``, but with a bearer token — the archive routes need one."""
    store = MemoryStore(path=str(tmp_path / "chroma"), collection="surface_http")
    server = ThreadingHTTPServer(("127.0.0.1", 0),
                                 build_handler(store, auth_token="s3cret"))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def call(method, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = Request(f"{base}{path}", data=data, method=method,
                      headers={"Content-Type": "application/json",
                               "Authorization": "Bearer s3cret"})
        try:
            with urlopen(req) as resp:
                return resp.status, json.loads(resp.read() or b"{}")
        except HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    yield call, store
    server.shutdown()
    server.server_close()
    store.client.close()


class TestMcpSurface:
    def test_restore_undoes_a_supersede(self, mcp_store):
        old = srv.remember(text="the old policy")
        new = srv.remember(text="the new policy")
        srv.supersede(old_id=old["id"], new_id=new["id"])
        out = srv.restore(memory_id=old["id"])
        assert out["restored"] is True
        assert out["was_superseded_by"] == new["id"]
        assert srv.get(memory_id=old["id"])["superseded_by"] is None

    def test_restore_on_a_live_memory_is_false(self, mcp_store):
        mem = srv.remember(text="never superseded")
        assert srv.restore(memory_id=mem["id"])["restored"] is False

    def test_subgraph_walks_links(self, mcp_store):
        a = srv.remember(text="graph node a")
        b = srv.remember(text="graph node b")
        c = srv.remember(text="graph node c")
        srv.link(src_id=a["id"], dst_id=b["id"], rel="refines")
        srv.link(src_id=b["id"], dst_id=c["id"], rel="refines")

        one = srv.subgraph(memory_ids=[a["id"]], depth=1)
        assert {n["id"] for n in one["nodes"]} == {a["id"], b["id"]}
        two = srv.subgraph(memory_ids=[a["id"]], depth=2)
        assert {n["id"] for n in two["nodes"]} == {a["id"], b["id"], c["id"]}
        assert {"src": a["id"], "dst": b["id"], "rel": "refines"} in two["edges"]

    def test_undo_reverses_the_newest_mutation(self, mcp_store):
        mem = srv.remember(text="about to be edited")
        srv.edit(memory_id=mem["id"], text="edited text")
        assert srv.get(memory_id=mem["id"])["text"] == "edited text"
        out = srv.undo()
        assert out["ok"] is True and out["op"] == "edit"
        assert srv.get(memory_id=mem["id"])["text"] == "about to be edited"

    def test_undo_targets_one_memory(self, mcp_store):
        first = srv.remember(text="first subject")
        second = srv.remember(text="second subject")
        srv.edit(memory_id=first["id"], importance=0.9)
        srv.edit(memory_id=second["id"], importance=0.1)
        # The newest entry overall belongs to `second`; ask for `first`.
        out = srv.undo(memory_id=first["id"])
        assert out["ok"] is True and out["id"] == first["id"]
        assert srv.get(memory_id=first["id"])["importance"] == 0.5
        assert srv.get(memory_id=second["id"])["importance"] == 0.1

    def test_undo_by_exact_ts(self, mcp_store):
        mem = srv.remember(text="ts targeted")
        entry = list(srv.get_store().journal.read())[-1]
        out = srv.undo(ts=entry.ts)
        assert out["ok"] is True and out["op"] == "remember"
        assert srv.get(memory_id=mem["id"])["found"] is False

    def test_undo_with_nothing_to_undo(self, mcp_store):
        out = srv.undo()
        assert out["ok"] is False and "no journal entry" in out["error"]

    def test_nuke_requires_the_confirm_phrase(self, mcp_store):
        srv.remember(text="should survive a refused nuke")
        refused = srv.nuke()
        assert refused["ok"] is False and refused["deleted"] == 0
        assert srv.stats()["count"] == 1

        done = srv.nuke(confirm="DELETE ALL")
        assert done["ok"] is True and done["deleted"] == 1
        assert srv.stats()["count"] == 0

    def test_ready_reports_both_checks(self, mcp_store):
        out = srv.ready()
        assert out["ready"] is True
        assert out["checks"]["store"]["ok"] is True
        assert out["checks"]["embedder"]["ok"] is True
        assert out["checks"]["embedder"]["dim"] > 0


class TestHttpSurface:
    @pytest.mark.parametrize("rerank", [False, True])
    @pytest.mark.parametrize("endpoint", ["/recall", "/recall_pack"])
    def test_graph_filters_survive_http_and_context_packing(self, http, rerank, endpoint):
        call, _ = http
        status, seed = call("POST", "/memories", {"text": "trusted graph seed"})
        assert status == 201
        for text, fields in [
            ("untrusted graph content", {"trust": "untrusted"}),
            ("expired graph content", {"expires_at": 1}),
            ("retired graph content", {"valid_until": 1}),
        ]:
            status, child = call("POST", "/memories", {"text": text, **fields})
            assert status == 201
            status, _ = call("POST", "/links", {
                "src_id": seed["id"], "dst_id": child["id"], "rel": "related"})
            assert status == 200
        _, allowed = call("POST", "/memories", {"text": "eligible graph content"})
        call("POST", "/links", {"src_id": seed["id"], "dst_id": allowed["id"], "rel": "related"})
        status, result = call("POST", endpoint, {
            "query": "trusted graph seed", "k": 2, "max_items": 2,
            "mode": "semantic", "min_cosine": 0.999, "min_trust": "trusted",
            "token_budget": 1000,
            "expand": {"rels": ["related"], "cap": 1, "rerank": rerank},
        })
        assert status == 200
        rows = result["results"] if endpoint == "/recall" else result["items"]
        assert {r["id"] for r in rows} == {seed["id"], allowed["id"]}
        if endpoint == "/recall_pack":
            assert "eligible graph content" in result["text"]
            assert all(text not in result["text"] for text in (
                "untrusted graph content", "expired graph content", "retired graph content"))

    def test_journal_routes_and_default_undo_are_collection_scoped(self, http):
        call, store = http
        _, own = call("POST", "/memories", {"text": "own original"})
        call("PATCH", f"/memories/{own['id']}", {"text": "own edited"})
        foreign = MemoryStore(path=store.path, collection="foreign",
                              embedding_function=store._embed_fn)
        try:
            other = foreign.remember("foreign private fact")
            foreign.forget(other.id)
            status, result = call("GET", "/journal?n=10")
            assert status == 200 and result["count"] == 2
            assert {e["id"] for e in result["entries"]} == {own["id"]}
            status, state = call("GET", f"/state_at?ts={time.time() + 1}")
            assert status == 200 and [m["id"] for m in state["memories"]] == [own["id"]]
            assert call("GET", f"/memories/{other.id}/history")[0] == 404
            assert call("POST", "/undo", {"memory_id": other.id})[0] == 404
            status, undone = call("POST", "/undo", {})
            assert status == 200 and undone["ok"] is True and undone["id"] == own["id"]
            assert call("GET", f"/memories/{own['id']}")[1]["text"] == "own original"
            assert call("GET", f"/memories/{other.id}")[0] == 404
            assert foreign.get(other.id) is None
        finally:
            foreign.client.close()

    def test_import_overwrite_failure_is_reported_without_losing_memory(self, http_tokened, tmp_path):
        call, _ = http_tokened
        _, original = call("POST", "/memories", {"text": "preserve this row", "tags": ["keep"]})
        path = str(tmp_path / "invalid-vector.ahkai")
        assert call("POST", "/export", {"path": path})[0] == 200
        with gzip.open(path, "rt") as f:
            rows = [json.loads(line) for line in f]
        rows[1]["text"] = "invalid replacement"
        rows[1]["vector"] = [1.0, 0.0]
        with gzip.open(path, "wt") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        status, result = call("POST", "/import", {"path": path, "on_conflict": "overwrite"})
        assert status == 200 and result["overwritten"] == 0
        assert [e["id"] for e in result["errors"]] == [original["id"]]
        status, fetched = call("GET", f"/memories/{original['id']}")
        assert status == 200 and fetched["text"] == "preserve this row"
        assert fetched["tags"] == ["keep"]

    def test_import_collision_rejects_nonconflicting_rows_too(self, http_tokened, tmp_path):
        call, store = http_tokened
        _, original = call("POST", "/memories", {"text": "existing fact"})
        path = str(tmp_path / "collision.ahkai")
        assert call("POST", "/export", {"path": path})[0] == 200
        with gzip.open(path, "rt") as f:
            rows = [json.loads(line) for line in f]
        rows.insert(1, {"id": "new-nonconflicting-id", "text": "must not be inserted"})
        with gzip.open(path, "wt") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        before = list(store.journal.read())
        status, result = call("POST", "/import", {"path": path, "on_conflict": "error"})
        assert status == 409 and "collision" in result["error"]
        assert call("GET", "/memories")[1]["memories"][0]["id"] == original["id"]
        assert store.count() == 1 and list(store.journal.read()) == before

    def test_restore(self, http):
        call, _ = http
        _, old = call("POST", "/memories", {"text": "http old policy"})
        _, new = call("POST", "/memories", {"text": "http new policy"})
        call("POST", "/supersede", {"old_id": old["id"], "new_id": new["id"]})
        status, body = call("POST", "/restore", {"memory_id": old["id"]})
        assert status == 200 and body["restored"] is True

    def test_restore_unknown_id_is_404(self, http):
        call, _ = http
        status, _ = call("POST", "/restore", {"memory_id": "nope"})
        assert status == 404

    def test_subgraph(self, http):
        call, _ = http
        _, a = call("POST", "/memories", {"text": "http graph a"})
        _, b = call("POST", "/memories", {"text": "http graph b"})
        call("POST", "/links", {"src_id": a["id"], "dst_id": b["id"], "rel": "refines"})
        status, body = call("POST", "/subgraph", {"memory_ids": [a["id"]], "depth": 1})
        assert status == 200
        assert {n["id"] for n in body["nodes"]} == {a["id"], b["id"]}
        assert body["edges"] == [{"src": a["id"], "dst": b["id"], "rel": "refines"}]

    def test_subgraph_requires_ids(self, http):
        call, _ = http
        status, body = call("POST", "/subgraph", {})
        assert status == 400 and "memory_ids" in body["error"]

    def test_undo(self, http):
        call, _ = http
        _, mem = call("POST", "/memories", {"text": "http undo subject"})
        call("PATCH", f"/memories/{mem['id']}", {"text": "http changed"})
        status, body = call("POST", "/undo", {})
        assert status == 200 and body["ok"] is True and body["op"] == "edit"
        _, fetched = call("GET", f"/memories/{mem['id']}")
        assert fetched["text"] == "http undo subject"

    def test_undo_nothing_is_404(self, http):
        call, _ = http
        status, _ = call("POST", "/undo", {})
        assert status == 404

    def test_nuke_is_guarded(self, http):
        call, _ = http
        call("POST", "/memories", {"text": "http nuke subject"})
        status, body = call("POST", "/nuke", {})
        assert status == 400 and "DELETE ALL" in body["error"]
        status, body = call("POST", "/nuke", {"confirm": "DELETE ALL"})
        assert status == 200 and body["deleted"] == 1

    def test_journal_tail(self, http):
        call, _ = http
        call("POST", "/memories", {"text": "http journal one"})
        call("POST", "/memories", {"text": "http journal two"})
        status, body = call("GET", "/journal?n=1")
        assert status == 200 and body["count"] == 1
        assert body["entries"][0]["op"] == "remember"
        _, filtered = call("GET", "/journal?op=forget")
        assert filtered["count"] == 0

    def test_export_then_import_roundtrip(self, http_tokened, tmp_path):
        # Archive routes require a configured token — see TestArchiveRouteGate
        # in test_http_server.py for the refusal side.
        call, store = http_tokened
        call("POST", "/memories", {"text": "http archive subject", "tags": ["a"]})
        archive = str(tmp_path / "dump.ahkai")

        status, exported = call("POST", "/export", {"path": archive})
        assert status == 200 and exported["count"] == 1
        assert exported["path"] == archive

        call("POST", "/nuke", {"confirm": "DELETE ALL"})
        status, imported = call("POST", "/import", {"path": archive})
        assert status == 200 and imported["imported"] == 1
        assert store.count() == 1

    def test_import_missing_archive_is_404(self, http_tokened, tmp_path):
        call, _ = http_tokened
        status, _ = call("POST", "/import", {"path": str(tmp_path / "absent.ahkai")})
        assert status == 404


class TestCliSurface:
    """history / state-at / get-at / metrics had no CLI command at all."""

    def _run(self, tmp_path, *args):
        return CliRunner().invoke(
            app, ["--store", str(tmp_path / "chroma"), *args])

    def test_history_and_metrics_and_time_travel(self, tmp_path):
        assert self._run(tmp_path, "remember", "cli surface subject").exit_code == 0
        listing = self._run(tmp_path, "list", "--format", "json")
        mid = json.loads(listing.stdout)[0]["id"]

        hist = self._run(tmp_path, "history", mid, "--json")
        assert hist.exit_code == 0
        assert json.loads(hist.stdout)[0]["op"] == "remember"

        met = self._run(tmp_path, "metrics", "--json")
        assert met.exit_code == 0 and json.loads(met.stdout)["count"] == 1

        state = self._run(tmp_path, "state-at", "now", "--json")
        assert state.exit_code == 0 and json.loads(state.stdout)["count"] == 1

        at = self._run(tmp_path, "get-at", mid, "now", "--json")
        assert at.exit_code == 0 and json.loads(at.stdout)["id"] == mid

    def test_bad_timestamp_is_a_clean_error(self, tmp_path):
        self._run(tmp_path, "remember", "cli ts subject")
        res = self._run(tmp_path, "state-at", "not-a-time")
        assert res.exit_code == 1
        # The message goes to stderr; a traceback would mean we let the
        # ValueError from parse_timestamp escape.
        assert "invalid timestamp" in (res.stderr or res.stdout)
        assert "Traceback" not in (res.stderr or "")

    def test_history_unknown_id(self, tmp_path):
        self._run(tmp_path, "remember", "cli anchor")
        res = self._run(tmp_path, "history", "deadbeef")
        assert res.exit_code == 1

    def test_journal_undo_last(self, tmp_path):
        self._run(tmp_path, "remember", "cli undo-last subject")
        res = self._run(tmp_path, "journal", "undo-last", "-y")
        assert res.exit_code == 0 and "Undone." in res.stdout
        listing = self._run(tmp_path, "list", "--format", "json")
        assert json.loads(listing.stdout or "[]") == []

    def test_journal_undo_last_only_changes_the_selected_collection(self, tmp_path):
        for collection in ("alpha", "bravo"):
            result = self._run(tmp_path, "--collection", collection,
                               "remember", f"private {collection} fact")
            assert result.exit_code == 0, result.output
        state = self._run(tmp_path, "--collection", "alpha", "state-at", "now", "--json")
        assert state.exit_code == 0, state.output
        assert [m["text"] for m in json.loads(state.stdout)["memories"]] == ["private alpha fact"]
        undone = self._run(tmp_path, "--collection", "alpha", "journal", "undo-last", "-y")
        assert undone.exit_code == 0, undone.output
        alpha = self._run(tmp_path, "--collection", "alpha", "list", "--format", "json")
        bravo = self._run(tmp_path, "--collection", "bravo", "list", "--format", "json")
        assert alpha.exit_code == bravo.exit_code == 0
        assert json.loads(alpha.stdout) == []
        assert [m["text"] for m in json.loads(bravo.stdout)] == ["private bravo fact"]


class TestCliRegressionFixes:
    """CLI defects found in the 2026-08 review: prefix resolution against
    the wrong universe."""

    def _run(self, tmp_path, *args):
        return CliRunner().invoke(
            app, ["--store", str(tmp_path / "chroma"), *args])

    def test_trash_purge_resolves_displayed_prefix(self, tmp_path):
        """`trash list` shows 8-char ids; purge used to exact-match, so the
        displayed id confirmed destructively and then purged nothing."""
        self._run(tmp_path, "remember", "purge prefix subject")
        listing = self._run(tmp_path, "list", "--format", "json")
        mid = json.loads(listing.stdout)[0]["id"]
        self._run(tmp_path, "trash", "put", mid)

        res = self._run(tmp_path, "trash", "purge", mid[:8], "-y")
        assert res.exit_code == 0
        assert "Purged 1 entries" in res.stdout

    def test_trash_purge_unknown_prefix_fails_before_confirm(self, tmp_path):
        self._run(tmp_path, "remember", "unrelated")
        res = self._run(tmp_path, "trash", "purge", "deadbeef", "-y")
        assert res.exit_code == 1
        assert "not in the trash" in (res.stderr or res.stdout)

    def test_undo_last_by_prefix_after_forget(self, tmp_path):
        """--id used to resolve against live memories only, failing exactly
        when the newest entry is the forget the operator wants undone."""
        self._run(tmp_path, "remember", "deleted by mistake")
        listing = self._run(tmp_path, "list", "--format", "json")
        mid = json.loads(listing.stdout)[0]["id"]
        assert self._run(tmp_path, "forget", mid, "-y").exit_code == 0

        res = self._run(tmp_path, "journal", "undo-last",
                        "--id", mid[:8], "-y")
        assert res.exit_code == 0 and "Undone." in res.stdout
        listing = self._run(tmp_path, "list", "--format", "json")
        assert [m["id"] for m in json.loads(listing.stdout)] == [mid]


class TestTimeTravelPrefixResolution:
    """history / get-at exist precisely for memories that may no longer be
    live, so their id prefixes resolve against the journal (archives
    included), falling back to the live store only when the journal has no
    match."""

    def _run(self, tmp_path, *args):
        return CliRunner().invoke(
            app, ["--store", str(tmp_path / "chroma"), *args])

    def test_history_by_prefix_after_forget(self, tmp_path):
        self._run(tmp_path, "remember", "historied then deleted")
        listing = self._run(tmp_path, "list", "--format", "json")
        mid = json.loads(listing.stdout)[0]["id"]
        self._run(tmp_path, "forget", mid, "-y")

        res = self._run(tmp_path, "history", mid[:8], "--json")
        assert res.exit_code == 0, res.stdout + (res.stderr or "")
        ops = [e["op"] for e in json.loads(res.stdout)]
        assert ops == ["remember", "forget"]

    def test_get_at_by_prefix_after_forget(self, tmp_path):
        self._run(tmp_path, "remember", "reconstruct me later")
        listing = self._run(tmp_path, "list", "--format", "json")
        mem = json.loads(listing.stdout)[0]
        # An instant strictly between the journaled remember and the forget —
        # created_at can precede the remember entry's own journal ts.
        between = time.time()
        self._run(tmp_path, "forget", mem["id"], "-y")

        res = self._run(tmp_path, "get-at", mem["id"][:8], str(between),
                        "--json")
        assert res.exit_code == 0, res.stdout + (res.stderr or "")
        assert json.loads(res.stdout)["id"] == mem["id"]

    def test_unknown_prefix_still_errors(self, tmp_path):
        self._run(tmp_path, "remember", "anchor row")
        res = self._run(tmp_path, "history", "deadbeef")
        assert res.exit_code == 1
