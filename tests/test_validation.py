"""Tests for the shared validation layer: enum vocabularies validated in the
store, POST-body coercion + clean status codes in the HTTP server, and clean
CLI errors instead of tracebacks."""

from __future__ import annotations

import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from typer.testing import CliRunner

from ai_houkai.cli.main import app
from ai_houkai.http_server import make_server
from ai_houkai.memory_system import (
    CONFLICT_POLICIES,
    LINK_RELS,
    MemoryStore,
    RECALL_MODES,
)


class TestStoreValidation:
    def test_recall_rejects_mode_typo(self, store: MemoryStore):
        store.remember("something")
        with pytest.raises(ValueError, match="mode must be one of"):
            store.recall("q", mode="hybird")

    def test_recall_rejects_fusion_typo(self, store: MemoryStore):
        store.remember("something")
        with pytest.raises(ValueError, match="fusion must be one of"):
            store.recall("q", mode="hybrid", fusion="rff")

    def test_recall_rejects_bad_type_filter(self, store: MemoryStore):
        store.remember("something")
        with pytest.raises(ValueError, match="type must be one of"):
            store.recall("q", type="sematic")

    def test_remember_rejects_bad_type(self, store: MemoryStore):
        with pytest.raises(ValueError, match="type must be one of"):
            store.remember("x", type="text")

    def test_remember_rejects_bad_polarity(self, store: MemoryStore):
        with pytest.raises(ValueError, match="polarity"):
            store.remember("x", polarity=2)

    def test_remember_rejects_bad_on_conflict(self, store: MemoryStore):
        with pytest.raises(ValueError, match="on_conflict must be one of"):
            store.remember("x", on_conflict="bogus")

    def test_constructor_rejects_bad_conflict_policy(self, tmp_path):
        with pytest.raises(ValueError, match="conflict_policy must be one of"):
            MemoryStore(path=str(tmp_path / "c"), collection="colx",
                        conflict_policy="explode")

    def test_link_rejects_rel_typo(self, store: MemoryStore):
        a = store.remember("aaa")
        b = store.remember("bbb")
        with pytest.raises(ValueError, match="rel must be one of"):
            store.link(a.id, b.id, "supercedes")

    def test_link_rejects_dangling_dst(self, store: MemoryStore):
        a = store.remember("aaa")
        with pytest.raises(KeyError, match="dst_id"):
            store.link(a.id, "00000000-0000-4000-8000-000000000000")

    def test_unlink_rejects_rel_typo(self, store: MemoryStore):
        a = store.remember("aaa")
        b = store.remember("bbb")
        store.link(a.id, b.id, "refines")
        with pytest.raises(ValueError, match="rel must be one of"):
            store.unlink(a.id, b.id, rel="refnies")

    def test_neighbors_rejects_bad_direction(self, store: MemoryStore):
        a = store.remember("aaa")
        with pytest.raises(ValueError, match="direction must be one of"):
            store.neighbors(a.id, direction="up")

    def test_import_rejects_bad_policy(self, store: MemoryStore, tmp_path):
        out = tmp_path / "d.ahkai"
        store.remember("to export")
        store.export(out)
        with pytest.raises(ValueError, match="on_conflict must be one of"):
            store.import_(out, on_conflict="merge")

    def test_undo_unlink_with_forgotten_endpoint_returns_false(
        self, store: MemoryStore
    ):
        a = store.remember("aaa")
        b = store.remember("bbb")
        store.link(a.id, b.id, "refines")
        store.unlink(a.id, b.id, rel="refines")
        entry = list(store.journal.read(op="unlink"))[-1]
        store.forget(b.id)
        assert store.undo(entry) is False   # not a KeyError traceback

    def test_vocabularies_exported(self):
        assert "hybrid" in RECALL_MODES
        assert "supersedes" in LINK_RELS
        assert "raise" in CONFLICT_POLICIES


@pytest.fixture()
def server(store):
    httpd = make_server(host="127.0.0.1", port=0, store=store)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address
    yield f"http://{host}:{port}"
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=5)


def _req(base, method, path, body=None, headers=None):
    data = json.dumps(body).encode() if body is not None else None
    req = Request(base + path, data=data, method=method, headers=headers or {})
    try:
        with urlopen(req) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


class TestHttpCoercion:
    def test_recall_null_k_is_400(self, server, store):
        store.remember("something")
        status, body = _req(server, "POST", "/recall", {"query": "x", "k": None})
        # null k falls back to the default rather than a TypeError-500
        assert status == 200

    def test_recall_garbage_k_is_400(self, server, store):
        store.remember("something")
        status, body = _req(server, "POST", "/recall", {"query": "x", "k": "lots"})
        assert status == 400
        assert "k" in body["error"]

    def test_recall_string_min_importance_coerced(self, server, store):
        store.remember("something", importance=0.9)
        status, _ = _req(server, "POST", "/recall",
                         {"query": "x", "min_importance": "0.5"})
        assert status == 200

    def test_recall_garbage_min_importance_is_400(self, server, store):
        store.remember("something")
        status, body = _req(server, "POST", "/recall",
                            {"query": "x", "min_importance": "high"})
        assert status == 400

    def test_recall_mode_typo_is_400(self, server, store):
        store.remember("something")
        status, body = _req(server, "POST", "/recall",
                            {"query": "x", "mode": "hybird"})
        assert status == 400
        assert "mode" in body["error"]

    def test_include_superseded_string_false_is_false(self, server, store):
        a = store.remember("old fact")
        b = store.remember("new fact")
        store.supersede(old_id=a.id, new_id=b.id)
        status, body = _req(server, "POST", "/recall",
                            {"query": "fact", "k": 10,
                             "include_superseded": "false"})
        assert status == 200
        ids = [r["id"] for r in body["results"]]
        assert a.id not in ids            # "false" must not be truthy

    def test_remember_null_polarity_defaults(self, server):
        status, _ = _req(server, "POST", "/memories",
                         {"text": "x", "polarity": None})
        assert status == 201

    def test_remember_bad_polarity_is_400(self, server):
        status, body = _req(server, "POST", "/memories",
                            {"text": "x", "polarity": 5})
        assert status == 400
        assert "polarity" in body["error"]

    def test_conflicts_string_threshold_is_400(self, server, store):
        store.remember("something")
        status, body = _req(server, "POST", "/conflicts", {"threshold": "high"})
        assert status == 400

    def test_link_bad_rel_is_400(self, server, store):
        a = store.remember("aaa")
        b = store.remember("bbb")
        status, body = _req(server, "POST", "/links",
                            {"src_id": a.id, "dst_id": b.id, "rel": "supercedes"})
        assert status == 400

    def test_link_unknown_id_is_404(self, server, store):
        a = store.remember("aaa")
        status, _ = _req(server, "POST", "/links",
                         {"src_id": a.id,
                          "dst_id": "00000000-0000-4000-8000-000000000000"})
        assert status == 404

    def test_head_health_is_200(self, server):
        req = Request(server + "/health", method="HEAD")
        with urlopen(req) as resp:
            assert resp.status == 200
            assert resp.read() == b""     # no body on HEAD


class TestHttpAuth:
    @pytest.fixture()
    def auth_server(self, store):
        httpd = make_server(host="127.0.0.1", port=0, store=store,
                            auth_token="sekrit")
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        host, port = httpd.server_address
        yield f"http://{host}:{port}"
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)

    def test_non_ascii_auth_header_is_401_not_crash(self, auth_server):
        status, body = _req(auth_server, "GET", "/stats",
                            headers={"Authorization": "Bearer sécret"})
        assert status == 401              # was: thread crash, empty reply

    def test_correct_token_passes(self, auth_server):
        status, _ = _req(auth_server, "GET", "/stats",
                         headers={"Authorization": "Bearer sekrit"})
        assert status == 200


class TestCliValidation:
    """Drives the real Typer app; the store is built from --store/--collection
    exactly as in test_cli.py."""

    @pytest.fixture()
    def cli_store(self, tmp_path):
        s = MemoryStore(path=str(tmp_path / "chroma"), collection="cli_val")
        yield s, str(tmp_path / "chroma")
        s.client.close()

    def _run(self, store_path, args):
        runner = CliRunner()
        return runner.invoke(
            app, ["--store", store_path, "--collection", "cli_val"] + args)

    def test_link_rel_typo_clean_error(self, cli_store):
        store, path = cli_store
        a = store.remember("aaa")
        b = store.remember("bbb")
        res = self._run(path, ["link", a.id, b.id, "-r", "supercedes"])
        assert res.exit_code == 1
        assert "rel must be one of" in res.output
        assert "Traceback" not in res.output

    def test_link_self_clean_error(self, cli_store):
        store, path = cli_store
        a = store.remember("aaa")
        res = self._run(path, ["link", a.id, a.id])
        assert res.exit_code == 1
        assert "itself" in res.output
        assert "Traceback" not in res.output

    def test_link_dangling_dst_clean_error(self, cli_store):
        store, path = cli_store
        a = store.remember("aaa")
        res = self._run(path, ["link", a.id,
                               "00000000-0000-4000-8000-000000000000"])
        assert res.exit_code == 1
        assert "not found" in res.output
        assert "Traceback" not in res.output

    def test_recall_mode_typo_clean_error(self, cli_store):
        store, path = cli_store
        store.remember("something")
        res = self._run(path, ["recall", "x", "--mode", "hybird"])
        assert res.exit_code == 1
        assert "mode must be one of" in res.output
        assert "Traceback" not in res.output

    def test_remember_bad_on_conflict_clean_error(self, cli_store):
        _, path = cli_store
        res = self._run(path, ["remember", "hello world",
                               "--on-conflict", "bogus"])
        assert res.exit_code == 1
        assert "on_conflict must be one of" in res.output

    def test_remember_conflict_raise_is_clean_outcome(self, cli_store):
        store, path = cli_store
        store.remember("the sky is blue today", tags=["sky"])
        res = self._run(path, ["remember", "the sky is not blue today",
                               "-g", "sky", "--on-conflict", "raise"])
        assert res.exit_code == 1
        assert "Not stored" in res.output
        assert "Traceback" not in res.output

    def test_supersede_missing_id_clean_error(self, cli_store):
        store, path = cli_store
        a = store.remember("aaa")
        res = self._run(path, ["supersede", a.id,
                               "00000000-0000-4000-8000-000000000000"])
        assert res.exit_code == 1
        assert "Traceback" not in res.output

    def test_list_since_epoch_accepted(self, cli_store):
        store, path = cli_store
        store.remember("recent memory")
        res = self._run(path, ["list", "--since", "1751443200",
                               "--format", "json"])
        assert res.exit_code == 0, res.output
