"""Integration tests for the stdlib HTTP/REST API server.

Spins up a real ThreadingHTTPServer on an ephemeral port backed by the shared
`store` fixture, then drives it with urllib.
"""

from __future__ import annotations

import http.client
import json
import os
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

import ai_houkai.http_server.server as http_mod
from ai_houkai.http_server import make_server
from ai_houkai.http_server.server import _compress_threshold


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
    req = Request(base + path, data=data, method=method,
                  headers=headers or {})
    try:
        with urlopen(req) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


class TestHttpServer:
    def test_health(self, server):
        status, body = _req(server, "GET", "/health")
        assert status == 200
        assert body["status"] == "ok"
        assert body["count"] == 0

    def test_health_does_not_leak_collection(self, server):
        # /health is liveness-only and must not expose the collection name/topology.
        status, body = _req(server, "GET", "/health")
        assert status == 200 and body["status"] == "ok"
        assert "collection" not in body

    def test_remember_and_get(self, server):
        status, body = _req(server, "POST", "/memories",
                            {"text": "http memory", "type": "semantic",
                             "tags": ["x"], "source": "api"})
        assert status == 201
        assert body["stored"] is True
        mid = body["id"]

        status, got = _req(server, "GET", f"/memories/{mid}")
        assert status == 200
        assert got["text"] == "http memory"
        assert got["source"] == "api"
        assert got["tags"] == ["x"]

    def test_recall_get_and_post(self, server):
        _req(server, "POST", "/memories", {"text": "neon authentication flow"})
        _req(server, "POST", "/memories", {"text": "banana bread recipe"})

        status, body = _req(server, "GET", "/recall?query=authentication&k=1")
        assert status == 200
        assert body["results"]
        assert "authentication" in body["results"][0]["text"]

        status, body = _req(server, "POST", "/recall",
                            {"query": "authentication", "k": 1})
        assert status == 200
        assert body["results"][0]["score"] <= 1.0

    def test_recall_source_filter(self, server):
        _req(server, "POST", "/memories", {"text": "alpha", "source": "repo-a"})
        _req(server, "POST", "/memories", {"text": "alpha", "source": "repo-b"})
        status, body = _req(server, "GET", "/recall?query=alpha&k=10&source=repo-a")
        assert status == 200
        assert {r["source"] for r in body["results"]} == {"repo-a"}

    def test_recall_pack(self, server):
        _req(server, "POST", "/memories", {"text": "pack me into context"})
        status, body = _req(server, "POST", "/recall_pack",
                            {"query": "context", "token_budget": 200})
        assert status == 200
        assert "## Relevant memory" in body["text"]
        assert body["budget"] == 200

    def test_list_and_forget(self, server):
        status, body = _req(server, "POST", "/memories", {"text": "ephemeral"})
        mid = body["id"]
        status, body = _req(server, "GET", "/memories?limit=5")
        assert any(m["id"] == mid for m in body["memories"])

        status, body = _req(server, "DELETE", f"/memories/{mid}")
        assert status == 200 and body["forgotten"] is True
        status, _ = _req(server, "GET", f"/memories/{mid}")
        assert status == 404

    def test_links_and_neighbors(self, server):
        _, a = _req(server, "POST", "/memories", {"text": "source node"})
        _, b = _req(server, "POST", "/memories", {"text": "target node"})
        status, _ = _req(server, "POST", "/links",
                        {"src_id": a["id"], "dst_id": b["id"], "rel": "refines"})
        assert status == 200
        status, body = _req(server, "GET",
                           f"/memories/{a['id']}/neighbors?direction=out")
        assert status == 200
        assert any(n["id"] == b["id"] for n in body["neighbors"])

    def test_missing_query_is_400(self, server):
        status, body = _req(server, "GET", "/recall")
        assert status == 400
        assert "query" in body["error"]

    def test_unknown_route_is_404(self, server):
        status, _ = _req(server, "GET", "/nope")
        assert status == 404

    def test_method_not_allowed(self, server):
        status, _ = _req(server, "DELETE", "/recall")
        assert status == 405

    def test_bad_timestamp_is_400(self, server):
        status, body = _req(server, "GET", "/recall?query=x&since=garbage")
        assert status == 400


class TestHttpAuth:
    @pytest.fixture()
    def auth_server(self, store):
        httpd = make_server(host="127.0.0.1", port=0, store=store,
                            auth_token="s3cret")
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        host, port = httpd.server_address
        yield f"http://{host}:{port}"
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)

    def test_health_open(self, auth_server):
        status, _ = _req(auth_server, "GET", "/health")
        assert status == 200

    def test_protected_without_token(self, auth_server):
        status, body = _req(auth_server, "GET", "/stats")
        assert status == 401

    def test_protected_with_token(self, auth_server):
        status, _ = _req(auth_server, "GET", "/stats",
                        headers={"Authorization": "Bearer s3cret"})
        assert status == 200

    def test_wrong_token(self, auth_server):
        status, _ = _req(auth_server, "GET", "/stats",
                        headers={"Authorization": "Bearer nope"})
        assert status == 401


class TestHttpTagCoercion:
    def test_remember_string_tag_becomes_single_tag(self, server):
        status, body = _req(server, "POST", "/memories",
                            {"text": "tagged fact", "tags": "solo"})
        assert status == 201
        assert body["tags"] == ["solo"]        # not ["s","o","l","o"]

    def test_edit_string_tag_becomes_single_tag(self, server):
        _, created = _req(server, "POST", "/memories", {"text": "to edit"})
        status, body = _req(server, "PATCH", f"/memories/{created['id']}",
                            {"tags": "renamed"})
        assert status == 200
        assert body["tags"] == ["renamed"]

    def test_remember_tag_list_passes_through(self, server):
        status, body = _req(server, "POST", "/memories",
                            {"text": "multi", "tags": ["a", "b"]})
        assert status == 201
        assert body["tags"] == ["a", "b"]

    def test_remember_non_string_tags_is_400(self, server):
        status, body = _req(server, "POST", "/memories",
                            {"text": "bad tags", "tags": 123})
        assert status == 400
        assert "tags" in body["error"]

    def test_remember_mixed_tag_list_is_400(self, server):
        status, _ = _req(server, "POST", "/memories",
                         {"text": "bad tags", "tags": ["ok", 5]})
        assert status == 400

    def test_edit_empty_list_clears_tags(self, server):
        _, created = _req(server, "POST", "/memories",
                          {"text": "clear me", "tags": ["x"]})
        status, body = _req(server, "PATCH", f"/memories/{created['id']}",
                            {"tags": []})
        assert status == 200
        assert body["tags"] == []


class TestHttpRecallKnobs:
    def test_recall_get_overfetch(self, server):
        _req(server, "POST", "/memories", {"text": "overfetch target"})
        status, body = _req(server, "GET", "/recall?query=overfetch&overfetch=8")
        assert status == 200
        assert body["results"]

    def test_recall_post_overfetch(self, server):
        _req(server, "POST", "/memories", {"text": "overfetch target"})
        status, body = _req(server, "POST", "/recall",
                            {"query": "overfetch", "overfetch": 2})
        assert status == 200
        assert body["results"]

    def test_recall_bad_overfetch_is_400(self, server):
        status, _ = _req(server, "GET", "/recall?query=x&overfetch=lots")
        assert status == 400

    def test_recall_pack_ranking_knobs(self, server):
        _req(server, "POST", "/memories", {"text": "deploy the api"})
        _req(server, "POST", "/memories", {"text": "deploy the api again"})
        status, body = _req(server, "POST", "/recall_pack",
                            {"query": "deploy", "fusion": "rrf",
                             "diversity": 0.5, "dedup_threshold": 0.99,
                             "min_cosine": -1.0})
        assert status == 200
        assert "compressed_groups" in body

    def test_recall_pack_compress(self, server):
        for i in range(4):
            _req(server, "POST", "/memories",
                 {"text": f"deploy the payments api step {i} of the runbook"})
        status, body = _req(server, "POST", "/recall_pack",
                            {"query": "deploy payments", "token_budget": 1,
                             "compress": True, "compress_min_group": 2,
                             "compress_threshold": 0.2})
        assert status == 200
        assert body["truncated"] is True

    def test_recall_pack_bad_diversity_is_400(self, server):
        status, body = _req(server, "POST", "/recall_pack",
                            {"query": "x", "diversity": 3.0})
        assert status == 400

    def test_recall_pack_bad_fusion_is_400(self, server):
        status, _ = _req(server, "POST", "/recall_pack",
                         {"query": "x", "fusion": "blend"})
        assert status == 400


class TestHttpAutoContext:
    def test_auto_context_returns_pack(self, server):
        _req(server, "POST", "/memories",
             {"text": "The deploy pipeline runs through GitHub Actions."})
        status, body = _req(server, "POST", "/auto_context",
                            {"task": "deploy the api to production"})
        assert status == 200
        assert "text" in body and "items" in body
        assert body["queries"][0] == "deploy the api to production"
        assert len(body["queries"]) >= 1

    def test_auto_context_missing_task_is_400(self, server):
        status, body = _req(server, "POST", "/auto_context", {})
        assert status == 400
        assert "task" in body["error"]

    def test_auto_context_bad_mode_is_400(self, server):
        status, _ = _req(server, "POST", "/auto_context",
                         {"task": "x", "mode": "psychic"})
        assert status == 400


def test_run_expands_tilde_in_env_path(monkeypatch, tmp_path):
    """The env-driven console entrypoint must expanduser AI_HOUKAI_PATH."""
    seen = {}
    monkeypatch.setattr(http_mod, "serve", lambda **kw: seen.update(kw))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("AI_HOUKAI_PATH", "~/tilde-store/.chroma")
    http_mod.run()
    assert seen["path"] == os.path.join(str(tmp_path), "tilde-store", ".chroma")


class TestKeepAliveBodyDrain:
    """Unread request bodies must not desync HTTP/1.1 keep-alive connections.

    urllib opens a fresh connection per request, so these use
    http.client.HTTPConnection, which reuses one socket like every pooling
    client (requests.Session, Go http.Client, reverse proxies).
    """

    @staticmethod
    def _conn(base):
        host, port = base.removeprefix("http://").rsplit(":", 1)
        return http.client.HTTPConnection(host, int(port), timeout=10)

    def _roundtrip(self, conn, method, path, body=None, headers=None):
        hdrs = {"Content-Type": "application/json", **(headers or {})}
        conn.request(method, path,
                     body=json.dumps(body) if body is not None else None,
                     headers=hdrs)
        resp = conn.getresponse()
        data = resp.read()
        return resp.status, json.loads(data or b"{}")

    def test_unrouted_post_body_does_not_poison_connection(self, server):
        conn = self._conn(server)
        try:
            status, _ = self._roundtrip(conn, "POST", "/nope",
                                        {"text": "hello"})
            assert status == 404
            # Before the drain fix the leftover body bytes were parsed as the
            # start of this request → 400 Bad request syntax + closed socket.
            status, body = self._roundtrip(conn, "GET", "/health")
            assert status == 200
            assert body["status"] == "ok"
        finally:
            conn.close()

    def test_405_with_body_does_not_poison_connection(self, server):
        conn = self._conn(server)
        try:
            status, _ = self._roundtrip(conn, "POST", "/health",
                                        {"junk": "x" * 200})
            assert status == 405
            status, body = self._roundtrip(conn, "GET", "/health")
            assert status == 200 and body["status"] == "ok"
        finally:
            conn.close()

    def test_bodied_get_does_not_poison_connection(self, server):
        conn = self._conn(server)
        try:
            status, _ = self._roundtrip(conn, "GET", "/memories",
                                        {"unexpected": "body"})
            assert status == 200
            status, body = self._roundtrip(conn, "GET", "/health")
            assert status == 200 and body["status"] == "ok"
        finally:
            conn.close()

    def test_401_with_body_does_not_poison_connection(self, store):
        httpd = make_server(host="127.0.0.1", port=0, store=store,
                            auth_token="s3cret")
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        host, port = httpd.server_address
        conn = self._conn(f"http://{host}:{port}")
        try:
            status, _ = self._roundtrip(conn, "POST", "/memories",
                                        {"text": "no token"})
            assert status == 401
            status, _ = self._roundtrip(
                conn, "GET", "/stats",
                headers={"Authorization": "Bearer s3cret"})
            assert status == 200
        finally:
            conn.close()
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=5)


class TestBodyValidationFixes:
    def test_non_string_text_is_400_not_500(self, server):
        status, body = _req(server, "POST", "/memories", {"text": 123})
        assert status == 400
        assert "text" in body["error"]

    def test_non_string_query_is_400_not_500(self, server):
        status, body = _req(server, "POST", "/recall", {"query": ["a"]})
        assert status == 400
        assert "query" in body["error"]

    def test_null_type_uses_default(self, server):
        status, body = _req(server, "POST", "/memories",
                            {"text": "null type", "type": None})
        assert status == 201
        assert body["type"] == "semantic"

    def test_negative_limit_is_400(self, server):
        status, body = _req(server, "GET", "/memories?limit=-1")
        assert status == 400
        assert "limit" in body["error"]

    def test_neighbors_of_unknown_id_is_404(self, server):
        status, _ = _req(server, "GET",
                         "/memories/00000000-0000-4000-8000-000000000000"
                         "/neighbors")
        assert status == 404

    def test_compress_threshold_zero_not_replaced_by_default(self):
        # `or 0.30` used to swallow an explicit 0.0 ("cluster everything").
        assert _compress_threshold({"compress_threshold": 0.0}) == 0.0
        assert _compress_threshold({"compress_threshold": 0.9}) == 0.9
        assert _compress_threshold({}) == 0.30


class TestMetricsEndpoint:
    def test_metrics_reports_counters(self, server):
        _req(server, "POST", "/memories", {"text": "metric me"})
        _req(server, "GET", "/recall?query=metric&k=3")
        status, body = _req(server, "GET", "/metrics")
        assert status == 200
        assert body["calls"]["remember"] == 1
        assert body["calls"]["recall"] == 1
        assert body["recall_latency_ms"]["count"] == 1
        assert "uptime_seconds" in body


class TestTTLEndpoints:
    def test_remember_ttl_and_get_exposes_expires_at(self, server):
        status, body = _req(server, "POST", "/memories",
                            {"text": "ephemeral", "ttl_seconds": 500})
        assert status == 201
        assert body["expires_at"] is not None
        status, got = _req(server, "GET", "/memories/" + body["id"])
        assert got["expires_at"] is not None

    def test_recall_hides_expired_and_include_expired_shows(self, server):
        _req(server, "POST", "/memories",
             {"text": "soon gone deployment note", "expires_at": 1.0})  # long past
        s, hidden = _req(server, "GET", "/recall?query=deployment%20note&k=5")
        assert all(h["text"] != "soon gone deployment note"
                   for h in hidden["results"])
        s, shown = _req(server, "GET",
                        "/recall?query=deployment%20note&k=5&include_expired=true")
        assert any(h["text"] == "soon gone deployment note"
                   for h in shown["results"])

    def test_edit_sets_expiry(self, server):
        _, mem = _req(server, "POST", "/memories", {"text": "to expire"})
        s, edited = _req(server, "PATCH", "/memories/" + mem["id"],
                         {"expires_at": 1.0})
        assert s == 200 and edited["expires_at"] is not None

    def test_purge_expired(self, server):
        _req(server, "POST", "/memories", {"text": "expired doc", "expires_at": 1.0})
        s, dry = _req(server, "POST", "/purge_expired", {"dry_run": True})
        assert s == 200 and dry["purged"] == 1 and dry["dry_run"] is True
        s, done = _req(server, "POST", "/purge_expired", {})
        assert done["purged"] == 1


class TestExplainEndpoint:
    def test_recall_explain_attaches_breakdown(self, server):
        _req(server, "POST", "/memories", {"text": "explainable memory"})
        s, body = _req(server, "GET",
                       "/recall?query=explainable&k=1&mode=hybrid&explain=true")
        assert s == 200
        hit = body["results"][0]
        assert "explain" in hit
        assert hit["explain"]["mode"] == "hybrid"
        assert "cosine" in hit["explain"]

    def test_recall_without_explain_has_no_breakdown(self, server):
        _req(server, "POST", "/memories", {"text": "quiet memory"})
        s, body = _req(server, "GET", "/recall?query=quiet&k=1")
        assert "explain" not in body["results"][0]


class TestHistoryEndpoints:
    def test_history_timeline(self, server):
        _, mem = _req(server, "POST", "/memories", {"text": "v1"})
        _req(server, "PATCH", "/memories/" + mem["id"], {"text": "v2"})
        s, body = _req(server, "GET", "/memories/" + mem["id"] + "/history")
        assert s == 200
        assert [e["op"] for e in body["history"]] == ["remember", "edit"]

    def test_history_unknown_id_404(self, server):
        s, _ = _req(server, "GET",
                    "/memories/00000000-0000-4000-8000-000000000000/history")
        assert s == 404

    def test_state_at_requires_ts(self, server):
        s, _ = _req(server, "GET", "/state_at")
        assert s == 400

    def test_state_at_reconstructs(self, server):
        _req(server, "POST", "/memories", {"text": "present one"})
        time.sleep(0.02)
        s, body = _req(server, "GET", "/state_at?ts=" + str(time.time()))
        assert s == 200
        assert any(m["text"] == "present one" for m in body["memories"])

    def test_get_at_single_memory(self, server):
        _, mem = _req(server, "POST", "/memories", {"text": "time traveler"})
        time.sleep(0.02)
        s, body = _req(server, "GET",
                       "/memories/" + mem["id"] + "/at?ts=" + str(time.time()))
        assert s == 200 and body["text"] == "time traveler"

    def test_get_at_before_creation_404(self, server):
        t = time.time()
        time.sleep(0.02)
        _, mem = _req(server, "POST", "/memories", {"text": "future"})
        s, _ = _req(server, "GET",
                    "/memories/" + mem["id"] + "/at?ts=" + str(t))
        assert s == 404
