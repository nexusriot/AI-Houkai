"""Integration tests for the stdlib HTTP/REST API server.

Spins up a real ThreadingHTTPServer on an ephemeral port backed by the shared
`store` fixture, then drives it with urllib.
"""

from __future__ import annotations

import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from ai_houkai.http_server import make_server


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
