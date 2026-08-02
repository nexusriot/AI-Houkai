"""Pluggable embedding backends (B).

Python previously hard-wired SentenceTransformerEmbeddingFunction with no
override, so the library could not run without torch, could not use a hosted
embedder, and could not be tested without loading a real model — while the Go
port had injected `embed.Embedder` all along.

The HTTP-speaking embedders are exercised against a local stub server, never
a real provider.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from ai_houkai.embed import (
    DIGITALOCEAN_BASE_URL,
    EmbeddingError,
    OllamaEmbedder,
    OpenAICompatibleEmbedder,
    build_embedder,
    local_embedder,
)
from ai_houkai.memory_system import MemoryStore
from ai_houkai.memory_system.store import _get_embed_fn
from ai_houkai.testing import FakeEmbedder, make_store


class _StubHandler(BaseHTTPRequestHandler):
    """Serves both /v1/embeddings (OpenAI) and /api/embed (Ollama)."""

    routes: dict = {}
    seen: list = []

    def log_message(self, *_args):  # silence the default stderr access log
        pass

    def do_POST(self):  # noqa: N802 — BaseHTTPRequestHandler naming
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        type(self).seen.append({
            "path": self.path,
            "body": body,
            "auth": self.headers.get("Authorization"),
        })
        status, payload = type(self).routes.get(self.path, (404, {"error": "nope"}))
        if callable(payload):
            payload = payload(body)
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture()
def stub():
    """A local HTTP server standing in for an embedding provider."""

    class Handler(_StubHandler):
        routes = {}
        seen = []

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", Handler
    server.shutdown()
    server.server_close()


def _openai_payload(body):
    return {"data": [
        {"index": i, "embedding": [float(i), 0.5, 0.25]}
        for i in range(len(body["input"]))
    ]}


class TestOpenAICompatibleEmbedder:
    def test_embeds_and_sends_auth(self, stub):
        base, handler = stub
        handler.routes["/v1/embeddings"] = (200, _openai_payload)
        emb = OpenAICompatibleEmbedder("sk-test", "text-embedding-3-small", base)

        out = emb(["alpha", "beta"])
        assert out == [[0.0, 0.5, 0.25], [1.0, 0.5, 0.25]]
        req = handler.seen[-1]
        assert req["auth"] == "Bearer sk-test"
        assert req["body"]["model"] == "text-embedding-3-small"
        assert req["body"]["input"] == ["alpha", "beta"]

    def test_no_auth_header_without_key(self, stub):
        base, handler = stub
        handler.routes["/v1/embeddings"] = (200, _openai_payload)
        OpenAICompatibleEmbedder("", "m", base)(["x"])
        assert handler.seen[-1]["auth"] is None

    def test_empty_input_makes_no_request(self, stub):
        base, handler = stub
        assert OpenAICompatibleEmbedder("k", "m", base)([]) == []
        assert handler.seen == []

    def test_batches_large_inputs(self, stub):
        base, handler = stub
        handler.routes["/v1/embeddings"] = (200, _openai_payload)
        emb = OpenAICompatibleEmbedder("k", "m", base, batch_size=2)
        out = emb(["a", "b", "c", "d", "e"])
        assert len(out) == 5
        assert [len(r["body"]["input"]) for r in handler.seen] == [2, 2, 1]

    def test_out_of_order_results_are_reindexed(self, stub):
        base, handler = stub
        handler.routes["/v1/embeddings"] = (200, {"data": [
            {"index": 1, "embedding": [9.0]},
            {"index": 0, "embedding": [1.0]},
        ]})
        assert OpenAICompatibleEmbedder("k", "m", base)(["a", "b"]) == [[1.0], [9.0]]

    def test_count_mismatch_raises(self, stub):
        base, handler = stub
        handler.routes["/v1/embeddings"] = (
            200, {"data": [{"index": 0, "embedding": [1.0]}]})
        with pytest.raises(EmbeddingError, match="1 embeddings for 2 inputs"):
            OpenAICompatibleEmbedder("k", "m", base)(["a", "b"])

    def test_http_error_is_wrapped(self, stub):
        base, handler = stub
        handler.routes["/v1/embeddings"] = (500, {"error": "boom"})
        with pytest.raises(EmbeddingError, match="HTTP 500"):
            OpenAICompatibleEmbedder("k", "m", base)(["a"])

    def test_unreachable_host_is_wrapped(self):
        emb = OpenAICompatibleEmbedder("k", "m", "http://127.0.0.1:1", timeout=1.0)
        with pytest.raises(EmbeddingError, match="unreachable"):
            emb(["a"])

    def test_model_is_required(self):
        with pytest.raises(ValueError, match="model is required"):
            OpenAICompatibleEmbedder("k", "")


class TestOllamaEmbedder:
    def test_uses_the_native_route(self, stub):
        base, handler = stub
        handler.routes["/api/embed"] = (200, lambda b: {
            "embeddings": [[0.1, 0.2] for _ in b["input"]]})
        out = OllamaEmbedder("nomic-embed-text", base)(["a", "b"])
        assert out == [[0.1, 0.2], [0.1, 0.2]]
        assert handler.seen[-1]["path"] == "/api/embed"
        assert handler.seen[-1]["auth"] is None

    def test_count_mismatch_raises(self, stub):
        base, handler = stub
        handler.routes["/api/embed"] = (200, {"embeddings": [[0.1]]})
        with pytest.raises(EmbeddingError, match="1 embeddings for 2 inputs"):
            OllamaEmbedder("m", base)(["a", "b"])


class TestBuildEmbedder:
    def test_empty_spec_is_none(self):
        assert build_embedder(None) is None
        assert build_embedder("") is None

    def test_unknown_provider(self):
        with pytest.raises(ValueError, match="Unknown embedder provider"):
            build_embedder("cohere:embed-v3")

    def test_missing_model(self):
        with pytest.raises(ValueError, match="Missing model"):
            build_embedder("openai:")

    def test_openai_and_digitalocean_bases(self, monkeypatch):
        monkeypatch.delenv("AI_HOUKAI_EMBED_BASE_URL", raising=False)
        monkeypatch.setenv("AI_HOUKAI_EMBED_API_KEY", "sk-env")
        oa = build_embedder("openai:text-embedding-3-small")
        assert oa.base_url == "https://api.openai.com" and oa.api_key == "sk-env"
        do = build_embedder("digitalocean:gte-large-en-v1.5")
        assert do.base_url == DIGITALOCEAN_BASE_URL

    def test_base_url_override(self, monkeypatch):
        monkeypatch.setenv("AI_HOUKAI_EMBED_BASE_URL", "http://gpu.internal:8000")
        assert build_embedder("openai:m").base_url == "http://gpu.internal:8000"
        assert build_embedder("ollama:m").base_url == "http://gpu.internal:8000"

    def test_credentials_never_come_from_the_spec(self, monkeypatch):
        """A spec string lands in config files and process listings."""
        monkeypatch.delenv("AI_HOUKAI_EMBED_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert build_embedder("openai:m").api_key == ""

    def test_local_returns_the_cached_loader(self):
        assert build_embedder("local:all-MiniLM-L6-v2") is local_embedder(
            "all-MiniLM-L6-v2")


class TestStoreSeam:
    def test_injected_function_is_used(self, tmp_path):
        emb = FakeEmbedder(dim=8)
        store = MemoryStore(path=str(tmp_path / "c"), collection="seam",
                            embedding_function=emb)
        try:
            assert store._embed_fn is emb
            mem = store.remember("injected embedder subject")
            assert store.recall("injected embedder subject", k=1)[0][0].id == mem.id
            assert store.probe_embedding() == {
                "ok": True, "dim": 8,
                "latency_ms": pytest.approx(store.probe_embedding()["latency_ms"],
                                            abs=1000),
            }
        finally:
            store.client.close()

    def test_env_var_selects_the_embedder(self, tmp_path, monkeypatch, stub):
        base, handler = stub
        handler.routes["/api/embed"] = (200, lambda b: {
            "embeddings": [[0.1, 0.2, 0.3] for _ in b["input"]]})
        monkeypatch.setenv("AI_HOUKAI_EMBED_BASE_URL", base)
        monkeypatch.setenv("AI_HOUKAI_EMBEDDER", "ollama:nomic-embed-text")
        store = MemoryStore(path=str(tmp_path / "c"), collection="envseam")
        try:
            assert isinstance(store._embed_fn, OllamaEmbedder)
            store.remember("routed through the env-configured embedder")
            assert handler.seen and handler.seen[-1]["path"] == "/api/embed"
        finally:
            store.client.close()

    def test_explicit_function_beats_the_env_var(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AI_HOUKAI_EMBEDDER", "openai:should-not-be-used")
        emb = FakeEmbedder()
        store = MemoryStore(path=str(tmp_path / "c"), collection="prec",
                            embedding_function=emb)
        try:
            assert store._embed_fn is emb
        finally:
            store.client.close()

    def test_legacy_private_alias_still_resolves(self):
        assert _get_embed_fn is local_embedder

    def test_readiness_reports_the_injected_backend(self, tmp_path):
        store = make_store(tmp_path / "c", dim=12)
        try:
            out = store.readiness()
            assert out["ready"] is True
            assert out["checks"]["embedder"]["dim"] == 12
        finally:
            store.client.close()

    def test_a_failing_embedder_surfaces_in_readiness(self, tmp_path):
        def broken(_texts):
            raise EmbeddingError("provider down")

        store = MemoryStore(path=str(tmp_path / "c"), collection="broken",
                            embedding_function=broken)
        try:
            out = store.readiness()
            assert out["ready"] is False
            assert "provider down" in out["checks"]["embedder"]["error"]
        finally:
            store.client.close()


class TestFakeEmbedder:
    def test_is_deterministic(self):
        a, b = FakeEmbedder(), FakeEmbedder()
        assert a(["repeatable"]) == b(["repeatable"])

    def test_is_normalised(self):
        [vec] = FakeEmbedder(dim=64)(["normalise me"])
        assert len(vec) == 64
        assert sum(v * v for v in vec) == pytest.approx(1.0, abs=1e-9)

    def test_distinct_texts_differ(self):
        out = FakeEmbedder()(["alpha", "beta"])
        assert out[0] != out[1]

    def test_rejects_a_bad_dim(self):
        with pytest.raises(ValueError, match="dim must be > 0"):
            FakeEmbedder(dim=0)

    def test_a_full_store_roundtrip_needs_no_model(self, tmp_path):
        """The point of the seam: real store behaviour, no torch, no download."""
        started = time.perf_counter()
        store = make_store(tmp_path / "c")
        try:
            a = store.remember("first fact", tags=["x"])
            b = store.remember("second fact", type="procedural")
            store.link(src_id=a.id, dst_id=b.id, rel="refines")

            assert store.count() == 2
            assert {m.id for m, _ in store.recall("first fact", k=2)} == {a.id, b.id}
            assert store.neighbors(a.id, direction="out") == [(store.get(b.id), "refines")]
            assert store.recall_pack("first fact").items
        finally:
            store.client.close()
        # A real model load alone is multiple seconds; this whole test is well
        # under a second. Generous bound — the assertion is "no model load".
        assert time.perf_counter() - started < 5.0


class TestPytestFixtures:
    def test_fake_store_fixture(self, fake_store):
        mem = fake_store.remember("fixture-provided store")
        assert fake_store.get(mem.id) is not None

    def test_fake_embedder_fixture(self, fake_embedder):
        assert len(fake_embedder(["x"])[0]) == 32
