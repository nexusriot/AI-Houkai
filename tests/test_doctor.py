"""Unit tests for the diagnostics surface: MemoryStore.probe_embedding /
readiness, the HTTP /ready endpoint, and the `houkai doctor` CLI command."""

from __future__ import annotations

import json

from typer.testing import CliRunner

import ai_houkai.http_server.server as http_mod
from ai_houkai.cli.main import app
from ai_houkai.memory_system import MemoryStore

runner = CliRunner()


class TestProbeAndReadiness:
    def test_probe_embedding_reports_dim(self, store: MemoryStore):
        probe = store.probe_embedding()
        assert probe["ok"] is True
        assert probe["dim"] > 0
        assert probe["latency_ms"] >= 0

    def test_readiness_ok(self, store: MemoryStore):
        r = store.readiness()
        assert r["ready"] is True
        assert r["checks"]["store"]["ok"] is True
        assert r["checks"]["embedder"]["ok"] is True

    def test_readiness_reports_embedder_failure(self, store: MemoryStore):
        # A broken embedding backend flips readiness to not-ready without raising.
        def boom(_texts):
            raise RuntimeError("provider unreachable")

        store._embed_fn = boom
        r = store.readiness()
        assert r["ready"] is False
        assert r["checks"]["embedder"]["ok"] is False
        assert "provider unreachable" in r["checks"]["embedder"]["error"]


class TestReadyEndpoint:
    def test_ready_returns_200_when_ready(self, store: MemoryStore):
        status, payload = http_mod._ready(store, None, None, None)
        assert status == 200
        assert payload["ready"] is True

    def test_ready_returns_503_when_embedder_down(self, store: MemoryStore):
        def boom(_texts):
            raise RuntimeError("down")

        store._embed_fn = boom
        status, payload = http_mod._ready(store, None, None, None)
        assert status == 503
        assert payload["ready"] is False

    def test_ready_route_is_auth_exempt(self, store: MemoryStore):
        # /ready must be reachable without the bearer token (like /health) so
        # readiness probes work without the secret. The exemption short-circuits
        # before any header lookup, so a bare instance is enough to assert it.
        handler_cls = http_mod.build_handler(store, auth_token="s3cret")
        inst = handler_cls.__new__(handler_cls)
        assert inst._authorized("/ready") is True
        assert inst._authorized("/health") is True

    def test_ready_body_is_sanitized(self, store: MemoryStore):
        # Auth-exempt, so /ready must not leak error strings / paths / provider
        # dim+latency — only per-check ok bools.
        def boom(_texts):
            raise RuntimeError("secret /etc/foo provider host unreachable")

        store._embed_fn = boom
        status, payload = http_mod._ready(store, None, None, None)
        assert status == 503
        emb = payload["checks"]["embedder"]
        assert emb == {"ok": False}
        assert payload["checks"]["store"] == {"ok": True}


class TestReadinessCache:
    def test_cache_avoids_reembedding(self, store: MemoryStore):
        calls = {"n": 0}
        orig = store._embed_fn

        def counting(texts):
            calls["n"] += 1
            return orig(texts)

        store._embed_fn = counting
        store.readiness(cache_ttl=60.0)
        store.readiness(cache_ttl=60.0)
        assert calls["n"] == 1, "a ready result within the TTL must be cached"
        store.readiness(cache_ttl=0.0)
        assert calls["n"] == 2, "cache_ttl=0 must always re-probe"

    def test_failure_is_not_cached(self, store: MemoryStore):
        real = store._embed_fn

        def boom(_texts):
            raise RuntimeError("down")

        store._embed_fn = boom
        assert store.readiness(cache_ttl=60.0)["ready"] is False
        # Recovery must be seen immediately — a not-ready result is never cached.
        store._embed_fn = real
        assert store.readiness(cache_ttl=60.0)["ready"] is True


class TestDoctorCommand:
    def _invoke(self, tmp_path, *extra):
        return runner.invoke(
            app,
            ["--store", str(tmp_path / "chroma"), "--collection", "doc_test",
             "doctor", *extra],
        )

    def test_doctor_reports_ready(self, tmp_path):
        result = self._invoke(tmp_path)
        assert result.exit_code == 0, result.output
        assert "OK — ready" in result.output

    def test_doctor_json(self, tmp_path):
        result = self._invoke(tmp_path, "--json")
        assert result.exit_code == 0, result.output
        # The JSON object is the last brace-delimited block in the output.
        start = result.output.index("{")
        report = json.loads(result.output[start:])
        assert report["ok"] is True
        names = {c["name"] for c in report["checks"]}
        assert {"config", "store", "embedder"} <= names
