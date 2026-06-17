"""End-to-end functional tests — black-box, against the *installed* artifacts.

Unlike the unit suite in ``tests/`` (which imports ``MemoryStore`` in-process),
these drive the real deployment surface:

  * the ``houkai`` CLI console script, as a subprocess, over a real on-disk
    ChromaDB store — exercising arg parsing, JSON output and the full
    remember → recall → link → supersede → export/import → prune lifecycle;
  * the ``ai-houkai-serve`` HTTP server, started as its own process and hit
    over a real socket — including a concurrency regression test.

They are intentionally kept *out* of ``testpaths`` (see pyproject) so a plain
``pytest`` run stays fast and never spawns servers. Run them explicitly:

    pytest functional_tests/ -v

or, hermetically, inside the container (see functional_tests/Dockerfile):

    ./functional_tests/run.sh

Requires the package installed with the ``cli`` extra (``pip install ".[cli]"``)
so the ``houkai`` / ``ai-houkai-serve`` entry points are on PATH.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

import pytest

COLLECTION = "func"

# Every CLI subprocess gets a generous timeout: the first invocation in a fresh
# process pays a one-off cost to load the sentence-transformers model.
_CLI_TIMEOUT = 180


# ── CLI helpers ───────────────────────────────────────────────────────────────

def _have_cli() -> bool:
    return shutil.which("houkai") is not None and shutil.which("ai-houkai-serve") is not None


pytestmark = pytest.mark.skipif(
    not _have_cli(),
    reason="needs the installed 'houkai' + 'ai-houkai-serve' console scripts "
           "(pip install \".[cli]\")",
)


def houkai(store: str, *args: str, collection: str = COLLECTION,
           input_text: str | None = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run ``houkai -S <store> -C <collection> <args...>`` and return the result."""
    cmd = ["houkai", "-S", store, "-C", collection, *args]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, input=input_text, timeout=_CLI_TIMEOUT,
    )
    if check and proc.returncode != 0:
        raise AssertionError(
            f"`{' '.join(cmd)}` exited {proc.returncode}\n"
            f"── stdout ──\n{proc.stdout}\n── stderr ──\n{proc.stderr}"
        )
    return proc


def houkai_json(store: str, *args: str, collection: str = COLLECTION):
    """Run a CLI command with ``--format json`` already appended and parse stdout."""
    proc = houkai(store, *args, collection=collection)
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:  # pragma: no cover — surfaces bad output
        raise AssertionError(
            f"expected JSON from `houkai {' '.join(args)}`, got:\n{proc.stdout}"
        ) from exc


def remember(store: str, text: str, *extra: str) -> str:
    """Store a memory and return its id (the CLI echoes the id on success)."""
    proc = houkai(store, "remember", text, *extra)
    return proc.stdout.strip().splitlines()[-1].strip()


def test_cli_full_lifecycle(tmp_path):
    store = str(tmp_path / "chroma")

    # — empty store —
    empty = houkai_json(store, "stats", "--format", "json")
    assert empty["total"] == 0

    sem1 = remember(store, "Python uses the GIL to serialise bytecode execution",
                    "-t", "semantic", "-g", "python", "-i", "0.85")
    sem2 = remember(store, "ChromaDB persists embedding vectors to disk via SQLite",
                    "-t", "semantic", "-g", "infra", "-i", "0.70")
    proc = remember(store, "Deploy the service with `make release`",
                    "-t", "procedural", "-g", "ops", "-i", "0.60")
    remember(store, "Fixed a flaky timeout in the auth integration test",
             "-t", "episodic", "-g", "testing", "-i", "0.40")
    remember(store, "Fixed another flaky timeout in the auth integration test",
             "-t", "episodic", "-g", "testing", "-i", "0.40")

    assert all(len(i) >= 8 for i in (sem1, sem2, proc))

    stats = houkai_json(store, "stats", "--format", "json")
    assert stats["total"] == 5
    assert stats["active"] == 5
    assert stats["by_type"]["semantic"] == 2
    assert stats["by_type"]["episodic"] == 2
    assert stats["by_type"]["procedural"] == 1

    # — recall finds the right memory (hybrid blends cosine + BM25) —
    hits = houkai_json(store, "recall", "global interpreter lock threading",
                       "-k", "3", "--mode", "hybrid", "--format", "json")
    assert any(h["id"] == sem1 for h in hits), [h["text"] for h in hits]

    # metadata filters
    infra_hits = houkai_json(store, "recall", "storage backend",
                             "-k", "5", "-t", "semantic", "-g", "infra",
                             "--format", "json")
    assert all(h["type"] == "semantic" for h in infra_hits)
    assert all("infra" in h["tags"] for h in infra_hits)

    # — pack into a token budget —
    pack = houkai_json(store, "pack", "vectors on disk", "-b", "200", "--format", "json")
    assert pack["text"].strip() != ""
    assert pack["used_tokens"] <= pack["budget"]

    # — link + neighbors —
    houkai(store, "link", sem1, sem2, "--rel", "refines")
    nbrs = houkai_json(store, "neighbors", sem1, "--format", "json")
    assert sem2 in {n["id"] for n in nbrs}

    # — export the active set, then import into a fresh collection —
    archive = str(tmp_path / "backup.ahkai")
    houkai(store, "export", archive)
    assert os.path.exists(archive)

    imp = houkai(store, "import", archive, "--yes", collection="func_restored")
    assert "imported" in imp.stdout.lower() or imp.returncode == 0
    restored = houkai_json(store, "stats", "--format", "json", collection="func_restored")
    assert restored["total"] == 5

    # — supersede hides a memory from default views but keeps it on disk —
    houkai(store, "supersede", sem2, sem1)
    active = houkai_json(store, "list", "-n", "50", "--format", "json")
    assert sem2 not in {m["id"] for m in active}
    with_super = houkai_json(store, "list", "-n", "50",
                             "--include-superseded", "--format", "json")
    assert sem2 in {m["id"] for m in with_super}

    # — the audit journal recorded the writes —
    jrnl = houkai(store, "journal", "tail", "-n", "50")
    assert "remember" in jrnl.stdout.lower()


def test_cli_stats_health_protects_procedural(tmp_path):
    """A zero-importance memory scores below the prune threshold, but a
    *procedural* one is protected by DecayEngine and must NOT be counted
    at-risk — and the health report must agree with `prune`."""
    store = str(tmp_path / "chroma")

    remember(store, "Throwaway scratch note", "-t", "semantic", "-i", "0.0")
    remember(store, "Protected runbook: rotate the TLS certs quarterly",
             "-t", "procedural", "-i", "0.0")

    health = houkai_json(store, "stats", "--health", "--format", "json")["health"]
    # Only the semantic memory is at risk; the procedural one is protected.
    assert health["at_risk_count"] == 1, health

    # And `prune` (default protect_types=procedural) agrees: one candidate.
    dry = houkai(store, "prune", "--min-score", "0.05")
    assert "Prune candidates (1)" in dry.stdout, dry.stdout


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _http(method: str, url: str, payload: dict | None = None, timeout: float = 10,
          retries: int = 4):
    """Issue a JSON request, retrying only transient connection failures.

    A fresh TCP connection per call plus many parallel callers occasionally hits
    a transient reset; retrying keeps the test measuring *store* behaviour rather
    than raw socket luck. Application errors (4xx/5xx) are never retried.
    """
    data = json.dumps(payload).encode() if payload is not None else None
    last_exc: Exception | None = None
    for attempt in range(retries):
        req = urllib.request.Request(
            url, data=data, method=method,
            headers={"Content-Type": "application/json"} if data else {},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode()
                return resp.status, (json.loads(body) if body else {})
        except urllib.error.HTTPError as exc:  # a real status code — don't retry
            body = exc.read().decode()
            return exc.code, (json.loads(body) if body else {})
        except (ConnectionError, urllib.error.URLError, OSError) as exc:
            last_exc = exc
            time.sleep(0.1 * (attempt + 1))
    raise AssertionError(f"{method} {url} failed after {retries} retries: {last_exc}")


@pytest.fixture()
def http_server(tmp_path):
    """Start ai-houkai-serve as a real subprocess; yield its base URL."""
    port = _free_port()
    env = {
        **os.environ,
        "AI_HOUKAI_PATH": str(tmp_path / "chroma"),
        "AI_HOUKAI_COLLECTION": "func_http",
        "AI_HOUKAI_HTTP_HOST": "127.0.0.1",
        "AI_HOUKAI_HTTP_PORT": str(port),
    }
    proc = subprocess.Popen(
        ["ai-houkai-serve"], env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        # Wait for liveness — the first request also loads the embedding model.
        # Probe with a raw urlopen (not _http) so a not-yet-listening socket is
        # tolerated rather than raising.
        deadline = time.time() + 120
        while time.time() < deadline:
            if proc.poll() is not None:
                raise AssertionError(
                    f"server exited early ({proc.returncode}):\n{proc.stdout.read()}"
                )
            try:
                with urllib.request.urlopen(f"{base}/health", timeout=5) as resp:
                    if resp.status == 200:
                        break
            except (urllib.error.URLError, ConnectionError, OSError):
                time.sleep(0.5)
        else:
            raise AssertionError("server did not become healthy within 120s")
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover
            proc.kill()


def test_http_roundtrip(http_server):
    base = http_server

    status, mem = _http("POST", f"{base}/memories",
                        {"text": "remember me over http", "type": "semantic",
                         "tags": ["web"], "importance": 0.7})
    assert status == 201 and mem["stored"] is True
    mid = mem["id"]

    status, got = _http("GET", f"{base}/memories/{mid}")
    assert status == 200 and got["text"] == "remember me over http"

    status, res = _http("POST", f"{base}/recall",
                        {"query": "remember http", "k": 3, "mode": "hybrid"})
    assert status == 200
    assert any(r["id"] == mid for r in res["results"])


def test_http_concurrent_links_no_lost_updates(http_server):
    """Regression: link/_touch are read-modify-write against ChromaDB. Without
    serialisation, concurrent requests clobber each other's updates. Fire many
    parallel /links at one hub and assert none are lost."""
    base = http_server

    _, hub = _http("POST", f"{base}/memories", {"text": "hub node", "type": "semantic"})
    hub_id = hub["id"]

    n = 25
    target_ids = []
    for i in range(n):
        _, t = _http("POST", f"{base}/memories",
                     {"text": f"spoke {i}", "type": "semantic"})
        target_ids.append(t["id"])

    errors: list[str] = []

    def add_link(tid: str) -> None:
        try:
            status, _ = _http("POST", f"{base}/links",
                              {"src_id": hub_id, "dst_id": tid, "rel": "related"})
            if status != 200:
                errors.append(f"status {status} for {tid}")
        except Exception as exc:  # pragma: no cover
            errors.append(f"{type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=add_link, args=(t,)) for t in target_ids]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    _, hub_after = _http("GET", f"{base}/memories/{hub_id}")
    assert len(hub_after["links"]) == n, (
        f"lost {n - len(hub_after['links'])} of {n} concurrent links — "
        "store access is not serialised"
    )


if __name__ == "__main__":  # allow `python functional_tests/test_e2e.py`
    sys.exit(pytest.main([__file__, "-v"]))
