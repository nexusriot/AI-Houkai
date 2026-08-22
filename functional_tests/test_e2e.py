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

import contextlib
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
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

COLLECTION = "func"

# Every CLI subprocess gets a generous timeout: the first invocation in a fresh
# process pays a one-off cost to load the sentence-transformers model.
_CLI_TIMEOUT = 180


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

    # Empty store.
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

    # Recall finds the right memory (hybrid blends cosine + BM25)
    hits = houkai_json(store, "recall", "global interpreter lock threading",
                       "-k", "3", "--mode", "hybrid", "--format", "json")
    assert any(h["id"] == sem1 for h in hits), [h["text"] for h in hits]

    # metadata filters
    infra_hits = houkai_json(store, "recall", "storage backend",
                             "-k", "5", "-t", "semantic", "-g", "infra",
                             "--format", "json")
    assert all(h["type"] == "semantic" for h in infra_hits)
    assert all("infra" in h["tags"] for h in infra_hits)

    # Pack into a token budget.
    pack = houkai_json(store, "pack", "vectors on disk", "-b", "200", "--format", "json")
    assert pack["text"].strip() != ""
    assert pack["used_tokens"] <= pack["budget"]

    # Link + neighbors.
    houkai(store, "link", sem1, sem2, "--rel", "refines")
    nbrs = houkai_json(store, "neighbors", sem1, "--format", "json")
    assert sem2 in {n["id"] for n in nbrs}

    # Export the active set, then import into a fresh collection.
    archive = str(tmp_path / "backup.ahkai")
    houkai(store, "export", archive)
    assert os.path.exists(archive)

    imp = houkai(store, "import", archive, "--yes", collection="func_restored")
    assert "imported" in imp.stdout.lower() or imp.returncode == 0
    restored = houkai_json(store, "stats", "--format", "json", collection="func_restored")
    assert restored["total"] == 5

    # Supersede hides a memory from default views but keeps it on disk.
    houkai(store, "supersede", sem2, sem1)
    active = houkai_json(store, "list", "-n", "50", "--format", "json")
    assert sem2 not in {m["id"] for m in active}
    with_super = houkai_json(store, "list", "-n", "50",
                             "--include-superseded", "--format", "json")
    assert sem2 in {m["id"] for m in with_super}

    # The audit journal recorded the writes.
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


def test_cli_cjk_recall(tmp_path):
    """CJK text round-trips through the CLI subprocess boundary, and hybrid
    recall (which now emits CJK character bigrams for lexical matching) surfaces
    a Japanese memory for a Japanese query over English distractors."""
    store = str(tmp_path / "chroma")

    jp = remember(store, "日本語のテストメモリ", "-t", "semantic")
    remember(store, "the quick brown fox jumps over the lazy dog", "-t", "semantic")
    remember(store, "deploy the service with make release", "-t", "procedural")

    hits = houkai_json(store, "recall", "日本語", "-k", "3",
                       "--mode", "hybrid", "--format", "json")
    assert hits, "no hits for the CJK query"
    assert hits[0]["id"] == jp, [h["text"] for h in hits]


def test_cli_stats_health_frequency_weight_reinforces(tmp_path):
    """`stats --health --frequency-weight` feeds the recall-reinforcement term
    into the decay-score histogram: a frequently-recalled memory scores into a
    strictly higher band with reinforcement than without it."""
    store = str(tmp_path / "chroma")

    remember(store, "reinforced fact about widgets", "-t", "semantic", "-i", "0.5")
    # Each recall bumps access_count; build it up so reinforcement has an effect.
    for _ in range(3):
        houkai_json(store, "recall", "reinforced widgets", "-k", "1", "--format", "json")

    def _top_band(freq_weight: str) -> int:
        h = houkai_json(store, "stats", "--health", "--decay-rate", "0.1",
                        "--frequency-weight", freq_weight, "--format", "json")["health"]
        counts = list(h["decay_histogram"].values())  # bands ordered low → high
        return max(i for i, c in enumerate(counts) if c > 0)

    base_band = _top_band("0.0")
    reinforced_band = _top_band("2.0")
    assert reinforced_band > base_band, (base_band, reinforced_band)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _http(method: str, url: str, payload: dict | None = None, timeout: float = 10,
          retries: int = 4, headers: dict | None = None):
    """Issue a JSON request, retrying only transient connection failures.

    A fresh TCP connection per call plus many parallel callers occasionally hits
    a transient reset; retrying keeps the test measuring *store* behaviour rather
    than raw socket luck. Application errors (4xx/5xx) are never retried.
    """
    data = json.dumps(payload).encode() if payload is not None else None
    base_headers = {"Content-Type": "application/json"} if data else {}
    if headers:
        base_headers.update(headers)
    last_exc: Exception | None = None
    for attempt in range(retries):
        req = urllib.request.Request(
            url, data=data, method=method, headers=dict(base_headers),
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


@contextlib.contextmanager
def _serve(tmp_path, extra_env: dict | None = None):
    """Start ai-houkai-serve as a real subprocess; yield its base URL.

    ``extra_env`` overlays the base environment (e.g. to set
    ``AI_HOUKAI_HTTP_TOKEN`` for the auth-enabled server).
    """
    port = _free_port()
    env = {
        **os.environ,
        "AI_HOUKAI_PATH": str(tmp_path / "chroma"),
        "AI_HOUKAI_COLLECTION": "func_http",
        "AI_HOUKAI_HTTP_HOST": "127.0.0.1",
        "AI_HOUKAI_HTTP_PORT": str(port),
        **(extra_env or {}),
    }
    proc = subprocess.Popen(
        ["ai-houkai-serve"], env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        # Wait for liveness — the first request also loads the embedding model.
        # Probe with a raw urlopen (not _http) so a not-yet-listening socket is
        # tolerated rather than raising. /health is open even with auth set.
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


@pytest.fixture()
def http_server(tmp_path):
    """A running ai-houkai-serve with no auth; yields its base URL."""
    with _serve(tmp_path) as base:
        yield base

# Bearer token used by the auth-enabled server fixture.
_AUTH_TOKEN = "func-secret-token"


@pytest.fixture()
def http_server_auth(tmp_path):
    """A running ai-houkai-serve with AI_HOUKAI_HTTP_TOKEN set; yields (base, token)."""
    with _serve(tmp_path, {"AI_HOUKAI_HTTP_TOKEN": _AUTH_TOKEN}) as base:
        yield base, _AUTH_TOKEN


def test_http_roundtrip(http_server):
    base = http_server

    # — /health is liveness-only: it must expose status + count but NOT the
    #   collection name (topology), per the HTTP hardening. —
    status, health = _http("GET", f"{base}/health")
    assert status == 200 and health.get("status") == "ok" and "count" in health
    assert "collection" not in health, health

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


def test_http_stress_concurrent_add_and_fetch(http_server):
    """Stress: hammer the server with many concurrent writers *and* readers at
    once, then concurrently fetch every item back.

    Asserts the store stays consistent under load — every add lands exactly
    once (unique id, server count matches), reads running alongside writes never
    error, and each item is individually retrievable with its exact text. Guards
    against lost writes, dropped connections and cross-item corruption.
    """
    base = http_server
    n_items = 120
    n_workers = 16

    def _text(i: int) -> str:
        return f"stress item {i} marker{i:05d}"

    errors: list[str] = []
    ids: dict[int, str] = {}
    ids_lock = threading.Lock()

    def add(i: int) -> None:
        try:
            status, mem = _http("POST", f"{base}/memories",
                                {"text": _text(i), "type": "semantic",
                                 "tags": ["stress"]})
            if status != 201 or not mem.get("stored"):
                errors.append(f"add {i}: status {status} {mem}")
                return
            with ids_lock:
                ids[i] = mem["id"]
        except Exception as exc:  # pragma: no cover
            errors.append(f"add {i}: {type(exc).__name__}: {exc}")

    def read_noise(_: int) -> None:
        # Readers running *concurrently* with the writers. /recall also bumps
        # access_count (a read-modify-write), so this exercises the store lock
        # under mixed read/write load. None of it may error.
        try:
            _http("GET", f"{base}/stats")
            _http("POST", f"{base}/recall", {"query": "stress item", "k": 5})
        except Exception as exc:  # pragma: no cover
            errors.append(f"read: {type(exc).__name__}: {exc}")

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(add, i) for i in range(n_items)]
        futures += [pool.submit(read_noise, i) for i in range(n_items // 2)]
        for f in as_completed(futures):
            f.result()

    assert not errors, errors[:10]
    assert len(ids) == n_items, f"only {len(ids)} of {n_items} adds recorded an id"
    assert len(set(ids.values())) == n_items, "duplicate ids returned across adds"

    # Stored exactly once — the server's own count agrees.
    _, stats = _http("GET", f"{base}/stats")
    assert stats["count"] == n_items, stats

    fetch_errors: list[str] = []

    def fetch(item: tuple[int, str]) -> None:
        i, mid = item
        try:
            status, got = _http("GET", f"{base}/memories/{mid}")
            if status != 200:
                fetch_errors.append(f"fetch {mid}: status {status}")
            elif got.get("text") != _text(i):
                fetch_errors.append(f"fetch {mid}: wrong text {got.get('text')!r}")
        except Exception as exc:  # pragma: no cover
            fetch_errors.append(f"fetch {mid}: {type(exc).__name__}: {exc}")

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        list(pool.map(fetch, list(ids.items())))

    assert not fetch_errors, fetch_errors[:10]


def test_http_recall_bumps_all_hits(http_server):
    """Regression guard for the batched access-bump (`_touch_many`): a single
    /recall that returns several memories must bump *every* hit's access_count
    exactly once — none dropped by the one batched Chroma write."""
    base = http_server

    ids = []
    for i in range(3):
        _, mem = _http("POST", f"{base}/memories",
                       {"text": f"alpha widget variant {i}", "type": "semantic"})
        ids.append(mem["id"])

    # Freshly stored → never recalled yet.
    for mid in ids:
        _, got = _http("GET", f"{base}/memories/{mid}")
        assert got["access_count"] == 0, got

    status, res = _http("POST", f"{base}/recall", {"query": "alpha widget", "k": 5})
    assert status == 200
    assert set(ids) <= {r["id"] for r in res["results"]}, res["results"]

    # Every hit bumped exactly once by the single batched write.
    for mid in ids:
        _, got = _http("GET", f"{base}/memories/{mid}")
        assert got["access_count"] == 1, got


def test_http_auth_enforced(http_server_auth):
    """With AI_HOUKAI_HTTP_TOKEN set: /health stays open, but every other route
    requires the correct bearer token (constant-time compared); an absent or
    wrong token is rejected with 401."""
    base, token = http_server_auth

    # Liveness probe is always open, even with auth configured.
    status, _ = _http("GET", f"{base}/health")
    assert status == 200

    # Protected route, no token → 401.
    status, _ = _http("GET", f"{base}/stats")
    assert status == 401

    # Wrong token → 401.
    status, _ = _http("GET", f"{base}/stats",
                      headers={"Authorization": "Bearer wrong-token"})
    assert status == 401

    # Correct token → 200.
    status, body = _http("GET", f"{base}/stats",
                         headers={"Authorization": f"Bearer {token}"})
    assert status == 200 and "count" in body

if __name__ == "__main__":  # allow `python functional_tests/test_e2e.py`
    sys.exit(pytest.main([__file__, "-v"]))

#
# The unit suite already pins the semantics of everything below, in-process.
# What only these can catch is the deployment wiring: console-script argument
# parsing, and state that has to stay consistent across process boundaries —
# which every in-process test papers over by holding one long-lived store.


def list_ids(store: str) -> list[str]:
    """Ids in the store, newest first.

    Parsed strictly: `list --format json` prints a valid JSON array to stdout
    even when the store is empty (`[]`; the human "No memories found." goes to
    stderr), so a lenient fallback here would only mask a regression to
    un-parseable output.
    """
    proc = houkai(store, "list", "-n", "100", "--format", "json")
    return [m["id"] for m in json.loads(proc.stdout)]


def test_cli_curation_lifecycle(tmp_path):
    """merge / versions / tags / path / trash through the real console script."""
    store = str(tmp_path / "chroma")

    target = remember(store, "the deploy runbook", "--tag", "ops")
    other = remember(store, "the rollback runbook", "--tag", "ops")
    pointer = remember(store, "points at the rollback one")
    houkai(store, "link", pointer, other, "--rel", "refines")

    # Path: undirected, so the arrow direction does not matter.
    path = houkai_json(store, "path", other, pointer, "--json")
    assert path["found"] is True and path["length"] == 1

    # Merge: the incoming edge must survive, re-pointed at the target.
    houkai(store, "merge", target, other, "-y")
    nbrs = houkai_json(store, "neighbors", pointer, "--direction", "out",
                       "--format", "json")
    assert [n["id"] for n in nbrs] == [target]
    assert houkai(store, "show", other, check=False).returncode != 0

    # Versions: the merge rewrote the target's text, so it has history.
    versions = houkai_json(store, "versions", target, "--json")
    assert [v["text"] for v in versions] == ["the deploy runbook"]

    # Tags.
    assert houkai_json(store, "tags", "list", "--json") == [
        {"tag": "ops", "count": 1}]
    houkai(store, "tags", "rename", "ops", "operations")
    assert houkai_json(store, "tags", "list", "--json")[0]["tag"] == "operations"


def test_cli_trash_roundtrip_across_processes(tmp_path):
    """A trashed memory must survive as a file, not just in a live store."""
    store = str(tmp_path / "chroma")
    mid = remember(store, "recoverable by trash")

    houkai(store, "trash", "put", mid)
    assert list_ids(store) == []
    # Each CLI call is a fresh process: the trash file is the only carrier.
    assert [e["memory_id"] for e in
            houkai_json(store, "trash", "list", "--json")] == [mid]

    houkai(store, "trash", "restore", mid[:8])
    assert list_ids(store) == [mid]

    houkai(store, "trash", "put", mid)
    houkai(store, "trash", "purge", "-y")
    assert houkai_json(store, "trash", "list", "--json") == []
    assert houkai(store, "trash", "restore", mid[:8], check=False).returncode != 0


def test_cli_pinned_trust_and_idempotent(tmp_path):
    """The three write-time flags, across the subprocess boundary."""
    store = str(tmp_path / "chroma")

    houkai(store, "remember", "ALWAYS run make lint", "--pin",
           "--type", "procedural")
    houkai(store, "remember", "scraped from a blog", "--trust", "untrusted")
    houkai(store, "remember", "ordinary gardening note")

    # Pinned memories lead the pack and are marked; so is untrusted provenance.
    packed = houkai(store, "pack", "gardening", "--include-pinned").stdout
    lines = [l for l in packed.splitlines() if l.startswith("- ")]
    assert lines[0].startswith("- (procedural) [pinned]")
    assert any("[untrusted]" in l for l in lines)

    # min-trust drops the scraped memory.
    trusted = houkai_json(store, "recall", "scraped", "--min-trust", "trusted",
                          "--format", "json")
    assert all("scraped" not in m["text"] for m in trusted)

    # Idempotent writes dedupe by normalised text — across processes.
    before = len(list_ids(store))
    houkai(store, "remember", "Use ruff for linting", "--idempotent")
    houkai(store, "remember", "use  ruff for linting", "--idempotent")
    assert len(list_ids(store)) == before + 1


def test_cli_eval_scores_a_goldset(tmp_path):
    """`houkai eval` over a real gold-set file."""
    store = str(tmp_path / "chroma")
    mid = remember(store, "the deploy runbook lives in ops/deploy.md")
    remember(store, "postgres vacuum runs nightly")

    gold = tmp_path / "gold.jsonl"
    gold.write_text(json.dumps({
        "query": "the deploy runbook lives in ops/deploy.md",
        "relevant_ids": [mid],
    }) + "\n")

    out = houkai_json(store, "eval", str(gold), "--json")
    assert out["n"] == 1
    assert out["recall_at_k"] == 1.0 and out["mrr"] == 1.0
    # The config is recorded so A/B runs are attributable.
    assert out["config"]["mode"] == "hybrid"


def test_cli_time_travel_and_metrics(tmp_path):
    """history / state-at / get-at / metrics against a real journal on disk."""
    store = str(tmp_path / "chroma")
    mid = remember(store, "subject with a history")
    houkai(store, "bump", mid, "=0.9")

    history = houkai_json(store, "history", mid, "--json")
    assert [e["op"] for e in history] == ["remember", "edit"]

    state = houkai_json(store, "state-at", "now", "--json")
    assert state["count"] == 1

    at = houkai_json(store, "get-at", mid, "now", "--json")
    assert at["id"] == mid

    metrics = houkai_json(store, "metrics", "--json")
    assert metrics["count"] == 1 and "p95" in metrics["recall_latency_ms"]

    # undo-last reverses the edit, not the remember.
    houkai(store, "journal", "undo-last", "-y")
    after_undo = json.loads(
        houkai(store, "list", "-n", "10", "--format", "json").stdout)[0]
    assert after_undo["id"] == mid and after_undo["importance"] == 0.5


def test_http_new_routes(http_server):
    """The routes added alongside the CLI commands, over a real socket."""
    base = http_server

    status, a = _http("POST", f"{base}/memories", {"text": "http merge target"})
    status2, b = _http("POST", f"{base}/memories", {"text": "http merge source"})
    assert status == 201 and status2 == 201

    # Subgraph.
    _http("POST", f"{base}/links", {"src_id": a["id"], "dst_id": b["id"],
                                    "rel": "refines"})
    status, graph = _http("POST", f"{base}/subgraph",
                          {"memory_ids": [a["id"]], "depth": 1})
    assert status == 200 and len(graph["nodes"]) == 2

    # find_path.
    status, path = _http("POST", f"{base}/find_path",
                         {"from_id": b["id"], "to_id": a["id"]})
    assert status == 200 and path["found"] is True

    # Tags.
    status, tags = _http("GET", f"{base}/tags")
    assert status == 200 and tags["tags"] == []

    # Trash roundtrip.
    status, _ = _http("POST", f"{base}/trash", {"memory_id": b["id"]})
    assert status == 200
    status, listing = _http("GET", f"{base}/trash")
    assert status == 200 and [e["memory_id"] for e in listing["entries"]] == [b["id"]]
    status, _ = _http("POST", f"{base}/trash/restore", {"memory_id": b["id"]})
    assert status == 200

    # Versions (the merge below will create one)
    status, merged = _http("POST", f"{base}/merge",
                           {"target_id": a["id"], "other_id": b["id"]})
    assert status == 200 and "http merge source" in merged["text"]
    status, versions = _http("GET", f"{base}/memories/{a['id']}/versions")
    assert status == 200 and versions["versions"][0]["text"] == "http merge target"

    # Journal + undo.
    status, journal = _http("GET", f"{base}/journal?n=1")
    assert status == 200 and journal["count"] == 1
    status, undone = _http("POST", f"{base}/undo", {})
    assert status == 200 and undone["ok"] is True

    # Nuke is guarded.
    status, _ = _http("POST", f"{base}/nuke", {})
    assert status == 400
    status, wiped = _http("POST", f"{base}/nuke", {"confirm": "DELETE ALL"})
    assert status == 200 and wiped["ok"] is True
