# Functional (end-to-end) tests

These complement the in-process unit suite in [`../tests/`](../tests/). Instead
of importing `MemoryStore` directly, they exercise the **installed deployment
surface** as a black box:

- the **`houkai` CLI** — driven as a subprocess over a real on-disk ChromaDB
  store, covering the full lifecycle: `remember → recall → pack → link →
  neighbors → export → import → supersede → list → stats → journal` (`prune` is
  exercised separately by `test_cli_stats_health_protects_procedural`), plus
  **CJK recall** (a Japanese query surfaces a Japanese memory via the bigram
  tokenizer) and the **`stats --health --frequency-weight`** reinforcement flag;
- the **`ai-houkai-serve` HTTP server** — started as its own process and hit
  over a real socket: a **concurrency regression test** (25 parallel
  `POST /links`, none lost), a **stress test** (120 items added by 16
  concurrent workers while readers run in parallel, then every item fetched back
  concurrently), the **hardened `/health`** (liveness only — no collection name),
  **bearer-token auth** (`/health` open; protected routes reject an absent/wrong
  token), and a **batched access-bump** check (one `/recall` bumps every hit's
  `access_count` exactly once);
- the **operator-facing failure modes** that only show up in a rendered
  subprocess: every shape of broken `.ahkai` exiting 1 with one `Error:` line
  instead of a traceback, `houkai info` refusing a file `import` would refuse,
  `houkai info` building no store at all (proved by a directory that must not
  appear and a run that must not take a model load), and `list --limit 0`
  listing nothing.

They also lock in specific regressions:

| Test | Guards against |
|---|---|
| `test_http_concurrent_links_no_lost_updates` | read-modify-write races in the threaded HTTP server (lost link/`_touch` updates) |
| `test_http_recall_bumps_all_hits` | the batched `_touch_many` dropping an access-bump for some hits of a single recall |
| `test_http_auth_enforced` | auth bypass — every non-`/health` route must reject an absent/wrong bearer token |
| `test_http_roundtrip` | `/health` leaking the collection name (topology) to callers |
| `test_cli_stats_health_protects_procedural`  | `stats --health` counting protected (`procedural`) memories as at-risk |
| `test_cli_rejects_every_shape_of_broken_archive` | a gzip failure escaping as `OSError` — the command catches `ImportError`, so a plain `.json` printed a traceback panel where every sibling corruption printed one line |
| `test_cli_info_refuses_a_file_import_would_refuse` | `info` reporting "memories on disk: N" for a file that is not an ai-houkai export |
| `test_cli_info_opens_no_store` | the root callback building a `MemoryStore` before *every* subcommand — twelve seconds and a stray `chroma.sqlite3` to read a gzip header |
| `test_cli_list_limit_zero_returns_nothing` | `--limit 0` reading as "unbounded" and printing the whole store |

## Why they live outside `tests/`

`pyproject.toml` sets `testpaths = ["tests"]`, so a plain `pytest` run never
collects these — it stays fast and never spawns servers or subprocesses. Run
them explicitly.

## Running locally

Needs the package installed with the `cli` extra so the console scripts are on
`PATH`:

```bash
pip install ".[cli,dev]"
pytest functional_tests/ -v
```

(If `houkai` / `ai-houkai-serve` aren't installed, the tests skip themselves.)

## Running in Docker (recommended — hermetic)

The image installs a clean copy of the package, pre-downloads the embedding
model at build time, and runs offline (`TRANSFORMERS_OFFLINE=1`):

```bash
./functional_tests/run.sh            # unit + functional, in the container
./functional_tests/run.sh functional # functional suite only

# or by hand, from the repo root:
docker build -f functional_tests/Dockerfile -t ai-houkai-functional .
docker run --rm ai-houkai-functional
```

> The first build is large and slow — it pulls CPU-only torch and the
> sentence-transformers model. Subsequent builds reuse the cached layers.
