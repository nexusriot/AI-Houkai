# AI-Houkai — Agent Memory System

A long-term memory system for AI agents backed by **ChromaDB** and
exposed over **MCP**.  Agents can remember, recall, and forget
information across sessions — with automatic decay of stale memories
and periodic reflection that condenses experience into knowledge.

<p align="center" width="100%">
    <img width="70%" src="https://raw.githubusercontent.com/nexusriot/AI-Houkai/main/logo.png">
</p>

## Features

| Feature | Description |
|---|---|
| **Vector search** | Cosine-space HNSW via ChromaDB + sentence-transformers |
| **Memory types** | `episodic` · `semantic` · `procedural` · `feedback` |
| **Rich metadata** | `importance`, `tags`, `source`, access tracking |
| **In-place edit** | `edit()` — update text/metadata keeping id, links, history; journaled + undoable |
| **Validated vocabularies** | Enum params (`type`, `rel`, `mode`, `fusion`, policies, `direction`) checked once in the store — typos raise, never silently degrade |
| **Decay** | Exponential forgetting — prune old, unimportant memories |
| **Reflection** | Cluster episodic memories → condense into semantic summaries |
| **Memory linking** | Typed directed edges — `refines`, `supersedes`, `derived_from`, … |
| **Conflict detection** | Duplicate / contradiction scan with configurable policies |
| **Hybrid retrieval** | Cosine + BM25 + recency (creation-time by default) + importance + polarity blended scoring; CJK/Korean lexical tokenization |
| **Retrieval controls** | RRF fusion · MMR diversity · near-duplicate dedup · `min_cosine` relevance gate · read-only/`explain` recall |
| **Graph-proximity fusion** | Opt-in `HybridWeights.graph` — PPR-lite spread over the link graph lifts candidates connected to strong hits; gated expansion (`ExpandSpec.rerank`) routes expanded nodes through the same dedup/MMR/top-k as primary hits |
| **Reranking** | Pluggable second-stage `reranker` (e.g. a cross-encoder) rescores the candidate pool before the top-k cut |
| **TTL / expiry** | Per-memory `ttl_seconds`/`expires_at` — expired memories hidden from recall, reclaimed by `purge_expired` |
| **History / point-in-time** | `history()` timeline per memory + `state_at()`/`get_at()` journal replay ("what did I know as of T?") |
| **Runtime metrics** | `metrics()` — op counters (all mutators) + recall latency (avg/max + p50/p95/p99), via `GET /metrics` and the `metrics` MCP tool |
| **Retrieval eval** | `houkai eval <goldset.jsonl>` + the `eval_recall` MCP tool — stdlib-only harness scoring recall/precision/MRR/MAP/nDCG against a gold set, in both ports |
| **Pluggable embedder** | `MemoryStore(embedding_function=…)` or `AI_HOUKAI_EMBEDDER=provider:model` — stdlib OpenAI-compatible / Ollama backends in `ai_houkai.embed`; the local sentence-transformers model is now the `[local]` extra, so the core install needs no torch |
| **Full-corpus lexical recall** | Opt-in `lexical_index="corpus"` — pulls candidates whose text contains the query's tokens into the pool via Chroma's own `where_document`, so an exact-token match with a weak embedding is reachable at all. No second database |
| **Curation** | `merge` (re-points incoming links) · `versions` · tag rename/merge/delete · `find_path` — graduated out of ai-houkai-service, which had been reaching through the library's private API |
| **Trash** | Recoverable delete between `supersede` and `forget` — `trash` / `trash_list` / `trash_restore` / `trash_purge`; decay pruning routes here instead of hard-deleting |
| **Pinned working set** | `pinned` memories lead `recall_pack`/`auto_context` inside the budget and are exempt from decay and eviction — a standing-instruction slot, so `importance` stops doing three jobs |
| **Idempotent writes** | `remember(idempotent=True)` dedupes by normalised-text content hash instead of writing a near-duplicate every session |
| **Provenance trust** | `trust` = `trusted` / `reported` / `untrusted`, set at ingest boundaries; `recall(min_trust=…)` filters and the packer annotates untrusted lines |
| **Tiered reflection** | `ReflectionEngine(types=…)` instead of episodic-only, with a `level` tag so reflections-of-reflections form a hierarchy under a `max_level` guard |
| **Test helpers** | `ai_houkai.testing` — hash-based `FakeEmbedder` + `fake_store` fixtures, so a suite runs with no torch and no model download |
| **Context packing** | `recall_pack` — assemble top-ranked memories into a token-budgeted, ready-to-inject block |
| **Importance auto-assignment** | Heuristic 0–1 scoring from text (instructions > decisions > completions > observations) |
| **Bulk ingest** | `houkai ingest` — chunk files/stdin into memories (markdown-aware) |
| **Batch write** | `remember_many` / `POST /memories/batch` / MCP `remember_many` — bulk store with batched embedding (`ceil(N/batch)` encode passes instead of N) |
| **Collections** | `houkai collections` — list/create/delete/copy namespaces in one store |
| **TUI browser** | `houkai tui` — Textual UI: search, detail pane, link-graph walking |
| **Scheduled maintenance** | Automatic decay + reflection daemon — cron, foreground loop, or background daemon |
| **Audit journal** | Append-only JSONL log of every mutation — `journal tail` / `journal show` / `journal undo` |
| **Portable import/export** | Gzipped `.ahkai` archives with embedded vectors, conflict policies, dry-run |
| **Recall filters** | Narrow search by `source` provenance and a `since`/`until` creation-time window |
| **MCP server** | 41 tools for any MCP client (Claude Code, Claude Desktop) |
| **HTTP/REST API** | `houkai serve` — stdlib JSON server (remember/recall/pack/links/history/metrics), `/health` + `/ready` probes, optional bearer-token auth |
| **Diagnostics** | `houkai doctor` — active embedder probe (latency + dim), embed-dim guardrail, store/journal checks; `GET /ready` readiness endpoint (200/503, auth-exempt, minimal body, ~5 s cached) |
| **CLI (`houkai`)** | 39 commands + 5 sub-groups — CRUD, graph, curation, trash, tags, maintenance, diagnostics, eval, I/O |
| **Port parity, enforced** | `parity.json` is the single source of truth for the MCP tool list, HTTP routes and recall knobs; both ports assert against it in their own suites, so drift fails a build instead of going unnoticed |
| **Multi-provider** | Claude · OpenAI · Ollama (local) agent examples |

## Layout

```
AI-Houkai/
├── ai_houkai/
│   ├── __init__.py               # convenience re-exports
│   ├── memory_system/
│   │   ├── __init__.py
│   │   ├── store.py              # MemoryStore + Memory dataclass (+ edit/export/import/undo)
│   │   ├── async_store.py        # AsyncMemoryStore — coroutine wrapper (single-threaded executor)
│   │   ├── journal.py            # Append-only audit journal (JSONL, gzipped on rotate)
│   │   ├── decay.py              # DecayEngine — exponential forgetting
│   │   ├── reflection.py         # ReflectionEngine — episodic → semantic
│   │   ├── summarizers.py        # build_summarizer("ollama:…|openai:…|anthropic:…")
│   │   ├── importance.py         # score_importance — heuristic auto-assignment
│   │   └── ingest.py             # chunk_text — markdown-aware chunking
│   ├── maintenance/
│   │   ├── __init__.py
│   │   ├── durations.py          # parse/format human duration strings
│   │   ├── state.py              # MaintenanceState — JSON run history
│   │   ├── scheduler.py          # MaintenanceScheduler — tick + run_forever
│   │   └── daemon.py             # PID file helpers + spawn_detached
│   ├── mcp_server/
│   │   ├── __init__.py
│   │   └── server.py             # FastMCP server (41 tools)
│   ├── http_server/
│   │   ├── __init__.py
│   │   └── server.py             # stdlib JSON HTTP/REST server (houkai serve)
│   ├── cli/
│   │   ├── __init__.py
│   │   ├── __main__.py           # python -m ai_houkai.cli
│   │   ├── main.py               # Typer app + _main() entrypoint
│   │   ├── config.py             # store path / collection resolution
│   │   ├── output.py             # rich tables, TSV, JSON, id prefix helpers
│   │   └── commands/
│   │       ├── remember.py       # houkai remember
│   │       ├── recall.py         # houkai recall
│   │       ├── pack.py           # houkai pack / auto-context
│   │       ├── list_cmd.py       # houkai list
│   │       ├── show.py           # houkai show
│   │       ├── forget.py         # houkai forget
│   │       ├── nuke.py           # houkai nuke
│   │       ├── edit.py           # houkai edit / tag / bump
│   │       ├── link.py           # houkai link / unlink / neighbors / graph
│   │       ├── conflicts.py      # houkai conflicts / supersede / restore
│   │       ├── decay.py          # houkai prune
│   │       ├── reflect.py        # houkai reflect
│   │       ├── maintenance.py    # houkai maintenance tick/run/start/stop/status
│   │       ├── journal.py        # houkai journal tail/show/undo
│   │       ├── io.py             # houkai export / import / info / backup
│   │       ├── ingest.py         # houkai ingest
│   │       ├── serve.py          # houkai serve
│   │       ├── collections.py    # houkai collections list/create/delete/copy
│   │       ├── tui_cmd.py        # houkai tui
│   │       ├── stats.py          # houkai stats
│   │       └── doctor.py         # houkai doctor — diagnostics / readiness
│   ├── tui/
│   │   ├── data.py               # view models: recent/search/neighbors + Navigator
│   │   └── app.py                # HoukaiTui — Textual memory browser
│   └── installers/
│       ├── __init__.py
│       ├── common.py             # shared command resolver / JSON patcher / memory guide
│       ├── claude_code.py        # ClaudeCodeInstaller — register MCP w/ Claude Code
│       ├── cursor.py             # CursorInstaller   — register MCP w/ Cursor
│       └── opencode.py           # OpenCodeInstaller — register MCP w/ OpenCode
├── go/                           # Go port — static houkai + ai-houkai-mcp binaries
│   ├── cmd/                      # entry points (CLI, MCP server)
│   ├── internal/                 # memory core, embedders, CLI, MCP, TUI, installers
│   ├── README.md                 # user guide for the Go port
│   └── DESIGN.md                 # Go port internals
├── examples/
│   ├── 01_standalone.py          # pure-Python walkthrough, no LLM
│   ├── 02_ollama_local_network.py  # Ollama on LAN, fully offline
│   ├── 03_claude_desktop.py      # MCP auto-install for Claude Desktop
│   ├── 04_openai.py              # OpenAI GPT-4o / gpt-4o-mini
│   ├── 05_decay_reflection.py    # decay + reflection demo
│   ├── 06_claude_code.py         # Claude Code MCP integration
│   ├── claude_agent.py           # Claude Sonnet REPL (Anthropic SDK)
│   └── pip_package_example.py   # post-install usage walkthrough
├── tests/                        # 1094 tests across 44 files
│   ├── conftest.py               # isolated MemoryStore fixture (tmp_path)
│   ├── test_memory.py            # MemoryStore unit tests (remember/forget/nuke/recall)
│   ├── test_decay.py             # DecayEngine unit tests
│   ├── test_reflection.py        # ReflectionEngine unit tests
│   ├── test_dispatch.py          # cross-provider _dispatch_tool tests
│   ├── test_hybrid.py            # hybrid retrieval scoring
│   ├── test_graph_fusion.py      # graph-proximity fusion + gated expansion
│   ├── test_doctor.py            # doctor CLI + /ready readiness probe
│   ├── test_links.py             # typed links / neighbors / subgraph
│   ├── test_conflicts.py         # conflict detection + supersede/restore
│   ├── test_cli.py               # houkai CLI round-trips (incl. nuke)
│   ├── test_pack.py              # recall_pack token budgeting
│   ├── test_journal.py           # audit journal tail/show/undo
│   ├── test_export_import.py     # portable .ahkai archives
│   ├── test_maintenance.py       # scheduler, daemon, durations, state
│   ├── test_summarizers.py       # LLM summarizer specs, fallback, wiring
│   ├── test_importance.py        # heuristic importance tiers + wiring
│   ├── test_ingest.py            # chunk_text + ingest/collections commands
│   ├── test_async_store.py       # AsyncMemoryStore coroutine wrapper
│   ├── test_http_server.py       # HTTP/REST API round-trips
│   ├── test_stats_health.py      # houkai stats + --health report
│   ├── test_recall_filters.py    # source/since/until recall filters
│   ├── test_timeparse.py         # parse_timestamp (epoch/ISO/relative spans)
│   └── test_tui.py               # TUI view models + Textual pilot runs
├── pyproject.toml
└── requirements.txt
```

## Install

Modern Linux distros protect the system Python (PEP 668).  Pick whichever
approach fits your workflow — none requires `--break-system-packages`.

> **Pick an embedder.** The core install is deliberately light: it carries
> ChromaDB and the MCP SDK, but **not** the local sentence-transformers model,
> because that pulls torch (~2 GB installed). Add `[local]` for the batteries-
> included local model, or point the store at a hosted embedder — see
> [Embedding backends](#embedding-backends).
>
> ```bash
> pip install "ai-houkai[local]"    # local model, no API key, fully offline
> pip install ai-houkai             # bring your own embedder (see below)
> ```

### Virtual environment (recommended for development)

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install "ai-houkai[local]"
```

### pipx (recommended for the MCP server / CLI tool)

`pipx` installs CLI tools into isolated venvs and puts the script on your
PATH automatically — no activation step needed.

```bash
sudo apt install pipx            # or: pip install --user pipx
pipx ensurepath                  # adds ~/.local/bin to PATH (one-time)

pipx install "ai-houkai[local]"
ai-houkai-mcp                    # available everywhere
```

### uv (fastest, modern)

```bash
curl -Lsf https://astral.sh/uv/install.sh | sh   # one-time install

uv venv && uv pip install "ai-houkai[local]"      # project venv
# or run a script without a persistent install:
uv run --with ai-houkai python examples/pip_package_example.py
```

### Extras

```bash
pip install "ai-houkai[local]"    # + local sentence-transformers embeddings
pip install "ai-houkai[claude]"   # + Anthropic SDK (reflection summarizer)
pip install "ai-houkai[openai]"   # + OpenAI SDK (reflection summarizer)
pip install "ai-houkai[cli]"      # + houkai terminal CLI (typer + rich)
pip install "ai-houkai[tui]"      # + houkai tui memory browser (textual)
pip install "ai-houkai[all]"      # local model + all providers + CLI + TUI
pip install "ai-houkai[dev]"      # everything above + pytest
pip install "ai-houkai[test]"     # pytest + CLI, no torch — the fast suite
```

> With `[local]`, the embedding model (`all-MiniLM-L6-v2`) downloads
> automatically on first use (~90 MB) and everything runs offline — no API key
> required for the memory layer itself. Note that `[openai]` is only needed for
> the OpenAI *summarizer*; the OpenAI-compatible **embedder** in
> `ai_houkai.embed` is stdlib-only and needs no SDK.

### Embedding backends

The store resolves its embedder in this order — most explicit first:

```python
from ai_houkai.embed import OpenAICompatibleEmbedder
from ai_houkai.memory_system import MemoryStore

# 1. an injected callable — anything fn(list[str]) -> list[list[float]]
store = MemoryStore(embedding_function=OpenAICompatibleEmbedder(
    api_key=os.environ["OPENAI_API_KEY"], model="text-embedding-3-small"))
```

```bash
# 2. AI_HOUKAI_EMBEDDER=provider:model — moves an existing deployment onto a
#    hosted embedder without touching any call site.
export AI_HOUKAI_EMBEDDER=ollama:nomic-embed-text
export AI_HOUKAI_EMBEDDER=openai:text-embedding-3-small
export AI_HOUKAI_EMBEDDER=digitalocean:gte-large-en-v1.5
export AI_HOUKAI_EMBEDDER=local:all-MiniLM-L6-v2     # 3. the default
```

Credentials come from the environment (`AI_HOUKAI_EMBED_API_KEY`, falling back
to `OPENAI_API_KEY` / `DIGITALOCEAN_API_KEY`), never from the spec string, so a
spec is safe to put in a config file or a process listing.
`AI_HOUKAI_EMBED_BASE_URL` overrides the endpoint — that is how you point at
vLLM, llama.cpp's OpenAI-compat server, or any other compatible service.

> **Changing the embedding model changes the vector space.** Vectors written by
> one model are meaningless to another. `houkai doctor` catches the obvious
> case (a different dimension) via its embed-dim guardrail, but two models of
> the same width will silently produce nonsense rankings. Re-embed with
> `houkai export` followed by `houkai import --regenerate-vectors` when
> switching. `houkai doctor` reports the backend actually in use, not just the
> configured local model name.

## Quick-start

```python
from ai_houkai.memory_system import MemoryStore

store = MemoryStore()                  # persists to ./.chroma

mem = store.remember("Python's GIL blocks CPU parallelism",
                     type="semantic", importance=0.85, tags=["python"])

# Fix or refine in place — same id, links, and history; re-embeds on text
# change; journaled and undoable (unlike forget()+remember()):
store.edit(mem.id, text="CPython's GIL blocks CPU-bound thread parallelism",
           importance=0.9)

for mem, score in store.recall("parallel execution", k=3):
    print(f"{score:.3f}  {mem.text}")

# Or assemble a token-budgeted block ready to drop into a prompt:
pack = store.recall_pack("parallel execution", token_budget=600)
print(pack.text)          # "## Relevant memory\n- (semantic) Python's GIL …"
print(pack.used_tokens, "/", pack.budget, "truncated:", pack.truncated)

# Multi-angle counterpart to recall_pack — fans out over the task plus
# extracted key phrases, dedupes by id, then packs (no fusion="rrf" here):
pack = store.auto_context_pack("deploy the API to production", token_budget=800)
```

### Tuning retrieval

```python
# Scale-free Reciprocal Rank Fusion instead of the weighted blend
hits = store.recall("deploy", k=5, mode="hybrid", fusion="rrf")
# Maximal Marginal Relevance re-rank + near-duplicate drop
hits = store.recall("deploy", k=5, diversity=0.7, dedup_threshold=0.92)
# Absolute relevance gate — return nothing rather than weak hits
hits = store.recall("off-topic", k=5, min_cosine=0.2)
# Read-only recall (no access-count bump) + score breakdowns
for mem, score, why in store.recall("deploy", k=3, touch=False, explain=True):
    print(score, why)   # why: {'mode':'hybrid','fusion':'weighted','cosine':…,'weights':{…}}

# Second-stage rerank: a stronger model rescores the pool before the top-k cut.
# A reranker takes (query, [Memory]) and returns one score per memory.
def rerank(query, mems):
    return [cross_encoder.predict((query, m.text)) for m in mems]
hits = store.recall("deploy", k=5, overfetch=10, reranker=rerank)
# …or set it once on the store: MemoryStore(..., reranker=rerank)
```

`diversity`/`dedup_threshold` must be in `[0, 1]` and `min_cosine` in `[-1, 1]`
(else `ValueError`). `recall_pack(...)` accepts the same
`fusion`/`diversity`/`dedup_threshold`/`min_cosine`/`touch` and forwards them to
`recall`.

### Expiry, history & metrics

```python
# TTL: a memory that expires (and disappears from recall) after a window
store.remember("staging deploy token abc123", ttl_seconds=3600)   # or expires_at=<epoch>
store.recall("token")                       # expired memories are hidden…
store.recall("token", include_expired=True) # …unless you ask for them
store.purge_expired()                       # hard-delete expired rows (reclaim storage)

# Point-in-time: the audit journal doubles as a time machine
store.history(mem.id)          # every event that touched this memory, oldest→newest
store.state_at(some_epoch)     # the whole store reconstructed as of a past instant
store.get_at(mem.id, epoch)    # one memory as it was then (or None)

# Runtime metrics: op counters + recall latency since the store was created
store.metrics()   # {'uptime_seconds':…, 'count':…, 'calls':{…}, 'recall_latency_ms':{…}}
```

### Async usage

`AsyncMemoryStore` is a coroutine wrapper around `MemoryStore` for use inside an
event loop. Every method offloads the blocking ChromaDB call to a dedicated
**single-threaded** executor, so the event loop stays unblocked and concurrent
callers are serialised onto one thread (ChromaDB's SQLite backend is not safe
under concurrent writes).

```python
from ai_houkai.memory_system import AsyncMemoryStore

async with AsyncMemoryStore(path=".chroma") as store:
    mem  = await store.remember("Python favours duck typing.", type="semantic")
    hits = await store.recall("typing", k=3)
    pack = await store.recall_pack("typing", token_budget=600)
```

The constructor takes the same arguments as `MemoryStore`. Use `await store.aclose()`
(or the `async with` block) to flush and shut the executor down; `store.close()`
is the synchronous equivalent for non-async teardown. The underlying sync store
is reachable as `store.sync`, and any not-yet-wrapped method can be offloaded via
`await store.run(store.sync.<method>, ...)`. `recall`/`recall_pack`/
`auto_context_pack` accept the same tuning params as the sync store (`fusion`,
`diversity`, `dedup_threshold`, `min_cosine`, `touch`, `explain`; the pack
methods also `compress`/`compress_threshold`/`compress_min_group`) and forward
them through.

> **Note:** the constructor itself runs synchronously — building the store loads
> the embedding model, so construct it before entering your hot path (or wrap the
> construction in `asyncio.to_thread`) rather than on a latency-sensitive request.

---

## CLI — `houkai`

A full-featured terminal interface for managing memories directly.
Requires the `cli` extra:

```bash
pip install "ai-houkai[cli]"
houkai --help
```

All commands accept `--store PATH` / `--collection NAME` (or env vars
`AI_HOUKAI_PATH` / `AI_HOUKAI_COLLECTION`) to target any store.
IDs can be abbreviated to any unique 8-character prefix.

### Core commands

```bash
# Print the installed version
houkai --version          # or -V

# Store a memory
houkai remember "Python's GIL blocks CPU parallelism" \
  --type semantic --tag python --importance 0.85

# Read from stdin (pipe-friendly)
echo "Deploy with: make release" | houkai remember --stdin --type procedural

# Time-to-live: expires (and drops out of recall) after N seconds
houkai remember "staging token abc123" --ttl 3600

# Score importance heuristically from the text (instructions/corrections ≈ 0.9,
# decisions ≈ 0.75, completions ≈ 0.6, hedged observations ≈ 0.35)
houkai remember "Never push directly to main" --auto-importance
# …or make it the default: default_importance = "auto" in config.toml

# Bulk-ingest files (or stdin): blank-line chunking, markdown headings kept
# with their paragraph, long paragraphs re-packed on sentence boundaries
houkai ingest notes.md meeting.txt --type semantic --tag project --yes
houkai ingest notes.md --dry-run                 # preview chunks
git log --oneline -20 | houkai ingest --tag git --auto-importance --yes

# Semantic search
houkai recall "parallel execution" -k 5
houkai recall "deploy" --mode hybrid --format json
houkai recall "token" --include-expired          # also return TTL-expired hits

# Filter by provenance and creation time (ISO date, epoch, or a relative span)
houkai recall "auth" --source git --since 7d           # last week, from git ingests
houkai recall "decision" --since 2026-01-01 --until 2026-03-31
houkai pack  "deploy" --source runbook --since 30d --budget 500

# Pack the most relevant memories into a token-budgeted context block.
# The block prints to stdout (pipe it into a prompt); a summary goes to stderr.
houkai pack "how do we deploy and test" --budget 800
houkai pack "deploy" --budget 500 > context.md      # clean block, no summary
houkai pack "deploy" --format json                  # block + per-item scores/tokens

# Multi-angle packing: fan recall out over the task plus extracted key
# phrases, dedupe, and pack (wraps auto_context_pack; same output shape)
houkai auto-context "deploy the api to production" --budget 800
houkai auto-context "deploy the api" --max-phrases 3 --min-cosine 0.3 -f json

# List recent memories
houkai list
houkai list --type episodic --tag python --since 7d --sort importance

# Inspect one memory (full metadata + link chain)
houkai show 72be7903

# Delete memories (confirms unless --yes)
houkai forget 72be7903
houkai forget id1 id2 id3 --yes

# Delete ALL memories in the current collection (irreversible)
houkai nuke                  # shows count, confirms before deleting
houkai nuke --yes            # skip confirmation
```

### Curation

```bash
# Edit text or metadata in $EDITOR (re-embeds only if text changes)
houkai edit 72be7903

# Add / remove tags without re-embedding
houkai tag 72be7903 --add hardware --add risc-v --remove old

# Adjust importance
houkai bump 72be7903 +0.2      # relative delta
houkai bump 72be7903 =0.9      # absolute value
```

All three go through the store's `edit()` API: the change keeps the memory's
id, links, and access history, lands in the audit journal (`houkai journal
tail --op edit`), and can be reversed with `houkai journal undo <ts>`.

```bash
# Fold one memory into another. Incoming links are re-pointed at the target,
# not dropped — `forget` does not clean those up, so a naive merge would
# silently strand every relationship pointing at the absorbed memory.
houkai merge 72be7903 a1b2c3d4

# Past text states, recovered from the journal (oldest first)
houkai versions 72be7903

# Tag curation across the whole collection
houkai tags list
houkai tags rename ops operations
houkai tags merge infra platform --into infrastructure
houkai tags delete stale

# Shortest undirected link path between two memories
houkai path 72be7903 a1b2c3d4
```

#### Trash — the recoverable delete

`supersede` is soft but means "replaced by X"; `forget` is irreversible. Trash
is the missing middle, and decay pruning routes here instead of hard-deleting,
so a mis-tuned `min_score` is no longer unrecoverable.

```bash
houkai trash put 72be7903      # soft-delete, recoverable
houkai trash list
houkai trash restore 72be7903
houkai trash purge --yes       # irreversible, empties this collection's trash

# Retention: drop entries trashed more than N days ago. Without this a
# "recoverable delete" is really a permanent archive and the file grows
# forever. The maintenance scheduler applies it on the same tick as the TTL
# purge (`trash_ttl_days` in config, default 30; 0 keeps everything).
houkai trash purge --older-than 30 --yes
```

#### Write-time flags

```bash
# Pinned: leads recall_pack/auto_context inside the budget, is exempt from
# decay pruning, and follows a supersede onto the replacement. A
# standing-instruction slot, so `importance` stops doing triple duty
# (ranking + decay + filtering).
houkai remember "ALWAYS run make lint before pushing" --pin --type procedural

# Trust: provenance tier for anything that did not come from the user.
houkai ingest notes.md --trust reported
houkai recall "deploy" --min-trust trusted

# Idempotent: dedupe by normalised-text content hash instead of writing a
# near-duplicate every session.
houkai remember "Use ruff for linting" --idempotent
```

### Memory graph

```bash
# Link two memories — rel must be one of:
# related | refines | derived_from | example_of | contradicts | supersedes
# (typos are rejected, as are links to ids that don't exist)
houkai link src-id dst-id --rel refines

# Remove a link
houkai unlink src-id dst-id --rel refines

# Show neighbors (BFS, configurable depth and direction)
houkai neighbors 72be7903 --direction out --depth 2

# Render a subgraph
houkai graph 72be7903 --depth 2 --format ascii
houkai graph 72be7903 --format dot | dot -Tsvg -o graph.svg
houkai graph 72be7903 --format json
```

### Conflicts & supersedes

```bash
# Full pairwise conflict scan (duplicates + contradictions)
houkai conflicts

# Check one memory
houkai conflicts --id 72be7903 --threshold 0.85

# Interactive resolution
houkai conflicts --resolve interactive

# Mark old memory as superseded by new
houkai supersede old-id new-id

# Undo a supersede
houkai restore old-id
```

### Maintenance

```bash
# Preview what the decay engine would prune (dry-run by default)
houkai prune
houkai prune --decay-rate 0.05 --min-score 0.1

# Reinforcement: let frequently-recalled memories resist decay (0 = off)
houkai prune --frequency-weight 0.2

# Actually delete (requires --apply)
houkai prune --apply --yes

# Purge TTL-expired memories (dry-run by default; ignores protect-types)
houkai purge
houkai purge --apply --yes

# Preview reflection clusters
houkai reflect
houkai reflect --threshold 0.8 --min-cluster-size 3

# Write reflection summaries (requires --apply)
houkai reflect --apply --consolidate soft   # supersede source episodics
houkai reflect --apply --consolidate hard   # hard-delete source episodics

# Summarise clusters with an LLM instead of the extractive default
houkai reflect --summarizer ollama:llama3.1 --apply
```

### Import / export / backup

Export uses the portable **`.ahkai`** format — gzipped JSONL with a
header line on line 1 (format/version/source/options) followed by one
memory per line. Embeddings are included by default so an import can
reuse them without re-running the model.

```bash
# Export everything (memories + vectors) to a portable archive
houkai export dump.ahkai

# Filter & shrink — omit embeddings for a smaller file
houkai export dump.ahkai --type episodic --tag project --no-vectors

# Peek at an archive header without touching the store
houkai info dump.ahkai

# Import — default policy skips id collisions
houkai import dump.ahkai --yes

# Other conflict policies: overwrite | rename | error
houkai import dump.ahkai --on-conflict overwrite --yes

# Re-embed text on the way in (required if the export's model differs)
houkai import dump.ahkai --regenerate-vectors --yes

# Preview without writing anything
houkai import dump.ahkai --dry-run

# Snapshot the Chroma store directory itself
houkai backup   # → ~/.ai_houkai/backups/<ISO timestamp>/
```

### Audit journal

Every mutation (`remember`, `forget`, `edit`, `supersede`, `restore`,
`link`, `unlink`, `import`, `export`, `reflect`, `decay`, `undo`) is
appended to an append-only JSONL journal next to the store (`journal.log`,
rotated and gzipped at 64 MB, 90-day retention by default). Entries
carry the actor (`cli` / `mcp` / `http` / `reflection` / `decay` /
`import` / `lib`) plus `before`/`after` snapshots where applicable.

```bash
# Tail recent entries (newest first)
houkai journal tail
houkai journal tail -n 50 --op supersede --actor reflection
houkai journal tail --all          # include rotated archives

# Pretty-print one entry by timestamp
houkai journal show 1748284800.123

# Reverse a single operation (remember/forget/edit/supersede/restore/link/unlink)
houkai journal undo 1748284800.123 --yes

# Reverse the most recent mutation — the "undo my last change" shortcut
houkai journal undo-last --yes
houkai journal undo-last --id 72be7903 --yes    # newest entry touching one memory
```

Point-in-time queries have CLI commands too:

```bash
houkai history 72be7903            # full journaled timeline of one memory
houkai state-at 7d                 # every live memory as it stood a week ago
houkai get-at 72be7903 2026-07-01  # one memory, as of a date
houkai metrics                     # op counters + recall-latency percentiles
```

### Collections

One Chroma store can hold many collections (namespaces) — e.g. one per
agent or per project (`claude_code` / `cursor` / `opencode`).

```bash
houkai collections list                    # names, counts, active marker
houkai collections create scratch
houkai collections copy ai_houkai backup --yes   # embeddings included, no re-embed
houkai collections delete scratch --yes          # refuses the active collection
```

All other commands target a collection via `--collection NAME`.

### TUI browser

```bash
pip install "ai-houkai[tui]"
houkai tui
```

A Textual full-screen browser: memory list with detail pane, `/` for
semantic search, `n` to walk the link graph from the selected memory
(neighbors view with rel labels), `b` to go back along the breadcrumb,
`r` for recent, `X` to nuke all memories (two-press confirmation),
`q` to quit.

### Stats

```bash
houkai stats                 # rich table
houkai stats --format json   # machine-readable

# Detailed health report: decay-score histogram, at-risk / stale / never-recalled
# counts, episodic clusters ripe for reflection, link density and top-recalled.
houkai stats --health
houkai stats --health --stale-days 14 --decay-rate 0.05 --frequency-weight 0.2
houkai stats --health --format json   # health block nested under "health"
```

The health view scores each active memory with the same formula as the engine:
`importance × exp(-decay_rate × days_idle) × (1 + frequency_weight × ln(1 + access_count))`.
`--decay-rate`, `--frequency-weight`, the at-risk threshold, and the protected
types all **default to the `[maintenance.decay]` config** (fallbacks `0.1` /
`0.0` / `0.05` / `["procedural"]`), so the report matches what `houkai prune`
and the maintenance daemon would actually remove. **At-risk** counts active,
non-protected memories whose score has fallen below the prune threshold — i.e.
those `prune` would remove; protected types (default `procedural`) are never
counted as at-risk, matching `DecayEngine`'s `protect_types`. The new
`--frequency-weight` flag, plus `--decay-rate` and `--stale-days`, only affect
this report; they do not mutate the store.

### Output formats

Every listing command accepts `--format auto|rich|tsv|json`.
- **auto** — rich table on a TTY, TSV otherwise (pipe-safe).
- **json** — structured JSON array; pair with `jq` for scripting.
- `NO_COLOR=1` disables colour in rich output.

### Config file

`~/.config/ai_houkai/config.toml` (all optional):

```toml
store_path          = "~/.ai_houkai/.chroma"
collection          = "ai_houkai"
default_type        = "semantic"
default_importance  = 0.5        # or "auto" — heuristic scoring per memory
editor              = "nvim"
```

---

## HTTP / REST API

For web apps, shell scripts, n8n flows or any non-MCP agent, `houkai serve`
exposes the same store over a small JSON HTTP API. It is **standard-library
only** — no extra dependency beyond the core memory layer.

```bash
houkai serve --port 8077                 # uses the --store / --collection in effect
# or, dependency-free console script driven by env vars:
AI_HOUKAI_HTTP_PORT=8077 ai-houkai-serve
```

Endpoints (all JSON in / JSON out):

| Method & path | Purpose |
|---|---|
| `GET /health` | liveness + memory count (always open) |
| `GET /ready` | readiness — backend + embedder probe; `200`/`503`; auth-exempt, minimal body, ~5 s cached |
| `GET /stats` | store statistics |
| `GET /metrics` | runtime counters + recall latency |
| `GET /memories?limit=&include_superseded=&include_expired=` | recent memories |
| `POST /memories` | store a memory (`remember`; accepts `ttl_seconds`/`expires_at`) — `201`/`stored:true` when a row is created, `200`/`stored:false` when `idempotent` matched an existing one, `409` only for a real conflict rejection |
| `POST /memories/batch` | bulk store (`remember_many`; `{items:[…], batch_size?, on_conflict?}`) with batched embedding |
| `GET /memories/{id}` | fetch one |
| `PATCH /memories/{id}` | edit fields in place (journaled, undoable; `expires_at`) |
| `DELETE /memories/{id}` | forget one |
| `GET /memories/{id}/neighbors?rel=&direction=&depth=` | linked memories |
| `GET /memories/{id}/history` | journaled timeline of one memory |
| `GET /memories/{id}/at?ts=` | reconstruct one memory as of a past time |
| `GET /state_at?ts=` | reconstruct all live memories as of a past time |
| `POST /purge_expired` | hard-delete TTL-expired memories (`{dry_run?}`) |
| `GET\|POST /recall` | search — supports `source`, `since`, `until`, `include_expired`, `explain`, `overfetch`, `as_of`, `min_trust`, `lexical_index`, `touch`; the POST body additionally takes `fusion`, `diversity`, `dedup_threshold`, `min_cosine`, `graph` and a nested `expand` object |
| `POST /recall_pack` | token-budgeted context block |
| `POST /auto_context` | multi-angle context block (`auto_context_pack`) |
| `POST /links` · `POST /unlink` | manage the link graph |
| `POST /subgraph` | link graph reachable from seed ids (`{memory_ids, depth?}`) |
| `POST /find_path` | shortest undirected link path between two memories |
| `POST /supersede` · `POST /restore` · `POST /conflicts` | curation |
| `POST /merge` | fold one memory into another, re-pointing incoming links |
| `GET /memories/{id}/versions` | past text states, recovered from the journal |
| `GET /tags` · `POST /tags/rename` · `POST /tags/merge` · `DELETE /tags/{tag}` | tag curation |
| `GET /trash` · `POST /trash` · `POST /trash/restore` · `POST /trash/purge` | recoverable delete |
| `GET /journal?n=&op=&since=` | audit-journal tail |
| `POST /undo` | reverse a journaled mutation (`{ts?, memory_id?}`) |
| `POST /nuke` | delete every memory — guarded by `{"confirm": "DELETE ALL"}` |
| `POST /export` · `POST /import` | `.ahkai` archives (server-side paths). Because they reach past the store onto the filesystem, a tokenless server answers them for **loopback callers only** — configure a token to use them remotely |

```bash
curl -s localhost:8077/health
curl -s localhost:8077/metrics
curl -s 'localhost:8077/recall?query=auth&k=3&since=7d&source=git&explain=true'
curl -s localhost:8077/memories -d '{"text":"session token","type":"semantic","ttl_seconds":3600}'
curl -s localhost:8077/recall_pack -d '{"query":"deploy","token_budget":500}'
curl -s "localhost:8077/state_at?ts=7d"          # store as it was a week ago
curl -s -X POST localhost:8077/purge_expired -d '{"dry_run":true}'
curl -s -X PATCH localhost:8077/memories/<id> -d '{"importance":0.9}'
curl -s localhost:8077/tags                       # tag usage counts
curl -s localhost:8077/trash                      # recoverable deletes
curl -s -X POST localhost:8077/undo -d '{}'       # reverse the last mutation
```

**Auth:** pass `--token <secret>` (or set `AI_HOUKAI_HTTP_TOKEN`) and every
request must carry `Authorization: Bearer <secret>`; `/health` and `/ready`
stay open for liveness/readiness probes. The server binds `127.0.0.1` by default — set `--host 0.0.0.0`
(or `AI_HOUKAI_HTTP_HOST`) only behind a trusted network or reverse proxy.

**Concurrency:** the server is multi-threaded but serialises all store access
through a single lock, because `MemoryStore` writes (links, supersede, the
access-count bump on every recall) are read-modify-write and would otherwise
race under concurrent requests. ChromaDB already serialises internally, so the
lock costs little; for true parallelism run multiple processes against separate
stores. (`AsyncMemoryStore` gives the same guarantee in-process via a
single-threaded executor.)

**Validation:** malformed query-string *and* JSON-body parameters (a string
`k`, a garbage `threshold`, `"include_superseded": "false"`) are coerced or
rejected with `400` — never a 500. Unknown enum values (`mode`, `type`,
`rel`, `on_conflict`, `direction`, `polarity`) also return `400` with a
message naming the allowed vocabulary; unknown ids return `404`. `HEAD` is
supported on every GET route (probes get `200`, body suppressed).

**Hardening:** `GET /health` returns only `{"status":"ok","count":N}` and
deliberately omits the collection name / topology. Bearer-token checks use a
constant-time comparison (`hmac.compare_digest` over UTF-8 bytes, so a
non-ASCII header gets a clean `401`). Unhandled errors return
`500 {"error":"internal server error","request_id":"<hex>"}` — no exception
type, message, or traceback is leaked to the client.

**Ranking/compression knobs over HTTP:** `/recall` accepts `overfetch`;
`/recall_pack` additionally accepts `fusion` (`weighted`|`rrf`), `diversity`,
`dedup_threshold`, `min_cosine`, and the compression trio
`compress`/`compress_threshold`/`compress_min_group` (responses include a
`compressed_groups` list). `/auto_context` takes `task` (required) plus
`token_budget`, `max_phrases`, `mode`, `min_cosine`, `header`, and the same
compression trio; its response adds the fan-out `queries`. String `tags` in
`POST /memories` / `PATCH /memories/{id}` are coerced to a one-element list
(anything other than a string or list of strings is a `400`). `touch=false`
(GET or POST) makes recall read-only — access counters aren't bumped, so
eval/monitoring traffic doesn't feed decay reinforcement. Every serialised
memory carries `pinned` and `trust` explicitly — including their defaults
(`false` / `"trusted"`), because an absent key would be indistinguishable
from "not pinned" / "unlabelled".

---

## Go port

A full Go port lives in [`go/`](go/) — two static binaries (`houkai` CLI +
`ai-houkai-mcp` MCP server), no Python runtime, packaged as a Debian `.deb`
and macOS tarballs.

**The two ports are at parity on the remote surface, and it is enforced.**
[`parity.json`](parity.json) is the single source of truth for the MCP tool list
(41 tools), the HTTP routes (41) and the recall knobs; each port asserts against
it in its own test suite (`tests/test_parity.py`,
`go/internal/parity/`), so a tool added to only one side fails that side's
build rather than going unnoticed. That guard exists because this drift really
happened: the Python MCP `recall` had silently fallen five ranking knobs behind
the Go one.

Also ported: hybrid retrieval with the full ranking suite (RRF fusion, MMR
diversity/dedup, `min_cosine`, read-only/`explain` recall, created-based
recency, multi-hop link decay, CJK tokenization), graph-proximity fusion and
gated expansion, context packing with compression, reranking, TTL/expiry,
point-in-time history, runtime metrics, the curation set (merge/versions/tags/
find_path), trash, pinned/trust/idempotent writes, tiered reflection, the
retrieval-eval harness (`go/internal/eval`, metric-for-metric with the Python
one and reading the same gold-set format), decay/reflection with LLM
summarizers, bulk ingest, collections, importance auto-assignment, installers
for Claude Code / Cursor / OpenCode, a Bubble Tea TUI, and diagnostics
(`houkai doctor` + `GET /ready`).

Deliberately **not** at parity: the CLIs differ (Python ships installers as
console scripts, Go has `houkai install` and `houkai config`), so `parity.json`
does not assert command equality — that would be a lie.

Embeddings are delegated to Ollama (default), OpenAI, or DigitalOcean instead of
bundled sentence-transformers — the same pluggable model Python now has via
`ai_houkai.embed`. The store is
[chromem-go](https://github.com/philippgille/chromem-go) — not binary-compatible
with ChromaDB, but the portable `.ahkai` export/import format bridges the two.
See [go/README.md](go/README.md).

## Design docs

In-depth design notes live in [docs/DESIGN.md](https://raw.githubusercontent.com/nexusriot/AI-Houkai/main/docs/DESIGN.md).
The original feature proposals for hybrid retrieval, conflict detection, and
memory linking — all now shipped — are archived in
[PROPOSALS.md](https://raw.githubusercontent.com/nexusriot/AI-Houkai/main/PROPOSALS.md).
The Go port has its own internals doc in [go/DESIGN.md](go/DESIGN.md); the
original porting feasibility study is archived in
[GO_PORT_DESIGN.md](GO_PORT_DESIGN.md).
Forward-looking feature recommendations live in
[docs/ROADMAP.md](https://raw.githubusercontent.com/nexusriot/AI-Houkai/main/docs/ROADMAP.md).

## Run the tests

```bash
pytest tests/ -q                       # 1094 tests across 44 files
pytest tests/ -q -m "not needs_model"  # the fast subset — no torch, no download
pytest functional_tests -v             # 16 black-box e2e tests (needs an install)
```

Most of the suite runs against `ai_houkai.testing.FakeEmbedder`, a hash-based
embedder that needs no model. Only the ~29 tests marked `needs_model` load real
sentence-transformers — their assertions depend on genuine semantic similarity
(conflict thresholds, reflection clustering, ranking quality). To re-derive that
marker set after a change, force the real model everywhere:

```bash
AI_HOUKAI_TEST_REAL_EMBEDDER=1 pytest tests/ -q
```

The functional suite drives the installed `houkai` / `ai-houkai-serve` console
scripts as subprocesses. Run it hermetically in a container — which is also the
only thing that exercises a genuinely clean install:

```bash
./functional_tests/run.sh              # build the image, run unit + functional
```

Go: `cd go && go test ./...` (add `-race` for the concurrent packages).
CI runs both ports plus a port-parity job — see
[.github/workflows/ci.yml](.github/workflows/ci.yml).

---

## Examples

### 01 · Standalone (no LLM)

Full memory lifecycle — seed → recall with filters → access tracking → forget.

```bash
python examples/01_standalone.py
```

### 02 · Ollama (local network)

Conversational REPL using a local model over Ollama's OpenAI-compatible
endpoint.  No API key, no internet.

```bash
ollama pull llama3.1
OLLAMA_MODEL=llama3.1 python examples/02_ollama_local_network.py
```

| Env var | Default |
|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` |
| `OLLAMA_MODEL` | `llama3.1` |
| `AI_HOUKAI_PATH` | `./.chroma` |

### 03 · Claude Desktop (MCP)

Auto-installs the MCP server into Claude Desktop's config.

```bash
python examples/03_claude_desktop.py            # preview config
python examples/03_claude_desktop.py --install  # write config
python examples/03_claude_desktop.py --demo     # simulated session
```

### 04 · OpenAI

GPT-4o / gpt-4o-mini with function calling.

```bash
export OPENAI_API_KEY=sk-...
python examples/04_openai.py
OPENAI_MODEL=gpt-4o AI_HOUKAI_PATH=~/.ai_houkai python examples/04_openai.py
```

| Env var | Default |
|---|---|
| `OPENAI_MODEL` | `gpt-4o-mini` |
| `AI_HOUKAI_PATH` | temp dir |

### 05 · Decay + Reflection

Shows both cognitive maintenance features with backdated timestamps.

```bash
python examples/05_decay_reflection.py
```

### 06 · Claude Code (MCP)

Gives the Claude Code CLI a persistent memory so it remembers project
conventions, past debug sessions, and your preferences across every
coding session.

```bash
# Option A — one-liner (recommended)
claude mcp add ai-houkai -- ai-houkai-mcp

# Option B — installer console script (after `pip install ai-houkai`)
ai-houkai-install-claude-code --install
ai-houkai-install-claude-code --verify
ai-houkai-install-claude-code --claudemd

# Option C — auto-register via the example script
python examples/06_claude_code.py --install

# Option D — preview the config block
python examples/06_claude_code.py

# Smoke-test
python examples/06_claude_code.py --verify

# Simulated coding session
python examples/06_claude_code.py --demo

# Print a CLAUDE.md snippet that teaches Claude how to use memory
python examples/06_claude_code.py --claudemd
```

The installer registers through `claude mcp add --scope user` (Claude Code
reads MCP servers from `~/.claude.json` / a project's `.mcp.json`, **not**
`settings.json`); when the `claude` CLI is not on PATH it merges this block
into `~/.claude.json` directly:

```json
{
  "mcpServers": {
    "ai-houkai": {
      "type": "stdio",
      "command": "ai-houkai-mcp",
      "args": [],
      "env": {
        "AI_HOUKAI_PATH": "~/.ai_houkai/.chroma",
        "AI_HOUKAI_COLLECTION": "claude_code"
      }
    }
  }
}
```

The install logic lives in the reusable `ai_houkai.installers.claude_code`
module — drop it into your own scripts:

```python
from ai_houkai.installers import ClaudeCodeInstaller

ClaudeCodeInstaller(memory_path="~/.ai_houkai/.chroma").install()
```

#### Recommended CLAUDE.md addition

Add the following to your project's `CLAUDE.md` so Claude Code knows when and
how to use memory tools (run `ai-houkai-install-claude-code --claudemd`
or `python examples/06_claude_code.py --claudemd` to generate it):

```markdown
## Memory (AI-Houkai MCP)

- **remember** — store conventions, decisions, preferences
- **recall** — search before starting any task
- **edit** — update a memory in place (keeps id, links, history)
- **forget** — remove outdated facts

| Situation | Action |
|---|---|
| User states a convention | `remember` with `type="procedural"` |
| A stored fact is outdated or has a typo | `edit` it in place |
| User corrects you | `remember` correction, `forget` old fact |
| Starting a new task | `recall` relevant context first |
```

### Claude agent (Anthropic SDK REPL)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
AI_HOUKAI_PATH=/tmp/my_memory python examples/claude_agent.py
```

REPL commands: `memories` to list recent memories · `quit` to exit.

---

## Retrieval evaluation

`ai_houkai.eval` is a dependency-free (stdlib-only) harness for scoring
retrieval quality against a gold set. It is reachable from the library, the
CLI, MCP **and the Go port** — which matters because every ranking constant in
the project (the graph damping and iteration count, the β=0.20 lexical weight,
the MMR defaults) was otherwise set by intuition with no way to measure a
change.

```bash
# Gold sets are JSONL — one case per line. `relevant_ids` accepts 8-char
# prefixes as well as full UUIDs, so you can write one by eye from `houkai list`.
cat > gold.jsonl <<'EOF'
{"query": "how do we deploy", "relevant_ids": ["72be7903"]}
{"query": "test isolation",   "relevant_ids": ["a1b2c3d4", "e5f6a7b8"], "k": 3}
EOF

houkai eval gold.jsonl
houkai eval gold.jsonl --mode hybrid --fusion rrf --graph 0.15 --per-case
houkai eval gold.jsonl --json          # for CI / A/B scripting
```

The `--json` output records the config under test alongside the scores, so
numbers from two runs are attributable. The same gold-set format works with the
Go port's `houkai eval` and with the read-only `eval_recall` MCP tool.

```python
from ai_houkai.eval import EvalCase, evaluate

cases = [
    EvalCase(query="how do we deploy", relevant_ids=[deploy_mem.id]),
    EvalCase(query="test isolation",   relevant_ids=[a.id, b.id], k=3),
]
result = evaluate(store, cases, default_mode="hybrid")
print(result.summary())          # n=… recall@…=… P@…=… MRR=… MAP=… nDCG@…=…
print(result.recall_at_k, result.mrr, result.ndcg_at_k)
```

Metrics (binary relevance; a duplicated retrieved id is credited once):
`recall_at_k` · `precision_at_k` · `reciprocal_rank` · `average_precision` ·
`dcg_at_k` · `ndcg_at_k`. The metric functions also work standalone on any
ranked list of ids. `evaluate()` calls `recall(..., touch=False)`, so scoring
never perturbs access tracking. Extra keyword args (e.g. `weights=`, `fusion=`,
`diversity=`) are forwarded to `recall` so you can A/B ranking configs.

---

## Full-corpus lexical recall

The vector over-fetch pool is chosen by embedding distance alone, so a memory
carrying the query's exact tokens but embedding weakly never enters it — and can
therefore never be surfaced by the BM25 term, at any corpus size.

```python
store.recall("quetzalcoatlus", mode="hybrid", lexical_index="corpus")
```

`lexical_index="corpus"` also pulls in candidates whose *text* contains the
query's tokens, using Chroma's own `where_document` filter, and lets the existing
pool-relative BM25 score them. Unioned candidates keep their real cosine
distance (computed against the query vector) rather than a fabricated one, and
they obey the same `type` / `source` / date filters as a vector hit.

It costs one server-side scan per token — measured ~4.5 ms at 25k memories,
growing linearly — so it is off by default (`"pool"`). Probes are capped at the
four longest tokens, those being the most selective.

> **Why not a real inverted index?** A SQLite FTS5 index beside `.chroma` was
> measured and rejected. Its lookup was genuinely faster and flat (0.03 ms
> regardless of corpus size), but it would be a second source of truth for data
> Chroma already holds — needing verify-on-open, disable-on-mismatch and rebuild
> machinery to stay safe — and it made `list_recent` *slower*, because serving
> ids from the index and then fetching them from Chroma is two round-trips
> instead of one. On a path already dominated by an embedding call, that does not
> pay. If you need a true inverted index at scale, it belongs somewhere it can be
> shared across collections — see [docs/ROADMAP.md](docs/ROADMAP.md).


---

## MCP server

Exposes the memory store to any MCP client.

```bash
ai-houkai-mcp
# or: python -m ai_houkai.mcp_server.server
```

Exposed tools (41):

- **Core** — `remember` · `remember_many` · `get` · `edit` · `recall` · `recall_pack` · `auto_context` · `forget` · `purge_expired` · `list_recent` · `stats` · `metrics`
- **Linking & graph** — `link` · `unlink` · `neighbors` · `subgraph` · `find_path`
- **Conflicts & lifecycle** — `find_conflicts` · `supersede` · `restore` · `merge` · `versions`
- **Tags** — `list_tags` · `rename_tag` · `merge_tags` · `delete_tag`
- **Trash** — `trash` · `trash_list` · `trash_restore` · `trash_purge`
- **History** — `history` · `state_at` · `get_at`
- **Maintenance & audit** — `maintenance_tick` · `journal_tail` · `undo` · `nuke` · `export` · `import`
- **Diagnostics** — `ready` · `eval_recall`

`recall` and `recall_pack` expose the full ranking surface — `fusion`,
`diversity`, `dedup_threshold`, `min_cosine`, `graph`, `lexical_index`,
`min_trust`, `touch` and the flat `expand_*` graph-walk knobs. (These were
library- and HTTP-only until recently: an MCP client could not reach RRF, MMR,
dedup, the relevance gate or graph fusion at all, which is the drift
`parity.json` now guards against.)

`remember`/`edit` accept a TTL (`ttl_seconds`/`expires_at`), and `remember`
also takes `pinned`, `trust` and `idempotent`.
`history`/`state_at`/`get_at` replay the audit journal for point-in-time
queries; `metrics` reports op counters + recall latency percentiles;
`eval_recall` scores a gold set read-only; `nuke` is guarded behind
`confirm="DELETE ALL"`.

`auto_context` fans out recall over several angles extracted from a task
description, dedupes by id, and packs the result within a token budget.

`edit` updates a memory in place — same id, links, and access history;
the text is re-embedded only when it changed. The change is journaled and
reversible, so agents should `edit` to fix or refine a stored fact instead
of a `forget` + `remember` round-trip (which would discard the graph).
Omitted fields stay unchanged; pass `clear_source=true` to remove the
provenance string (a `source` of `null` means "leave unchanged").

Environment variables:

| Variable | Default |
|---|---|
| `AI_HOUKAI_PATH` | `./.chroma` |
| `AI_HOUKAI_COLLECTION` | `ai_houkai` |
| `AI_HOUKAI_AUTO_IMPORTANCE` | unset — set `1` to heuristically score `remember` calls that omit `importance` |
| `AI_HOUKAI_SUMMARIZER` | unset — e.g. `ollama:llama3.1` for `maintenance_tick` reflection |

**Claude Code** (global):

```bash
claude mcp add ai-houkai -- ai-houkai-mcp
```

**Claude Code** (manual — `~/.claude.json` user scope, or a project `.mcp.json`):

```json
{
  "mcpServers": {
    "ai-houkai": {
      "type": "stdio",
      "command": "ai-houkai-mcp",
      "args": [],
      "env": { "AI_HOUKAI_PATH": "/your/memory/path" }
    }
  }
}
```

**Claude Desktop** — use `examples/03_claude_desktop.py --install`.

**Cursor** (`~/.cursor/mcp.json`, or `--project` for `./.cursor/mcp.json`):

```bash
ai-houkai-install-cursor --install      # patch ~/.cursor/mcp.json
ai-houkai-install-cursor --verify       # smoke-test the server + registration
ai-houkai-install-cursor --rule         # print a .cursor/rules/*.mdc snippet
ai-houkai-install-cursor                # preview the JSON block (no write)
```

Cursor uses the same `mcpServers` schema as Claude. The written block:

```json
{
  "mcpServers": {
    "ai-houkai": {
      "command": "ai-houkai-mcp",
      "env": {
        "AI_HOUKAI_PATH": "~/.ai_houkai/.chroma",
        "AI_HOUKAI_COLLECTION": "cursor"
      }
    }
  }
}
```

After installing, reload Cursor and confirm under **Settings → MCP**. Drop the
`--rule` output into `.cursor/rules/ai-houkai-memory.mdc` so the agent knows
when to call the memory tools.

**OpenCode** (`~/.config/opencode/opencode.json`, or `--project` for
`./opencode.json`):

```bash
ai-houkai-install-opencode --install    # patch ~/.config/opencode/opencode.json
ai-houkai-install-opencode --verify     # smoke-test the server + registration
ai-houkai-install-opencode --agents     # print an AGENTS.md snippet
ai-houkai-install-opencode              # preview the JSON block (no write)
```

OpenCode uses its own `mcp` schema (note the `command` array and `environment`):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "ai-houkai": {
      "type": "local",
      "command": ["ai-houkai-mcp"],
      "enabled": true,
      "environment": {
        "AI_HOUKAI_PATH": "~/.ai_houkai/.chroma",
        "AI_HOUKAI_COLLECTION": "opencode"
      }
    }
  }
}
```

Restart OpenCode after installing. Append the `--agents` output to your
project (or global `~/.config/opencode/`) `AGENTS.md` to teach the agent the
memory workflow.

Both installers are reusable library classes, mirroring `ClaudeCodeInstaller`:

```python
from ai_houkai.installers import CursorInstaller, OpenCodeInstaller

CursorInstaller(memory_path="~/.ai_houkai/.chroma").install()
OpenCodeInstaller(memory_path="~/.ai_houkai/.chroma", settings_path="opencode.json").install()
```

---

## Decay

Memories fade over time based on age and importance, and — optionally — how
often they are recalled.

```
score = importance × exp(−λ × days_since_last_access) × (1 + frequency_weight × ln(1 + access_count))
```

Default `λ = 0.1` → half-life ≈ 7 days for a 0.5-importance memory.
`procedural` memories are protected and never pruned. `frequency_weight`
defaults to `0.0` (recency-only); raise it so frequently-recalled memories
resist decay.

```python
from ai_houkai.memory_system import MemoryStore, DecayEngine

store  = MemoryStore()
engine = DecayEngine(store, decay_rate=0.1, min_score=0.05,
                     frequency_weight=0.0)   # >0 → recall reinforcement

engine.prune(dry_run=True)   # preview
engine.prune()               # delete stale memories
```

## Reflection

Clusters of semantically similar episodic memories are condensed into a
single `semantic` summary (the Generative Agents pattern).

```python
from ai_houkai.memory_system import MemoryStore, ReflectionEngine

store  = MemoryStore()
engine = ReflectionEngine(store, similarity_threshold=0.75)

engine.clusters()                    # inspect clusters
engine.reflect(dry_run=True)         # preview
engine.reflect(consolidate=True)     # create summaries + delete sources
```

By default an extractive summarizer is used (no LLM — most-important
text first, concatenated). To get real condensation, plug in an LLM
summarizer via a `provider:model` spec:

```python
from ai_houkai.memory_system import MemoryStore, ReflectionEngine, build_summarizer

store  = MemoryStore()
engine = ReflectionEngine(store, summarizer=build_summarizer("ollama:llama3.1"))
```

Supported specs:

| Spec | Backend | Requires |
|---|---|---|
| `extractive` (default) | built-in, no LLM | — |
| `ollama:llama3.1` | Ollama OpenAI-compat endpoint (stdlib HTTP, no SDK) | `OLLAMA_BASE_URL` (default `http://localhost:11434`) |
| `openai:gpt-4o-mini` | OpenAI SDK | `pip install "ai-houkai[openai]"`, `OPENAI_API_KEY` |
| `anthropic:claude-haiku-4-5` | Anthropic SDK | `pip install "ai-houkai[claude]"`, `ANTHROPIC_API_KEY` |

LLM summarizers **fall back to extractive** (with a logged warning) if
the call fails or returns empty — reflection degrades rather than
crashes when running unattended.

Configure once in `~/.config/ai_houkai/config.toml` and both
`houkai reflect` and the maintenance daemon will use it
(env override: `AI_HOUKAI_SUMMARIZER`):

```toml
[maintenance.reflect]
summarizer = "ollama:llama3.1"
```

Or pass it ad hoc on the CLI:

```bash
houkai reflect --summarizer ollama:llama3.1 --apply
```

A custom callable still works for anything else:

```python
def my_summarizer(memories):
    return call_llm("Summarise:\n" + "\n".join(m.text for m in memories))

engine = ReflectionEngine(store, summarizer=my_summarizer)
```

## Scheduled Maintenance

Decay and reflection are powerful but only useful if they run regularly.
The maintenance daemon orchestrates both automatically so you never have to
think about it.

### Three usage modes

**Mode A — one-shot tick (recommended for cron)**

```bash
houkai maintenance tick
```

Run one tick synchronously: prune stale memories and (optionally) reflect.
Jobs only execute when their configured interval has elapsed since the last
run — safe to call as often as you like, and safe to call **concurrently**:
the tick cycle holds a cross-process file lock, so a cron tick racing the
daemon (or the MCP tool) serialises instead of double-running a job. This
also holds for **dry-run reflection**: a dry-run still pays for clustering
(and the LLM summarizer, if one is configured), so it advances the schedule
like a real run; only the persisted-summaries total is reserved for
`apply` runs.

Crontab example (daily at 03:00):

```cron
0 3 * * * houkai --store ~/.ai_houkai/.chroma maintenance tick
```

**Mode B — foreground loop**

```bash
houkai maintenance run
```

Runs the scheduler in the foreground.  Use this to observe what the daemon
would do before committing to `start`.  Press Ctrl-C (or send SIGTERM) to
stop cleanly.

**Mode C — background daemon**

```bash
houkai maintenance start    # detach into the background
houkai maintenance status   # show pid, last runs, next schedules
houkai maintenance stop     # send SIGTERM
```

Logs go to `~/.ai_houkai/maintenance.log` (configurable).

### Configuration

Add a `[maintenance]` section to `~/.config/ai_houkai/config.toml`:

```toml
[maintenance]
enabled       = true      # master switch (default). false → tick/run/start
                          # and the MCP maintenance_tick are no-ops
decay_every   = "24h"     # or "off" to disable decay
reflect_every = "7d"      # or "off" to disable reflection
tick_interval = "5m"      # how often the loop wakes to check schedules

[maintenance.decay]
decay_rate       = 0.1
min_score        = 0.05
protect_types    = ["procedural"]   # never pruned
frequency_weight = 0.0   # >0 → frequently-recalled memories resist decay

[maintenance.reflect]
min_cluster_size = 3
apply            = false   # set true to write reflection summaries
consolidate      = "soft"  # none | soft (default) | hard — what happens to the
                           # source episodics when a summary is written
summarizer       = "ollama:llama3.1"   # optional; default extractive (no LLM)
```

Supported duration units: `s` · `m` · `h` · `d` · `w`.
Default `decay_every = "24h"`, `reflect_every = "7d"`, `tick_interval = "5m"`.

> **enabled = false** turns every scheduled-maintenance surface into a clear
> no-op: `houkai maintenance tick|run` print that maintenance is disabled,
> `houkai maintenance start` refuses to daemonize, and the MCP
> `maintenance_tick` tool returns `{"enabled": false, ...}` without running
> anything. `houkai prune` / `houkai reflect` stay available for manual runs.

> **reflect.apply = false** (default): reflection runs in observe-only mode —
> it logs how many summaries *would* be created but writes nothing.  Flip to
> `true` once you're happy with the clusters.

> **reflect.consolidate** controls the source episodics when `apply = true`:
> `soft` (default) supersedes them under the new summary — without this a
> scheduled reflection would re-create the same summaries every interval;
> `hard` deletes them permanently; `none` leaves them untouched (expect
> duplicate summaries on repeat runs).

### Programmatic usage

```python
import threading
from ai_houkai.memory_system import MemoryStore
from ai_houkai.maintenance import MaintenanceScheduler

store = MemoryStore()
sched = MaintenanceScheduler(
    store,
    decay_every=86_400,     # 24 h
    reflect_every=604_800,  # 7 d
    tick_interval=300,      # wake every 5 min
    reflect_apply=False,    # dry-run reflect
)

# One-shot (e.g. from cron or an agent):
result = sched.tick()
print(result.summary())   # "decay pruned 3 | reflect would create 2 (dry-run)"

# Blocking loop:
stop = threading.Event()
sched.run_forever(stop)   # blocks; call stop.set() to exit
```

### MCP tool

The `maintenance_tick` tool lets any MCP client trigger a tick without
running the CLI. `reflect_apply` defaults to the config's
`[maintenance.reflect] apply` setting; the result reports which mode ran:

```
maintenance_tick()
→ {"summary": "decay pruned 2 | reflect would create 1 (dry-run)",
   "decayed": 2, "reflected": 1, "reflect_applied": false, ...}
```

### systemd unit (optional)

If you prefer systemd over the built-in daemon:

```ini
# ~/.config/systemd/user/ai-houkai-maintenance.service
[Unit]
Description=AI-Houkai maintenance scheduler

[Service]
ExecStart=houkai --store %h/.ai_houkai/.chroma maintenance run
Restart=on-failure
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
```

```bash
systemctl --user enable --now ai-houkai-maintenance
journalctl --user -u ai-houkai-maintenance -f
```
