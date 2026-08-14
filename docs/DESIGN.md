# AI-Houkai — Architecture & Design

> This document covers the **Python** implementation (`ai_houkai/`). The Go
> port under `go/` has the same remote surface (41 MCP tools, 41 HTTP routes,
> identical `.ahkai` format and journal line format) but its own internals — see
> [go/DESIGN.md](../go/DESIGN.md). Parity is not a claim, it is asserted:
> [`parity.json`](../parity.json) is the single source of truth and both ports
> check themselves against it (§23).

## Table of Contents

1. [Motivation](#1-motivation)
2. [System Overview](#2-system-overview)
3. [Data Model](#3-data-model)
4. [Storage Layer](#4-storage-layer)
5. [Memory Lifecycle](#5-memory-lifecycle)
6. [Decay Engine](#6-decay-engine)
7. [Reflection Engine](#7-reflection-engine)
8. [MCP Server](#8-mcp-server)
9. [Installers](#9-installers)
10. [Agent Integrations](#10-agent-integrations)
11. [Test Architecture](#11-test-architecture)
12. [Memory Linking](#12-memory-linking)
13. [Conflict / Contradiction Detection](#13-conflict--contradiction-detection)
14. [Hybrid Retrieval](#14-hybrid-retrieval)
15. [Extension Points](#15-extension-points)
16. [CLI — houkai](#16-cli--houkai)
17. [Scheduled Maintenance](#17-scheduled-maintenance)
18. [Audit Journal](#18-audit-journal)
19. [Portable Import / Export](#19-portable-import--export)
20. [HTTP / REST API](#20-http--rest-api)
21. [Memory Expiry (TTL)](#21-memory-expiry-ttl)
22. [Runtime Metrics](#22-runtime-metrics)
23. [Port Parity](#23-port-parity)
24. [Pluggable Embedding Backends](#24-pluggable-embedding-backends)
25. [Full-Corpus Lexical Recall](#25-full-corpus-lexical-recall)
26. [Curation & Trash](#26-curation--trash)
27. [Working Set: Pinned, Trust, Idempotent](#27-working-set-pinned-trust-idempotent)
28. [Retrieval Evaluation](#28-retrieval-evaluation)

---

## 1. Motivation

LLM context windows are finite and stateless.  Every new conversation
starts from scratch.  AI-Houkai gives an agent a **persistent,
searchable memory** that survives across sessions — without requiring
cloud services or API keys for the memory layer itself.

Four cognitive operations model how humans manage long-term memory:

| Operation | Human analogy | AI-Houkai component |
|---|---|---|
| **Store** | Encoding an experience | `MemoryStore.remember()` |
| **Retrieve** | Remembering relevant context | `MemoryStore.recall()` |
| **Forget** | Natural fading of unimportant events | `DecayEngine.prune()` |
| **Reflect** | Summarising experiences into knowledge | `ReflectionEngine.reflect()` |

---

## 2. System Overview

```
┌──────────────────────────────────────────────────────────────┐
│                         Agent / LLM                          │
│   (Claude · OpenAI · Ollama · any tool-use capable model)    │
└───────────────┬───────────────────────────┬──────────────────┘
                │ tool calls                │ tool results
                ▼                           │
┌──────────────────────────┐                │   ┌─────────────────────┐
│      _dispatch_tool()    │◄───────────────┘   │   houkai CLI        │
│  (examples/claude_agent, │                    │  (ai_houkai.cli)    │
│   04_openai, 02_ollama)  │                    └──────────┬──────────┘
└───────────┬──────────────┘                               │
            │  or via MCP                                  │ direct Python
            ▼                                              ▼
┌──────────────────────────────────────────────────────────────┐
│                      MemoryStore                             │
│   remember()  edit()  recall()  recall_pack()                │
│   auto_context_pack()  forget()  nuke()  count()             │
│   list_recent()  link()  unlink()  neighbors()  subgraph()   │
│   supersede()  restore()  find_conflicts()  undo()           │
│   purge_expired()  history()  state_at()  get_at()  metrics()│
└───────────────────────────┬──────────────────────────────────┘
                            │
            ┌───────────────┼────────────────┐
            ▼               ▼                ▼
┌─────────────────┐ ┌──────────────┐ ┌──────────────────────┐
│  ChromaDB HNSW  │ │ DecayEngine  │ │  ReflectionEngine    │
│  (cosine space) │ │  prune()     │ │  clusters()          │
│ PersistentClient│ │  score_all() │ │  reflect()           │
└────────┬────────┘ └──────────────┘ └──────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  sentence-transformers          │
│  all-MiniLM-L6-v2  (local)      │
│  384-dim cosine embeddings      │
└─────────────────────────────────┘
```

The **MCP server** (`ai_houkai/mcp_server/server.py`) wraps `MemoryStore`
and exposes the same operations as MCP tools so any MCP client (Claude Code,
Claude Desktop, custom agents) can call them without Python glue code.

### Package structure

```
ai_houkai/                        pip package name: ai-houkai
├── __init__.py                   convenience re-exports
├── memory_system/
│   ├── __init__.py               exports Memory, MemoryStore, MemoryType,
│   │                             Link, Graph, HybridWeights, ExpandSpec,
│   │                             Conflict, ConflictError, ConflictFn,
│   │                             PackResult, PackedMemory,
│   │                             ExportSummary, ImportSummary,
│   │                             ImportConflictError, Journal, JournalEntry,
│   │                             DecayEngine, ReflectionEngine, build_summarizer,
│   │                             AsyncMemoryStore, CompressedGroup, Reranker,
│   │                             extract_key_phrases, score_importance,
│   │                             MEMORY_TYPES, LINK_RELS, RECALL_MODES,
│   │                             FUSIONS, CONFLICT_POLICIES,
│   │                             IMPORT_POLICIES, DIRECTIONS
│   ├── store.py                  MemoryStore + dataclasses + BM25 + conflict
│   ├── async_store.py            AsyncMemoryStore — coroutine wrapper, single-threaded executor
│   ├── journal.py                Journal — append-only JSONL audit log
│   ├── decay.py                  DecayEngine
│   ├── reflection.py             ReflectionEngine
│   ├── summarizers.py            build_summarizer — LLM summarizer specs
│   ├── importance.py             score_importance — heuristic auto-assignment
│   └── ingest.py                 chunk_text — markdown-aware chunking
├── tui/
│   ├── data.py                   View/Navigator view models (no textual import)
│   └── app.py                    HoukaiTui — Textual browser (houkai tui)
├── maintenance/
│   ├── __init__.py
│   ├── durations.py              parse/format human duration strings
│   ├── state.py                  MaintenanceState — JSON run history
│   ├── scheduler.py              MaintenanceScheduler — tick + run_forever
│   └── daemon.py                 PID file helpers + spawn_detached
├── mcp_server/
│   ├── __init__.py
│   └── server.py                 FastMCP server — 41 tools
├── http_server/
│   ├── __init__.py
│   └── server.py                 stdlib JSON HTTP/REST server (houkai serve)
├── cli/
│   ├── __init__.py
│   ├── __main__.py               python -m ai_houkai.cli
│   ├── main.py                   Typer app, shared --store/--collection flags
│   ├── config.py                 env → config file → defaults resolution
│   ├── output.py                 rich / TSV / JSON, id prefix, fmt_age
│   └── commands/
│       ├── remember.py           houkai remember
│       ├── recall.py             houkai recall
│       ├── pack.py               houkai pack / auto-context
│       ├── list_cmd.py           houkai list
│       ├── show.py               houkai show
│       ├── forget.py             houkai forget
│       ├── nuke.py               houkai nuke  (bulk delete all memories)
│       ├── edit.py               houkai edit / tag / bump
│       ├── link.py               houkai link / unlink / neighbors / graph
│       ├── conflicts.py          houkai conflicts / supersede / restore
│       ├── decay.py              houkai prune / purge  (DecayEngine + TTL)
│       ├── reflect.py            houkai reflect (wraps ReflectionEngine)
│       ├── maintenance.py        houkai maintenance tick/run/start/stop/status
│       ├── journal.py            houkai journal tail/show/undo
│       ├── io.py                 houkai export / import / info / backup
│       ├── ingest.py             houkai ingest
│       ├── serve.py              houkai serve  (HTTP/REST front-end)
│       ├── stats.py              houkai stats
│       ├── doctor.py             houkai doctor  (diagnostics / readiness)
│       ├── tui_cmd.py            houkai tui  (Textual browser)
│       └── collections.py        houkai collections (group)
└── installers/
    ├── __init__.py               re-exports ClaudeCodeInstaller, CursorInstaller, OpenCodeInstaller
    ├── common.py                 shared command resolver, JSON patcher, verify_server, memory guide
    ├── claude_code.py            ClaudeCodeInstaller — claude mcp add / ~/.claude.json / .mcp.json
    ├── cursor.py                 CursorInstaller — patches ~/.cursor/mcp.json
    └── opencode.py               OpenCodeInstaller — patches ~/.config/opencode/opencode.json
```

Import styles:

```python
# Subpackage (canonical)
from ai_houkai.memory_system import MemoryStore, DecayEngine, ReflectionEngine

# Top-level convenience re-export (from ai_houkai/__init__.py)
from ai_houkai import MemoryStore, DecayEngine, ReflectionEngine, AsyncMemoryStore
```

---

## 3. Data Model

### Memory dataclass

```python
@dataclass
class Memory:
    id:            str          # UUID-4
    text:          str          # the memory content
    type:          MemoryType   # see below
    tags:          list[str]    # freeform topic labels
    importance:    float        # 0.0 – 1.0
    created_at:    float        # Unix timestamp
    last_accessed: float        # updated on every recall hit
    access_count:  int          # total recall hits
    source:        str | None   # optional provenance label
    # linking
    links:         list[Link]   # directed edges to other memories
    # conflict management
    superseded_by: str          # id of superseding memory, or ""
    superseded_at: float        # epoch when superseded
    polarity:      int          # -1 / 0 / +1
    # expiry (TTL)
    expires_at:    float        # epoch after which the memory is expired; 0 = never
```

`polarity` plays two roles. Beyond driving `polarity_diff` conflict detection
(§13 — two non-zero opposite polarities at high similarity flag a
contradiction), it also contributes a small **additive ranking bonus** at
recall time via `HybridWeights.polarity_weight` (default `0.05`): `+ζ` for
`+1`, `−ζ` for `−1`, nothing for neutral. See §14.

### Link dataclass

```python
@dataclass
class Link:
    to:  str   # destination memory id
    rel: str   # relation vocabulary below
```

Standard `rel` vocabulary:

| `rel` | Meaning | Created by |
|---|---|---|
| `supersedes`    | replaces another memory             | `supersede()` |
| `refines`       | adds detail to another memory       | manual / agent |
| `derived_from`  | reflection summary ← source episodic | `ReflectionEngine` |
| `example_of`    | concrete instance of a rule         | manual / agent |
| `contradicts`   | intentional disagreement            | `find_conflicts()` + user |
| `related`       | catch-all weak association          | manual / agent |

`rel` is an open string — callers may add their own.

### Memory types

| Type | Intended use |
|---|---|
| `episodic` | Time-stamped events: "Deployed v2.1 on Monday" |
| `semantic` | Distilled facts: "Python's GIL blocks CPU parallelism" |
| `procedural` | How-to knowledge: "Run `make release` to deploy" |
| `feedback` | User preferences: "User prefers concise answers" |

Types affect two behaviours:

- **Filtering** — `recall(type="procedural")` narrows the vector search.
- **Protection** — `DecayEngine` never prunes `procedural` memories by
  default (configurable via `protect_types`).

### Enum vocabularies (validated in the store)

The store is the single validation point for every enum-ish parameter, so
all surfaces (CLI, HTTP, MCP, TUI) reject typos in one place instead of
silently degrading (e.g. `mode="hybird"` used to fall through to semantic
search; an unknown `on_conflict` used to scan and then discard the
conflicts). The vocabularies are exported from `ai_houkai.memory_system`:

| Constant | Values | Validated in |
|---|---|---|
| `MEMORY_TYPES` | `episodic, semantic, procedural, feedback` | `remember`, `edit`, `recall(type=)` |
| `LINK_RELS` | `related, refines, derived_from, example_of, contradicts, supersedes` | `link`, `unlink`, `neighbors(rel=)` |
| `RECALL_MODES` | `semantic, hybrid` | `recall`, `recall_pack`, `auto_context_pack` |
| `FUSIONS` | `weighted, rrf` | `recall(fusion=)` |
| `CONFLICT_POLICIES` | `ignore, warn, supersede, raise` | `MemoryStore(conflict_policy=)`, `remember(on_conflict=)` |
| `IMPORT_POLICIES` | `skip, overwrite, rename, error` | `import_(on_conflict=)` |
| `DIRECTIONS` | `out, in, both` | `neighbors(direction=)` |

A bad value raises `ValueError` naming the parameter and the allowed
vocabulary; `polarity` is likewise restricted to `-1/0/+1`. `link()` also
rejects a `dst_id` that does not resolve — graph walkers skip unresolvable
targets, so a dangling edge would be stored but permanently unreachable.

### Metadata serialisation

ChromaDB metadata values must be scalar.  Tags are stored as a
comma-joined string (which is why `remember()`/`edit()` reject a comma
inside a tag — it would silently split into two tags on the next read);
`links` are JSON-encoded:

```python
# write
{
    "tags":          "deploy,api,prod",
    "links":         '[{"to":"a8...","rel":"refines"}]',
    "superseded_by": "",
    "superseded_at": 0.0,
    "polarity":      0,
    "expires_at":    0.0,
}

# read — all new fields default safely for old records
tags       = [t for t in meta.get("tags", "").split(",") if t]
links      = json.loads(meta.get("links") or "[]")
expires_at = float(meta.get("expires_at", 0.0))   # 0 = never expires
```

Every field added over the project's life (`links`, `superseded_by`,
`superseded_at`, `polarity`, `expires_at`) is read with a safe default, so a
`.chroma` written by an older version loads unchanged — no migration step.

---

## 4. Storage Layer

### ChromaDB

`MemoryStore` uses a `PersistentClient` which writes to a local
directory (default `./.chroma`).  This gives:

- **Persistence** across Python process restarts.
- **Test isolation** — each test gets its own `tmp_path` directory so
  collections never share state.

```python
chromadb.PersistentClient(
    path=path,
    settings=Settings(anonymized_telemetry=False),
)
```

The collection is created with `hnsw:space=cosine` so distances are
cosine distances (0 = identical, 2 = opposite).  The store converts to
similarity at query time: `similarity = 1.0 − distance`.

### Embedding function

`SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")`
produces 384-dimensional vectors.  The model runs fully offline once
downloaded.  To swap to a different provider pass a custom
`chromadb.EmbeddingFunction` to the collection.

### HNSW index

ChromaDB uses the HNSW (Hierarchical Navigable Small World) graph for
approximate nearest-neighbour search.  At the scale of a single agent's
memory (hundreds to low thousands of entries), exact search would also
be fine — HNSW ensures queries stay fast as collections grow.

---

## 5. Memory Lifecycle

```
                 ┌─────────────────────┐
                 │    remember(text)   │  ← optional ttl_seconds / expires_at
                 └────────┬────────────┘
                          │  UUID assigned
                          │  text embedded (384-dim)
                          │  metadata written
                          ▼
                 ┌─────────────────────┐
                 │   ChromaDB HNSW     │◄── persists to disk
                 └────────┬────────────┘
          ┌───────────────┼──────────────┬─────────────────┬───────────────┐
          ▼               ▼              ▼                 ▼               ▼
   recall(query)    list_recent()   edit(id, …)       forget(id)    purge_expired()
          │               │              │                 │               │
     vector search   chronological  in-place update   hard delete   hard-delete every
     metadata filter    sort        re-embeds on      returns bool   memory past its TTL
   hide superseded  hide superseded text change                     (journaled, actor
    + expired         + expired                                      "purge")
          │
          ▼
    _touch(memory)
    ├── last_accessed = now
    └── access_count += 1
```

**Expiry (TTL).** `remember(ttl_seconds=…)` or `remember(expires_at=…)` stamps
an `expires_at` epoch. Once it passes, the memory is **soft-hidden** — excluded
from `recall()` and `list_recent()` by default (like superseded), overridable
with `include_expired=True` — but still physically present and fetchable by id.
`purge_expired()` reclaims the storage with a per-row journaled `forget`
(actor `"purge"`); unlike decay's `prune()` it ignores `protect_types` (an
explicit TTL is a stronger signal than the decay heuristic). The scheduled
maintenance tick runs it automatically on a `purge_every` cadence (§17).

`edit()` updates fields of an existing memory **keeping its id**: omitted
fields stay unchanged (`source` uses a sentinel so `source=None` explicitly
clears it), text changes are re-embedded, and links / supersede state /
access tracking / `created_at` all survive — unlike a `forget()` +
`remember()` round-trip. The change is journaled (op `edit`, with
before/after snapshots) and reversible via `undo()`; a call that changes
nothing is a true no-op (no write, no journal entry).

#### recall() filtering pipeline

1. Build the `where` clause (`_build_where`) from `type`, `min_importance`,
   `source`, `since`, `until` and over-fetch a cosine pool.
2. Post-filter by `tag` (ChromaDB only supports `$eq` on scalar fields,
   not array membership — tag filtering happens in Python).
3. Score the pool: weighted hybrid (`_hybrid_score`), `fusion="rrf"`
   (`_rrf_score`), or semantic (`_semantic_filter`); `min_cosine` drops any
   candidate below an absolute cosine floor.
4. **Drop expired** memories (`expires_at != 0 and expires_at <= now`) unless
   `include_expired=True` — a post-filter, so pre-TTL records (no key) are
   unaffected (§21).
5. **Rerank** (optional): if a `reranker` is configured (per-store or per-call),
   rescore the surviving pool with it and re-sort before the cut (§14).
6. Optional re-selection: when `diversity` or `dedup_threshold` is set,
   `_mmr_select` re-orders/drops near-duplicates (min-max-normalising
   relevance first); otherwise take the top-k.
7. Unless `touch=False`, batch-bump access tracking for **all** hits in ONE
   Chroma write (`_touch_many`).
8. Optional multi-hop graph expansion out to `expand.depth` with a per-hop
   `expand.decay` multiplier (`_expand_links`).
9. Return `(Memory, score)` pairs — or `(Memory, score, breakdown)` triples
   when `explain=True` (§14).

#### recall_pack() — token-budgeted assembly

`recall_pack()` is a thin read-path layer over `recall()` that solves the
agent's real consumption pattern: *"fill ~N tokens of context with the most
useful memory,"* not *"give me exactly k rows."*

1. Rank candidates via `recall(query, k=max_items, …)` — inheriting hybrid
   scoring (the pack default), tag/type filters, link expansion, superseded
   exclusion, and `_touch()` for free. No query logic is duplicated.
2. Walk candidates in rank order, rendering each as `- (type) text` and
   estimating its token cost. Greedily admit items while the running total
   stays within `token_budget`; a candidate that doesn't fit is skipped but
   the walk continues, so a smaller lower-ranked memory can still slot in.
3. Return a `PackResult` — the rendered block (`text`), the admitted `items`
   (each with its `score` and `tokens`), `used_tokens`, `budget`, and a
   `truncated` flag (true when any ranked candidate was dropped to fit).

Token estimation is tokenizer-free (`max(1, round(len/4))`), so `token_budget`
is a **soft ceiling** covering the memory lines only — the header is excluded.
Callers needing exact budgets pass `token_counter=`. The budget has no
data-model impact and is purely additive on the read path.

`recall_pack()` forwards `fusion` / `diversity` / `dedup_threshold` /
`min_cosine` / `touch` straight through to `recall()`, so all the ranking
controls from §14 are available on the pack path too. With `compress=True`,
candidates that were dropped for exceeding the budget are clustered by
token-Jaccard similarity (`compress_threshold`, default `0.30`); each cluster
of ≥ `compress_min_group` (default `2`) members is folded into a single
`- (compressed)` summary line that is packed if it fits the remaining budget,
surfaced separately as `compressed_groups`.

![recall_pack greedy token-budget packing, default vs compress=True](resources/recall_pack_budget.png)

*Greedy fit to the default `token_budget = 800`: items are admitted in rank
order until the budget is exhausted; an item too large for the remaining space
is skipped while the walk continues, so a smaller lower-ranked memory can still
slot in. `compress=True` folds the dropped candidates into one `- (compressed)`
summary line when they cluster.*

#### auto_context_pack() — multi-angle fan-out

`auto_context_pack()` sits alongside `recall_pack()` for tasks with compound
concepts. It extracts up to `max_phrases` key bigram/keyword phrases from the
task (`extract_key_phrases`), runs `recall()` for the task **and** each phrase
angle independently, deduplicates by id (keeping the **best** score seen per
memory), then packs greedily through the **same** packer (`_pack_ranked`) that
`recall_pack()` uses — so `compress` works here too. It also accepts
`min_cosine` (an absolute floor applied to every fan-out query, so an
off-topic task injects nothing rather than weak padding) but deliberately
omits `fusion="rrf"`, whose pool-relative scores are not comparable across the
different fan-out pools (see §14).

---

## 6. Decay Engine

### Formula

```
score(m) = importance × exp(−λ × days_since_last_access) × reinforcement
reinforcement = 1 + frequency_weight × ln(1 + access_count)
```

| Parameter | Default | Effect |
|---|---|---|
| `decay_rate` (λ) | `0.1` | Half-life ≈ 7 days for importance=0.5 |
| `min_score` | `0.05` | Prune threshold |
| `protect_types` | `("procedural",)` | Types immune to pruning |
| `frequency_weight` | `0.0` | Recall reinforcement strength (0 = off) |

![Exponential decay curves for importance 0.90/0.50/0.30 against the 0.05 prune line](resources/decay_curves.png)

*With λ = 0.1 a memory crosses the `min_score = 0.05` prune line at ~29 days
(importance 0.90), ~23 days (0.50), or ~18 days (0.30); the dotted line marks
the ≈6.9-day half-life.*

![Decay survival region over importance × idle days](resources/decay_heatmap.png)

*The same score as a 2-D field — everything above the red `0.05` contour is
retained, everything below is pruned.*

### Recall reinforcement

`access_count` (incremented on every `recall()` hit) feeds back into the
score: a frequently-recalled memory ages out more slowly than an untouched
one of equal importance and age. With the default `frequency_weight = 0.0`
the reinforcement factor is exactly `1.0`, so scoring matches the
recency-only behaviour and nothing changes. As `frequency_weight` rises,
often-used memories resist pruning — e.g. `0.2` lets a memory recalled ~10
times survive a prune that drops its never-reread twin. Note the score can
then exceed `importance`; `min_score` is compared against the reinforced
value. Configurable via `[maintenance.decay].frequency_weight`,
`houkai prune --frequency-weight`, and the `MaintenanceScheduler`.

![Reinforcement multiplier and the extra survival days recalls buy](resources/reinforcement.png)

*The multiplier grows with `ln(1 + access_count)`, so each extra recall resists
decay a little less than the last; the right panel converts that into days of
added lifetime before the prune line.*

Score examples with λ=0.1 (no reinforcement, `frequency_weight=0`):

| importance | age | score | verdict |
|---|---|---|---|
| 0.9 | 1 day | 0.81 | kept |
| 0.9 | 7 days | 0.45 | kept |
| 0.9 | 30 days | 0.04 | **pruned** |
| 0.5 | 1 day | 0.45 | kept |
| 0.1 | 7 days | 0.05 | **pruned** |

### Tuning λ

| λ | Half-life (imp=0.5) | Use case |
|---|---|---|
| 0.01 | ~69 days | Long-lived knowledge bases |
| 0.05 | ~14 days | Normal agent memory |
| 0.1 | ~7 days | Fast-changing environments |
| 0.2 | ~3.5 days | Ephemeral session contexts |

![Decay curves for λ = 0.05/0.10/0.20 at importance 0.50](resources/halflife.png)

*Turning λ up steepens the forgetting curve: 0.05 → ~14-day half-life,
0.10 → ~7, 0.20 → ~3.5.*

![Half-life as a function of decay_rate, t½ = ln2/λ](resources/halflife_vs_lambda.png)

*Half-life is `ln(2)/λ`, so it falls off hyperbolically as λ rises — the three
marked points are the defaults from the table above.*

### API

```python
from ai_houkai.memory_system import MemoryStore, DecayEngine

engine = DecayEngine(store, decay_rate=0.1, min_score=0.05,
                     protect_types=("procedural",),
                     frequency_weight=0.0)   # >0 → frequent recalls resist decay

score     = engine.score(mem)            # single memory
pairs     = engine.score_all()           # all, sorted desc
candidates = engine.prune(dry_run=True)  # preview
removed    = engine.prune()              # delete stale memories
```

`now` can be overridden in both `score()` and `prune()` for
deterministic testing or time-travel simulations.

---

## 7. Reflection Engine

Implements the **Generative Agents** reflection pattern: cluster
semantically similar episodic memories and condense them into a single
semantic "summary" memory.

### Algorithm

```
1. Fetch candidates of the configured types from ChromaDB (with embeddings).
2. Sort by importance descending — highest importance seeds first.
3. Greedy single-linkage clustering:
      for each unseeded memory (highest importance first):
          start a new cluster with this memory as seed
          absorb every other unseeded memory whose cosine
          similarity to the seed ≥ similarity_threshold
4. Discard clusters with fewer than min_cluster_size members.
5. Skip any cluster whose members are already at max_level.
6. For each qualifying cluster:
      text       = summarizer(cluster_members)
      tags       = ["reflection", "level:N"] + union of source tags
      importance = mean(source importances)
      trust      = worst trust among the sources
      store new semantic memory  →  MemoryStore.remember()
7. consolidate: none = leave sources; soft = supersede them;
   hard = delete them.
```

### What gets clustered, and how deep

`types` was hard-coded to `("episodic",)`, which had two consequences. Semantic
memories — **including reflections themselves** — never consolidated, so a
long-lived store accumulated summaries without bound; and `feedback` /
`procedural` never benefited at all. It is now a parameter, defaulting to
`("episodic",)` so existing behaviour is unchanged.

Letting reflections re-cluster raises the opposite risk: a summary of summaries
of summaries, eating the store. Each summary is tagged `level:N` and a cluster
whose members already sit at `max_level` is skipped. `max_level=1` (the default)
reproduces the old behaviour — summaries are produced but never re-summarised —
and raising it builds a deliberate tier hierarchy.

A summary inherits the **worst** trust among its sources: reflecting over
content the agent did not author would otherwise launder it into a `trusted`
memory that `min_trust="trusted"` happily returns (§27). It is born `pinned`
only when `consolidate` removes the sources from the working set — see §27.

Candidates are filtered before clustering, and the reasons rhyme. A
**superseded** row was already consolidated, so re-clustering it would emit a
duplicate summary every run. A **lapsed** row (§21) is hidden from recall and
waiting for `purge_expired`; folding it in would copy its text into a fresh
summary carrying no TTL of its own, turning a deliberate lifetime into a
permanent row. In both cases a summary must not grant its sources more reach
than they had — the same rule as the trust inheritance above.

**Known limitation.** `_cluster` is O(n²) over a pure-Python `_cosine`: fine at
a thousand memories, painful at twenty thousand. The cheap mitigation, if it
ever bites, is to cluster within a vector-index candidate pool rather than the
full cross-product.

### Clustering properties

- **Seed-based**: highest-importance memory anchors each cluster.
- **Single-linkage**: a memory joins if similar to the *seed*, not all
  existing members — O(n) per seed, avoids chaining artefacts.
- **Non-overlapping**: each memory belongs to at most one cluster.

### Similarity threshold guide

| threshold | Effect |
|---|---|
| 0.95 | Only near-duplicates |
| 0.80 | Same topic, similar phrasing |
| 0.75 | Same topic, varied phrasing (default) |
| 0.60 | Broadly related content |

### Default summarizer (extractive)

```python
def _default_summarizer(memories):
    ordered = sorted(memories, key=lambda m: m.importance, reverse=True)
    body = " | ".join(m.text for m in ordered)
    return ("[Reflection ×N] " + body)[:512]
```

### LLM summarizers (`summarizers.py`)

`build_summarizer(spec)` turns a `provider:model` string into a
summarizer callable:

```python
from ai_houkai.memory_system import ReflectionEngine, build_summarizer

engine = ReflectionEngine(store, summarizer=build_summarizer("ollama:llama3.1"))
```

| Provider | Transport | Notes |
|---|---|---|
| `extractive` | — | the built-in default; also used for `None`/`""` |
| `ollama:MODEL` | stdlib `urllib` → OpenAI-compat `/v1/chat/completions` | no SDK dependency; `OLLAMA_BASE_URL` env (default `http://localhost:11434`); model may contain colons (`ollama:llama3.1:8b`) |
| `openai:MODEL` | `openai` SDK (lazy import) | raises ImportError with an `ai-houkai[openai]` hint if absent |
| `anthropic:MODEL` | `anthropic` SDK (lazy import) | `ai-houkai[claude]` hint; joins `text` content blocks |

All providers share one prompt (`render_prompt()`): events ordered by
importance, an instruction to produce a 1–3 sentence summary without
inventing facts.

**Fallback wrapper** (default on): if the LLM call raises or returns
empty/whitespace output, the extractive summarizer is used for that
cluster and a warning is logged.  This matters because reflection runs
unattended inside the maintenance daemon — a down Ollama box must not
fail the tick.  Pass `fallback=False` to surface errors instead.

Spec errors (`unknown provider`, missing model) raise `ValueError` at
build time, so a typo'd config fails fast at CLI/daemon startup — not
silently at 3 AM inside a tick.

The single config key `[maintenance.reflect].summarizer` (env override
`AI_HOUKAI_SUMMARIZER`) feeds all three consumers: `houkai reflect`
(overridable per-run with `--summarizer`), `houkai maintenance
tick/run/start`, and the MCP `maintenance_tick` tool.

### Custom summarizer

Any `Callable[[list[Memory]], str]` still works:

```python
def my_summarizer(memories):
    prompt = "\n".join(m.text for m in memories)
    return call_llm(f"Summarise these events into one insight:\n{prompt}")

engine = ReflectionEngine(store, summarizer=my_summarizer)
```

### API

```python
from ai_houkai.memory_system import MemoryStore, ReflectionEngine

engine = ReflectionEngine(store,
                          similarity_threshold=0.75,
                          min_cluster_size=2,
                          summarizer=None,
                          types=("episodic",),   # what to cluster
                          max_level=1)           # tiers of reflection-of-reflection

clusters = engine.clusters()              # list[list[Memory]], no writes
previews = engine.reflect(dry_run=True)   # list[Memory], not persisted
created  = engine.reflect()               # persist summaries, sources untouched
created  = engine.reflect(consolidate=True)   # + supersede the sources
created  = engine.reflect(consolidate="hard") # + delete the sources
```

### ChromaDB numpy array guard

ChromaDB returns embeddings as numpy arrays.  Using `raw or []` raises
`ValueError: The truth value of an array is ambiguous`.  The engine
uses an explicit `None` check:

```python
raw  = res.get("embeddings")
embs = [] if raw is None else raw   # safe for numpy arrays
```

---

## 8. MCP Server

`ai_houkai/mcp_server/server.py` uses **FastMCP** to expose twenty-three tools:

**Core tools**

| Tool | Key parameters | Returns |
|---|---|---|
| `remember` | `text`, `type?`, `tags?`, `importance?`, `source?`, `on_conflict?`, `polarity?`, `expires_at?`, `ttl_seconds?` | `{id, stored, importance, expires_at}` or `{stored:false, conflicts:[…]}` |
| `remember_many` | `items` (each `{text, type?, tags?, importance?, source?, polarity?, expires_at?, ttl_seconds?}`), `batch_size?`, `on_conflict?` | `{stored, ids}` — batched bulk write (`ceil(N/batch)` encodes); `on_conflict="raise"` unsupported |
| `edit` | `memory_id`, `text?`, `type?`, `tags?`, `importance?`, `polarity?`, `source?`, `expires_at?` | `{ok, id, text, type, tags, importance, polarity, source, expires_at}` or `{ok:false, error}` — in-place, journaled, undoable |
| `recall` | `query`, `k?`, `type?`, `tag?`, `min_importance?`, `source?`, `since?`, `until?`, `mode?`, `overfetch?`, `include_superseded?`, `include_expired?`, `explain?` | `list[{id,text,type,tags,importance,score,created_at,superseded_by,expires_at,explain?}]` |
| `recall_pack` | `query`, `token_budget?`, `type?`, `tag?`, `min_importance?`, `source?`, `since?`, `until?`, `mode?`, `max_items?`, `include_superseded?`, `compress?`, `compress_threshold?`, `compress_min_group?` | `{text, used_tokens, budget, truncated, items:[{id,text,type,tags,importance,score,tokens}], compressed_groups:[{ids,text,tokens,count}]}` |
| `auto_context` | `task`, `token_budget?`, `max_phrases?`, `mode?` | `{text, queries, used_tokens, budget, truncated, items:[{id,text,type,tags,importance,score,tokens}]}` |
| `forget` | `memory_id` | `{deleted}` |
| `purge_expired` | `dry_run?` | `{purged, dry_run, ids}` — hard-delete TTL-expired memories (§21) |
| `list_recent` | `limit?`, `include_superseded?`, `include_expired?` | `list[{…,superseded_by,expires_at}]` |
| `stats` | — | `{count, path, collection}` |
| `metrics` | — | `{uptime_seconds, count, calls, recall_latency_ms}` — runtime counters (§22) |

**Linking tools**

| Tool | Key parameters | Returns |
|---|---|---|
| `link`      | `src_id`, `dst_id`, `rel?` | `{ok, src_id, dst_id, rel}` |
| `unlink`    | `src_id`, `dst_id`, `rel?` | `{removed}` |
| `neighbors` | `memory_id`, `rel?`, `direction?`, `depth?` | `list[{id,text,type,tags,importance,rel}]` |

**Conflict tools**

| Tool | Key parameters | Returns |
|---|---|---|
| `find_conflicts` | `memory_id?`, `threshold?` | `list[{kind,reason,similarity,a,b}]` |
| `supersede`      | `old_id`, `new_id`         | `{ok, old_id, new_id}` |

**Maintenance & audit tools**

| Tool | Key parameters | Returns |
|---|---|---|
| `maintenance_tick` | `reflect_apply?` (omit → config `[maintenance.reflect] apply`) | `{summary, ran_decay, ran_reflect, ran_purge, decayed, reflected, purged, trash_purged, reflect_applied, decay_error, reflect_error, purge_error}` |
| `journal_tail` | `n?`, `op?`, `since_seconds?` | `list[{ts,op,actor,id,summary,meta}]` (newest first) |
| `history` | `memory_id` | `list[{ts,op,actor,id,before,after,meta,summary}]` — full timeline, oldest first (§18) |
| `state_at` | `ts` | `{ts, count, memories:[…]}` — store reconstructed as of `ts` (§18) |
| `get_at` | `memory_id`, `ts` | `{ok, ts, …}` — one memory as of `ts`, or `{ok:false, error}` (§18) |
| `export` | `path`, `include_vectors?`, `include_superseded?`, `type?`, `tag?`, `since?` | `{path, count, bytes, elapsed}` |
| `import` | `path`, `on_conflict?`, `regenerate_vectors?`, `dry_run?` | `{ok, imported, skipped, overwritten, renamed, errors, vectors_regenerated}` |

`maintenance_tick` honours the `[maintenance]` schedule from
`~/.config/ai_houkai/config.toml` — jobs only run when their interval has
elapsed, **including dry-run reflection** (the schedule gates the work, not
the writes), so MCP clients may call it freely (see §17). `reflect_apply`
defaults to the config's `[maintenance.reflect] apply` setting; when it
resolves to false, `reflected` is the number of summaries a real run
*would* create and `reflect_applied` is `false`.

Configuration via environment variables:

| Variable | Default |
|---|---|
| `AI_HOUKAI_PATH` | `./.chroma` |
| `AI_HOUKAI_COLLECTION` | `ai_houkai` |

The `run()` function is the **console-script entry point**:

```python
# ai_houkai/mcp_server/server.py
def run() -> None:
    mcp.run()

# pyproject.toml
[project.scripts]
ai-houkai-mcp = "ai_houkai.mcp_server.server:run"
```

### Claude Code integration

Claude Code discovers MCP servers from `~/.claude.json` (user scope) or a
project-level `.mcp.json` — **not** from `settings.json`, which has no
`mcpServers` key. The supported registration interface is
`claude mcp add --scope user|project`, and the installer prefers it.

```
Claude Code CLI
    │
    │  reads at startup / per-invocation
    ▼
~/.claude.json (user scope)  /  ./.mcp.json (project scope)
    │
    │  spawns subprocess
    ▼
ai-houkai-mcp                         (console script)
    │
    │  stdio transport (JSON-RPC 2.0)
    ▼
ai_houkai.mcp_server.server  ──►  MemoryStore  ──►  ChromaDB on disk
```

**Quickest setup** — one command:

```bash
claude mcp add ai-houkai -- ai-houkai-mcp
```

**Programmatic setup** — three equivalent paths, all backed by the same
installer module (`ai_houkai.installers.claude_code`):

```bash
# console script (installed by pip)
ai-houkai-install-claude-code --install

# python -m
python -m ai_houkai.installers.claude_code --install

# example wrapper (also offers --demo)
python examples/06_claude_code.py --install
```

**Library usage** — embed installation in your own bootstrap scripts:

```python
from ai_houkai.installers import ClaudeCodeInstaller

inst = ClaudeCodeInstaller(
    memory_path = "~/.ai_houkai",
    collection  = "my_project",          # per-project namespace
)
inst.install()                   # `claude mcp add --scope user`, or ~/.claude.json
inst.install(scope="project")    # project scope → ./.mcp.json
inst.verify()
print(inst.claudemd_snippet())
```

The installer is a small dataclass (no global state, idempotent
registration, atomic JSON-merge into existing `mcpServers`) so it composes
well with project scaffolding tooling.

#### CLAUDE.md guidance

Adding memory instructions to `CLAUDE.md` teaches Claude Code *when*
to use the tools autonomously — without the user needing to prompt:

```markdown
## Memory (AI-Houkai MCP)
- recall() before starting any task
- remember() conventions, decisions, corrections
- edit() to fix or refine an existing memory (keeps id, links, history)
- forget() outdated facts
```

Generate a full snippet: `python examples/06_claude_code.py --claudemd`

### Claude Desktop integration

```
Claude Desktop
    │
    │  reads at startup
    ▼
~/.config/claude/claude_desktop_config.json   (Linux)
~/Library/Application Support/Claude/…        (macOS)
%APPDATA%\Claude\…                             (Windows)
    │
    │  spawns subprocess
    ▼
python -m ai_houkai.mcp_server.server
    │
    │  stdio transport (JSON-RPC 2.0)
    ▼
MemoryStore ──► ChromaDB on disk
```

`examples/03_claude_desktop.py --install` locates the platform-specific
config path and patches the `mcpServers` block automatically.

---

## 9. Installers

The `ai_houkai.installers` subpackage isolates the boilerplate needed to
register the MCP server with various MCP clients.  Three installers ship
today — **Claude Code**, **Cursor**, and **OpenCode** — all built on a
shared `common.py` (command resolution, JSON load/patch/write,
`verify_server()` smoke test, and the memory-guide snippet).  The logic
used to live inside `examples/06_claude_code.py`; promoting it to a real
module means:

- Third-party tools can import the installers as a library.
- The console scripts (`ai-houkai-install-claude-code`,
  `ai-houkai-install-cursor`, `ai-houkai-install-opencode`) are available
  immediately after `pip install ai-houkai` — no example file required.
- Tests can target the installers directly without `spec_from_file_location`
  hacks for digit-prefixed examples.

### `ClaudeCodeInstaller`

```python
@dataclass
class ClaudeCodeInstaller:
    memory_path: str = "~/.ai_houkai"
    collection:  str = "claude_code"
    config_path: str = "~/.claude.json"   # direct-write fallback (user scope)
    server_name: str = "ai-houkai"
    extra_env:   dict = {}

    def build_env(self)            -> dict
    def build_mcp_block(self)      -> dict       # {"type": "stdio", ...}
    def build_settings_block(self) -> dict
    def install(self, *, scope="user") -> str    # what was written (cmd or path)
    def print_config(self)         -> None
    def verify(self)               -> bool
    @staticmethod
    def claudemd_snippet()         -> str
```

### Behaviour

- **CLI-first**: when `claude` is on PATH, `install()` shells out to
  `claude mcp remove` + `claude mcp add --scope user|project` — robust to
  future config-layout changes. Without the CLI it merges the stdio block
  into `~/.claude.json` (user) or `./.mcp.json` (project) directly.
- **Idempotent**: re-running `install()` replaces the `ai-houkai` entry
  but preserves all other servers and top-level keys.
- **Atomic writes**: direct config edits go through write-to-temp +
  `os.replace` (`common.write_json`) so a crash can never truncate the
  user's client config.
- **Unparseable config**: by default replaces a corrupted JSON file
  rather than aborting (toggle with `overwrite_unparseable=False`).
- **No import side effects**: the MCP server module creates its store
  lazily (`get_store()`, on first tool use), so importing the installers —
  or `ai_houkai.mcp_server.server` itself — never materialises a stray
  `./.chroma` or loads the embedding model. `verify()` reports the count of
  the store actually being installed (`memory_path`/`collection`), not the
  env-default one.
- **Common shape**: all installers expose the same surface
  (`build_mcp_block` / `build_settings_block` / `install` / `print_config`
  / `verify` + a client-specific guidance snippet) so callers can dispatch
  generically; future installers follow the same pattern.

### `CursorInstaller` and `OpenCodeInstaller`

Both mirror `ClaudeCodeInstaller` with client-specific defaults and
guidance snippets:

| | Cursor | OpenCode |
|---|---|---|
| Settings file | `~/.cursor/mcp.json` (`--project` → `./.cursor/mcp.json`) | `~/.config/opencode/opencode.json` (`--project` → `./opencode.json`) |
| Config schema | same `mcpServers` block as Claude | own `mcp` schema: `command` array, `environment`, `enabled`, `type: local` |
| Default collection | `cursor` | `opencode` |
| Guidance snippet | `rule_snippet()` → `.cursor/rules/*.mdc` | `agents_snippet()` → `AGENTS.md` |
| Console script | `ai-houkai-install-cursor` | `ai-houkai-install-opencode` |

### Console script wiring (`pyproject.toml`)

```toml
[project.scripts]
ai-houkai-mcp                 = "ai_houkai.mcp_server.server:run"
ai-houkai-install-claude-code = "ai_houkai.installers.claude_code:_main"
ai-houkai-install-cursor      = "ai_houkai.installers.cursor:_main"
ai-houkai-install-opencode    = "ai_houkai.installers.opencode:_main"
houkai                        = "ai_houkai.cli.main:_main"
```

---

## 10. Agent Integrations

All agent examples share the same `_dispatch_tool(name, arguments)`
interface.  Only the SDK and message format differ.

### Unified dispatch signature

```python
def _dispatch_tool(name: str, arguments: str) -> str:
    inputs: dict = json.loads(arguments)   # JSON string in, JSON string out
    if name == "remember":   ...
    elif name == "recall":   ...
    elif name == "forget":   ...
    else: return json.dumps({"error": f"unknown tool: {name}"})
```

This JSON-string interface matches the OpenAI/Ollama function-calling
format natively.  The Claude example serialises its dict input before
calling dispatch.

### Provider comparison

| | Claude (`claude_agent.py`) | OpenAI (`04_openai.py`) | Ollama (`02_ollama_local_network.py`) |
|---|---|---|---|
| SDK | `anthropic` | `openai` | `openai` (compat endpoint) |
| Tool definition | `{"name":…,"input_schema":{…}}` | `{"type":"function","function":{…}}` | same as OpenAI |
| Tool call access | `block.name`, `block.input` (dict) | `tc.function.name`, `tc.function.arguments` (str) | same as OpenAI |
| Arguments | dict → `json.dumps()` | JSON string | JSON string |
| Endpoint | `api.anthropic.com` | `api.openai.com` | `localhost:11434/v1` |
| API key required | yes | yes | no |

### Message flow (generic)

```
user message
     │
     ▼
LLM API  ──►  tool_call: {name, arguments}
     │
     ▼
_dispatch_tool(name, arguments)
     ├── "remember" ──► store.remember()  ──► {"id":…, "stored":true}
     ├── "recall"   ──► store.recall()    ──► {"results":[…]}
     └── "forget"   ──► store.forget()    ──► {"deleted":true/false}
     │
     ▼
tool result appended to messages
     │
     ▼
LLM API  ──►  assistant reply to user
```

---

## 11. Test Architecture

### 1232 tests across 45 files

| File | Tests | What it covers |
|---|---|---|
| `test_maintenance.py` | 74 | Maintenance scheduler/daemon: tick, run-forever, duration parsing, state history, PID files, dry-run reflect schedule gating, **TTL purge job** (gating, state persistence, disabled) |
| `test_http_server.py` | 65 | HTTP/REST API: all endpoints, auth token, 404/405/400/413 handling, keep-alive body-drain, `/health` topology-leak guard, plus `/metrics`, `/purge_expired`, `history`, `state_at`, `get_at`, TTL + `include_expired` + `explain`, POST-`/recall` advanced knobs (`graph` weight + `expand` rerank gating), and `POST /memories/batch` bulk write (`remember_many`) |
| `test_hybrid.py` | 55 | Hybrid retrieval: BM25 pool scoring, `HybridWeights`, blended ranking, link expansion, RRF fusion, MMR diversity & near-duplicate dedup, `min_cosine` gate, `explain` breakdowns, `recency_basis`, multi-hop expansion decay, CJK tokenization |
| `test_graph_fusion.py` | 15 | Graph-proximity fusion (`HybridWeights.graph`): PPR-lite spread math, weighted/RRF no-op at `graph=0`, hub lift, `explain` graph term; gated expansion (`ExpandSpec.rerank`): respects `k`, dedups, RRF scale-remap doesn't bury primaries, drops un-embeddable nodes, `seen_ids` shielding |
| `test_pack.py` | 54 | `recall_pack` / `auto_context_pack`: token-budget packing, truncation, custom counter, filters, rank-order preservation, near-duplicate compression, `min_cosine` |
| `test_validation.py` | 46 | Shared validation layer: store enum vocabularies (incl. `lexical_index`, so the retired `"fts"` spelling errors instead of silently reading as `"pool"`), dangling-link rejection, HTTP body coercion + status codes (400/404, HEAD, non-ASCII auth), clean CLI errors |
| `test_cli.py` | 38 | CLI round-trips: remember → list → show → forget → nuke, tag/bump, link/neighbors/unlink, supersede/restore, export/import, stats, prune dry-run, stdin, pack, edit re-embed, interactive conflict resolution, **`--ttl` + `purge`** |
| `test_conflicts.py` | 39 | Conflict/contradiction detection, `on_conflict` policies, supersede/restore, negation heuristic, and that a **lapsed** candidate never clashes with a new write |
| `test_memory.py` | 30 | `MemoryStore`: remember, forget, nuke, recall (filters, touch control), list_recent, `Memory` dataclass serialisation |
| `test_links.py` | 28 | Typed links: `link`/`unlink`/`neighbors`/`subgraph`, direction, depth, cycles, dangling targets |
| `test_reflection.py` | 27 | `ReflectionEngine`: clustering, reflect (dry-run, consolidate, tags, custom summarizer), skips superseded **and lapsed** sources, polarity-cluster separation |
| `test_async_store.py` | 51 | `AsyncMemoryStore`: coroutine API parity with sync store (incl. `probe_embedding`/`readiness`), executor lifecycle, aclose, plus a mechanical signature check — walking the MRO, so the twelve curation methods inherited from `CurationMixin` are covered too — so a wrapper cannot quietly drop a parameter the sync method gained |
| `test_summarizers.py` | 24 | `build_summarizer`: spec parsing, ollama/openai/anthropic providers (stubbed), extractive fallback, config + scheduler wiring |
| `test_journal.py` | 22 | Append-only audit journal: tail/show/undo (incl. restore-after-forget), rotation, actor attribution |
| `test_ingest.py` | 26 | `chunk_text` + `houkai ingest` / `houkai collections` round-trips, and that splitting a long paragraph never loses a short fragment to the noise filter |
| `test_ttl.py` | 21 | **Expiry/TTL**: `remember(ttl_seconds/expires_at)`, recall + list hide-expired / `include_expired`, `edit`, `purge_expired` (dry-run, custom now, journaled), serialisation round-trip + migration |
| `test_edit.py` | 21 | `MemoryStore.edit()`: in-place update, re-embedding, journaling, undo, no-op detection, async wrapper, CLI edit/tag/bump, HTTP `PATCH` |
| `test_decay.py` | 26 | `DecayEngine`: score formula, score_all sorting, prune (dry-run, protect, custom now, empty store, routing to the trash, and not counting a row whose removal failed), recall reinforcement (`frequency_weight`) |
| `test_remember_many.py` | 20 | **Batch write**: `remember_many` input order/field mapping, batched embed (`ceil(N/batch)` `collection.add` calls), one-journal-entry-per-id + per-id undo, validation aborts before any write, `ignore`/`warn`/`supersede` (earlier-wins, no cycle) + `raise` rejected, TTL, async wrapper |
| `test_mcp_server.py` | 22 | MCP tools in-process: lazy `get_store()` env honoring, `edit` dicts, `maintenance_tick` config, plus `metrics`, `purge_expired`, `history`, `state_at`, `get_at`, TTL + `explain`, and `remember_many` (bulk store, bad-item + `raise` rejection) |
| `test_importance.py` | 18 | `score_importance`: tier matching, modifiers, clamping, store/config wiring |
| `test_export_import.py` | 17 | Portable `.ahkai` archives: export filters, import conflict policies, dry-run, vector regen |
| `test_tui.py` | 15 | TUI view models, Navigator stack, Textual pilot runs (list/detail, neighbors, search), parallel-link row collapse |
| `test_stats_health.py` | 15 | `houkai stats` and `--health` report: decay histogram, at-risk/stale counts, cluster detection, decay-formula alignment with `DecayEngine` (frequency reinforcement) and protected-type exclusion |
| `test_installers.py` | 14 | Claude Code installer: `claude mcp add` invocation, `~/.claude.json` / `.mcp.json` direct writes, config preservation, atomic `write_json`, side-effect-free import of installers and the MCP server module |
| `test_eval.py` | 13 | Retrieval-quality metrics for `ai_houkai/eval.py`: recall/precision@k, MRR, (n)DCG, the `evaluate()` harness over a gold set |
| `test_trust.py` | 13 | **Provenance trust** rules: `worst_trust` takes the worst case (so a derived memory cannot launder its sources), `trust_rank` separates an *absent* level — old rows, which read as trusted — from an *unrecognised* one, which fails safe as least-trusted so a hand-edited or future-version row is filtered out rather than crashing a `min_trust` recall (ranked pool and the `include_pinned` lane alike) |
| `test_recall_filters.py` | 12 | `source`/`since`/`until` metadata filters pushed into ChromaDB `where` clauses |
| `test_doctor.py` | 11 | **Diagnostics**: `probe_embedding`/`readiness` (incl. embedder-failure, cache TTL, no-cache-on-failure), `GET /ready` (200/503, auth-exempt, sanitized body), `houkai doctor` CLI (text + `--json`) |
| `test_history.py` | 10 | **History / point-in-time**: `history()` timeline (incl. link/supersede counterparts), `state_at` / `get_at` replay, nuke reset, link-delta replay |
| `test_timeparse.py` | 8 | `parse_timestamp`: epoch, ISO-8601, relative spans (`7d`, `24h`), error cases |
| `test_dispatch.py` | 24 | `_dispatch_tool` for all three providers × remember / recall / forget / unknown tool |
| `test_reranker.py` | 7 | **Reranking**: reorders results, promotes below-top-k candidates, per-store + per-call hooks, explain `rerank` block, wrong-length error |
| `test_metrics.py` | 7 | **Runtime metrics**: op counters (incl. link/unlink/restore/purge mutators), recall-latency recording + p50/p95/p99 percentiles, empty-recall counting, backend count |
| `test_curation.py` | 49 | **Curation**: `merge` (text join, outgoing-link transfer, **incoming-link re-pointing**, self-merge + missing-id errors, journaling), `versions` from the journal incl. archives, tag list/rename/merge/delete, `find_path` (undirected, depth cap, no-path, trivial), **trash** put/list/restore/purge and the decay-prune routing |
| `test_corpus_lexical.py` | 17 | **Full-corpus lexical recall**: reaches a candidate outside the vector pool, real-cosine (not fabricated) distance for unioned hits, no duplication of a pool member, metadata/superseded/expired filters still applied, short-token skip + probe cap, punctuation safety, the merged pool keeping its embeddings so MMR/dedup still apply, and the Chroma-native `purge_expired` range query |
| `test_working_set.py` | 36 | **Pinned** tier (packing order, budget interaction, decay/eviction exemption), **trust** tier (default, filtering, packer annotation, ingest boundaries), **idempotent** writes (normalised-text content hash, `remember_many`) |
| `test_mcp_retrieval_knobs.py` | 35 | Every ranking knob reaches MCP `recall`/`recall_pack`/`auto_context` and forwards correctly; omitting them is byte-for-byte the previous behaviour |
| `test_embed.py` | 31 | **Pluggable embedder**: OpenAI-compatible + Ollama backends against a stub server (auth, batching, out-of-order reindexing, count mismatch, wrapped errors), `build_embedder` spec grammar, credentials never from the spec, the store seam + env var + precedence, `FakeEmbedder` determinism/normalisation, and a full store round-trip with no model |
| `test_eval_wiring.py` | 23 | Gold-set parsing (comments, malformed lines, id-prefix resolution, unresolvable id is an error not a zero score), `houkai eval` scores + recorded config + read-only guarantee, the `eval_recall` MCP tool |
| `test_surface_coverage.py` | 23 | Store capabilities that had no remote surface: MCP + HTTP `undo` (newest / by ts / by memory), `restore`, `subgraph`, guarded `nuke`, `journal`, `export`/`import`, and the CLI `history`/`state-at`/`get-at`/`metrics` commands |
| `test_code_style.py` | 31 | House rules enforced mechanically: no banner comments (trailing rules **and** dash-bookended labels, across .py/.go/.sh/.yml/.toml/Makefile/Dockerfile), no import below module top, plus the detectors' own discrimination cases |
| `test_public_get.py` | 9 | `MemoryStore.get()` as public API: plain read (no touch, no journal), returns superseded/expired, async wrapper, deprecated `_get_by_id` alias, MCP `get` tool |
| `test_parity.py` | 8 | The Python surface matches `parity.json` — MCP tool list, HTTP routes, recall knobs — and every tool/route appears in its module docstring |
| `test_fd_hygiene.py` | 1 | ChromaDB file-descriptor reclamation across many store open/close cycles |

### The embedder seam and the fast suite

Most of the suite runs against `ai_houkai.testing.FakeEmbedder`, which hashes
text into a deterministic unit vector. An autouse `conftest.py` fixture patches
`local_embedder` — the single place the store resolves its default — so *every*
store built during a test inherits it: the MCP server's lazy store, CLI runs
under `CliRunner`, the HTTP fixtures, and direct constructions alike.

Exactly **29 tests** are marked `needs_model` and opt back into real
sentence-transformers, because their assertions depend on genuine semantic
similarity: conflict thresholds, reflection clustering, and ranking quality.
Everything else is testing plumbing, where hash vectors are indistinguishable
from real ones. Re-derive the marker set with
`AI_HOUKAI_TEST_REAL_EMBEDDER=1 pytest tests/ -q` — whatever fails needs the
marker.

This is what makes CI viable: the fast job installs the `[test]` extra, which
deliberately omits `sentence-transformers`, and asserts the module is absent so
the promise cannot rot.

### Test isolation strategy

`EphemeralClient()` shares an in-process SQLite database.  All tests
use `PersistentClient` with a `tmp_path`-backed directory:

```python
# tests/conftest.py
@pytest.fixture()
def store(tmp_path) -> MemoryStore:
    s = MemoryStore(path=str(tmp_path / "chroma"), collection="test_memory")
    yield s
    s.client.close()      # PersistentClient leaks FDs without explicit close
```

pytest's `tmp_path` fixture creates a unique directory per test and
cleans it up afterwards.  The fixture `yield`s and then closes the
client — `PersistentClient` holds file handles open until closed, and
without teardown the OS FD limit (~1024) is exhausted after ~100 tests.

### Loading digit-prefixed modules

`importlib.import_module("04_openai")` raises `ModuleNotFoundError`.
`test_dispatch.py` uses `spec_from_file_location` instead:

```python
path = os.path.join(_EXAMPLES_DIR, filename)
spec = importlib.util.spec_from_file_location(module_name, path)
mod  = importlib.util.module_from_spec(spec)
sys.modules[module_name] = mod
spec.loader.exec_module(mod)
```

### SDK stubbing

Agent examples import `openai` and `anthropic` at module level.
Tests inject stubs before loading so no network calls happen:

```python
fake_client = types.SimpleNamespace(
    chat=types.SimpleNamespace(
        completions=types.SimpleNamespace(create=lambda **kw: None)),
    messages=types.SimpleNamespace(create=lambda **kw: None),
)
sys.modules["openai"].OpenAI = lambda **kw: fake_client
sys.modules["anthropic"].Anthropic = lambda: fake_client
```

---

## 12. Memory Linking

Typed directed edges between memories — `Link(to, rel)` stored as a
JSON string in ChromaDB metadata.

### API

```python
store.link(src_id, dst_id, rel="related")    # idempotent
store.unlink(src_id, dst_id, rel=None)       # rel=None → remove all
store.neighbors(memory_id, rel=None,
                direction="both", depth=1)   # BFS, returns [(Memory, rel)]
store.subgraph(memory_ids, depth=1)          # Graph(nodes, edges)
```

`rel` must be one of `LINK_RELS` (§3) and `dst_id` must resolve — both are
validated in the store, so a typo'd relation or a dangling target raises
instead of silently creating an edge no graph walker can follow.

Two memories may be joined by several differently-typed edges;
`neighbors()` reports one `(memory, rel)` pair per edge while still
visiting/expanding each node once. `subgraph()` tracks the best remaining
hop budget per node (not a plain visited set), so diamond shapes expand
fully — a node first reached at the depth limit is re-expanded when a
shorter path reaches it with budget to spare. `unlink()` journals the
removed rels, so undoing a rel=None unlink restores every edge it dropped.

### Supersede (soft-delete)

```python
store.supersede(old_id, new_id)   # marks old as superseded + adds "supersedes" link
store.restore(memory_id)          # undo a supersede
```

`superseded_by != ""` hides a memory from default `recall()` / `list_recent()`.
Pass `include_superseded=True` to see them.

---

## 13. Conflict / Contradiction Detection

### Detection algorithm

```
candidates(a) = recall(a.text, n=12)
for b in candidates:
    if b.superseded_by             → skip
    if b has lapsed (expires_at)   → skip
    if b.type != a.type            → skip
    if sim(a,b) < threshold        → skip   (default 0.80)
    if both have tags & disjoint   → skip   (an untagged side never triggers
                                             the guard)
    if both polarities nonzero
       and opposite                → kind="contradiction", reason="polarity_diff"
    elif negation_diff(a, b)       → kind="contradiction", reason="negation_diff"
    elif contradiction_fn(a, b)    → kind="contradiction", reason="custom_fn"
    else                           → kind="duplicate",     reason="similarity"
```

`negation_diff`: strips apostrophes, tokenises, counts negation words
(`not`, `never`, `no`, `dont`, …), returns True if parity differs.

Superseded and **lapsed** candidates are skipped for the same reason
`find_by_content_hash` skips them: they are hidden from recall, list and stats,
so letting one clash with a new write would reject it under
`on_conflict="raise"` with a conflict the caller cannot see or resolve — and
under `"supersede"` would re-label a row that is already on its way out.

### on_conflict policies

| Policy | Effect on `remember()` |
|---|---|
| `ignore` (default) | no check |
| `warn` | `warnings.warn()` listing conflicts |
| `supersede` | auto-supersede conflicting memories |
| `raise` | raises `ConflictError(conflicts)` |

Any other value raises `ValueError` (see `CONFLICT_POLICIES`, §3) — both in
the constructor's `conflict_policy` and per-call `on_conflict`.

```python
store = MemoryStore(conflict_policy="warn", conflict_threshold=0.85)
store.find_conflicts()                       # global pairwise scan (O(n²))
store.find_conflicts(memory_id=x)            # check one memory
```

---

## 14. Hybrid Retrieval

### Score formula

```
final = α·cosine + β·BM25_local + γ·recency + δ·importance + ζ·polarity
```

| Weight | Default |
|---|---|
| α cosine          | 0.55 |
| β lexical         | 0.20 |
| γ recency         | 0.15 |
| δ importance      | 0.10 |
| ζ polarity_weight | 0.05 |
| η graph           | 0.00 |

![Hybrid blend weights and a worked candidate ranking](resources/hybrid_weights.png)

*Cosine dominates at 0.55; the worked example shows how the additive blend
ranks an exact-and-fresh hit (A, 0.95) above a strong keyword-only match
(C, 0.61) and a stale semantic paraphrase (B, 0.52).*

The polarity term is `ζ · mem.polarity` (so `+ζ` for `polarity=+1`, `−ζ`
for `−1`, nothing for neutral). It is **not** confined to hybrid mode: the
default `mode="semantic"` path also adds the same `ζ·polarity` bonus and
re-sorts its results whenever `polarity_weight ≠ 0` (with `ζ=0` the order is
the pure cosine ranking, unchanged).

`recency = exp(−λ · age_days)` — same λ as `DecayEngine` (default 0.1).
By default `age_days` is measured from `created_at`
(`HybridWeights.recency_basis="created"`), so a memory's recency score is
**stable across recalls** — it reflects how recently the fact was *learned*.
Setting `recency_basis="accessed"` measures from `last_accessed` instead,
restoring the older retrieved-recency behaviour, which is self-reinforcing
because every recall hit `_touch`-bumps `last_accessed`.

**BM25 is computed locally** over the cosine over-fetch pool only — no
second index, O(1) additional storage.

![BM25 term-frequency saturation for k1 = 1.0/1.5/2.5](resources/bm25_saturation.png)

*BM25's `k1 = 1.5, b = 0.75` saturate term frequency: the 10th occurrence of a
term adds far less than the 1st, unlike a linear tf count.*

#### Graph-proximity fusion (`η graph`)

Off by default (`graph = 0.0`, a byte-for-byte no-op). When set, a candidate's
**connectedness to the other strong hits in the pool** lifts its score — a
lightweight, HippoRAG-style associative signal. `_graph_spread` runs
personalised-PageRank-lite (`damping = 0.5`, 3 iterations) over the links
*within the candidate pool only*, treated as **undirected** (so both a memory's
outgoing links and their reverse are followed), seeded by each candidate's
min-max-normalised base relevance. The spread is min-max normalised to `[0, 1]`
and folded in per fusion mode: an additive `η · spread` term for the weighted
blend, and a rank-transformed extra signal for RRF (so it stays scale-free).
Restricting the walk to the pool keeps it `O(pool · links)` — no full-store scan
and no per-node `neighbors()` call. Returns nothing (and the term is skipped)
when the pool has no internal edges.

> **Scope.** Like the other `HybridWeights` terms, `graph` only takes effect in
> `mode="hybrid"` — a default `semantic` recall ignores custom weights. And both
> `graph` and `ExpandSpec.rerank` are reachable from every surface: the
> library, `POST /recall` / `POST /recall_pack` (a `graph` field and a nested
> `expand` object), and the MCP `recall` / `recall_pack` tools (a `graph`
> number plus flat `expand_*` parameters — MCP tool schemas are flat, so the
> nested object is spelled one parameter per field). `parity.json` asserts the
> full knob list against both ports (§23).

#### Gated graph expansion (`ExpandSpec.rerank`)

Graph-walk expansion (below) normally *appends* neighbours after the top-`k`
cut, where they bypass the relevance floor, dedup and diversity selection and
can overflow `k`. Setting `ExpandSpec.rerank=True` instead **merges** the
expanded neighbours into the candidate pool *before* `min_cosine` / dedup / MMR
/ top-`k`, so they compete for the `k` slots on equal footing and can neither
inject near-duplicates nor exceed `k`. Expanded nodes carry no query-embedding,
so they are re-fetched/re-embedded for MMR/dedup; `min_cosine` (a query-relevance
gate) does not apply to them since they are graph-justified, not cosine-justified.
`rerank=False` (default) preserves the original append-after behaviour.

#### Fusion modes

| `fusion` | Blend |
|---|---|
| `"weighted"` (default) | the weighted sum above (`α·cosine + … + ζ·polarity`). |
| `"rrf"` | Reciprocal Rank Fusion — each signal ranks the pool independently and the fused score is `Σ weight_s / (rrf_k + rank_s)` with `rrf_k=60`; polarity stays a tiny additive nudge. |

RRF is **scale-free** (it consumes only ordinal ranks, so it is immune to
the BM25-vs-cosine magnitude mismatch), but its scores are **pool-relative**
— comparable only across identically-pooled fan-outs, not arbitrary recalls.
That is why `auto_context_pack` (which fans out over several different query
pools and dedupes by score) deliberately does **not** expose `fusion="rrf"`.

![Reciprocal Rank Fusion contributions per signal](resources/rrf_fusion.png)

*RRF sums `weight / (60 + rank)` across signals. Scores are tiny by design —
only the relative order matters — which is exactly what makes it immune to the
BM25-vs-cosine magnitude mismatch the weighted blend has to normalise around.*

#### Re-selection, dedup, and gating

- `diversity` (0..1) re-selects results with Maximal Marginal Relevance
  (`_mmr_select`): `score = λ·relevance − (1−λ)·max_cosine_to_selected`,
  higher → more relevance, lower → more novelty. Relevance is min-max
  normalised to [0,1] so the trade-off is on the same scale as the cosine
  novelty penalty (critical for the tiny RRF scores).
- `dedup_threshold` hard-drops a candidate whose cosine to an
  already-selected result is ≥ the threshold (e.g. `0.92`).
- `min_cosine` is an **absolute** cosine floor; candidates below it are
  dropped so a caller can receive *nothing* rather than weak hits. Out of
  the `[-1, 1]` range → `ValueError` (as do out-of-range `diversity` /
  `dedup_threshold`).
- `touch=False` makes recall read-only (no access-count / `last_accessed`
  bump) — e.g. for evaluation.
- `explain=True` returns `(Memory, score, breakdown)` triples.
- `diversity` / `dedup_threshold` also apply in `mode="semantic"`.

![MMR relevance-versus-novelty trade-off and the crossover λ](resources/mmr_tradeoff.png)

*MMR's `λ · relevance − (1 − λ) · redundancy`: below the crossover λ a novel,
slightly-less-relevant item beats a near-duplicate of something already
selected; above it, raw relevance wins.*

#### Lexical renormalisation

When a query yields **zero** BM25 across the whole pool (e.g. an
all-stopword or out-of-vocabulary query), the lexical weight is dropped and
the remaining core weights (`cosine`, `recency`, `importance`) are
renormalised so scores are not artificially depressed — unless the config is
lexical-only (nothing to scale the weight into), in which case the weights
are left untouched.

#### CJK / Korean tokenization

`_tokenize` emits character **bigrams** for non-Latin runs (Hiragana,
Katakana, CJK ideographs, and Hangul), so BM25 and the Jaccard similarity
used elsewhere still produce a lexical signal for CJK/Korean text that has no
whitespace word boundaries.

### API

```python
store.recall(query, k, mode="hybrid")                    # default weights
store.recall(query, k, mode="hybrid",
             weights=HybridWeights(cosine=0.8, lexical=0.1,
                                   recency=0.05, importance=0.05))
store.recall(query, k, overfetch=6)                      # larger cosine pool

# Ranking controls
store.recall(query, k, fusion="rrf")                     # Reciprocal Rank Fusion
store.recall(query, k, diversity=0.7, dedup_threshold=0.92)  # MMR + near-dup drop
store.recall(query, k, min_cosine=0.2)                   # absolute relevance gate
store.recall(query, k, touch=False, explain=True)        # read-only + breakdowns
store.recall(query, k, reranker=my_cross_encoder)        # second-stage rerank
store.recall(query, k, include_expired=True)             # keep TTL-expired hits
store.recall(query, k, mode="hybrid",
             weights=HybridWeights(graph=0.15))          # graph-proximity fusion
store.recall(query, k, mode="hybrid", lexical_index="corpus") # full-corpus lexical (§25)
store.recall(query, k, min_trust="trusted")              # provenance floor (§27)

# Graph-walk expansion after scoring (multi-hop BFS)
store.recall(query, k,
             expand=ExpandSpec(rels=("refines","example_of"),
                               depth=2, cap=5, score=0.70, decay=0.8))
#   hop-h neighbour scored score·decay**(h-1); decay=1.0 (default) = old
#   distance-independent behaviour; depth>1 now does a real BFS over links.
store.recall(query, k,
             expand=ExpandSpec(rels=("refines",), cap=5, rerank=True))
#   rerank=True merges expanded nodes into the pool BEFORE dedup/MMR/top-k
#   (they compete for the k slots); rerank=False (default) appends after.

# Metadata filters — provenance + creation-time window
store.recall(query, k, source="git", since=ts_a, until=ts_b)
```

Default `mode="semantic"` is unchanged — zero risk for existing callers.

The `explain=True` breakdown shape **varies by scoring path**:

| Path | Breakdown keys |
|---|---|
| weighted hybrid | `cosine`, `lexical`, `recency`, `importance`, `polarity`, `weights`, `score` (plus `graph` + `weights.graph` when `graph > 0`) |
| `fusion="rrf"`  | `signals` (per-signal `{rank, contribution}`, incl. `graph` when `graph > 0`), `polarity`, `rrf_k`, `score` |
| semantic        | `cosine`, `polarity`, `score` |
| graph-expanded  | `{source:"graph_expansion", rel, hop, score}` |

Both servers surface `explain` on request (HTTP `?explain=true` / body
`{"explain": true}`; MCP `explain: true`), attaching the breakdown under an
`explain` key on each hit.

#### Second-stage reranking

The blended score is a cheap first stage. A **reranker** — a stronger, usually
cross-encoder relevance model — can rescore the candidate pool before the
top-`k` cut, the standard retrieve-then-rerank pattern (Elasticsearch's
`rescore`, ColBERT, cross-encoders). It is a pluggable hook, not a bundled
model, so the core stays dependency-free:

```python
Reranker = Callable[[str, list[Memory]], list[float]]   # query, pool → one score/mem

MemoryStore(..., reranker=fn)          # per-store default
store.recall(query, k, reranker=fn)    # per-call override (wins over the default)
```

The reranker receives the query and the surviving first-stage pool (so raise
`overfetch` to give it more to work with) and returns one score per memory in
order; `recall()` re-sorts by those scores — the reranker's score **replaces**
the blended score — then applies the top-`k` cut. Under `explain=True` each
hit gains a `rerank` block recording `first_stage_score` / `first_stage_rank`
and the new `score` / `rank`, so a promotion is auditable. A reranker is a live
callable, so it is a **library / server-config** concern — it cannot cross the
JSON boundary, and the servers simply honour whatever reranker their store was
built with (no per-request reranker parameter).

#### The recall pipeline

Where each stage from §5 sits — and the exact insertion points for expiry
filtering, reranking, and explain:

```
  query
    │  embed
    ▼
 ┌──────────────────────┐   over-fetch  k · overfetch  (default 4×)
 │  ChromaDB HNSW query  │──────────────────────────────┐
 └──────────────────────┘                               ▼
                                       ┌───────────────────────────────────┐
   where-clause pool  ◄────────────────│ type · min_importance · source ·  │
   (type/imp/src/time)                 │ since · until                     │
                                       └───────────────┬───────────────────┘
                                                       ▼
                                   tag · superseded · min_cosine  (in scorer)
                                                       ▼
                          ┌────────────────────────────────────────────┐
                          │ score: _hybrid_score / _rrf_score /         │──► explain{}
                          │        _semantic_filter                     │    (per hit,
                          └────────────────────────┬───────────────────┘     if explain)
                                                    ▼
                          drop expired  (unless include_expired)   ◄── §21 TTL
                                                    ▼
                          reranker(query, pool)  →  re-sort          ◄── §14 rerank
                                                    ▼
                          MMR / dedup  (if diversity|dedup)  ELSE  top-k
                                                    ▼
                          _touch_many  (unless touch=False)
                                                    ▼
                          graph expand  (if expand)  →  append neighbours
                                                    ▼
                          [(Memory, score)]  or  [(Memory, score, explain)]
```

#### Metadata filters

`recall` (and `recall_pack`) accept `source`, `since` and `until` alongside
the existing `type` / `tag` / `min_importance`. `source` matches the exact
provenance string set at `remember` time; `since`/`until` are Unix timestamps
bounding `created_at` (inclusive). `type`, `min_importance`, `source`, `since`
and `until` push down into Chroma's `where` clause; `tag` is still matched
post-query against the comma-joined tag list.

ChromaDB ≥ 1.x rejects a multi-key `where` (`{"a":1,"b":2}`) and a
multi-operator leaf (`{"created_at":{"$gte":x,"$lte":y}}`) — each leaf must
carry exactly one operator, with conjunctions expressed via an explicit
`$and`. `store._build_where()` centralises this: it returns `None` for no
filters, a flat single-condition clause for one, and an `$and` of
single-operator leaves otherwise (the `since`/`until` range becoming two
separate `created_at` leaves). This also fixed a latent bug where combining
`type` with `min_importance` previously emitted a rejected two-key clause.

The user-facing layers parse human time inputs through
`ai_houkai/timeparse.py::parse_timestamp`, which accepts epoch seconds, an
ISO-8601 date/datetime (naive values read as UTC, trailing `Z` honoured), or a
relative span like `"7d"` / `"24h"` (→ now minus that span).

---

## 15. Extension Points

> The designs in [PROPOSALS.md](https://raw.githubusercontent.com/nexusriot/AI-Houkai/main/PROPOSALS.md) are now implemented (§12–14).
> Hybrid retrieval scoring is built in (§14, `mode="hybrid"`), and scheduled
> maintenance shipped as the `ai_houkai.maintenance` subsystem (§17) — no
> hand-rolled threads needed.  The sketches below remain valid as quick
> recipes for further customisation.

### Multi-user / multi-agent

Each `MemoryStore` targets a single collection.  Isolate agents by
passing distinct collection names:

```python
from ai_houkai.memory_system import MemoryStore

alice = MemoryStore(path=".chroma", collection="agent_alice")
bob   = MemoryStore(path=".chroma", collection="agent_bob")
```

### Pluggable embeddings

```python
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

store.collection = store.client.get_or_create_collection(
    name="ai_houkai",
    embedding_function=OpenAIEmbeddingFunction(
        api_key=os.environ["OPENAI_API_KEY"],
        model_name="text-embedding-3-small",
    ),
    metadata={"hnsw:space": "cosine"},
)
```

### LLM reflection summarizer

Built in — see §7 (`build_summarizer("ollama:llama3.1")` /
`openai:…` / `anthropic:…`, configured via
`[maintenance.reflect].summarizer`).  For anything beyond the three
providers, pass any callable:

```python
from ai_houkai.memory_system import ReflectionEngine

def my_summarizer(memories):
    prompt = "\n".join(f"- {m.text}" for m in memories)
    return call_llm(f"Distil these events into one insight:\n{prompt}")

engine = ReflectionEngine(store, summarizer=my_summarizer)
```

### Importance auto-assignment

Built in — `ai_houkai/memory_system/importance.py` implements the tiers
below as a deterministic regex heuristic (`score_importance(text, type,
tags)`), wired via `MemoryStore(importance_fn=…)`, the CLI
(`--auto-importance` or `default_importance = "auto"`), and the MCP
server (`AI_HOUKAI_AUTO_IMPORTANCE=1`):

- **High** (0.9): explicit user instructions, corrections, preferences
- **Medium-high** (0.75): decisions, conventions, policies
- **Medium** (0.6): task completions, durable project facts
- **Low** (0.35): hedged/passing observations
- Modifiers: +0.10 procedural/feedback, −0.15 questions, −0.10 fragments

![Heuristic importance tiers with example phrases](resources/importance_tiers.png)

*Each bar is the score the shipped `score_importance()` actually returns for
the quoted phrase; the highest matching tier wins.*

![Importance base tier through modifiers to the final clamped value](resources/importance_waterfall.png)

*Base tier → modifiers → final, clamped to `[0.05, 0.98]`. A procedural
instruction saturates at the 0.98 ceiling (0.90 tier + 0.10 type bonus), while
the short question "Slow?" drops to 0.25 (−0.15 question, −0.10 fragment).*

Still open as an extension: **LLM-based** scoring (ask the model to rate
0–1 before storing) — pass any callable as `importance_fn`.

---

## 16. CLI — houkai

The `houkai` command gives a human operator direct terminal access to
the same `MemoryStore` that agents and MCP clients use.  It is an
optional dependency (`pip install "ai-houkai[cli]"`) so the core memory
library stays dep-light.

### Dependency strategy

```toml
# pyproject.toml
[project.optional-dependencies]
cli = ["typer>=0.12", "rich>=13.7"]

[project.scripts]
houkai = "ai_houkai.cli.main:_main"
```

`typer` and `rich` are not imported at the module level of any
`memory_system` or `mcp_server` code — the CLI subtree is the only
consumer.  A `try/except ImportError` guard at the entrypoint prints a
friendly install hint if the extras are absent.

### Architecture

```
houkai (bin)
  └── ai_houkai/cli/main.py          Typer app; registers 39 commands plus
                                     five sub-command groups (maintenance,
                                     journal, collections, tags, trash);
                                     shared --store / --collection flags
        ├── config.py                Config resolution chain:
        │                             1. --store / --collection CLI flags
        │                             2. AI_HOUKAI_PATH / AI_HOUKAI_COLLECTION env
        │                             3. ~/.config/ai_houkai/config.toml
        │                             4. ~/.ai_houkai/.chroma  /  ai_houkai
        ├── output.py                Output layer:
        │                             - rich Table (TTY)
        │                             - TSV (non-TTY / pipe)
        │                             - JSON (--format json)
        │                             - id prefix resolution (8-char → UUID)
        │                             - fmt_age(), fmt_importance() helpers
        └── commands/
              *.py                   One module per logical group;
                                     each file exports plain functions
                                     registered in main.py via
                                     app.command("name")(fn)
```

All command functions take `ctx: typer.Context` as their first
parameter.  The shared callback stores `{"store": MemoryStore, "config":
Config}` in `ctx.obj` before any subcommand runs.

### Command inventory

| Command | Module | Wraps |
|---|---|---|
| `remember` | `remember.py` | `store.remember()` |
| `recall` | `recall.py` | `store.recall()` |
| `pack` | `pack.py` | `store.recall_pack()` |
| `auto-context` | `pack.py` | `store.auto_context_pack()` — multi-angle fan-out packing |
| `list` | `list_cmd.py` | `store.list_recent()` + Python filters |
| `show` | `show.py` | `store._get_by_id()` |
| `forget` | `forget.py` | `store.forget()` |
| `nuke` | `nuke.py` | `store.nuke()` — deletes all memories; confirms unless `--yes` |
| `edit` | `edit.py` | `store.edit()` in place (re-embeds when text changed; id + links preserved; journaled + undoable) |
| `tag` | `edit.py` | `store.edit(tags=…)` — journaled + undoable |
| `bump` | `edit.py` | `store.edit(importance=…)` — journaled + undoable |
| `link` | `link.py` | `store.link()` |
| `unlink` | `link.py` | `store.unlink()` |
| `neighbors` | `link.py` | `store.neighbors()` |
| `graph` | `link.py` | `store.subgraph()` |
| `conflicts` | `conflicts.py` | `store.find_conflicts()` |
| `supersede` | `conflicts.py` | `store.supersede()` |
| `restore` | `conflicts.py` | `store.restore()` |
| `prune` | `decay.py` | `DecayEngine.prune()` |
| `purge` | `decay.py` | `store.purge_expired()` — hard-delete TTL-expired memories (dry-run by default, `--apply`) |
| `reflect` | `reflect.py` | `ReflectionEngine.reflect()` |
| `export` | `io.py` | `store.list_recent()` → JSONL |
| `import` | `io.py` | JSONL → `store.remember()` |
| `info` | `io.py` | inspect a `.ahkai` archive without importing |
| `backup` | `io.py` | `shutil.copytree(.chroma → backups/<ts>/)` |
| `stats` | `stats.py` | `store.list_recent()` + Counter; `--health` reuses the engine decay formula (incl. frequency reinforcement) with `decay_rate` / `min_score` / `protect_types` / `frequency_weight` loaded from `[maintenance.decay]` config (overridable via `--decay-rate` / `--frequency-weight`) so its at-risk count matches `prune()` |
| `doctor` | `doctor.py` | `store.readiness()` + config/embed-dim/journal checks — active embedder probe (latency + dim), embed-dim guardrail; `--json`; exits non-zero on failure |
| `ingest` | `ingest.py` | `chunk_text()` → one `store.remember()` per chunk |
| `serve` | `serve.py` | `http_server.serve()` — starts JSON HTTP API on `--host`/`--port`, optional `--token` |
| `tui` | `tui_cmd.py` | `HoukaiTui` (Textual; needs the `tui` extra) |
| `maintenance` (group) | `maintenance.py` | `MaintenanceScheduler` — tick/run/start/stop/status |
| `journal` (group) | `journal.py` | `Journal` — tail/show/undo |
| `collections` (group) | `collections.py` | Chroma client — list/create/delete/copy (embeddings copied verbatim) |

### Output system

`output.py` selects the renderer at call time (not import time) by
checking `sys.stdout.isatty()` and `NO_COLOR`:

```
stdout isatty + no NO_COLOR  →  rich Table  (colour, box-drawing)
stdout not a TTY              →  TSV         (tab-separated, machine-readable)
--format json                 →  JSON array  (explicit override)
```

Every table row uses an 8-char id prefix.  Full UUIDs are accepted on
input; `resolve_id_prefix()` performs a linear scan of all memories and
raises `ValueError` on ambiguous or missing prefixes.

### Safety rules

- **Destructive commands** (`forget`, `nuke`, `prune --apply`, `reflect --apply
  --consolidate hard`, `import`) confirm interactively unless `--yes`.
- `nuke` deletes **all** memories in the active collection in one call;
  it shows the count before prompting, and returns the number deleted.
- `prune` and `reflect` default to **dry-run** — matching the underlying
  engine conventions — and require an explicit `--apply` flag to write.
- `tag` and `bump` update only metadata (via `store.edit()`); the embedding
  vector is unchanged, keeping the semantic index consistent.
- `edit` updates the record in place via `store.edit()`, keeping the same
  id and all links; the text is re-embedded only when it changed. All three
  curation commands are journaled (op `edit`) and reversible with
  `houkai journal undo`.
- **Clean errors**: store validation failures (bad enum value, unknown id,
  self-link, supersede cycle) surface as a one-line `Error: …` + exit 1 via
  `output.friendly_errors()`, never a traceback. `remember --on-conflict
  raise` hitting a real conflict prints the conflicts and exits — the
  policy outcome, not a crash.

### Config file

`~/.config/ai_houkai/config.toml` (read with `tomllib`, stdlib ≥ 3.11):

```toml
store_path         = "~/.ai_houkai/.chroma"
collection         = "ai_houkai"
default_type       = "semantic"
default_importance = 0.5
editor             = "nvim"    # fallback: $EDITOR env, then "nano"
```

### Testing strategy

`tests/test_cli.py` uses Typer's `CliRunner` — no subprocess, no disk
I/O outside `tmp_path`.  Each test creates a fresh isolated store via
`--store tmp_path/chroma --collection cli_test`.

HF Hub model load warnings are emitted to stdout through the runner;
UUID extraction uses a regex (`_first_uuid()`) rather than assuming the
output is only the UUID, making tests robust to logging noise.

---

## 17. Scheduled Maintenance

Decay and reflection only matter if they run regularly.  The
`ai_houkai.maintenance` subpackage orchestrates both on a schedule
without any external dependency (no APScheduler, no celery).

### Components

```
maintenance/
├── durations.py   parse_duration("24h") → seconds; format_duration() back
├── state.py       MaintenanceState — JSON file with last-run timestamps
│                  and bounded run history; next_run_at() schedule math
├── scheduler.py   MaintenanceScheduler — tick() + run_forever(stop_event)
│                  TickResult — per-job ran/count/error + summary()
└── daemon.py      PID-file helpers (get/write/remove/is_alive/stop)
                   + spawn_detached() for `houkai maintenance start`
```

### Tick semantics

`tick()` is the single unit of work and is **idempotent with respect to
the schedule**: each job (decay, reflect, **purge**) runs only if its
configured interval (`decay_every`, default 24 h; `reflect_every`, default
7 d; `purge_every`, default 24 h) has elapsed since the last run recorded in
`MaintenanceState`.
Callers may therefore invoke it as often as they like — from cron, from
the foreground loop (`tick_interval`, default 5 min), from an MCP client
via the `maintenance_tick` tool, or ad hoc.

The **purge job** reclaims TTL-expired memories: it calls
`store.purge_expired()` (§21), stamping `last_purge_at` and accumulating
`total_purged` in `MaintenanceState` (both fields default to 0, so older state
files load unchanged). It is safe to run always-on: expired memories are a new
concept, so an existing store has none, and the job only ever deletes memories
whose TTL has already passed.

**Trash retention rides the same job**: after a successful purge the tick calls
`trash_purge_expired(trash_ttl_days)` (§26, default 30; `<= 0` disables it) and
accumulates `total_trash_purged`. This matters more now that decay pruning
routes into the trash — without retention the file would grow with every prune.
Both ports report the sweep as `trash_purged` on the tick result and in the
one-line summary; the Go port reads it from `trash_ttl_days` in
`[maintenance]`.

**Concurrent tickers serialise.** The whole load→run→save cycle holds an
exclusive `flock` on `<state file>.lock`, so the daemon loop, a cron tick,
and the MCP tool can share one state file without double-running a job or
clobbering each other's timestamps: a blocked ticker waits, re-loads the
fresh state, and sees the job as no longer due. (On non-POSIX platforms
without `fcntl`, ticks run unlocked as before.)

**The schedule gates the work, not the writes.** A reflection run stamps
`last_reflect_at` even in dry-run mode: clustering is O(n²) and the
summarizer may call an LLM, and both costs are paid on a dry-run too.
(Before this rule, a dry-run-configured caller re-ran full reflection —
LLM call included — on *every* tick, forever.) `TickResult.reflect_applied`
records which mode ran, and `summary()` says `reflect would create N
(dry-run)` vs `reflect created N`. Totals (`total_reflected`) only count
persisted summaries.

Errors in one job never block the others; they are captured in
`TickResult.decay_error` / `reflect_error` / `purge_error` and surfaced in
`summary()`. A failed job does **not** advance its schedule — it retries next
tick.

### Three deployment modes

| Mode | Command | Mechanism |
|---|---|---|
| cron one-shot | `houkai maintenance tick` | synchronous tick, exit |
| foreground loop | `houkai maintenance run` | `run_forever()` until SIGINT/SIGTERM |
| background daemon | `houkai maintenance start/status/stop` | `spawn_detached()` + PID file, logs to `~/.ai_houkai/maintenance.log` |

Configuration lives in the `[maintenance]` section of
`~/.config/ai_houkai/config.toml` (see README for the full key list);
`reflect.apply` defaults to **false** so reflection observes without
writing until explicitly enabled, and `reflect.summarizer` selects an
LLM summarizer spec (§7) for the daemon's reflection runs.

---

## 18. Audit Journal

Every mutation of the store is appended to a JSONL journal
(`journal.log` next to the Chroma directory) by
`ai_houkai/memory_system/journal.py`.

### Entry format

One compact JSON object per line:

```json
{"ts": 1748284800.123, "op": "supersede", "actor": "reflection",
 "id": "72be7903-…", "before": {…}, "after": {…}, "meta": {"new_id": "…"}}
```

- **op** — `remember | forget | edit | supersede | restore | link | unlink |
  reflect | decay | import | export | undo`
- **actor** — who performed it: `cli` / `mcp` / `http` / `reflection` /
  `decay` / `import` / `lib`
- **before / after** — full memory snapshots where applicable; these are
  what make `undo` possible.

### Design properties

- **Best-effort writes** — a journal failure is logged, never raised; the
  memory operation itself always wins.
- **Crash-safe appends** — `O_APPEND` line writes are atomic on POSIX
  (≤ PIPE_BUF); `read()` skips a truncated trailing line.
- **Size-based rotation** — file size is checked every 256 appends;
  beyond `rotate_mb` (64 MB) the log is gzipped to a timestamped archive,
  and archives older than `keep_days` (90) are pruned.

### Undo

`MemoryStore.undo(entry)` reverses a single entry where possible
(`remember` → forget, `forget` → re-add from the `before` snapshot,
`edit` → restore the `before` snapshot (re-embedding the text),
`supersede`/`restore`/`link`/`unlink` → inverse op).  It refuses rather
than clobbers: e.g. undoing a `forget` fails if the id already exists
again, and undoing an `edit` or `unlink` fails if the memory (or a link
endpoint) has since been forgotten.  Every successful undo is itself
journalled with `op="undo"`.

### History &amp; point-in-time queries

Because every mutating op writes a full `before`/`after` snapshot, the journal
doubles as a **time machine** for read-only auditing:

```python
store.history(memory_id)          # → [JournalEntry] touching this memory, oldest→newest
store.state_at(timestamp)         # → [Memory] the whole store, reconstructed as of t
store.get_at(memory_id, t)        # → Memory | None — one memory as of t
```

`history()` returns every entry that concerns a memory — as the op's subject
(`entry.id`) **and** as a counterpart recorded only in `meta`: a link target
(`dst_id`), the superseding memory (`new_id`), or a restore's `superseder_id`.
So `B`'s history shows the `link A→B` even though that entry is filed under `A`.

`state_at()` / `get_at()` **replay** the log up to `t`: `remember`/`import`/
`edit`/`supersede`/`restore` set the memory to their `after` snapshot,
`forget` deletes it, `link`/`unlink` apply the meta delta, and `nuke` clears
everything. It is a **best-effort audit tool, not an event-source of record**:
it sees only what was journaled, so it is accurate back to the oldest retained
archive (rotation prunes past `keep_days`), a `nuke` in the window resets the
reconstruction (it snapshots nothing), memories written with the journal
disabled are invisible, and reconstructed memories carry no embedding.

```
journal (oldest → newest, replayed until t):
  remember A {v1}   edit A {v2}   remember B   link A→B   forget B   nuke
  └─ state[A]=v1    └─ state[A]=v2  └ state[B]  └ A.links  └ del B   └ state={}
                    ▲                                                  ▲
              state_at(t₁) = {A:v2, …}                         state_at(t₂) = {}
```

Both are exposed over HTTP (`GET /memories/{id}/history`, `GET /state_at?ts=`,
`GET /memories/{id}/at?ts=`) and MCP (`history`, `state_at`, `get_at`).

---

## 19. Portable Import / Export

`MemoryStore.export()` / `import_()` move memories between stores via
the **`.ahkai`** format: gzipped JSONL with a header object on line 1
(format name, version, source store/collection, embedding model,
options) followed by one memory per line.

### Export

- Streams in `created_at` ascending order, so two exports of the same
  store are byte-identical modulo the header timestamp.
- Embeddings are included by default (`include_vectors=True`) so the
  importing side can skip re-running the model; `--no-vectors` shrinks
  the file when portability matters more than speed.
- Filterable by `types`, `tags`, `since`, `include_superseded`.

### Import

- **Conflict policies** on id collision: `skip` (default) ·
  `overwrite` · `rename` (new UUID) · `error`.
- **Model safety**: if the archive's embedding model differs from the
  importing store's, the import raises unless
  `regenerate_vectors=True`, which re-embeds text on the way in.
- **dry_run** previews counts without writing; all real imports are
  journalled with `actor="import"`.

`houkai info dump.ahkai` prints the header without touching the store,
and `houkai backup` snapshots the raw Chroma directory for
disaster recovery (orthogonal to `.ahkai`, which is portable data).

---

## 20. HTTP / REST API

`ai_houkai/http_server/` exposes the same `MemoryStore` over a small JSON
HTTP API, for clients that cannot speak MCP — web apps, shell scripts,
automation tools, non-MCP agents. It is **standard-library only**
(`http.server.ThreadingHTTPServer`), so it adds no dependency beyond the
core memory layer, mirroring the project's lean-deps stance (cf. the stdlib
`urllib` Ollama summarizer in §7).

### Surface

| Method & path | Store call |
|---|---|
| `GET /health` | `count()` — always reachable (liveness, skips auth); returns only `{status, count}`, omitting the collection name |
| `GET /ready` | `readiness()` — backend + embedder probe; **200** when ready, **503** otherwise; skips auth like `/health`. Body is deliberately minimal (overall flag + per-check `ok` only — no error strings / dim / latency / paths), and the probe is cached ~5 s so rapid polling can't hammer a billed remote embedder |
| `GET /stats` | path / collection / count |
| `GET /metrics` | `metrics()` — runtime counters + recall latency (§22) |
| `GET /memories` | `list_recent(limit, include_superseded, include_expired)` |
| `POST /memories` | `remember(...)` incl. `ttl_seconds` / `expires_at` → 201, or 409 on `ConflictError` |
| `POST /memories/batch` | `remember_many(items, batch_size?, on_conflict?)` → 201 `{stored, memories}`; 400 on a bad item or `on_conflict="raise"` |
| `GET /memories/{id}` | `_get_by_id` → 404 if absent |
| `PATCH /memories/{id}` | `edit(...)` incl. `expires_at` — in-place, journaled; 404 if absent, 400 on bad field |
| `DELETE /memories/{id}` | `forget` → 404 if absent |
| `GET /memories/{id}/neighbors` | `neighbors(rel, direction, depth)` |
| `GET /memories/{id}/history` | `history(id)` → journaled timeline; 404 if id unknown (§18) |
| `GET /memories/{id}/at?ts=` | `get_at(id, ts)` → 404 if it didn't exist then (§18) |
| `GET /state_at?ts=` | `state_at(ts)` → all live memories reconstructed as of `ts` (§18) |
| `POST /purge_expired` | `purge_expired(dry_run?)` → count + ids (§21) |
| `GET\|POST /recall` | `recall(...)` incl. `source`/`since`/`until`/`include_expired`/`explain`; the POST body also carries the advanced knobs (`fusion`, `diversity`, `dedup_threshold`, `min_cosine`, `graph` weight, and an `expand` object incl. `rerank`) that don't map onto a query string |
| `POST /recall_pack` | `recall_pack(...)` incl. `graph` weight + `expand` (rerank) |
| `POST /links` · `POST /unlink` | `link` / `unlink` |
| `POST /supersede` · `POST /conflicts` | `supersede` / `find_conflicts` |

The stable `/recall` parameter set is `query`, `k` / `token_budget`, `type`,
`tag`, `min_importance`, `source`, `since`, `until`, `mode`,
`include_superseded`, `include_expired`, and `explain`. The heavier ranking
and compression knobs (`fusion`, `diversity`, `dedup_threshold`, `min_cosine`,
`expand`, and `recall_pack`'s `compress*` group) are **not** exposed over
HTTP; a configured `reranker` still applies (it is store config, not a request
param). Callers needing the full ranking surface use the Python API or MCP.

### Design

- **Framework-free routing.** A single `_ROUTES` table of
  `(method, compiled_regex, handler, needs_body)` rows. Each handler is a
  plain function `(store, match, query, body) -> (status, payload)`. One
  `_dispatch` method serves every verb (`do_GET = do_POST = … = _dispatch`).
  `HEAD` is dispatched as `GET` with the response body suppressed, so
  `HEAD /health` liveness probes work. A path that matches a row but not
  the verb returns `405`; no match is `404`.
- **Input coercion.** Query-string params go through `_as_int` / `_as_float`
  / `_as_bool`; JSON-body params go through the matching `_body_int` /
  `_body_float` / `_body_bool` twins, which accept JSON-native types and
  numeric strings and raise `HttpError(400)` on anything else (a JSON
  `"false"` is falsy, `null` falls back to the default). Store-level
  validation errors (bad `mode` / `type` / `rel` / `polarity` /
  `on_conflict`, self-link, supersede cycle) surface as
  `400 {"error": "<message naming the allowed values>"}`; unknown ids are
  `404`.
- **Errors.** Handlers raise `HttpError(status, message)` for expected
  failures (400 bad input, 404 missing, 413 oversized body); any other
  exception is caught and rendered as
  `500 {"error": "internal server error", "request_id": "<hex>"}` so a single
  bad request never takes the server down. The exception type, message, and
  traceback are deliberately **not** leaked to the client — only the
  `request_id` is, so a failure can be correlated to a server-side log.
  Request bodies are capped at 4 MiB and parsed as a JSON object.
- **Auth.** Optional bearer token (`auth_token` arg / `AI_HOUKAI_HTTP_TOKEN`).
  When set, every route except `/health` requires
  `Authorization: Bearer <token>`, compared with `hmac.compare_digest` over
  UTF-8 **bytes** (constant-time, so a wrong token can't be recovered by
  timing; the str overload would raise on a non-ASCII header and kill the
  request thread instead of returning 401). Binds `127.0.0.1` by default;
  exposing `0.0.0.0` is left to a deliberate `--host` / env override behind a
  trusted network or reverse proxy.
- **Attribution.** The server sets the store's actor to `"http"`, so journal
  entries from API writes are distinguishable from `cli` / `mcp` / `lib`.

### Entry points

- `houkai serve [--host --port --token]` — reuses the CLI-selected store.
- `ai-houkai-serve` console script → `http_server.server:run`, configured
  purely by env (`AI_HOUKAI_PATH`, `AI_HOUKAI_COLLECTION`,
  `AI_HOUKAI_HTTP_HOST`, `AI_HOUKAI_HTTP_PORT`, `AI_HOUKAI_HTTP_TOKEN`) so it
  needs no CLI extras — symmetric with `ai-houkai-mcp`.
- `make_server(...)` / `serve(...)` / `build_handler(...)` for embedding the
  API in another process or test (tests drive a real server on port 0).

---

## 21. Memory Expiry (TTL)

Some memories are useful only for a bounded window — a sprint's context, a
session token, a "remind me until Friday" note. **Expiry** is an explicit,
absolute deadline that complements the statistical `DecayEngine` (§6): decay is
a soft, importance-and-recency curve; a TTL is a hard "gone after this instant."

### Data model

One field, `Memory.expires_at` (epoch seconds, `0` = never), serialised
alongside the other scalar metadata and read with a safe default so pre-TTL
records load as "never expires" (§3). A memory is **expired** when
`expires_at != 0 and expires_at <= now`.

### Setting a TTL

```python
store.remember(text, ttl_seconds=3600)      # relative — now + 1 h
store.remember(text, expires_at=epoch)       # absolute
store.edit(id, expires_at=epoch)             # set/extend; expires_at=0 clears it
```

`ttl_seconds` and `expires_at` are mutually exclusive (passing both, or a
non-positive `ttl_seconds` / negative `expires_at`, raises `ValueError`).

### Two-stage removal — hide, then reclaim

Expiry is a **soft-delete that mirrors supersede** (§12), not an immediate
`forget`:

1. **Hidden** — once `expires_at` passes, the memory is excluded from
   `recall()` and `list_recent()` by default (post-filter, so old records are
   unaffected; §5), overridable with `include_expired=True`. It is still in the
   store and fetchable by id (`GET /memories/{id}`, `get_at`), so nothing is
   lost the instant a deadline slips past.
2. **Reclaimed** — `store.purge_expired(now=None, dry_run=False)` hard-deletes
   every expired memory, returning those purged, each with a per-row journaled
   `forget` (actor `"purge"`) so it is auditable and individually undoable.
   Unlike decay's `prune()`, purge **ignores `protect_types`** — an explicit
   TTL is a stronger, user-set signal than the decay heuristic. The scheduled
   maintenance tick runs it on the `purge_every` cadence (§17); `houkai purge`
   (dry-run by default, `--apply` to delete) and the `purge_expired` MCP tool /
   `POST /purge_expired` endpoint expose it on demand.

### Surface

| Layer | Set TTL | Hide/show | Reclaim |
|---|---|---|---|
| Python | `remember(ttl_seconds=/expires_at=)`, `edit(expires_at=)` | `recall/list_recent(include_expired=)` | `purge_expired()` |
| MCP | `remember`, `edit` args | `recall`, `list_recent` `include_expired` | `purge_expired` tool |
| HTTP | `POST/PATCH` `ttl_seconds`/`expires_at` | `?include_expired=` | `POST /purge_expired` |
| CLI | `houkai remember --ttl <s>` | `--include-expired` | `houkai purge [--apply]` |

---

## 22. Runtime Metrics

`stats()` reports **content** aggregates (counts, type/tag breakdowns) computed
from the store's rows. `metrics()` is the complementary **operational** view —
process-local counters and recall latency accumulated since the store object
was created:

```python
store.metrics()
# {
#   "uptime_seconds": 1234.5,
#   "count": 812,                         # live backend count
#   "calls": {"remember": 40, "recall": 210, "forget": 3, "edit": 12,
#             "supersede": 5, "link": 18, "unlink": 2, "restore": 1,
#             "purge_expired": 4, "export": 1, "import": 1},
#   "recall_latency_ms": {"count": 210, "avg": 8.3, "max": 141.2,
#                         "p50": 6.1, "p95": 22.7, "p99": 58.4},
# }
```

Counters are bumped inside every mutating op — `remember` / `recall` / `forget`
/ `edit` / `supersede` plus `link` / `unlink` / `restore` / `purge_expired` /
`export` / `import`. Recall wraps its body to record wall-clock latency (the
empty-store early return still counts, so the numbers match call volume) and
keeps a bounded ring of the last 1024 samples to derive **p50/p95/p99**
percentiles (exact over the window) alongside `avg`/`max`. The registry is
**in-memory and per-instance** — not persisted, reset on restart, and not
shared across processes — so it is a cheap liveness/throughput signal for a
long-running server, not a durable analytics store. In the Go port it is
guarded by a mutex since handlers run concurrently; in Python the HTTP server's
`store_lock` already serialises access. Exposed as the `metrics` MCP tool and
`GET /metrics`.

---

## 23. Port Parity

The Python and Go ports are expected to expose the same remote surface. That
was previously an aspiration checked by hand, and it drifted: the Python MCP
`recall` silently fell five ranking knobs behind the Go one (`fusion`,
`diversity`, `dedup_threshold`, `min_cosine`, `touch`) and nothing noticed,
because nothing compared them.

[`parity.json`](../parity.json) at the repo root is now the single source of
truth for the MCP tool list, the HTTP routes and the recall knobs. Each port
asserts against it **in its own suite**:

| Port | Assertion |
|---|---|
| Python | `tests/test_parity.py` — introspects `mcp.list_tools()` and the `_ROUTES` table |
| Go | `go/internal/parity/` — calls `tools/list` over the real JSON-RPC handler, and probes each route through `httptest` |

Neither CI job needs the other toolchain, so a surface added to one side fails
that side's build. The manifest lists are kept sorted and unique (itself
asserted), which keeps diffs readable and duplicates impossible.

The Python test additionally checks that every tool and route appears in its
module's docstring inventory — the docstring is the thing agents and readers
actually see, and it had already drifted (`POST /memories/batch` was missing).

**CLI commands are deliberately not asserted.** The two CLIs legitimately
differ — Python ships installers as console scripts, Go has `houkai install`
and `houkai config` — so an equality assertion there would encode a false
claim. Parity is asserted exactly where it is real.

---

## 24. Pluggable Embedding Backends

`MemoryStore` previously hard-wired `SentenceTransformerEmbeddingFunction` with
no override, which made `sentence-transformers` (and therefore torch, ~2 GB
installed) a mandatory dependency. That cost more than disk: no hosted
embedder was reachable from Python at all, and every test had to load a real
model — the Go port's suite ran in under a second because its embedder was
injectable, while Python's took nearly six minutes.

Resolution order, most explicit first:

```
1. embedding_function=          an injected callable
2. AI_HOUKAI_EMBEDDER=          "provider:model" from the environment
3. embedding_model=             the local sentence-transformers default
```

`ai_houkai/embed.py` supplies stdlib-only backends — `OpenAICompatibleEmbedder`
(OpenAI, DigitalOcean, vLLM, llama.cpp's compat server, Ollama's `/v1`) and
`OllamaEmbedder` (the native `/api/embed`) — using `urllib`, so no SDK is
required. `local_embedder()` moved here from `store.py`; `_get_embed_fn`
remains as an alias.

### The Chroma adapter

Chroma's collection API needs more than a callable: the query path calls
`embed_query`, and the config serialiser reads `name()` (**as a staticmethod**)
plus `is_legacy`/`get_config`. Rather than force every caller to subclass
Chroma's Protocol, `as_chroma_embedding_function()` wraps anything that does not
already satisfy it, so `embedding_function=lambda texts: …` just works.

The provider classes deliberately do **not** subclass Chroma's
`EmbeddingFunction`: its `__init_subclass__` rewrites `__call__` to normalise
and validate into numpy arrays, which changes their return type and rejects an
empty batch. They stay plain callables; the adapter provides the protocol at
the boundary. `store._embed_fn` holds the *unwrapped* callable so diagnostics
see plain lists, and `store.embedder_name` reports the backend actually in use
— reading a stale `embedding_model` off a healthy-looking `doctor` is exactly
how an embedder swap goes unnoticed until rankings degrade.

### Packaging consequence

`sentence-transformers` moved from the core dependencies into a `[local]`
extra. `pip install ai-houkai` no longer provides an embedder; `[local]`,
`[all]` and `[dev]` do, and the new `[test]` extra deliberately omits it so the
fast suite cannot silently regain a torch dependency (CI asserts the module is
absent).

---

## 25. Full-Corpus Lexical Recall

`_bm25_score_pool` scores lexical relevance only over the vector over-fetch
pool. The pool is selected by embedding distance alone, so a memory carrying the
query's exact tokens but embedding weakly never enters it — and therefore can
never be surfaced by the BM25 term, at any corpus size.

`recall(lexical_index="corpus")` closes that gap with Chroma's own
`where_document={"$contains": token}` filter, which scans the collection
server-side and works on the ranked query path. `_lexical_candidates` probes the
four longest query tokens (longest = most selective, and each probe is a scan),
skips tokens under three characters, and unions the resulting ids into the pool
before scoring.

Unioned candidates get their **real** cosine distance, computed against a
freshly embedded query vector (Chroma does not return the vector it used).
Fabricating one is wrong in both directions: a neutral value invents vector
evidence the candidate never earned, and a worst-case value (−1 similarity ×
the 0.55 cosine weight) buries it below anything the 0.20 lexical weight could
recover, making the channel decorative. Metadata filters are applied during
candidate selection, so a lexical hit obeys `type`/`source`/`importance`/date
filters exactly as a vector hit does.

Off by default (`"pool"`), because the scan is real: ~4.5 ms at 25k memories,
growing linearly.

### Why there is no second index

Full-corpus lexical recall could instead be served by a SQLite metadata + FTS5
index beside `.chroma`, promising five payoffs: full-corpus BM25, cursor
pagination, O(1) reverse links, tag counts, and an indexed expiry sweep.
Measurement ruled it out:

| Operation | 3k rows | 25k rows |
|---|---|---|
| SQLite FTS lookup | 0.03 ms | 0.03 ms (flat — a real inverted index) |
| Chroma `$contains` | 0.91 ms | 4.52 ms (linear) |
| SQLite `list_recent(50)` | 3.24 ms | **13.86 ms** |
| Chroma `get(limit=50)` | 2.03 ms | **3.72 ms** |

Chroma does four of the five natively — and two of them *better*:

- **Pagination** — `limit`/`offset`. Going through an index was ~4× slower,
  because serving ids from it and then fetching them from Chroma is two
  round-trips.
- **Expiry sweep** — `where={"$and": [{"expires_at": {"$gt": 0}}, …]}`.
  `purge_expired` pushes the range down instead of loading the collection.
- **Type counts** — `where={"type": …}` with `include=[]`.
- **Lexical reachability** — `where_document`, as above.

What a linear scan gives up is the *flat* lookup curve: at 10⁶ memories it costs
~180 ms where an inverted index stays at 0.03 ms. That is a real trade, made
deliberately. Against paying for it: a second index is a second source of truth
for data Chroma already holds, so it needs verify-on-open, disable-on-mismatch
and rebuild machinery purely to keep a cache from lying; it duplicates the text;
it is one database *per collection*, which in a multi-tenant deployment means N
satellite files with N failure modes; and the primary consumer
(`ai-houkai-service`) already runs its own WAL SQLite. Three SQLite databases in
one stack — Chroma's, the service's, and one per collection — to save
milliseconds on a path dominated by a 50–200 ms hosted embedding call.

The remaining structural gap is reverse links: Chroma stores `links` as an
opaque metadata string, so "who points at me?" is not expressible as a `where`
clause and `neighbors(direction="in")` reads every memory, once per frontier
node per hop. The fix is to denormalise the edge — record a→b on both sides —
which needs no index at all. Tracked in [ROADMAP.md](ROADMAP.md).

If a true inverted index is ever needed at scale, it belongs in the service's
existing database where it can be shared across collections and maintained in
one place, not as satellites the library manages behind Chroma's back.

## 26. Curation & Trash

These operations grew up in `ai-houkai-service`, which implemented them by
reaching through the library's private API (`store._get_by_id`,
`store._get_all_memories`, `store.collection.update`, `store._journal`). Every
one is a store-level primitive that a downstream consumer cannot do correctly
from outside, so they were graduated into `ai_houkai/memory_system/curation.py`
and attach to `MemoryStore` as mixin methods.

| Operation | Why it belongs in the store |
|---|---|
| `merge(target, other)` | Combines text, transfers outgoing links, and **re-points every incoming link** `x → other` at `x → target`. `forget` does not clean up incoming edges, so without this a merge silently strands every relationship pointing at the absorbed memory. Needs a reverse-link walk. |
| `versions(id)` | Past text states recovered from the journal, including rotated archives. |
| `list_tags` / `rename_tag` / `merge_tags` / `delete_tag` | There was no tag maintenance anywhere, yet `tag` is a first-class recall filter and tags accumulate typos permanently. |
| `find_path(a, b)` | BFS over undirected links. The service's version carried a 2000-row cap that "silently returned wrong *no path* results on larger collections" — the class of bug that belongs behind a library API with a real index. |

`_repoint_incoming` writes each source's link list directly rather than going
through `unlink`+`link`: that path re-validates the rel vocabulary (rejecting a
legacy custom rel outright) and costs two journal entries per edge. A
pre-existing `new_dst → old_dst` edge is dropped rather than turned into a
self-loop.

### Trash

The store jumped straight from `supersede` (soft, but semantically "replaced by
X") to `forget` (irreversible); `restore` only undoes a supersede. Trash is the
missing middle — a recoverable delete, parked in a gzipped JSONL file beside
the store. **Decay pruning now routes here instead of hard-deleting**, so a
mis-tuned `min_score` is no longer unrecoverable.

Entries are **appended** one gzip member at a time rather than rewritten, so
binning N memories costs O(N) — the read-all-rewrite-all version made a
400-memory prune take 2.16 s against 0.12 s. Both ports' readers (and Python's
`gzip` module) consume concatenated members transparently, so a file written by
either is readable by the other.

`trash_purge_expired(ttl_days)` supplies retention, and the maintenance
scheduler drives it on the same tick as the TTL purge (`trash_ttl_days`,
default 30 days). Without it a recoverable delete is really a permanent
archive and the file grows without bound. `ttl_days <= 0` is a **no-op** rather
than "purge everything" — a misconfigured or unset retention must never be read
as "delete it all". Surfaced as `houkai trash purge --older-than`,
`older_than_days` on the `trash_purge` MCP tool, and `POST /trash/purge`.

This is also why the library's trash is the one that survives when it is
consumed by `ai-houkai-service`: the service's `trash` table had retention and
cascade-on-collection-delete, and retention was the only one of those the
library lacked. Cascade is implicit here, because the trash file lives beside
the store it belongs to.

---

## 27. Working Set: Pinned, Trust, Idempotent

Three write-time fields, each defaulting to the previous behaviour so old
stores load unchanged.

### `pinned`

The only lever for "always consider this" was `importance`, which
simultaneously drives ranking, decay survival and the `min_importance` filter —
three jobs, one number. A pinned memory leads `recall_pack`/`auto_context`
inside the token budget and is exempt from decay pruning, giving agents an
explicit standing-instruction slot. (Quota eviction is a ROADMAP item, not a
feature yet; when it lands, pinned rows are exempt there too.)

The pin **follows a supersede onto the replacement**, and `restore` hands it
back, and `merge` takes the **union** of the two sides' pins — `other` is
deleted by the merge, so a pin that does not travel is destroyed outright.
Reflection does the same when it consolidates: a summary of a pinned
source is born pinned under `soft` (which supersedes the sources) and `hard`
(which deletes them), but not under `none`, where the sources stay live and
pinned and a second pinned row would double the slot. Superseding a standing instruction is how you correct one, and a
superseded row is out of the working set — without the transfer the slot
silently emptied and the agent stopped seeing the rule. `trust` is deliberately
not inherited: both rows survive with their own provenance, unlike `merge`,
which folds two into one and must take the worse label.

### `trust`

AI-Houkai writes straight into agent context via `recall_pack`/`auto_context`,
so anything reaching `remember` — a scraped page, a tool result, another
agent's output — becomes durable, well-ranked context later. `source` is
free-text with no semantics and no filtering default.

`trust` is `trusted` (default, so old stores are byte-identical) / `reported` /
`untrusted`, set at ingest boundaries; `recall(min_trust=…)` filters and the
packer annotates untrusted lines so the consuming model can see the provenance
boundary. Best-effort labelling, not a security guarantee — the same framing as
the PII item in the roadmap.

### `idempotent`

Agents re-assert the same fact every session. The only defence was
`on_conflict="supersede"`, which costs a 12-neighbour vector query per write
and still creates a new row plus a supersede edge. A normalised-text
`content_hash` lets `remember(idempotent=True)` return the existing memory and
bump its access count instead, which matters most in `remember_many` where the
conflict scan is per item.

The surfaces have to **say which happened**. A dedupe hit creates nothing, so
`POST /memories` answers `200` with `stored: false` and the existing row (a new
write stays `201`/`stored: true`), and the `remember` MCP tool returns the row
with `stored: false`. Both ports reported every idempotent repeat as a fresh
write, so a client replaying its batch each session was told it had stored N new
rows when it had stored none — and the Go REST port went further and answered
`409` with an *empty* conflict list, turning the feature into a hard error
against a store that already knew the fact. `409` is now reserved for an actual
conflict rejection, which always carries its conflicts.
`MemoryStore.find_by_content_hash(text)` is the public form of the lookup, so a
caller can also ask before writing.

`remember_many` follows the same rule: `stored` counts the rows **created**, not
the items submitted — 0 for a fully replayed batch (`POST /memories/batch` then
answers `200`), and intra-batch duplicates count once because they collapse to
one row. Every input still maps to an entry in `ids` / `memories`, with
duplicates sharing an id.

---

## 28. Retrieval Evaluation

`ai_houkai/eval.py` could score a ranking for months but was reachable from
nothing — no CLI command, no MCP tool, no HTTP route, no Go port. The
consequence was that every ranking constant in the project (graph damping 0.5
and 3 iterations, the β=0.20 lexical weight, the MMR defaults, the `graph`
weight itself) was set by intuition, with no way to tell whether a change
helped.

Now wired to:

| Surface | Entry point |
|---|---|
| CLI | `houkai eval <goldset.jsonl>` with ranking flags + `--json` |
| MCP | the read-only `eval_recall` tool |
| Go | `go/internal/eval` + `houkai eval`, metric-for-metric with Python |

Gold sets are JSONL — one case per line, `#` comments and blank lines skipped —
to preserve the stdlib-only ethos. `relevant_ids` accepts 8-char prefixes so a
set can be written by eye from `houkai list`; an **unresolvable id is an error,
not a zero score**, because a typo would otherwise look exactly like a ranking
regression.

Recall runs with `touch=False` throughout (the Go adapter forces it), so
evaluating never perturbs access-count or recency — otherwise a second run of
the same gold set would score against a mutated store. The `--json` output
records the configuration under test alongside the scores, so numbers from two
runs are attributable.

Metrics (binary relevance, each relevant id credited once so duplicates cannot
inflate anything): recall@k · precision@k · MRR · MAP · nDCG@k. `k` is reported
as `-1` when cases used mixed values.
