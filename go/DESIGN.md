# AI-Houkai (Go) — Design

This document describes the Go port's internals. For the user-facing overview
see [README.md](README.md); for the original Python design see
[`../DESIGN.md`](../DESIGN.md); for the porting rationale see
[`../GO_PORT_DESIGN.md`](../GO_PORT_DESIGN.md).

## Goals

1. **Single static binary per command** — `ai-houkai-mcp`, `ai-houkai-serve`,
   and `houkai`, ~8 MB each, no Python runtime, no native deps beyond glibc.
2. **Same model as the Python version** — episodic/semantic/procedural/feedback
   memories, hybrid recall, decay-based pruning, reflection clusters, link
   graph, conflict detection. Tool names preserved so external MCP clients
   keep working.
3. **Distro-native deployment** — ship as a Debian package with conffile,
   systemd unit, and Ollama setup hint in the postinst.
4. **Embedder is pluggable.** Inference is delegated to Ollama (default),
   OpenAI, or DigitalOcean Serverless Inference. The binary never links a
   model.

## Component layout

```
cmd/ai-houkai-mcp ───┐
cmd/ai-houkai-serve ─┤── cli.ResolveConfig ──► embed.Embedder
cmd/houkai ──────────┘                      └─► vector.Backend  ──► memory.MemoryStore
                                                                        │
                                                                        ├─► decay.Engine
                                                                        ├─► reflect.Engine   (+ summarizers)
                                                                        ├─► maintenance.Daemon
                                                                        ├─► tui.Model        (Bubble Tea browser)
                                                                        ├─► mcpserver.New    (MCP tools)
                                                                        └─► httpserver.New   (JSON HTTP/REST API)
```

Side packages: `internal/ingest` (pure chunking functions for
`houkai ingest`), `internal/installer` (MCP-client config patchers) and
`internal/timeparse` (lenient epoch/ISO/relative-span parsing shared by the
CLI, MCP and HTTP front-ends) have no store dependency at all.

The three `cmd/` entry points are deliberately thin: each builds the same
`MemoryStore` object then attaches it to a front-end (MCP server, HTTP server,
or cobra CLI). `houkai serve` reuses the CLI-selected store as a fourth path
into `httpserver`.

### `internal/memory` — domain core

- `Memory`: 14-field record (id, text, type, tags, importance, timestamps,
  access count, source, links, supersedence pointers, polarity, `expires_at`
  TTL). Also exposes `ToDict()` / `MemoryFromDict()` for the journal and
  `.ahkai` payloads — these emit and consume Python's serialisation shape
  verbatim.
- `MemoryStore`: the only stateful object. Backed by a `vector.Backend` plus
  an `embed.Embedder`. All public operations are methods on it:
  `Remember`, `Recall`, `RecallPack`, `Forget`, `ListRecent`, `GetByID`,
  `UpdateMemory`, `Link`, `Unlink`, `Neighbors`, `Subgraph`, `FindConflicts`,
  `Supersede`, `Restore`, `Stats`, `AllRaw`, `Export`, `Import`, `Undo`,
  `PurgeExpired`, `History`, `StateAt`, `GetAt`, `Metrics`.
- `RecallPack` (`pack.go`): ranks via `Recall` (hybrid by default), then
  greedily packs `- (type) text` lines into a token budget. Token cost
  defaults to a ~4-chars/token estimate (`EstimateTokens`); callers can
  supply an exact `TokenCounter`. A candidate that doesn't fit sets
  `Truncated` but the loop keeps going — a smaller, lower-ranked item may
  still fit. The header is not counted against the budget.
- `ScoreImportance` (`importance.go`): heuristic importance scorer — regex
  tiers (0.90 instructions/corrections/preferences, 0.75 decisions, 0.60
  completions, 0.35 hedges; strongest tier wins) plus modifiers (+0.10
  procedural/feedback type, −0.15 question, −0.10 under-20-chars), clamped
  to [0.05, 0.98]. Deterministic by design. Wired in via
  `StoreConfig.ImportanceFn`, which `Remember` consults when
  `opts.Importance == 0`.
- `MetadataToMemory` / `MemoryToMetadata`: chromem-go only stores
  `map[string]string`, so every numeric / list / nested field is
  stringified at the boundary and parsed back on read. Links are stored as
  inline JSON. This is the only fragile serialisation surface in the
  codebase — touch with care.
- **ID resolution**: any caller may pass an 8-char prefix; `resolvePrefix`
  scans `backend.All()` and errors out on ambiguity. Matches Python behaviour.

### `internal/memory` — audit journal

`journal.go` defines `Journal` and `JournalEntry`. Every mutation on
`MemoryStore` (`remember`, `forget`, `supersede`, `restore`, `link`,
`unlink`, plus the higher-level `import` / `export` ops) is appended as one
JSON line to `journal.log` next to the store. Reflection and decay are *not*
op markers — they run through ordinary `remember` / `forget` / `link` entries
and are distinguished only by their `actor` tag (`reflection` / `decay`).

- **Best-effort writes.** `Journal.Append` swallows and logs all errors —
  a journal failure must never break the underlying memory op.
- **Size-based gzip rotation.** Hot file caps at `RotateMB` (64 by default);
  rotated archives carry an ISO timestamp suffix (`journal-YYYYMMDDThhmmss.log.gz`).
  Rotation truncates the live file in place so any concurrent writer's fd
  stays valid. Archives older than `KeepDays` (90) are pruned on the next
  rotation check.
- **Actor tagging.** Entries are written under a thread-local `actor`
  string. `MemoryStore.AsActor(name)` swaps the actor and returns a
  closure that restores the previous value — call sites use it via
  `defer store.AsActor("reflection")()`. `reflect` and `decay` set this
  through an optional `actorScoped` interface assertion on the store, so
  the engines stay testable with smaller fake stores.
- **Reads tolerate truncation.** A crash mid-write leaves an un-parseable
  last line; `Journal.Read` silently skips it. The journal is forensics,
  not consistency.

`store.Undo(entry)` reverses one entry where the semantics are well-defined
— `remember` ↔ `forget`, `supersede` ↔ `restore`, `link` ↔ `unlink`.
`reflect` / `decay` / `import` / `export` / `undo` themselves are not
undoable; the function returns `(false, nil)` for those.

### `internal/memory` — portable .ahkai export/import

The format is **gzipped JSONL** with a header line followed by one
memory row per line, in `created_at`-ascending order so two exports of
the same store are byte-identical modulo the header timestamp.

```
line 1     : {"format":"ai-houkai/export","version":1,"exported_at":…,
              "source":{"collection","embedding_model","embedding_dim","count"},
              "options":{"include_vectors","include_superseded","types","tags","since"}}
line 2..N  : {"id":…,"text":…,"meta":{full ToDict()},"vector":[…]?}
```

`Export(ctx, path, opts)` streams everything via `encoding/json.Encoder`
straight into a `gzip.Writer`, with type / tag / since / superseded
filters applied client-side. `Import(ctx, path, opts)` reads back the
header, validates `format` and `version ≤ 1`, then decodes rows one at a
time. Four `OnConflict` policies — `skip` (default), `overwrite`,
`rename` (assigns a fresh UUID), `error` (collects collisions and raises
`ImportConflictError`). On embedding-model mismatch, the importer
refuses unless `RegenerateVectors=true`, in which case `addImported`
re-embeds text on the way in.

`PeekExportHeader(path)` is used by `houkai info` to inspect a file
without instantiating a store.

### `internal/memory` — scoring

- `bm25.go`: pure-Go BM25 over the over-fetch pool returned by the vector
  backend. Min-max normalised to `[0, 1]`. IDF is computed locally on each
  query (no global term-frequency index) — fine for the small pool sizes
  involved.
- `hybrid.go`: linear combination `Cosine·0.55 + Lexical·0.20 + Recency·0.15 +
  Importance·0.10` (`DefaultWeights`). Recency is `exp(-λ · age_days)`.
- `conflict.go`: candidate B conflicts with A iff (same type) ∧ (cosine ≥
  threshold) ∧ (tags overlap). Within that filter, kind is `contradiction`
  when negation parity differs, else `duplicate`. Negation parity is a crude
  but effective bag-of-negation-words count mod 2.

### `internal/vector` — storage backend

`Backend` is an 8-method interface. The only implementation is
`ChromemBackend` over [`philippgille/chromem-go`](https://github.com/philippgille/chromem-go),
configured with a no-op embedding function — the caller is expected to supply
pre-computed vectors. Two quirks worth knowing:

- **`UpdateMetadata` is delete-then-add** because chromem-go has no in-place
  mutation API. Cheap because there's no rebuild step.
- **`All()` queries with a zero vector** of the configured dimension to pull
  every document. Works because chromem-go's cosine query degenerates to "all
  docs" when the query is the origin. If `dim` is set wrong the call returns
  garbage scores but correct documents — and we throw the scores away.

Swapping backends (sqlite-vec, qdrant, etc.) means implementing the 8 methods
and changing one constructor call in `cmd/*/main.go`.

Beyond the interface, `ChromemBackend` exposes collection-management
extras consumed by `houkai collections` (the CLI type-asserts to the
concrete type): `ListCollections`, `HasCollection`, `CreateCollection`,
`DeleteCollection`, and `CopyCollection` (which moves documents with their
embeddings — no re-embedding — via the same zero-vector retrieval trick).

### `internal/embed` — embedders

`Embedder` is two methods (`Dim` + `Embed`). Three implementations ship:

- `OllamaEmbedder` — POSTs to a local Ollama at `/api/embed`.
- `OpenAIEmbedder` — POSTs to `/v1/embeddings`. The base URL is a field, so
  the same struct is reused for any OpenAI-compatible host.
- `NewDigitalOcean` — thin constructor that returns an `OpenAIEmbedder`
  pointed at `https://inference.do-ai.run`. DO's Serverless Inference is
  wire-compatible with OpenAI's embeddings endpoint (same JSON request and
  response shapes, same `Bearer` auth), so no separate type is needed. Any
  future OpenAI-compatible provider (Together, vLLM, llamafile,
  `llama-server`) can be added the same way with one line.

All three lazily cache the dimension from the first response. A "warmup"
call is made by `ai-houkai-mcp` at startup so `Dim()` is correct before the
first real request.

The `embed_dim` config value is the **vector backend's** dimension and must
match the embedder; if it doesn't, the zero-vector trick in `All()` is the
first thing that misbehaves.

### `internal/decay` — pruning

`Engine.Score(m) = importance · exp(-λ · days_since_last_access) ·
reinforcement`, where `reinforcement = 1 + FrequencyWeight · ln(1 +
access_count)`. With the default `FrequencyWeight = 0` the factor is exactly
`1.0` (recency-only behaviour); raising it makes frequently-recalled memories
age out more slowly than untouched ones of equal importance and age — the score
can then exceed `importance`, and `MinScore` is compared against the reinforced
value. Memories below `MinScore` are deleted (`Prune`), unless their type is in
`ProtectTypes` (default `[procedural]`). The engine takes a narrow `Storable`
interface (`ListRecent` + `Forget`) so it's trivially testable. Exposed via
`houkai prune --frequency-weight`, the `maintenance_tick` MCP tool, and
`maintenance.Config`.

### `internal/reflect` — reflection

Greedy single-linkage clustering on episodic memories. For each unassigned
seed (highest importance first), absorb every other unassigned episodic with
`cosine(seed, other) ≥ threshold` (default 0.75). Clusters of size
≥ `MinClusterSize` (default 2) are summarised by `Summarizer` (the default
just concatenates with `|` separators and truncates at 512 chars). Each new
semantic memory is linked back to its sources via `derived_from`; with
`consolidate=true` the sources are deleted.

LLM summarizers live in `summarizers.go`:
`BuildSummarizer("provider:model", fallback)` returns a `Summarizer` for
`extractive` (the built-in default), `ollama:` (OpenAI-compatible
`/v1/chat/completions`, `OLLAMA_BASE_URL`), `openai:` (`OPENAI_API_KEY`,
`OPENAI_BASE_URL`), or `anthropic:` (`/v1/messages`, `ANTHROPIC_API_KEY`,
`ANTHROPIC_BASE_URL`) — all via plain `net/http`, no SDKs. With
`fallback=true` (the usual case) LLM failures and empty outputs degrade to
the extractive summarizer with a logged warning, so unattended maintenance
never crashes. The spec comes from the `summarizer` config key /
`AI_HOUKAI_SUMMARIZER` env / `houkai reflect --summarizer`.

### `internal/ingest` — bulk-ingestion chunking

`ChunkText(text, maxChars, minChars)` is deterministic and dependency-free:
split on blank lines, glue markdown headings to the paragraph that follows
(so a stored memory keeps its context), re-pack paragraphs longer than
`maxChars` on sentence boundaries (a single oversized sentence is kept whole
— splitting mid-sentence would destroy the embedding's meaning), drop chunks
shorter than `minChars`. Go's RE2 has no lookbehind, so sentence boundaries
are located with `FindAllStringIndex` and re-assembled manually. Embedding
and storage happen at the caller (`houkai ingest`, one `Remember` per chunk,
actor `import`, source `ingest:<filename>`).

### `internal/tui` — interactive browser

Bubble Tea port of the Python Textual TUI. Split in two files on the same
boundary as Python's `tui/data.py` vs `tui/app.py`:

- `data.go` — pure view-models, no bubbletea import, fully unit-testable:
  `Row`, `View` (kind/title/rows/id8→Memory map), `DetailText`, and
  `Navigator`, a breadcrumb stack of views (`OpenRecent` resets the stack,
  `OpenSearch`/`OpenNeighbors` push, `Back` pops but never below the root).
- `app.go` — the `tea.Model`: `bubbles/table` list + `bubbles/viewport`
  detail pane side by side, `bubbles/textinput` search bar. Keys: `/`
  search, `n` neighbors, `b` back, `r` recent, `q` quit. Store calls are
  synchronous (embedding a search query blocks the UI briefly — same
  trade-off as the Python version).

### `internal/maintenance` — scheduled ticks

One synchronous entry point: `maintenance.Tick(ctx, store, store, cfg,
statePath, now)` runs a decay prune, (optionally) a reflection pass, and a
**TTL purge** (`store.PurgeExpired`, reached through an `expirable`
type-assertion so a bare `decay.Storable` fake need not implement it), then
persists `State` (last-run timestamps + totals, incl. `LastPurgeAt` /
`TotalPurged`) to a JSON state file. Jobs are gated on a schedule
(`DecayEvery` / `ReflectEvery` / `PurgeEvery`, seconds since the job's last
recorded run — mirroring Python's `MaintenanceScheduler`), and the
whole load→run→save cycle holds an exclusive **flock on `<state>.lock`** so a
daemon loop, a cron `houkai maintenance tick`, and the MCP `maintenance_tick`
tool can all target the same state file without double-running jobs. The CLI
drives it via `houkai maintenance` (`tick`/`run`/`start`/`stop`/`status`) and
`ai-houkai-mcp` wires the same config through `mcpserver.SetMaintenance`.

### `internal/mcpserver` — MCP surface

Wraps `mark3labs/mcp-go`. Twenty-two tools:

```
remember        recall          recall_pack     auto_context    forget
purge_expired   edit            list_recent     stats           metrics
history         state_at        get_at          link            unlink
neighbors       find_conflicts  supersede       maintenance_tick
journal_tail    export          import
```

Every handler converts inputs via `req.GetString/GetFloat/...`, calls the
corresponding `MemoryStore` method, and returns a JSON text result via
`jsonText`. `ConflictError` is unwrapped so callers see
`{stored: false, conflicts: [...]}` rather than an opaque error.

These 22 tools mirror the Python tool names so existing MCP clients keep
working, and the surface is at full parity: the `auto_context` tool, the
advanced `recall` / `recall_pack` parameters (`fusion=rrf`, diversity/MMR,
`dedup_threshold`, `min_cosine`, `touch`, `explain`, `recall_pack`
compression via `compress`/`compress_threshold`/`compress_min_group`), and the
recent additions — reranking (store-config `Reranker` func), TTL/expiry
(`ttl_seconds`/`expires_at`, `include_expired`, `purge_expired`), point-in-time
`history`/`state_at`/`get_at`, and `metrics` — are all implemented (see
`internal/memory/scoring.go`, `pack.go`, `autocontext.go`, `store.go`).
`maintenance_tick`'s `consolidate` is a tri-state string (`none|soft|hard`).
`export` / `import`
take a server-local file path — there's no streaming over MCP yet, and
binary payloads (the gzipped bytes) are kept off the JSON wire. `recall` /
`recall_pack` accept the `source` / `since` / `until` metadata filters
(`since`/`until` parsed through `internal/timeparse`, so they take epoch
seconds, an ISO date/datetime, or a relative span like `"7d"`), and
`maintenance_tick` accepts `frequency_weight` for recall reinforcement.

`remember` passes `importance` through as a pointer: a missing value is
`nil` = unset, so the store's `ImportanceFn` (enabled via
`default_importance = "auto"` or `AI_HOUKAI_AUTO_IMPORTANCE=1`) can
auto-score it, while an explicit value — **including `0`** — is honoured and
clamped to `[0, 1]`; the response echoes the resolved `importance`. The
default memory type is `semantic` (matching Python), and enum inputs are
validated in the store, so a typo like `mode="hybird"` comes back as a
descriptive error instead of silently degrading. `maintenance_tick` is
schedule-gated through `SetMaintenance` (see `internal/maintenance`). `maintenance_tick`'s reflection step uses the
summarizer spec injected at startup via `SetSummarizerSpec` (from the
`summarizer` config key).

### `internal/httpserver` — JSON HTTP/REST API

A standard-library front-end (`net/http` only — no router dependency) over the
same `MemoryStore`, for clients that cannot speak MCP. `Server.Handler()`
registers routes on a `http.ServeMux` using Go 1.22+ method+path patterns
(`GET /memories/{id}`), so a path that matches a row but not the verb yields
`405` and an unmatched path `404` for free. Each handler has the signature
`func(*http.Request) (status int, payload any, error)`; the `wrap` adapter
renders an `*httpError` with its status and any other error as a `500`, so a
single bad request never takes the server down. A middleware layer adds optional
bearer-token auth (every route except `/health`), `panic`→`500` recovery, and a
buffering `captureWriter` that re-renders ServeMux's plain-text `404`/`405`
pages as the same `{"error": …}` JSON envelope. Bodies are capped at 4 MiB.
`since`/`until` query and body values flow through `internal/timeparse`.

Exposed two ways, symmetric with the MCP server: `houkai serve` (reuses the
CLI-selected store, sets actor `"http"`) and the env-configured
`cmd/ai-houkai-serve` binary (`AI_HOUKAI_HTTP_{HOST,PORT,TOKEN}`). Binds
`127.0.0.1` by default. Driven in tests by a real `httptest.Server`.

### `internal/cli` — CLI front-end

`cobra` root + subcommands (~35 leaves including the `journal {tail,show,undo}`,
`collections {list,create,delete,copy}`, and
`install {claude-code,cursor,opencode}` groups, plus `pack`, `ingest`, and
`tui`). The root's `PersistentPreRunE` is where the real work happens: it
resolves config, instantiates the embedder + backend + store (with
`Actor="cli"`, a qualified `EmbeddingModel` like
`openai:text-embedding-3-small` for the export header, and `ImportanceFn`
when `default_importance = "auto"`), and stuffs them into the command's
context via four typed keys (`storeKey`, `cfgKey`, `fmtKey`, `backendKey` —
the last giving `collections` access to the concrete `ChromemBackend`).
Subcommands pull them back out with `storeFromCtx` / `cfgFromCtx` /
`fmtFromCtx` / `backendFromCtx`. This keeps each command's `RunE` short and
free of construction boilerplate.

Output goes through `output.go`:
- `--format json` → encoder
- `--format tsv` → tab-separated
- `--format auto` → lipgloss-styled human output if stdout is a TTY, else TSV

### `internal/installer` — MCP client wiring

Three installers share the helpers in `common.go` (JSON load/merge/write,
binary resolution, the client-agnostic `MemoryGuide` instruction text):

- `ClaudeCodeInstaller` — merges an `mcpServers` entry into
  `~/.claude/settings.json` (or `.claude/settings.json` with `--project`).
- `CursorInstaller` — same `mcpServers` schema, but in `~/.cursor/mcp.json`
  (project: `.cursor/mcp.json`), default collection `cursor`. Also emits a
  `.cursor/rules/*.mdc` snippet (`CursorRuleSnippet`, `alwaysApply: true`).
- `OpenCodeInstaller` — OpenCode's own `mcp` schema (`type: "local"`,
  `command` *array*, `environment` map, `enabled` flag, top-level `$schema`)
  in `~/.config/opencode/opencode.json` (project: `opencode.json`), default
  collection `opencode`. Also emits an AGENTS.md snippet.

All merge rather than overwrite (existing keys and sibling servers are
preserved; unparseable files are replaced). Each block points at
`ai-houkai-mcp` on PATH and sets `AI_HOUKAI_PATH` / `AI_HOUKAI_COLLECTION`
env vars. Verification (`--verify`) checks the binary is reachable plus a
membership check on the merged map. Exposed as
`houkai install {claude-code|cursor|opencode}`; bare `houkai install` keeps
its historical Claude Code meaning.

## Config resolution

```
defaults → /etc/ai-houkai/config.toml
        → ~/.config/ai_houkai/config.toml
        → $AI_HOUKAI_CONFIG
        → AI_HOUKAI_* env vars
        → --store / --collection CLI flags
```

Implemented in [`internal/cli/config.go`](internal/cli/config.go). Missing
files are silently skipped; unparseable TOML errors are also swallowed
(matches Python behaviour — config should never break a working binary).

`default_importance` accepts a float or the literal string `"auto"` via a
custom `toml.Unmarshaler` (`ImportanceDefault{Value, Auto}`); `"auto"`
switches the store to the heuristic scorer. `summarizer` holds the
reflection summarizer spec (env override: `AI_HOUKAI_SUMMARIZER`). Note the
`summarizer` key is flat here, unlike Python's `[maintenance.reflect].summarizer`
table. There is a `[maintenance]` section (`interval_secs`, `reflect`,
`consolidate`, `state_path`/`pid_path`/`log_path`) with a `[maintenance.decay]`
sub-table (`decay_rate`, `min_score`, `protect_types`, `frequency_weight`) that
`houkai prune`, `stats --health`, and the `maintenance` command group read.

The same resolver is used by both `houkai` and `ai-houkai-mcp` so the two
binaries see identical settings.

## Packaging

### Debian / Ubuntu

`scripts/build-deb.sh` produces a `.deb` per arch:

- Binaries → `/usr/bin/{ai-houkai-mcp, ai-houkai-serve, houkai}`
- Default config → `/etc/ai-houkai/config.toml` (marked as a **conffile** so
  upgrades preserve user edits)
- systemd unit → `/lib/systemd/system/ai-houkai-mcp.service` (not enabled by
  default; for users who want a background MCP daemon)
- `postinst` hint that prompts the user to `ollama pull all-minilm` or to
  switch to OpenAI

No declared `Depends:` — the binary is statically linked Go and Ollama is
installed out-of-band (it isn't a Debian-archive package). See the `postinst`
script for the runtime guidance.

Version policy: `git describe --tags --always --dirty` is mapped to a
dpkg-friendly form by replacing `-` with `~` (so `0.3.4-dirty` →
`0.3.4~dirty`, which sorts *before* `0.3.4`). The `~` is backslash-escaped in
the `${var//-/\~}` replacement to suppress bash's tilde expansion.

### macOS

`scripts/build-macos.sh` produces a `.tar.gz` per arch, intended for manual
install or Homebrew tap distribution:

- Cross-compiles from any host via the same `build.sh` (CGO disabled, so no
  Xcode / osxcross toolchain is required).
- Apple Silicon → `..._darwin_arm64.tar.gz`; Intel → `..._darwin_x86_64.tar.gz`
  (Homebrew naming, despite Go's `GOARCH=amd64`).
- Layout inside the archive: `bin/{ai-houkai-mcp, ai-houkai-serve, houkai}`,
  `share/ai-houkai/config.toml.example`, `README.txt`.
- A sibling `.sha256` file is emitted next to each tarball so Homebrew
  formulae can reference the hash without recomputing it.
- **No code signing / notarisation.** Users downloading via a browser hit
  Gatekeeper's quarantine bit; the bundled `README.txt` instructs them to
  `xattr -d com.apple.quarantine` after install. Adding `codesign` /
  `notarytool` is a future improvement when an Apple Developer ID is
  available.
- The original Linux `~` version-quoting trick is **not** needed here: tar
  filenames don't care about `~` or `-`.

## Testing strategy

~20 test files / 100+ test functions across 9 packages, all offline
(`go test ./...` needs no network and no Ollama):

- `memory/store_test.go` — Remember→Recall→Supersede→Restore round-trips
  against a real chromem-go store in a tmpdir, using `stubEmbedder` (FNV
  hash → deterministic L2-normalised vector)
- `memory/bm25_test.go`, `memory/hybrid_test.go` — scoring known-answers
- `memory/conflict_test.go` — negation/duplicate cases
- `memory/links_test.go` — graph ops
- `memory/export_import_test.go` — `.ahkai` round-trip, conflict policies
- `memory/pack_test.go` — token budgets, truncation, rank-order
  preservation, custom token counters
- `memory/importance_test.go` — scoring tiers, modifiers, clamping, store
  wiring
- `ingest/ingest_test.go` — chunking: headings, sentence re-packing, CRLF,
  min-chars filtering
- `reflect/engine_test.go` — cluster boundaries;
  `reflect/summarizers_test.go` — spec parsing, all three providers against
  `httptest` servers, fallback-on-error/empty
- `decay/engine_test.go` — protected types, thresholds
- `vector/chromem_test.go` — backend round-trip + collection management
- `installer/*_test.go` — merge-don't-overwrite for all three clients
- `tui/data_test.go` — view-models and Navigator stack (no terminal needed)
- `cli/config_test.go` — resolution-order precedence, `"auto"` importance,
  summarizer env override
- `embed/openai_test.go` — request/response shapes

## Non-goals

- **No LLM inference inside the binary.** Embeddings only, via HTTP. Adding
  generation would mean dragging in an inference runtime, which contradicts
  goal #1.
- **No multi-user / multi-tenant store.** Single user per data directory.
  Use the `collection` knob to namespace within one user.
- **No binary compatibility with the Python store.** Use the portable
  `.ahkai` export/import bridge — same gzipped-JSONL format on both ports.

## Known rough edges

- `MemoryStore.Neighbors` with `direction in {in, both}` does a full
  `backend.All()` scan per BFS step. Fine at ≤10⁴ memories; O(n·d) at scale.
  A reverse-link index would fix it.
- `Recall` bumps access tracking via `touchMany` on the returned memories
  (one metadata write per hit; `RecallOpts.NoTouch` / the `touch=false`
  MCP/HTTP param skips it for read-only recall). Still a write per hit —
  chromem has no true batch-update, so this remains a candidate for a real
  bulk write if chromem gains one.
- `ChromemBackend.All()` relies on the zero-vector trick — see the
  `internal/vector` section above. If chromem-go ever changes its behaviour,
  this is the first thing to break.
- The maintenance daemon is exposed via the `houkai maintenance` command
  group (`tick` for a one-shot cron pass, `run` for a foreground loop, and
  `start`/`stop`/`status` for a detached pidfile-managed daemon that persists
  a small JSON state file). It can also be embedded via `maintenance.Tick`.

## Adding a new MCP tool

1. Add a method on `MemoryStore` (in `internal/memory/store.go`).
2. Add an `add<Name>` function in `internal/mcpserver/server.go` mirroring
   the existing tool wrappers.
3. Register the call inside `New()`.
4. Add a matching `houkai` subcommand in `internal/cli/commands.go` and wire
   it into `root.go`.

The split between CLI and MCP is deliberately minimal — they're two surfaces
over the exact same store, and divergence (Python has it) is what we're
trying to avoid here.
