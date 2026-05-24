# AI-Houkai (Go) — Design

This document describes the Go port's internals. For the user-facing overview
see [README.md](README.md); for the original Python design see
[`../DESIGN.md`](../DESIGN.md); for the porting rationale see
[`../GO_PORT_DESIGN.md`](../GO_PORT_DESIGN.md).

## Goals

1. **Single static binary per command** — `ai-houkai-mcp` and `houkai`,
   ~8 MB each, no Python runtime, no native deps beyond glibc.
2. **Same model as the Python version** — episodic/semantic/procedural/feedback
   memories, hybrid recall, decay-based pruning, reflection clusters, link
   graph, conflict detection. Tool names preserved so external MCP clients
   keep working.
3. **Distro-native deployment** — ship as a Debian package with conffile,
   systemd unit, and Ollama setup hint in the postinst.
4. **Embedder is pluggable.** Inference is delegated to Ollama (default) or
   OpenAI. The binary never links a model.

## Component layout

```
cmd/ai-houkai-mcp ──┐
                    ├─► cli.ResolveConfig ──► embed.Embedder
cmd/houkai ─────────┘                      └─► vector.Backend  ──► memory.MemoryStore
                                                                        │
                                                                        ├─► decay.Engine
                                                                        ├─► reflect.Engine
                                                                        ├─► maintenance.Daemon
                                                                        └─► mcpserver.New  (MCP tools)
```

The two `cmd/` entry points are deliberately thin: each builds the same
`MemoryStore` object then attaches it to a front-end (MCP server or cobra
CLI).

### `internal/memory` — domain core

- `Memory`: 13-field record (id, text, type, tags, importance, timestamps,
  access count, source, links, supersedence pointers, polarity).
- `MemoryStore`: the only stateful object. Backed by a `vector.Backend` plus
  an `embed.Embedder`. All public operations are methods on it:
  `Remember`, `Recall`, `Forget`, `ListRecent`, `GetByID`, `UpdateMemory`,
  `Link`, `Unlink`, `Neighbors`, `Subgraph`, `FindConflicts`, `Supersede`,
  `Restore`, `Stats`, `AllRaw`.
- `MetadataToMemory` / `MemoryToMetadata`: chromem-go only stores
  `map[string]string`, so every numeric / list / nested field is
  stringified at the boundary and parsed back on read. Links are stored as
  inline JSON. This is the only fragile serialisation surface in the
  codebase — touch with care.
- **ID resolution**: any caller may pass an 8-char prefix; `resolvePrefix`
  scans `backend.All()` and errors out on ambiguity. Matches Python behaviour.

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

`Engine.Score(m) = importance · exp(-λ · days_since_last_access)`. Memories
below `MinScore` are deleted (`Prune`), unless their type is in
`ProtectTypes` (default `[procedural]`). The engine takes a narrow
`Storable` interface (`ListRecent` + `Forget`) so it's trivially testable.

### `internal/reflect` — reflection

Greedy single-linkage clustering on episodic memories. For each unassigned
seed (highest importance first), absorb every other unassigned episodic with
`cosine(seed, other) ≥ threshold` (default 0.75). Clusters of size
≥ `MinClusterSize` (default 2) are summarised by `Summarizer` (the default
just concatenates with `|` separators and truncates at 512 chars). Each new
semantic memory is linked back to its sources via `derived_from`; with
`consolidate=true` the sources are deleted.

Plugging in an LLM summariser is a one-function swap — `Engine.Summarizer`
is a public field.

### `internal/maintenance` — background daemon

Goroutine ticker that calls `decay.Prune` (and optionally `reflect.Reflect`)
on an interval. Cancel the parent `context.Context` to stop. Currently
**unused by both binaries** — present so callers embedding the library can
opt in, and so `maintenance_tick` can be a synchronous MCP tool that exposes
the same logic on-demand. Worth wiring into `ai-houkai-mcp` later as an
optional `--daemon` flag.

### `internal/mcpserver` — MCP surface

Wraps `mark3labs/mcp-go`. Eleven tools:

```
remember        recall          forget          list_recent     stats
link            unlink          neighbors       find_conflicts  supersede
maintenance_tick
```

Every handler converts inputs via `req.GetString/GetFloat/...`, calls the
corresponding `MemoryStore` method, and returns a JSON text result via
`jsonText`. `ConflictError` is unwrapped so callers see
`{stored: false, conflicts: [...]}` rather than an opaque error.

The Python version exposes 14 tools; the missing three (`update`,
`get_by_id`, `subgraph`) are CLI-only here. If a client needs them, add them
— they're 20-line wrappers each.

### `internal/cli` — CLI front-end

`cobra` root + 23 subcommands. The root's `PersistentPreRunE` is where the
real work happens: it resolves config, instantiates the embedder + backend +
store, and stuffs them into the command's context via three typed keys
(`storeKey`, `cfgKey`, `fmtKey`). Subcommands pull them back out with
`storeFromCtx` / `cfgFromCtx` / `fmtFromCtx`. This keeps each command's
`RunE` short and free of construction boilerplate.

Output goes through `output.go`:
- `--format json` → encoder
- `--format tsv` → tab-separated
- `--format auto` → lipgloss-styled human output if stdout is a TTY, else TSV

### `internal/installer` — Claude Code wiring

`ClaudeCodeInstaller.Install()` merges (does not overwrite) an `mcpServers`
entry into `~/.claude/settings.json` (or `.claude/settings.json` with
`--project`). The block points at `ai-houkai-mcp` on PATH and sets
`AI_HOUKAI_PATH` / `AI_HOUKAI_COLLECTION` env vars. Verification is a
membership check on the merged map.

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

The same resolver is used by both `houkai` and `ai-houkai-mcp` so the two
binaries see identical settings.

## Packaging

### Debian / Ubuntu

`scripts/build-deb.sh` produces a `.deb` per arch:

- Binaries → `/usr/bin/{ai-houkai-mcp, houkai}`
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
- Layout inside the archive: `bin/{ai-houkai-mcp, houkai}`,
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

Currently sparse. The intended shape is:

- `memory/bm25_test.go` — known-answer queries on a small corpus
- `memory/conflict_test.go` — negation/duplicate cases
- `memory/store_test.go` — Remember→Recall→Supersede→Restore round-trip
  against a real chromem-go store in a tmpdir, with a stub Embedder that
  hashes text to a deterministic float vector
- `decay/engine_test.go` — protected types, threshold edge cases
- `reflect/engine_test.go` — cluster boundary at threshold ± ε
- `cli/config_test.go` — resolution-order precedence

A `Stub` embedder (deterministic hash → fixed-dim vector) is the missing
piece that would make all of the above table-driven and offline.

## Non-goals

- **No LLM inference inside the binary.** Embeddings only, via HTTP. Adding
  generation would mean dragging in an inference runtime, which contradicts
  goal #1.
- **No HTTP API.** MCP-over-stdio + CLI cover both interactive and
  programmatic use. An HTTP server would be a fourth front-end on the same
  `MemoryStore` and is easy to add later — but isn't required today.
- **No multi-user / multi-tenant store.** Single user per data directory.
  Use the `collection` knob to namespace within one user.
- **No binary compatibility with the Python store.** Use the JSONL
  export/import bridge.

## Known rough edges

- `MemoryStore.Neighbors` with `direction in {in, both}` does a full
  `backend.All()` scan per BFS step. Fine at ≤10⁴ memories; O(n·d) at scale.
  A reverse-link index would fix it.
- `Recall` calls `touch()` on every returned memory, which is a metadata
  write per hit (delete-then-add in chromem). Cheap in absolute terms but
  amplifies write volume — consider batching in a future revision.
- `ChromemBackend.All()` relies on the zero-vector trick — see the
  `internal/vector` section above. If chromem-go ever changes its behaviour,
  this is the first thing to break.
- The maintenance daemon is wired up in code but not enabled by either
  binary. Either enable it via a flag in `ai-houkai-mcp` or remove the
  package.

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
