# AI-Houkai — Go Port Feasibility & Design

Status: **implemented** — the port shipped in `go/` (merged 2026-06-10) and
has since reached full feature parity with the Python version (15 MCP tools,
pack/ingest/collections/TUI/installers/summarizers). This file is the
original 2026-05-20 feasibility study, kept as a historical record; the
authoritative docs for the shipped port are [go/README.md](go/README.md)
and [go/DESIGN.md](go/DESIGN.md).

Source baseline at writing: ai-houkai 0.3.4 (Python, ~3.6 kLOC under `ai_houkai/`)

## How it actually turned out

Where the implementation diverged from this proposal:

- **Vector store:** `chromem-go`, as recommended (§4.1). ✓
- **Embeddings:** the ONNX/MiniLM default (§4.2) was **never built** — the
  port went with open question #4's alternative: HTTP embedders only.
  Ollama (`all-minilm`) is the default, with OpenAI and DigitalOcean
  Serverless Inference as alternatives. No CGO anywhere; the binaries are
  fully static. Consequence: Python and Go stores use different vector
  spaces, so migration always re-embeds.
- **Migration (§9):** no `houkai migrate --from-chroma` command. The
  portable `.ahkai` export/import format (added to both ports later)
  covers migration instead — `houkai export` on one side,
  `houkai import --regenerate-vectors` on the other.
- **MCP SDK (§7):** `mark3labs/mcp-go` (the proposal's fallback), not the
  official `modelcontextprotocol/go-sdk`. Tool count grew from 10 to 15
  (`maintenance_tick`, `journal_tail`, `export`, `import`, `recall_pack`).
- **Layout (§5):** close to proposed, with deltas: no third
  `ai-houkai-install` binary (installers are `houkai install` subcommands);
  no `sqlitevec.go` / `onnx_minilm.go`; added `ingest/`, `tui/`,
  `maintenance/` packages; `supersede.go` folded into `store.go`.
- **Packaging:** Debian `.deb` (`scripts/build-deb.sh`, conffile + systemd
  unit) and macOS tarballs — not GoReleaser. No Windows builds.
- **Open questions (§12):** 1 — both ports live on in parallel, at parity;
  2 — same repo, `go/` subdir; 3 — stdio only, no HTTP transport;
  4 — yes: Ollama-by-default won, dropping MiniLM/ONNX entirely.

The original study follows, unedited.

---

## 1. Verdict

**Yes — a Go rewrite is feasible** and would yield a single-binary distribution
(no Python runtime, no pip, no HuggingFace cache bootstrap) at the cost of
replacing two Python-only pieces:

1. **ChromaDB** (vector store + HNSW + persistence)
2. **sentence-transformers** (`all-MiniLM-L6-v2` embeddings)

Everything else (MCP server, CLI, decay, reflection, BM25, hybrid scoring,
conflict detection, linking, installers) is plain logic that translates
directly. The MCP protocol has a maintained Go SDK.

The two hard pieces both have credible Go solutions; the design hinges on
which we pick. See §4.

---

## 2. Component-by-component mapping

| Python piece | LOC | Go replacement | Risk |
|---|---|---|---|
| `MemoryStore` (Chroma client wrapper) | 769 | `store` pkg over chosen vector backend | M |
| `DecayEngine` | 114 | pure Go, 1:1 port | none |
| `ReflectionEngine` (cosine clustering) | 213 | pure Go; needs vector access from backend | L |
| BM25 (in `store.py`) | inlined | pure Go (`blevesearch/bleve` BM25 or hand-rolled) | none |
| Conflict / negation detector | inlined | pure Go | none |
| Linking / supersede / subgraph | inlined | pure Go (JSON in metadata) | none |
| Hybrid retrieval (α·cos + β·bm25 + γ·recency + δ·imp) | inlined | pure Go | none |
| `mcp_server/server.py` (FastMCP, 10 tools) | 289 | `modelcontextprotocol/go-sdk` server | L |
| `installers/claude_code.py` | ~150 | pure Go (`encoding/json`) | none |
| CLI (`cli/`, 22 commands, 19 files) | ~1100 | `cobra` + `charmbracelet/lipgloss` tables | L |
| Config (`~/.config/ai_houkai/config.toml`) | 1 file | `BurntSushi/toml` | none |
| `sentence-transformers` embeddings | external | see §4 | **H** |
| ChromaDB persistence + HNSW | external | see §4 | **M** |

Test suite (168 tests, 5 files) — port to Go's testing pkg + `testify`. The
isolation pattern (per-test `tmp_path`) maps cleanly to `t.TempDir()`.

---

## 3. Public surface to preserve

The Go port must remain **wire-compatible at the MCP layer**:

- 10 tool names with identical JSON schemas (`remember`, `recall`, `forget`,
  `list_recent`, `stats`, `link`, `unlink`, `neighbors`, `find_conflicts`,
  `supersede`). This is what Claude Code / Desktop / arbitrary MCP clients
  call — breaking it breaks every downstream config.
- Env vars `AI_HOUKAI_PATH`, `AI_HOUKAI_COLLECTION`.
- Console-script name `ai-houkai-mcp` (Go binary keeps the name).

Out of scope for compatibility (Python-internal):
- `MemoryStore` Python API.
- The Chroma on-disk format. We are free to break it; offer a one-shot
  `houkai migrate --from-chroma <path>` importer that reads the Chroma
  SQLite + parquet files and re-embeds into the Go store.

---

## 4. The two hard choices

### 4.1 Vector store

| Option | Embedded? | HNSW | Mature | Notes |
|---|---|---|---|---|
| **chromem-go** | ✅ in-proc | ❌ brute-force cosine | ✅ | Pure Go, file-backed. Fine for ≤10k memories — exactly AI-Houkai's scale. Closest spiritual match to Chroma's API. |
| **sqlite-vec** | ✅ in-proc (CGO) | ❌ (flat) / IVF | ✅ | Battle-tested SQLite; needs CGO. Vec extension is young but active. Good if we want SQL queries for free. |
| **Qdrant (embedded mode)** | ✅ (as library, Rust FFI) | ✅ | ✅ | Heavy, drags Rust. Overkill. |
| **bbolt + custom HNSW** | ✅ | ✅ if we write it | — | Most work; reinvents the wheel. |
| **External Qdrant/Weaviate** | ❌ network | ✅ | ✅ | Breaks single-binary promise. |

**Recommendation: `chromem-go`** for v1. At AI-Houkai's working scale (the
DESIGN.md itself says "hundreds to low thousands of entries — exact search
would also be fine") brute-force cosine over 384-dim vectors is sub-ms and
keeps the dependency surface minimal. Reassess if a user reports >50k
memories.

Fallback if chromem-go proves limiting: swap to `sqlite-vec` behind the
same `VectorBackend` interface (see §5).

### 4.2 Embeddings — `all-MiniLM-L6-v2` in Go

This is the **only genuinely hard part** of the port. Options:

| Option | Single-binary? | Setup cost for user | Notes |
|---|---|---|---|
| **ONNX Runtime Go (`yalue/onnxruntime_go`) + MiniLM ONNX export** | ⚠️ needs `onnxruntime.so` | medium (auto-download .onnx + tokenizer.json on first run) | Same numerical output as Python. CGO + shared lib. Tokenizer: `daulet/tokenizers` (Rust HF tokenizers via CGO) or `sugarme/tokenizer` (pure Go). |
| **Ollama embed endpoint (`/api/embeddings`)** | ✅ Go binary alone | high (user must install Ollama + pull `nomic-embed-text` or similar) | Trivial Go code (HTTP call). Different vector space than MiniLM — old Chroma DBs cannot be reused. |
| **llama.cpp via `go-llama.cpp`** | ⚠️ CGO | medium | Can run GGUF MiniLM. Heavier than ONNX for this size of model. |
| **Pure-Go transformer inference** | ✅ | — | No mature option for BERT-class models. Not viable. |
| **Remote API (OpenAI, Voyage, …)** | ✅ | low | Breaks "no API key required" core promise. Optional, not default. |

**Recommendation: tiered embedder**, picked at startup by the same
`EmbeddingFunction` abstraction Chroma uses:

1. Default: **ONNX Runtime + MiniLM** for byte-identical-ish output to
   Python. First-run auto-fetches `model.onnx` + `tokenizer.json` from
   HF into `~/.ai_houkai/models/`.
2. Opt-in via config: `embedder = "ollama"` → POST to local Ollama.
3. Opt-in via config: `embedder = "openai"` → API embeddings.

This mirrors the §15 "pluggable embeddings" extension point in the Python
design and keeps the *default UX* identical: no API key, fully local.

---

## 5. Proposed Go package layout

```
ai-houkai/                              go.mod = github.com/nexusriot/ai-houkai
├── cmd/
│   ├── ai-houkai-mcp/main.go           MCP server entry (stdio)
│   ├── houkai/main.go                  CLI entry (cobra root)
│   └── ai-houkai-install/main.go       installers entry
├── internal/
│   ├── memory/
│   │   ├── memory.go                   Memory, Link, MemoryType, Graph, HybridWeights, ExpandSpec
│   │   ├── store.go                    Store struct, Remember/Recall/Forget/...
│   │   ├── bm25.go
│   │   ├── conflict.go                 negation diff, conflict policies, ConflictError
│   │   ├── links.go                    link/unlink/neighbors/subgraph
│   │   ├── supersede.go
│   │   ├── hybrid.go                   α/β/γ/δ scoring
│   │   └── store_test.go
│   ├── decay/
│   │   ├── engine.go
│   │   └── engine_test.go
│   ├── reflect/
│   │   ├── engine.go                   clustering + summarizer iface
│   │   └── engine_test.go
│   ├── vector/                         vector backend abstraction
│   │   ├── backend.go                  interface: Add, Query, Get, Delete, UpdateMetadata, All
│   │   ├── chromem.go                  default backend
│   │   └── sqlitevec.go                optional (build tag)
│   ├── embed/                          embedding backend abstraction
│   │   ├── embedder.go                 interface: Embed([]string) ([][]float32, error)
│   │   ├── onnx_minilm.go              default (build tag: !no_onnx)
│   │   ├── ollama.go
│   │   └── openai.go
│   ├── mcpserver/
│   │   ├── server.go                   wires 10 tools to *memory.Store
│   │   ├── tools.go                    tool schemas (mirror FastMCP)
│   │   └── server_test.go
│   ├── installer/
│   │   ├── claude_code.go
│   │   └── claude_desktop.go
│   ├── cli/
│   │   ├── root.go                     cobra root + persistent flags
│   │   ├── config.go                   TOML + env + flag resolution chain
│   │   ├── output.go                   lipgloss table / TSV / JSON renderer
│   │   └── cmd_*.go                    one file per command group (~6 files)
│   └── version/version.go
└── examples/                           parity with Python examples/
```

---

## 6. Key interfaces (Go)

```go
type Embedder interface {
    Dim() int
    Embed(ctx context.Context, texts []string) ([][]float32, error)
}

type VectorBackend interface {
    Add(ctx context.Context, items []Item) error          // Item{ID, Vector, Metadata}
    Query(ctx context.Context, vec []float32, k int, where Filter) ([]Hit, error)
    Get(ctx context.Context, ids []string) ([]Item, error)
    All(ctx context.Context) ([]Item, error)              // for reflection clustering
    UpdateMetadata(ctx context.Context, id string, md map[string]any) error
    Delete(ctx context.Context, ids []string) error
    Count(ctx context.Context) (int, error)
    Close() error
}

type Memory struct {
    ID            string    `json:"id"`
    Text          string    `json:"text"`
    Type          string    `json:"type"`           // episodic|semantic|procedural|feedback
    Tags          []string  `json:"tags"`
    Importance    float32   `json:"importance"`
    CreatedAt     float64   `json:"created_at"`
    LastAccessed  float64   `json:"last_accessed"`
    AccessCount   int       `json:"access_count"`
    Source        string    `json:"source,omitempty"`
    Links         []Link    `json:"links"`
    SupersededBy  string    `json:"superseded_by"`
    SupersededAt  float64   `json:"superseded_at"`
    Polarity      int       `json:"polarity"`
}

type Summarizer func(ctx context.Context, ms []Memory) (string, error)
```

Mirroring Python's `ChromaDB metadata = scalars only` rule isn't required
for chromem-go (it accepts `map[string]string`), but keeping the same
serialisation (`tags` as comma-string, `links` as JSON) eases the optional
Chroma-import migration tool.

---

## 7. MCP server

`github.com/modelcontextprotocol/go-sdk` (official) — or `mark3labs/mcp-go`
as a fallback. Either provides stdio JSON-RPC 2.0 transport and tool
registration. The 10 tools are thin wrappers that JSON-marshal args into
`memory.Store` method calls. Schemas must match Python byte-for-byte —
copy them from `mcp_server/server.py` and unit-test against a JSON
fixture per tool.

---

## 8. CLI

- Framework: `spf13/cobra` (matches the per-file-per-command shape).
- Tables: `charmbracelet/lipgloss` for TTY, plain TSV for pipes, JSON via
  `--format json`. Mirrors Python `output.py` rendering rules.
- Editor flow (`houkai edit`): same `$EDITOR` → `nano` chain.
- Confirmation for destructive ops: `--yes` to skip; default interactive.
- ID prefix resolution: 8-char prefix → full UUID; linear scan; ambiguity
  error — identical semantics to Python.

---

## 9. Migration story

A one-shot subcommand:

```
houkai migrate --from-chroma ~/.ai_houkai/.chroma --collection ai_houkai
```

Reads Chroma's SQLite + parquet directly (or shells out to a tiny Python
helper distributed alongside, if reading parquet/HNSW from Go is too
painful), pulls `(id, text, metadata)` triples, **re-embeds** with the
configured Go embedder, and writes into chromem-go. Re-embedding is
required because (a) we may not use MiniLM, and (b) different ONNX
exports of MiniLM can produce vectors that are *close* but not identical
to the Python sentence-transformers output. Recall quality is unaffected
as long as embedder and queries use the same model.

If the user stays on ONNX MiniLM and we ship a verified ONNX export, we
*could* skip re-embedding by reading the stored vectors. Not worth the
complexity for v1.

---

## 10. Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| ONNX MiniLM vectors drift from Python's | M | Pin ONNX export commit; ship cosine-sim sanity test against fixtures. |
| `daulet/tokenizers` requires CGO + Rust toolchain at build | M | Use `sugarme/tokenizer` (pure Go) as fallback, or ship prebuilt binaries via GoReleaser. |
| chromem-go scaling cliff at ~50k vectors | L | Backend is interfaced; swap to sqlite-vec without touching memory pkg. |
| MCP Go SDK still v0.x churn | M | Pin minor version; abstract tool registration in `internal/mcpserver`. |
| Reflection needs raw vectors; chromem-go must expose them | L | Verified: chromem-go stores and exposes embeddings. If not, cache vectors alongside in metadata. |
| Single-binary promise broken by `onnxruntime.so` | M | Use `goreleaser` to ship per-OS bundles including the .so; or fall back to Ollama-embedder mode (warn at startup). |
| 168-test parity gap | L | Port tests file-by-file; require green parity before deleting Python. |

---

## 11. Phased plan

**Phase 0 — Spike (1–2 days)**
- Stand up `embed/onnx_minilm.go` against a known ONNX export of
  all-MiniLM-L6-v2. Verify cosine sim ≥ 0.999 vs Python output on a
  10-sentence fixture.
- Stand up `vector/chromem.go` with Add/Query/Get/All round-trip.
- Gate: if either spike misses, revisit §4.

**Phase 1 — Core memory (3–4 days)**
- Port `Memory`, `Link`, `Store` (Remember/Recall/Forget/ListRecent/Stats).
- BM25 + hybrid scoring.
- Port `test_memory.py` (18 tests).

**Phase 2 — Decay + Reflection (2 days)**
- Port engines + 32 tests.

**Phase 3 — Linking + Conflict (2 days)**
- link/unlink/neighbors/subgraph/supersede/restore.
- find_conflicts + on_conflict policies.

**Phase 4 — MCP server (2 days)**
- 10 tools, stdio transport, env-var config.
- JSON-schema fixture tests verifying byte-for-byte parity with Python
  tool definitions.

**Phase 5 — CLI (3–4 days)**
- 22 commands, output renderer, config resolution, id-prefix resolver.
- Port `test_cli.py` (11 round-trip tests).

**Phase 6 — Installers + packaging (2 days)**
- Claude Code / Claude Desktop JSON patchers.
- GoReleaser config: linux/macos/windows × amd64/arm64.
- `houkai migrate --from-chroma`.

**Phase 7 — Hardening (1 week)**
- Cross-platform smoke tests.
- Real-world dogfood: replace the MCP server in this very `settings.json`
  and run for a week.

Total realistic estimate: **3–4 weeks** of focused part-time work for a
single developer comfortable in both languages.

---

## 12. Open questions

1. Keep the Python project alive in parallel, or hard-cut? Recommend
   parallel for one release cycle; mark Python "maintenance only".
2. Repo: same repo with a `go/` subdir, or new repo `ai-houkai-go`?
   Recommend same repo, `go/` subdir, shared `DESIGN.md` and issues.
3. Do we want a built-in HTTP MCP transport (in addition to stdio)? Easy
   to add in Go; not in scope for v1.
4. Is there appetite to drop MiniLM entirely and require Ollama? Would
   simplify packaging dramatically (no CGO/ONNX). Tradeoff: harder
   first-run UX.
