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
| **Decay** | Exponential forgetting — prune old, unimportant memories |
| **Reflection** | Cluster episodic memories → condense into semantic summaries |
| **Memory linking** | Typed directed edges — `refines`, `supersedes`, `derived_from`, … |
| **Conflict detection** | Duplicate / contradiction scan with configurable policies |
| **Hybrid retrieval** | Cosine + BM25 + recency + importance blended scoring |
| **Context packing** | `recall_pack` — assemble top-ranked memories into a token-budgeted, ready-to-inject block |
| **Importance auto-assignment** | Heuristic 0–1 scoring from text (instructions > decisions > completions > observations) |
| **Bulk ingest** | `houkai ingest` — chunk files/stdin into memories (markdown-aware) |
| **Collections** | `houkai collections` — list/create/delete/copy namespaces in one store |
| **TUI browser** | `houkai tui` — Textual UI: search, detail pane, link-graph walking |
| **Scheduled maintenance** | Automatic decay + reflection daemon — cron, foreground loop, or background daemon |
| **Audit journal** | Append-only JSONL log of every mutation — `journal tail` / `journal show` / `journal undo` |
| **Portable import/export** | Gzipped `.ahkai` archives with embedded vectors, conflict policies, dry-run |
| **MCP server** | 15 tools for any MCP client (Claude Code, Claude Desktop) |
| **CLI (`houkai`)** | Full-featured terminal interface — CRUD, graph, maintenance, I/O |
| **Multi-provider** | Claude · OpenAI · Ollama (local) agent examples |

## Layout

```
AI-Houkai/
├── ai_houkai/
│   ├── __init__.py               # convenience re-exports
│   ├── memory_system/
│   │   ├── __init__.py
│   │   ├── store.py              # MemoryStore + Memory dataclass (+ export/import/undo)
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
│   │   └── server.py             # FastMCP server (15 tools)
│   ├── cli/
│   │   ├── __init__.py
│   │   ├── __main__.py           # python -m ai_houkai.cli
│   │   ├── main.py               # Typer app + _main() entrypoint
│   │   ├── config.py             # store path / collection resolution
│   │   ├── output.py             # rich tables, TSV, JSON, id prefix helpers
│   │   └── commands/
│   │       ├── remember.py       # houkai remember
│   │       ├── recall.py         # houkai recall
│   │       ├── pack.py           # houkai pack
│   │       ├── list_cmd.py       # houkai list
│   │       ├── show.py           # houkai show
│   │       ├── forget.py         # houkai forget
│   │       ├── edit.py           # houkai edit / tag / bump
│   │       ├── link.py           # houkai link / unlink / neighbors / graph
│   │       ├── conflicts.py      # houkai conflicts / supersede / restore
│   │       ├── decay.py          # houkai prune
│   │       ├── reflect.py        # houkai reflect
│   │       ├── maintenance.py    # houkai maintenance tick/run/start/stop/status
│   │       ├── journal.py        # houkai journal tail/show/undo
│   │       ├── io.py             # houkai export / import / info / backup
│   │       ├── ingest.py         # houkai ingest
│   │       ├── collections.py    # houkai collections list/create/delete/copy
│   │       ├── tui_cmd.py        # houkai tui
│   │       └── stats.py          # houkai stats
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
├── tests/                        # 330 tests across 16 files
│   ├── conftest.py               # isolated MemoryStore fixture (tmp_path)
│   ├── test_memory.py            # MemoryStore unit tests
│   ├── test_decay.py             # DecayEngine unit tests
│   ├── test_reflection.py        # ReflectionEngine unit tests
│   ├── test_dispatch.py          # cross-provider _dispatch_tool tests
│   ├── test_hybrid.py            # hybrid retrieval scoring
│   ├── test_links.py             # typed links / neighbors / subgraph
│   ├── test_conflicts.py         # conflict detection + supersede/restore
│   ├── test_cli.py               # houkai CLI round-trips
│   ├── test_pack.py              # recall_pack token budgeting
│   ├── test_journal.py           # audit journal tail/show/undo
│   ├── test_export_import.py     # portable .ahkai archives
│   ├── test_maintenance.py       # scheduler, daemon, durations, state
│   ├── test_summarizers.py       # LLM summarizer specs, fallback, wiring
│   ├── test_importance.py        # heuristic importance tiers + wiring
│   ├── test_ingest.py            # chunk_text + ingest/collections commands
│   └── test_tui.py               # TUI view models + Textual pilot runs
├── pyproject.toml
└── requirements.txt
```

## Install

Modern Linux distros protect the system Python (PEP 668).  Pick whichever
approach fits your workflow — none requires `--break-system-packages`.

### Virtual environment (recommended for development)

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install ai-houkai
```

### pipx (recommended for the MCP server / CLI tool)

`pipx` installs CLI tools into isolated venvs and puts the script on your
PATH automatically — no activation step needed.

```bash
sudo apt install pipx            # or: pip install --user pipx
pipx ensurepath                  # adds ~/.local/bin to PATH (one-time)

pipx install ai-houkai
ai-houkai-mcp                    # available everywhere
```

### uv (fastest, modern)

```bash
curl -Lsf https://astral.sh/uv/install.sh | sh   # one-time install

uv venv && uv pip install ai-houkai               # project venv
# or run a script without a persistent install:
uv run --with ai-houkai python examples/pip_package_example.py
```

### Extras

```bash
pip install "ai-houkai[claude]"   # + Anthropic SDK
pip install "ai-houkai[openai]"   # + OpenAI SDK (also covers Ollama)
pip install "ai-houkai[cli]"      # + houkai terminal CLI (typer + rich)
pip install "ai-houkai[tui]"      # + houkai tui memory browser (textual)
pip install "ai-houkai[all]"      # all providers + CLI + TUI
pip install "ai-houkai[dev]"      # + pytest + CLI
```

> The embedding model (`all-MiniLM-L6-v2`) downloads automatically on
> first use (~90 MB).  Everything runs fully local — no API key required
> for the memory layer itself.

## Quick-start

```python
from ai_houkai.memory_system import MemoryStore

store = MemoryStore()                  # persists to ./.chroma

store.remember("Python's GIL blocks CPU parallelism",
               type="semantic", importance=0.85, tags=["python"])

for mem, score in store.recall("parallel execution", k=3):
    print(f"{score:.3f}  {mem.text}")

# Or assemble a token-budgeted block ready to drop into a prompt:
pack = store.recall_pack("parallel execution", token_budget=600)
print(pack.text)          # "## Relevant memory\n- (semantic) Python's GIL …"
print(pack.used_tokens, "/", pack.budget, "truncated:", pack.truncated)
```

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

# Pack the most relevant memories into a token-budgeted context block.
# The block prints to stdout (pipe it into a prompt); a summary goes to stderr.
houkai pack "how do we deploy and test" --budget 800
houkai pack "deploy" --budget 500 > context.md      # clean block, no summary
houkai pack "deploy" --format json                  # block + per-item scores/tokens

# List recent memories
houkai list
houkai list --type episodic --tag python --since 7d --sort importance

# Inspect one memory (full metadata + link chain)
houkai show 72be7903

# Delete memories (confirms unless --yes)
houkai forget 72be7903
houkai forget id1 id2 id3 --yes
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

### Memory graph

```bash
# Link two memories
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

# Actually delete (requires --apply)
houkai prune --apply --yes

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

Every mutation (`remember`, `forget`, `supersede`, `restore`, `link`,
`unlink`, `import`, `export`, `reflect`, `decay`, `undo`) is appended
to an append-only JSONL journal next to the store (`journal.log`,
rotated and gzipped at 64 MB, 90-day retention by default). Entries
carry the actor (`cli` / `mcp` / `reflection` / `decay` / `import` /
`lib`) plus `before`/`after` snapshots where applicable.

```bash
# Tail recent entries (newest first)
houkai journal tail
houkai journal tail -n 50 --op supersede --actor reflection
houkai journal tail --all          # include rotated archives

# Pretty-print one entry by timestamp
houkai journal show 1748284800.123

# Reverse a single operation (remember/forget/supersede/restore/link/unlink)
houkai journal undo 1748284800.123 --yes
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
`r` for recent, `q` to quit. Read-only — it never mutates the store.

### Stats

```bash
houkai stats                 # rich table
houkai stats --format json   # machine-readable
```

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

## Go port

A full Go port lives in [`go/`](go/) — two static binaries (`houkai` CLI +
`ai-houkai-mcp` MCP server), no Python runtime, packaged as a Debian `.deb`
and macOS tarballs. It is at feature parity with this Python version: the
same 15 MCP tools, hybrid retrieval, decay/reflection with LLM summarizers,
context packing, bulk ingest, collections, importance auto-assignment,
installers for Claude Code / Cursor / OpenCode, and a Bubble Tea TUI.
Embeddings are delegated to Ollama (default), OpenAI, or DigitalOcean
instead of bundled sentence-transformers, and the store is
[chromem-go](https://github.com/philippgille/chromem-go) — not
binary-compatible with ChromaDB, but the portable `.ahkai` export/import
format bridges the two. See [go/README.md](go/README.md).

## Design docs

In-depth design notes live in [DESIGN.md](https://raw.githubusercontent.com/nexusriot/AI-Houkai/main/DESIGN.md).
The original feature proposals for hybrid retrieval, conflict detection, and
memory linking — all now shipped — are archived in
[PROPOSALS.md](https://raw.githubusercontent.com/nexusriot/AI-Houkai/main/PROPOSALS.md).
The Go port has its own internals doc in [go/DESIGN.md](go/DESIGN.md); the
original porting feasibility study is archived in
[GO_PORT_DESIGN.md](GO_PORT_DESIGN.md).

## Run the tests

```bash
pytest tests/ -v
```

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

# Option C — auto-patch ~/.claude/settings.json via the example script
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

The installed MCP block in `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "ai-houkai": {
      "command": "ai-houkai-mcp",
      "env": {
        "AI_HOUKAI_PATH": "~/.ai_houkai",
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

ClaudeCodeInstaller(memory_path="~/.ai_houkai").install()
```

#### Recommended CLAUDE.md addition

Add the following to your project's `CLAUDE.md` so Claude Code knows when and
how to use memory tools (run `ai-houkai-install-claude-code --claudemd`
or `python examples/06_claude_code.py --claudemd` to generate it):

```markdown
## Memory (AI-Houkai MCP)

- **remember** — store conventions, decisions, preferences
- **recall** — search before starting any task
- **forget** — remove outdated facts

| Situation | Action |
|---|---|
| User states a convention | `remember` with `type="procedural"` |
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

## MCP server

Exposes the memory store to any MCP client.

```bash
ai-houkai-mcp
# or: python -m ai_houkai.mcp_server.server
```

Exposed tools (15):

- **Core** — `remember` · `recall` · `recall_pack` · `forget` · `list_recent` · `stats`
- **Linking** — `link` · `unlink` · `neighbors`
- **Conflicts** — `find_conflicts` · `supersede`
- **Maintenance & audit** — `maintenance_tick` · `journal_tail` · `export` · `import`

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

**Claude Code** (manual `~/.claude/settings.json`):

```json
{
  "mcpServers": {
    "ai-houkai": {
      "command": "ai-houkai-mcp",
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
        "AI_HOUKAI_PATH": "~/.ai_houkai",
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
        "AI_HOUKAI_PATH": "~/.ai_houkai",
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

CursorInstaller(memory_path="~/.ai_houkai").install()
OpenCodeInstaller(memory_path="~/.ai_houkai", settings_path="opencode.json").install()
```

---

## Decay

Memories fade over time based on age and importance.

```
score = importance × exp(−λ × days_since_last_access)
```

Default `λ = 0.1` → half-life ≈ 7 days for a 0.5-importance memory.
`procedural` memories are protected and never pruned.

```python
from ai_houkai.memory_system import MemoryStore, DecayEngine

store  = MemoryStore()
engine = DecayEngine(store, decay_rate=0.1, min_score=0.05)

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
run — safe to call as often as you like.

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
enabled       = true
decay_every   = "24h"     # or "off" to disable decay
reflect_every = "7d"      # or "off" to disable reflection
tick_interval = "5m"      # how often the loop wakes to check schedules

[maintenance.decay]
decay_rate    = 0.1
min_score     = 0.05
protect_types = ["procedural"]   # never pruned

[maintenance.reflect]
min_cluster_size = 3
apply            = false   # set true to write reflection summaries
summarizer       = "ollama:llama3.1"   # optional; default extractive (no LLM)
```

Supported duration units: `s` · `m` · `h` · `d` · `w`.
Default `decay_every = "24h"`, `reflect_every = "7d"`, `tick_interval = "5m"`.

> **reflect.apply = false** (default): reflection runs in observe-only mode —
> it logs how many summaries *would* be created but writes nothing.  Flip to
> `true` once you're happy with the clusters.

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
print(result.summary())   # "decay pruned 3 | reflect nothing to reflect"

# Blocking loop:
stop = threading.Event()
sched.run_forever(stop)   # blocks; call stop.set() to exit
```

### MCP tool

The `maintenance_tick` tool lets any MCP client trigger a tick without
running the CLI:

```
maintenance_tick(reflect_apply=false)
→ {"summary": "decay pruned 2", "decayed": 2, "reflected": 0, ...}
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
