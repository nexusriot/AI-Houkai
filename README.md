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
| **MCP server** | 10 tools for any MCP client (Claude Code, Claude Desktop) |
| **CLI (`houkai`)** | Full-featured terminal interface — CRUD, graph, maintenance, I/O |
| **Multi-provider** | Claude · OpenAI · Ollama (local) agent examples |

## Layout

```
AI-Houkai/
├── ai_houkai/
│   ├── __init__.py               # convenience re-exports
│   ├── memory_system/
│   │   ├── __init__.py
│   │   ├── store.py              # MemoryStore + Memory dataclass
│   │   ├── decay.py              # DecayEngine — exponential forgetting
│   │   └── reflection.py        # ReflectionEngine — episodic → semantic
│   ├── mcp_server/
│   │   ├── __init__.py
│   │   └── server.py             # FastMCP server (10 tools)
│   ├── cli/
│   │   ├── __init__.py
│   │   ├── __main__.py           # python -m ai_houkai.cli
│   │   ├── main.py               # Typer app + _main() entrypoint
│   │   ├── config.py             # store path / collection resolution
│   │   ├── output.py             # rich tables, TSV, JSON, id prefix helpers
│   │   └── commands/
│   │       ├── remember.py       # houkai remember
│   │       ├── recall.py         # houkai recall
│   │       ├── list_cmd.py       # houkai list
│   │       ├── show.py           # houkai show
│   │       ├── forget.py         # houkai forget
│   │       ├── edit.py           # houkai edit / tag / bump
│   │       ├── link.py           # houkai link / unlink / neighbors / graph
│   │       ├── conflicts.py      # houkai conflicts / supersede / restore
│   │       ├── decay.py          # houkai prune
│   │       ├── reflect.py        # houkai reflect
│   │       ├── io.py             # houkai export / import / backup
│   │       └── stats.py          # houkai stats
│   └── installers/
│       ├── __init__.py
│       └── claude_code.py        # ClaudeCodeInstaller — register MCP w/ Claude Code
├── examples/
│   ├── 01_standalone.py          # pure-Python walkthrough, no LLM
│   ├── 02_ollama_local_network.py  # Ollama on LAN, fully offline
│   ├── 03_claude_desktop.py      # MCP auto-install for Claude Desktop
│   ├── 04_openai.py              # OpenAI GPT-4o / gpt-4o-mini
│   ├── 05_decay_reflection.py    # decay + reflection demo
│   ├── 06_claude_code.py         # Claude Code MCP integration
│   ├── claude_agent.py           # Claude Sonnet REPL (Anthropic SDK)
│   └── pip_package_example.py   # post-install usage walkthrough
├── tests/
│   ├── conftest.py               # isolated MemoryStore fixture (tmp_path)
│   ├── test_memory.py            # MemoryStore unit tests
│   ├── test_decay.py             # DecayEngine unit tests
│   ├── test_reflection.py        # ReflectionEngine unit tests
│   └── test_dispatch.py          # cross-provider _dispatch_tool tests
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
pip install "ai-houkai[all]"      # all providers + CLI
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
# Store a memory
houkai remember "Python's GIL blocks CPU parallelism" \
  --type semantic --tag python --importance 0.85

# Read from stdin (pipe-friendly)
echo "Deploy with: make release" | houkai remember --stdin --type procedural

# Semantic search
houkai recall "parallel execution" -k 5
houkai recall "deploy" --mode hybrid --format json

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
```

### Import / export / backup

```bash
# Export to JSONL (stdout or file)
houkai export
houkai export dump.jsonl --type episodic --tag project

# Import from JSONL (deduplicates by text hash by default)
houkai import dump.jsonl
houkai import dump.jsonl --dedupe-by id --yes

# Snapshot the Chroma store
houkai backup   # → ~/.ai_houkai/backups/<ISO timestamp>/
```

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
default_importance  = 0.5
editor              = "nvim"
```

---

## Roadmap

Designs for upcoming features live in [PROPOSALS.md](PROPOSALS.md).

## Run the tests

```bash
pytest tests/ -v        # 168 tests
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

Exposed tools: `remember` · `recall` · `forget` · `list_recent` · `stats`.

Environment variables:

| Variable | Default |
|---|---|
| `AI_HOUKAI_PATH` | `./.chroma` |
| `AI_HOUKAI_COLLECTION` | `ai_houkai` |

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

Plug in any summarizer — including an LLM:

```python
def llm_summarizer(memories):
    prompt = "\n".join(m.text for m in memories)
    return openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Summarise: {prompt}"}],
    ).choices[0].message.content

engine = ReflectionEngine(store, summarizer=llm_summarizer)
```
