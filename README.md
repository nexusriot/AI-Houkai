# AI-Houkai — Agent Memory System

A long-term memory system for AI agents backed by **ChromaDB** and
exposed over **MCP**.  Agents can remember, recall, and forget
information across sessions — with automatic decay of stale memories
and periodic reflection that condenses experience into knowledge.

<p align="center" width="100%">
    <img width="70%" src="logo.png">
</p>

## Features

| Feature | Description |
|---|---|
| **Vector search** | Cosine-space HNSW via ChromaDB + sentence-transformers |
| **Memory types** | `episodic` · `semantic` · `procedural` · `feedback` |
| **Rich metadata** | `importance`, `tags`, `source`, access tracking |
| **Decay** | Exponential forgetting — prune old, unimportant memories |
| **Reflection** | Cluster episodic memories → condense into semantic summaries |
| **MCP server** | Five tools for any MCP client (Claude Code, Claude Desktop) |
| **Multi-provider** | Claude · OpenAI · Ollama (local) agent examples |

## Layout

```
AI-Houkai/
├── memory_system/
│   ├── __init__.py           # public exports
│   ├── store.py              # MemoryStore + Memory dataclass
│   ├── decay.py              # DecayEngine — exponential forgetting
│   └── reflection.py         # ReflectionEngine — episodic → semantic
├── mcp_server/
│   └── server.py             # FastMCP server (remember/recall/forget/…)
├── examples/
│   ├── 01_standalone.py      # pure-Python walkthrough, no LLM
│   ├── 02_ollama_local_network.py  # Ollama on LAN, fully offline
│   ├── 03_claude_desktop.py  # MCP auto-install for Claude Desktop
│   ├── 04_openai.py          # OpenAI GPT-4o / gpt-4o-mini
│   ├── 05_decay_reflection.py # decay + reflection demo
│   └── claude_agent.py       # Claude Sonnet REPL (Anthropic SDK)
├── tests/
│   ├── conftest.py           # isolated MemoryStore fixture (tmp_path)
│   ├── test_memory.py        # MemoryStore unit tests
│   ├── test_decay.py         # DecayEngine unit tests
│   ├── test_reflection.py    # ReflectionEngine unit tests
│   └── test_dispatch.py      # cross-provider _dispatch_tool tests
└── requirements.txt
```

## Install

```bash
pip install -r requirements.txt
```

> The embedding model (`all-MiniLM-L6-v2`) downloads automatically on
> first use (~90 MB).  Everything runs fully local — no API key required
> for the memory layer.

## Run the tests

```bash
pytest tests/ -v        # 79 tests
```

---

## Examples

### 01 · Standalone (no LLM)

Full memory lifecycle — seed → recall with filters → access tracking
→ forget.  Good starting point before adding an LLM.

```bash
python examples/01_standalone.py
```

### 02 · Ollama (local network)

Conversational REPL using a local model over Ollama's
OpenAI-compatible endpoint.  No API key, no internet.

```bash
# Install Ollama: https://ollama.com
ollama pull llama3.1

OLLAMA_MODEL=llama3.1 python examples/02_ollama_local_network.py
```

| Env var | Default |
|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` |
| `OLLAMA_MODEL` | `llama3.1` |
| `AI_HOUKAI_PATH` | `./.chroma` |

### 03 · Claude Desktop (MCP)

Auto-installs the MCP server into Claude Desktop's config so Claude
can call memory tools without any prompting.

```bash
# Preview the config block
python examples/03_claude_desktop.py

# Patch Claude Desktop config + restart Claude Desktop
python examples/03_claude_desktop.py --install
```

### 04 · OpenAI

GPT-4o / gpt-4o-mini with function calling.

```bash
export OPENAI_API_KEY=sk-...
python examples/04_openai.py

# override model or persist memory
OPENAI_MODEL=gpt-4o AI_HOUKAI_PATH=/tmp/oai python examples/04_openai.py
```

| Env var | Default |
|---|---|
| `OPENAI_MODEL` | `gpt-4o-mini` |
| `AI_HOUKAI_PATH` | `./.chroma` |

### 05 · Decay + Reflection

Shows both cognitive maintenance features with backdated timestamps so
results are visible without waiting days.

```bash
python examples/05_decay_reflection.py
```

### Claude agent (Anthropic SDK REPL)

Conversational REPL where Claude calls `recall`, `remember`, and
`forget` as native Anthropic tool-use calls.

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python examples/claude_agent.py

# persist memory across sessions
AI_HOUKAI_PATH=/tmp/my_memory python examples/claude_agent.py
```

REPL commands: `memories` to list recent memories · `quit` to exit.

---

## MCP server

Exposes the memory store to any MCP client.

```bash
python -m mcp_server.server
```

**Claude Code** — add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "AI-Houkai": {
      "command": "python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "/path/to/AI-Houkai",
      "env": { "AI_HOUKAI_PATH": "/path/to/AI-Houkai/.chroma" }
    }
  }
}
```

**Claude Desktop** — use `examples/03_claude_desktop.py --install`
(handles the platform-specific config path automatically).

Exposed tools: `remember` · `recall` · `forget` · `list_recent` · `stats`.

---

## Decay

Memories fade over time based on how long ago they were last accessed
and how important they are.

```
score = importance × exp(−λ × days_since_last_access)
```

Default `λ = 0.1` gives a half-life of ~7 days for a 0.5-importance
memory.  `procedural` memories are protected and never pruned.

```python
from memory_system import MemoryStore, DecayEngine

store  = MemoryStore()
engine = DecayEngine(store, decay_rate=0.1, min_score=0.05)

engine.prune(dry_run=True)   # preview what would be removed
engine.prune()               # delete stale memories
```

## Reflection

Clusters of semantically similar episodic memories are condensed into
a single `semantic` summary memory (the Generative Agents pattern).

```python
from memory_system import MemoryStore, ReflectionEngine

store  = MemoryStore()
engine = ReflectionEngine(store, similarity_threshold=0.75)

engine.clusters()            # inspect detected clusters
engine.reflect(dry_run=True) # preview summaries without writing
engine.reflect(consolidate=True)  # create summaries + delete sources
```

Plug in any summarizer — including an LLM call:

```python
def llm_summarizer(memories):
    prompt = "\n".join(m.text for m in memories)
    return openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Summarise: {prompt}"}],
    ).choices[0].message.content

engine = ReflectionEngine(store, summarizer=llm_summarizer)
```

---

## Quick-start (5 lines)

```python
from memory_system import MemoryStore

store = MemoryStore()                               # persists to ./.chroma
store.remember("Python's GIL limits CPU threads", type="semantic", importance=0.8)

hits = store.recall("parallel execution in Python", k=3)
for mem, score in hits:
    print(f"{score:.2f}  {mem.text}")
```
