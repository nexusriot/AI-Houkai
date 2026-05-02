# AI-Houkai — Agent Memory System

A minimal long-term memory system for AI agents, backed by **ChromaDB**
and exposed over **MCP**.  Four usage modes: standalone, Claude, OpenAI,
and Ollama (fully local).

<p align="center" width="100%">
    <img width="70%" src="logo.png">
</p>


## Layout

```
AI-Houkai/
├── memory_system/        # core library — MemoryStore + Memory dataclass
│   ├── __init__.py
│   └── store.py
├── mcp_server/
│   └── server.py         # FastMCP server (remember/recall/forget/stats)
├── examples/
│   ├── standalone.py     # pure Python walkthrough, no LLM
│   ├── claude_agent.py   # Claude Sonnet + Anthropic tool use
│   ├── openai_agent.py   # OpenAI GPT-4o / gpt-4o-mini + function calling
│   └── ollama_agent.py   # Ollama local LLM (llama3.1 / mistral-nemo)
├── tests/
│   ├── conftest.py       # in-memory ChromaDB fixture
│   ├── test_memory.py    # pytest unit tests for MemoryStore
│   └── test_dispatch.py  # cross-provider dispatch tests (Claude/OpenAI/Ollama)
└── requirements.txt
```

## Design

Each memory is a document with structured metadata:

| field           | purpose                                              |
| --------------- | ---------------------------------------------------- |
| `type`          | `episodic` / `semantic` / `procedural` / `feedback`  |
| `tags`          | freeform topical tags                                |
| `importance`    | 0..1 weight; supports `min_importance` filter        |
| `created_at`    | first write time                                     |
| `last_accessed` / `access_count` | bumped on every recall hit          |
| `source`        | optional origin (user, tool, agent)                  |

ChromaDB stores documents with cosine-space HNSW. Embeddings come from
`sentence-transformers/all-MiniLM-L6-v2` — runs fully local, no API
keys needed.  Swap `embedding_model` in `MemoryStore(...)` to use
OpenAI/Cohere/etc.

## Install

```bash
pip install -r requirements.txt
```

## Run the tests

```bash
cd AI-Houkai
pytest tests/ -v
```

## Standalone demo (no LLM)

Shows the full memory lifecycle — seed → recall with filters → access
tracking → forget.

```bash
python examples/standalone.py
```

## Claude agent

Conversational REPL where Claude calls `recall`, `remember`, and
`forget` as native tools (Anthropic tool-use API).

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python examples/claude_agent.py

# optional: persist memory across sessions
AI_HOUKAI_PATH=/tmp/my_memory python examples/claude_agent.py
```

Special commands in the REPL:
- `memories` — list recent stored memories
- `quit` — exit

## OpenAI agent

Conversational REPL using GPT-4o / gpt-4o-mini with OpenAI function
calling.  The `_dispatch_tool` logic is identical to the Claude and
Ollama examples — only the SDK and message format differ.

```bash
export OPENAI_API_KEY=sk-...
python examples/openai_agent.py

# Override model or persist memory:
OPENAI_MODEL=gpt-4o AI_HOUKAI_PATH=/tmp/oai python examples/openai_agent.py
```

Environment variables:
- `OPENAI_MODEL`      — default `gpt-4o-mini`
- `AI_HOUKAI_PATH` — ChromaDB persistence directory

Special commands in the REPL:
- `memories` — list recent stored memories (with access counts)
- `quit` — exit

## Ollama agent (fully local)

Same agent pattern but with a local model via Ollama's
OpenAI-compatible endpoint.  No API key, no internet.

```bash
# Install Ollama: https://ollama.com
ollama pull llama3.1        # 8B, good tool-call support
# or: ollama pull mistral-nemo

OLLAMA_MODEL=llama3.1 python examples/ollama_agent.py
```

Environment variables:
- `OLLAMA_BASE_URL` — default `http://localhost:11434/v1`
- `OLLAMA_MODEL`    — default `llama3.1`
- `AI_HOUKAI_PATH` — ChromaDB persistence directory

## MCP server

Exposes the memory store to any MCP client (e.g. Claude Code).

```bash
python -m mcp_server.server
```

Connect it to Claude Code by adding to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "AI-Houkai": {
      "command": "python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "/path/to/AI-Houkai",
      "env": {
        "AI_HOUKAI_PATH": "/path/to/AI-Houkai/.chroma"
      }
    }
  }
}
```

Exposed tools: `remember`, `recall`, `forget`, `list_recent`, `stats`.

## Extension ideas

- **Decay / forgetting** — prune by `last_accessed` + `importance`.
- **Reflection** — periodically summarise episodic clusters into
  semantic memories (Generative Agents pattern).
- **Hybrid scoring** — `score = α·sim + β·recency + γ·importance`.
- **Multi-user / multi-agent** — separate collections per user.
