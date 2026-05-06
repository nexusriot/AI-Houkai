# AI-Houkai — Architecture & Design

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
│   remember()  recall()  forget()  list_recent()  count()     │
│   link()  unlink()  neighbors()  subgraph()                  │
│   supersede()  restore()  find_conflicts()                   │
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
│   │                             DecayEngine, ReflectionEngine
│   ├── store.py                  MemoryStore + dataclasses + BM25 + conflict
│   ├── decay.py                  DecayEngine
│   └── reflection.py            ReflectionEngine
├── mcp_server/
│   ├── __init__.py
│   └── server.py                 FastMCP server — 10 tools
├── cli/
│   ├── __init__.py
│   ├── __main__.py               python -m ai_houkai.cli
│   ├── main.py                   Typer app, shared --store/--collection flags
│   ├── config.py                 env → config file → defaults resolution
│   ├── output.py                 rich / TSV / JSON, id prefix, fmt_age
│   └── commands/
│       ├── remember.py           houkai remember
│       ├── recall.py             houkai recall
│       ├── list_cmd.py           houkai list
│       ├── show.py               houkai show
│       ├── forget.py             houkai forget
│       ├── edit.py               houkai edit / tag / bump
│       ├── link.py               houkai link / unlink / neighbors / graph
│       ├── conflicts.py          houkai conflicts / supersede / restore
│       ├── decay.py              houkai prune  (wraps DecayEngine)
│       ├── reflect.py            houkai reflect (wraps ReflectionEngine)
│       ├── io.py                 houkai export / import / backup
│       └── stats.py              houkai stats
└── installers/
    ├── __init__.py               re-exports ClaudeCodeInstaller
    └── claude_code.py            patches ~/.claude/settings.json
```

Import styles:

```python
# Subpackage (canonical)
from ai_houkai.memory_system import MemoryStore, DecayEngine, ReflectionEngine

# Top-level convenience re-export
from ai_houkai import MemoryStore, DecayEngine, ReflectionEngine
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
```

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

### Metadata serialisation

ChromaDB metadata values must be scalar.  Tags are stored as a
comma-joined string; `links` are JSON-encoded:

```python
# write
{
    "tags":          "deploy,api,prod",
    "links":         '[{"to":"a8...","rel":"refines"}]',
    "superseded_by": "",
    "superseded_at": 0.0,
    "polarity":      0,
}

# read — all new fields default safely for old records
tags  = [t for t in meta.get("tags", "").split(",") if t]
links = json.loads(meta.get("links") or "[]")
```

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
                 │    remember(text)   │
                 └────────┬────────────┘
                          │  UUID assigned
                          │  text embedded (384-dim)
                          │  metadata written
                          ▼
                 ┌─────────────────────┐
                 │   ChromaDB HNSW     │◄── persists to disk
                 └────────┬────────────┘
          ┌───────────────┼────────────────────┐
          ▼               ▼                    ▼
   recall(query)    list_recent()          forget(id)
          │               │                    │
     vector search   chronological        hard delete
     metadata filter    sort              returns bool
          │
          ▼
    _touch(memory)
    ├── last_accessed = now
    └── access_count += 1
```

#### recall() filtering pipeline

1. Build `where` dict from `type` and `min_importance` args.
2. Call `collection.query(n_results=k, where=where)`.
3. Post-filter by `tag` (ChromaDB only supports `$eq` on scalar fields,
   not array membership — tag filtering happens in Python).
4. Call `_touch()` on every returned memory.
5. Convert cosine distance → similarity score and return.

---

## 6. Decay Engine

### Formula

```
score(m) = importance × exp(−λ × days_since_last_access)
```

| Parameter | Default | Effect |
|---|---|---|
| `decay_rate` (λ) | `0.1` | Half-life ≈ 7 days for importance=0.5 |
| `min_score` | `0.05` | Prune threshold |
| `protect_types` | `("procedural",)` | Types immune to pruning |

Score examples with λ=0.1:

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

### API

```python
from ai_houkai.memory_system import MemoryStore, DecayEngine

engine = DecayEngine(store, decay_rate=0.1, min_score=0.05,
                     protect_types=("procedural",))

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
1. Fetch all episodic memories from ChromaDB (with stored embeddings).
2. Sort by importance descending — highest importance seeds first.
3. Greedy single-linkage clustering:
      for each unseeded memory (highest importance first):
          start a new cluster with this memory as seed
          absorb every other unseeded memory whose cosine
          similarity to the seed ≥ similarity_threshold
4. Discard clusters with fewer than min_cluster_size members.
5. For each qualifying cluster:
      text       = summarizer(cluster_members)
      tags       = ["reflection"] + union of all source tags
      importance = mean(source importances)
      store new semantic memory  →  MemoryStore.remember()
6. If consolidate=True: delete all source episodic memories.
```

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

### Custom (LLM) summarizer

```python
from ai_houkai.memory_system import ReflectionEngine

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
                          summarizer=None)

clusters = engine.clusters()              # list[list[Memory]], no writes
previews = engine.reflect(dry_run=True)   # list[Memory], not persisted
created  = engine.reflect()               # persist semantic memories
created  = engine.reflect(consolidate=True)  # + delete source episodics
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

`ai_houkai/mcp_server/server.py` uses **FastMCP** to expose ten tools:

**Core tools**

| Tool | Key parameters | Returns |
|---|---|---|
| `remember` | `text`, `type?`, `tags?`, `importance?`, `source?`, `on_conflict?`, `polarity?` | `{id, stored}` or `{stored:false, conflicts:[…]}` |
| `recall` | `query`, `k?`, `type?`, `tag?`, `min_importance?`, `mode?`, `overfetch?`, `include_superseded?` | `list[{id,text,type,tags,importance,score,created_at,superseded_by}]` |
| `forget` | `memory_id` | `{deleted}` |
| `list_recent` | `limit?`, `include_superseded?` | `list[{…,superseded_by}]` |
| `stats` | — | `{count, path, collection}` |

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

Claude Code reads `~/.claude/settings.json` (global) or
`.claude/settings.json` (project-level) to discover MCP servers.

```
Claude Code CLI
    │
    │  reads at startup / per-invocation
    ▼
~/.claude/settings.json
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
    memory_path   = "~/.ai_houkai",
    collection    = "my_project",          # per-project namespace
    settings_path = ".claude/settings.json"  # project-scoped install
)
inst.install()
inst.verify()
print(inst.claudemd_snippet())
```

The installer is a small dataclass (no global state, idempotent writes,
JSON-merge into existing `mcpServers`) so it composes well with project
scaffolding tooling.

#### CLAUDE.md guidance

Adding memory instructions to `CLAUDE.md` teaches Claude Code *when*
to use the tools autonomously — without the user needing to prompt:

```markdown
## Memory (AI-Houkai MCP)
- recall() before starting any task
- remember() conventions, decisions, corrections
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
register the MCP server with various MCP clients.  It used to live inside
`examples/06_claude_code.py`; promoting it to a real module means:

- Third-party tools can import `ClaudeCodeInstaller` as a library.
- The console script `ai-houkai-install-claude-code` is available
  immediately after `pip install ai-houkai` — no example file required.
- Tests can target the installer directly without `spec_from_file_location`
  hacks for digit-prefixed examples.

### `ClaudeCodeInstaller`

```python
@dataclass
class ClaudeCodeInstaller:
    memory_path:   str = "~/.ai_houkai"
    collection:    str = "claude_code"
    settings_path: str = "~/.claude/settings.json"
    server_name:   str = "ai-houkai"
    extra_env:     dict = {}

    def build_mcp_block(self)      -> dict
    def build_settings_block(self) -> dict
    def install(self)              -> str        # returns settings_path
    def print_config(self)         -> None
    def verify(self)               -> bool
    @staticmethod
    def claudemd_snippet()         -> str
```

### Behaviour

- **Idempotent**: re-running `install()` overwrites the `ai-houkai` entry
  in `mcpServers` but preserves all other servers and top-level keys.
- **Unparseable settings**: by default replaces a corrupted JSON file
  rather than aborting (toggle with `overwrite_unparseable=False`).
- **Per-project install**: pass `settings_path=".claude/settings.json"`
  to scope the registration to a single repo.
- **Future installers** (e.g. `claude_desktop.py`) follow the same shape
  so that callers can dispatch generically.

### Console script wiring (`pyproject.toml`)

```toml
[project.scripts]
ai-houkai-mcp                 = "ai_houkai.mcp_server.server:run"
ai-houkai-install-claude-code = "ai_houkai.installers.claude_code:_main"
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

### 168 tests across 5 files

| File | Tests | What it covers |
|---|---|---|
| `test_memory.py` | 18 | `MemoryStore`: remember, forget, recall (filters, touch), list_recent, `Memory` dataclass serialisation |
| `test_decay.py` | 15 | `DecayEngine`: score formula, score_all sorting, prune (dry-run, protect, custom now, empty store) |
| `test_reflection.py` | 17 | `ReflectionEngine`: clustering, reflect (dry-run, consolidate, tags, custom summarizer), default summarizer |
| `test_dispatch.py` | 30 | `_dispatch_tool` for all three providers × remember / recall / forget / unknown tool |
| `test_cli.py` | 11 | CLI round-trips: remember → list → show → forget, tag/bump, link/neighbors/unlink, supersede/restore, export/import, stats, prune dry-run, stdin |

### Test isolation strategy

`EphemeralClient()` shares an in-process SQLite database.  All tests
use `PersistentClient` with a `tmp_path`-backed directory:

```python
# tests/conftest.py
@pytest.fixture()
def store(tmp_path) -> MemoryStore:
    return MemoryStore(path=str(tmp_path / "chroma"), collection="test_memory")
```

pytest's `tmp_path` fixture creates a unique directory per test and
cleans it up afterwards.

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
    if b.type != a.type          → skip
    if sim(a,b) < threshold      → skip   (default 0.80)
    if tags don't overlap        → skip   (both must share ≥1 tag, or both empty)
    if negation_diff(a, b)       → kind="contradiction", reason="negation_diff"
    elif contradiction_fn(a, b)  → kind="contradiction", reason="custom_fn"
    else                         → kind="duplicate",     reason="similarity"
```

`negation_diff`: strips apostrophes, tokenises, counts negation words
(`not`, `never`, `no`, `dont`, …), returns True if parity differs.

### on_conflict policies

| Policy | Effect on `remember()` |
|---|---|
| `ignore` (default) | no check |
| `warn` | `warnings.warn()` listing conflicts |
| `supersede` | auto-supersede conflicting memories |
| `raise` | raises `ConflictError(conflicts)` |

```python
store = MemoryStore(conflict_policy="warn", conflict_threshold=0.85)
store.find_conflicts()                       # global pairwise scan (O(n²))
store.find_conflicts(memory_id=x)            # check one memory
```

---

## 14. Hybrid Retrieval

### Score formula

```
final = α·cosine + β·BM25_local + γ·recency + δ·importance
```

| Weight | Default |
|---|---|
| α cosine    | 0.55 |
| β lexical   | 0.20 |
| γ recency   | 0.15 |
| δ importance| 0.10 |

`recency = exp(−λ · age_days)` — same λ as `DecayEngine` (default 0.1).

**BM25 is computed locally** over the cosine over-fetch pool only — no
second index, O(1) additional storage.

### API

```python
store.recall(query, k, mode="hybrid")                    # default weights
store.recall(query, k, mode="hybrid",
             weights=HybridWeights(cosine=0.8, lexical=0.1,
                                   recency=0.05, importance=0.05))
store.recall(query, k, overfetch=6)                      # larger cosine pool

# Graph-walk expansion after scoring
store.recall(query, k,
             expand=ExpandSpec(rels=("refines","example_of"),
                               depth=1, cap=5, score=0.70))
```

Default `mode="semantic"` is unchanged — zero risk for existing callers.

---

## 15. Extension Points

> The designs in [PROPOSALS.md](PROPOSALS.md) are now implemented (§12–14).
> The sketches below remain valid for further customisation.
> The sketches below remain valid as quick recipes.


### Hybrid retrieval score

```python
score = α·cosine_sim + β·(1 − age_days/max_age) + γ·importance
```

Implement as a post-ranking step after `recall()` returns.

### Scheduled cognitive maintenance

```python
import threading, time
from ai_houkai.memory_system import DecayEngine, ReflectionEngine

def _maintenance(store, interval=3600):
    decay   = DecayEngine(store)
    reflect = ReflectionEngine(store)
    while True:
        time.sleep(interval)
        decay.prune()
        reflect.reflect(consolidate=True)

threading.Thread(target=_maintenance, args=(store,), daemon=True).start()
```

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

```python
from ai_houkai.memory_system import ReflectionEngine

def gpt_summarizer(memories):
    prompt = "\n".join(f"- {m.text}" for m in memories)
    return openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user",
                   "content": f"Distil these events into one insight:\n{prompt}"}],
    ).choices[0].message.content

engine = ReflectionEngine(store, summarizer=gpt_summarizer)
```

### Importance auto-assignment

Assign importance based on context rather than caller-supplied values:

- **High** (0.9+): explicit user instructions, corrections, preferences
- **Medium** (0.6–0.8): task completions, project decisions
- **Low** (0.2–0.4): passing observations, intermediate steps
- **LLM-based**: ask the model to rate 0–1 before storing

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
  └── ai_houkai/cli/main.py          Typer app; registers 22 commands;
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
| `list` | `list_cmd.py` | `store.list_recent()` + Python filters |
| `show` | `show.py` | `store._get_by_id()` |
| `forget` | `forget.py` | `store.forget()` |
| `edit` | `edit.py` | forget + remember (re-embeds if text changed) |
| `tag` | `edit.py` | `collection.update()` metadata-only |
| `bump` | `edit.py` | `collection.update()` metadata-only |
| `link` | `link.py` | `store.link()` |
| `unlink` | `link.py` | `store.unlink()` |
| `neighbors` | `link.py` | `store.neighbors()` |
| `graph` | `link.py` | `store.subgraph()` |
| `conflicts` | `conflicts.py` | `store.find_conflicts()` |
| `supersede` | `conflicts.py` | `store.supersede()` |
| `restore` | `conflicts.py` | `store.restore()` |
| `prune` | `decay.py` | `DecayEngine.prune()` |
| `reflect` | `reflect.py` | `ReflectionEngine.reflect()` |
| `export` | `io.py` | `store.list_recent()` → JSONL |
| `import` | `io.py` | JSONL → `store.remember()` |
| `backup` | `io.py` | `shutil.copytree(.chroma → backups/<ts>/)` |
| `stats` | `stats.py` | `store.list_recent()` + Counter |

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

- **Destructive commands** (`forget`, `prune --apply`, `reflect --apply
  --consolidate hard`, `import`) confirm interactively unless `--yes`.
- `prune` and `reflect` default to **dry-run** — matching the underlying
  engine conventions — and require an explicit `--apply` flag to write.
- `tag` and `bump` update only ChromaDB metadata; the embedding vector
  is unchanged, keeping the semantic index consistent.
- `edit` re-embeds the text (delete + re-add) only when the text field
  changes; metadata-only edits go through `collection.update()`.

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
