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
9. [Agent Integrations](#9-agent-integrations)
10. [Test Architecture](#10-test-architecture)
11. [Extension Points](#11-extension-points)

---

## 1. Motivation

LLM context windows are finite and stateless.  Every new conversation
starts from scratch.  AI-Houkai gives an agent a **persistent,
searchable memory** that survives across sessions — without requiring
cloud services or API keys for the memory layer itself.

Three cognitive operations model how humans manage long-term memory:

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
└───────────────┬──────────────────────────┬───────────────────┘
                │ tool calls                │ tool results
                ▼                           │
┌──────────────────────────┐                │
│      _dispatch_tool()    │◄───────────────┘
│  (examples/claude_agent, │
│   04_openai, 02_ollama)  │
└───────────┬──────────────┘
            │  or via MCP
            ▼
┌──────────────────────────────────────────────────────────────┐
│                      MemoryStore                             │
│   remember()  recall()  forget()  list_recent()  count()     │
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

The **MCP server** (`mcp_server/server.py`) wraps `MemoryStore` and
exposes the same operations as MCP tools so any MCP client (Claude
Desktop, Claude Code, custom agents) can call them without any Python
glue code.

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
```

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
comma-joined string and re-split on read:

```python
# write
{"tags": "deploy,api,prod"}

# read
tags = [t for t in meta["tags"].split(",") if t]
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
downloaded.  To swap to a different provider:

```python
store = MemoryStore(embedding_model="text-embedding-3-small")
# or pass a custom chromadb.EmbeddingFunction subclass
```

### HNSW index

ChromaDB uses the HNSW (Hierarchical Navigable Small World) graph for
approximate nearest-neighbour search.  At the scale of a single agent's
memory (hundreds to low thousands of entries), exact search would also
be fine — HNSW just ensures the query stays fast as collections grow.

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
   not array membership — so tag filtering happens in Python).
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
| 0.1 | 1 day | 0.09 | borderline |
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
engine = DecayEngine(store, decay_rate=0.1, min_score=0.05,
                     protect_types=("procedural",))

# Inspect
score = engine.score(mem)                   # single memory
pairs = engine.score_all()                  # all memories, sorted desc

# Act
candidates = engine.prune(dry_run=True)     # preview
removed     = engine.prune()                # delete stale memories
```

`now` can be overridden in both `score()` and `prune()` for
deterministic testing or time-travel simulations.

---

## 7. Reflection Engine

Implements the **Generative Agents** reflection pattern: periodically
cluster semantically similar episodic memories and condense them into a
single semantic "summary" memory.

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
      text      = summarizer(cluster_members)
      tags      = ["reflection"] + union of all source tags
      importance = mean(source importances)
      store new semantic memory  →  MemoryStore.remember()
6. If consolidate=True: delete all source episodic memories.
```

### Clustering properties

- **Seed-based**: the most important memory anchors each cluster,
  which biases summaries toward high-signal events.
- **Single-linkage**: a memory joins a cluster if it is similar to
  the *seed*, not to all existing members.  This is fast (O(n) per
  seed) and avoids the chaining artefacts of full single-linkage.
- **Non-overlapping**: each memory belongs to at most one cluster
  (greedy `used[]` mask).

### Similarity threshold guide

| threshold | Effect |
|---|---|
| 0.95 | Only near-duplicates cluster |
| 0.80 | Same topic, similar phrasing |
| 0.75 | Same topic, varied phrasing (default) |
| 0.60 | Broadly related content |
| 0.40 | Almost everything clusters together |

### Default summarizer

```python
def _default_summarizer(memories):
    ordered = sorted(memories, key=lambda m: m.importance, reverse=True)
    body = " | ".join(m.text for m in ordered)
    return ("[Reflection ×N] " + body)[:512]
```

- **Extractive**: no LLM required.
- **Most-important first**: the highest-signal events appear earliest.
- **512-char cap**: keeps semantic memories compact.

### Custom (LLM) summarizer

```python
def my_summarizer(memories: list[Memory]) -> str:
    prompt = "\n".join(m.text for m in memories)
    return call_llm(f"Summarise these events into one insight:\n{prompt}")

engine = ReflectionEngine(store, summarizer=my_summarizer)
```

Any callable `(list[Memory]) → str` is accepted.

### API

```python
engine = ReflectionEngine(store,
                          similarity_threshold=0.75,
                          min_cluster_size=2,
                          summarizer=None)  # None → default extractive

# Inspect without writing
clusters = engine.clusters()               # list[list[Memory]]

# Preview without writing
previews = engine.reflect(dry_run=True)    # list[Memory] (not persisted)

# Create semantic reflections, keep episodics
created  = engine.reflect()

# Create semantic reflections, delete episodics
created  = engine.reflect(consolidate=True)
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

`mcp_server/server.py` uses **FastMCP** to expose five tools:

| Tool | Parameters | Returns |
|---|---|---|
| `remember` | `text`, `type?`, `tags?`, `importance?`, `source?` | `{id, stored}` |
| `recall` | `query`, `k?`, `type?`, `tag?`, `min_importance?` | `list[{id,text,type,tags,importance,score,created_at}]` |
| `forget` | `memory_id` | `{deleted}` |
| `list_recent` | `limit?` | `list[{id,text,type,tags,created_at}]` |
| `stats` | — | `{count, path, collection}` |

Configuration via environment variables:

| Variable | Default |
|---|---|
| `AI_HOUKAI_PATH` | `./.chroma` |
| `AI_HOUKAI_COLLECTION` | `ai_houkai` |

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
python -m mcp_server.server
    │
    │  stdio transport (JSON-RPC)
    ▼
MemoryStore ──► ChromaDB on disk
```

`examples/03_claude_desktop.py --install` locates the platform-specific
config path, patches the `mcpServers` block, and reports what Claude
will see.

---

## 9. Agent Integrations

All agent examples share the same `_dispatch_tool(name, arguments)` 
interface. Only the SDK and message format differs.

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
format natively. The Claude example serialises its dict input before
calling dispatch.

### Provider comparison

| | Claude (`claude_agent.py`) | OpenAI (`04_openai.py`) | Ollama (`02_ollama_local_network.py`) |
|---|---|---|---|
| SDK | `anthropic` | `openai` | `openai` (compat endpoint) |
| Tool definition format | `{"name":…,"input_schema":{…}}` | `{"type":"function","function":{…}}` | same as OpenAI |
| Tool call access | `block.name`, `block.input` (dict) | `tc.function.name`, `tc.function.arguments` (str) | same as OpenAI |
| Arguments format | dict → `json.dumps()` before dispatch | JSON string | JSON string |
| Endpoint | `api.anthropic.com` | `api.openai.com` | `localhost:11434/v1` |
| Requires API key | yes | yes | no |

### Message flow (generic)

```
user message
     │
     ▼
LLM API  ──►  tool_call: {name, arguments}
     │
     ▼
_dispatch_tool(name, arguments)
     │
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

## 10. Test Architecture

### 79 tests across 4 files

| File | Tests | What it covers |
|---|---|---|
| `test_memory.py` | 18 | `MemoryStore`: remember, forget, recall (filters, touch), list_recent, `Memory` dataclass serialisation |
| `test_decay.py` | 15 | `DecayEngine`: score formula, score_all sorting, prune (dry-run, protect, custom now, empty store) |
| `test_reflection.py` | 17 | `ReflectionEngine`: clustering (similarity threshold, min size), reflect (dry-run, consolidate, tags, custom summarizer), default summarizer |
| `test_dispatch.py` | 30 | `_dispatch_tool` for all three providers × remember / recall / forget / unknown tool |

### Test isolation strategy

`EphemeralClient()` shares an in-process SQLite database, so tests
in the same pytest session see each other's data.  All tests use
`PersistentClient` with a `tmp_path`-backed directory:

```python
# tests/conftest.py
@pytest.fixture()
def store(tmp_path) -> MemoryStore:
    return MemoryStore(path=str(tmp_path / "chroma"), collection="test_memory")
```

pytest's `tmp_path` fixture creates a unique directory per test
invocation and cleans it up afterwards.

### Loading digit-prefixed modules

`importlib.import_module("04_openai")` raises `ModuleNotFoundError`
because `04_openai` is not a valid Python identifier.
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
Tests inject stub modules before loading so no network calls happen:

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

## 11. Extension Points

### Hybrid retrieval score

Current retrieval ranks purely by cosine similarity.  A richer score:

```
score = α·cosine_sim + β·(1 − age_days/max_age) + γ·importance
```

could be implemented as a post-ranking step after `recall()` returns,
or by adding a custom ChromaDB embedding function that bakes recency
into the vector.

### Scheduled cognitive maintenance

Decay and reflection are currently called explicitly.  A production
agent could run them on a schedule:

```python
import threading, time

def _background_maintenance(store, interval=3600):
    decay = DecayEngine(store)
    reflect = ReflectionEngine(store)
    while True:
        time.sleep(interval)
        decay.prune()
        reflect.reflect(consolidate=True)

threading.Thread(target=_background_maintenance,
                 args=(store,), daemon=True).start()
```

### Multi-user / multi-agent

Each `MemoryStore` targets a single ChromaDB collection.  To isolate
users or agents, pass distinct collection names:

```python
alice_store = MemoryStore(path=".chroma", collection="agent_alice")
bob_store   = MemoryStore(path=".chroma", collection="agent_bob")
```

All agents can share the same ChromaDB directory; the HNSW index is
per-collection.

### Pluggable embeddings

Swap the local sentence-transformers model for any
`chromadb.EmbeddingFunction`:

```python
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

store = MemoryStore()
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

Replace the extractive default with a generative summary:

```python
def gpt_summarizer(memories: list[Memory]) -> str:
    prompt = "\n".join(f"- {m.text}" for m in memories)
    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user",
                   "content": f"Distil these events into one insight:\n{prompt}"}],
    )
    return resp.choices[0].message.content

engine = ReflectionEngine(store, summarizer=gpt_summarizer)
```

### Importance auto-assignment

Currently importance is caller-supplied.  An agent could estimate it
from context:

- High importance for explicit user instructions or corrections.
- Medium importance for task completions.
- Low importance for observations or passing mentions.
- LLM-based: ask the model to rate `0..1` before storing.
