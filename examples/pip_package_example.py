"""
AI-Houkai · pip package usage example

Install:
    pip install ai-houkai                  # core (no LLM provider)
    pip install "ai-houkai[claude]"        # + Anthropic SDK
    pip install "ai-houkai[openai]"        # + OpenAI SDK (also covers Ollama)
    pip install "ai-houkai[all]"           # everything

Then run this file:
    python examples/pip_package_example.py

No API keys required — the memory layer runs fully local.
"""

from __future__ import annotations

import shutil
import time
import tempfile

from ai_houkai.memory_system import (
    DecayEngine,
    Memory,
    MemoryStore,
    MemoryType,
    ReflectionEngine,
)

W = 64
sep = lambda t="": print("\n" + "─" * ((W - len(t) - 4) // 2) + ("  " + t + "  " if t else "─" * 4) + "─" * ((W - len(t) - 4) // 2))  # noqa: E731


sep("1 · MemoryStore")

tmp   = tempfile.mkdtemp(prefix="ai_houkai_demo_")
store = MemoryStore(path=tmp)          # persists to tmp dir

print(f"\n  Store created at: {tmp}")
print(f"  Embedding model : all-MiniLM-L6-v2 (local, no API key)")


sep("2 · remember()")

facts: list[tuple[str, MemoryType, float, list[str]]] = [
    ("Python's GIL prevents true CPU parallelism in threads",
     "semantic",    0.85, ["python", "concurrency"]),
    ("Use multiprocessing or asyncio to bypass the GIL",
     "procedural",  0.80, ["python", "concurrency"]),
    ("Deployed API v2.1 to production — no issues",
     "episodic",    0.70, ["deploy", "api"]),
    ("Deployed API v2.2 to production — latency −20 %%",
     "episodic",    0.75, ["deploy", "api"]),
    ("Alice prefers bullet-point summaries over prose",
     "feedback",    0.90, ["alice", "preferences"]),
]

memories: list[Memory] = []
for text, mtype, imp, tags in facts:
    m = store.remember(text, type=mtype, importance=imp, tags=tags,
                       source="pip_demo")
    memories.append(m)
    print(f"  + [{mtype:9s}] imp={imp:.2f}  {text[:55]}")

print(f"\n  Total stored: {store.count()}")


sep("3 · recall()")

print("\n  Query: 'parallel execution in Python'\n")
hits = store.recall("parallel execution in Python", k=3)
for mem, score in hits:
    print(f"  {score:.3f}  [{mem.type:9s}]  {mem.text}")


sep("4 · recall() with filters")

print("\n  type='episodic'  tag='api'\n")
hits = store.recall("production deployment", k=5, type="episodic", tag="api")
for mem, score in hits:
    print(f"  {score:.3f}  tags={mem.tags}  {mem.text[:55]}")


sep("5 · DecayEngine")

# Backdate the episodic memories to look 45 days old
for m in memories:
    if m.type == "episodic":
        t = time.time() - 45 * 86_400
        m.last_accessed = t
        m.created_at    = t
        store.collection.update(ids=[m.id], metadatas=[m.to_metadata()])

engine = DecayEngine(store, decay_rate=0.1, min_score=0.05,
                     protect_types=("procedural",))

print("\n  Scores after backdating episodics to 45 days old:\n")
for mem, score in engine.score_all():
    protected = "🔒" if mem.type in engine.protect_types else "  "
    print(f"  {score:.4f} {protected} [{mem.type:9s}]  {mem.text[:50]}")

removed = engine.prune()
print(f"\n  Pruned {len(removed)} memory(ies):")
for m in removed:
    print(f"  ✗ [{m.type}]  {m.text[:60]}")

print(f"\n  Remaining: {store.count()}")


sep("6 · ReflectionEngine")

# Add fresh episodic deployment memories (the old ones were pruned).
# Use consistent phrasing so embeddings cluster above the similarity threshold.
for text, imp in [
    ("Deployed API v2.1 to production on 2026-04-10 — no issues.",          0.70),
    ("Released API v2.2 to production on 2026-04-17, latency cut by 20%%.", 0.80),
    ("Rolled out API v2.3 to production environment on 2026-04-24.",         0.65),
]:
    store.remember(text, type="episodic", tags=["deploy", "api"], importance=imp)

ref_engine = ReflectionEngine(store, similarity_threshold=0.72, min_cluster_size=2)

clusters = ref_engine.clusters()
print(f"\n  Detected {len(clusters)} cluster(s):")
for i, cl in enumerate(clusters, 1):
    print(f"\n  Cluster {i} ({len(cl)} memories):")
    for m in cl:
        print(f"    • {m.text[:65]}")

created = ref_engine.reflect(consolidate=False)
print(f"\n  Created {len(created)} semantic reflection(s):")
for r in created:
    print(f"\n  ✨ importance={r.importance}  tags={r.tags}")
    print(f"     {r.text[:110]}")


sep("7 · list_recent() + forget()")

print("\n  Recent memories:")
for m in store.list_recent(limit=6):
    print(f"  [{m.type:9s}]  {m.text[:60]}")

first_id = memories[0].id
deleted  = store.forget(first_id)
print(f"\n  forget({first_id[:8]}…)  → deleted={deleted}")
print(f"  Remaining: {store.count()}")


sep("8 · MCP server")

print("""
  After pip install, launch the MCP server with:

      ai-houkai-mcp
      # or: python -m mcp_server.server

  Environment variables:
      AI_HOUKAI_PATH        path for ChromaDB  (default: ./.chroma)
      AI_HOUKAI_COLLECTION  collection name    (default: ai_houkai)

  Add to Claude Code (preferred):
      claude mcp add --scope user ai-houkai -- ai-houkai-mcp

  Or manually (~/.claude.json user scope / project .mcp.json):
      {
        "mcpServers": {
          "ai-houkai": {
            "type": "stdio",
            "command": "ai-houkai-mcp",
            "args": [],
            "env": { "AI_HOUKAI_PATH": "/your/memory/path" }
          }
        }
      }
""")

sep()
print(f"  Done.  Cleaned up {tmp}\n")
shutil.rmtree(tmp, ignore_errors=True)
