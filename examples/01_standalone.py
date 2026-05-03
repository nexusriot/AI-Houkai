"""
AI-Houkai · Example 01 · Standalone (no LLM)

Demonstrates the full memory API with no LLM involved.
Use-case: a personal knowledge-base for a Python developer.

Run:
    python examples/01_standalone.py

No API keys or running services required.
"""

from __future__ import annotations

import sys, os, shutil, tempfile, textwrap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from memory_system import MemoryStore

W = 64
def hr(title: str = "") -> None:
    pad = f"  {title}  " if title else ""
    side = "─" * ((W - len(pad)) // 2)
    print(f"\n{side}{pad}{side}")

def show(hits, label: str | None = None) -> None:
    if label:
        print(f"\n  ▸ {label}")
    if not hits:
        print("    (no results)")
        return
    for mem, score in hits:
        tags = f"[{', '.join(mem.tags)}]" if mem.tags else ""
        print(f"    {score:.3f}  ({mem.type})  {tags}")
        print(f"           {textwrap.shorten(mem.text, W - 11)}")


tmp = tempfile.mkdtemp(prefix="ai_houkai_01_")
store = MemoryStore(path=tmp)
print(f"\nAI-Houkai standalone demo  (db: {tmp})")


hr("1 · STORING MEMORIES")
# Four memory types:
#   semantic   → facts / domain knowledge
#   procedural → how-to steps
#   episodic   → things that happened (events)
#   feedback   → user preferences / corrections

store.remember(
    "Python's GIL prevents true multi-threading for CPU-bound work; "
    "use multiprocessing or asyncio for concurrency.",
    type="semantic", tags=["python", "concurrency"], importance=0.8,
)
store.remember(
    "FastAPI auto-generates OpenAPI docs at /docs and /redoc.",
    type="semantic", tags=["python", "fastapi", "api"], importance=0.7,
)
store.remember(
    "ChromaDB cosine distance ∈ [0, 2]; similarity = 1 − distance.",
    type="semantic", tags=["chromadb", "vectors"], importance=0.6,
)
store.remember(
    "To release: run `make test && make release` then "
    "`kubectl rollout restart deploy/api -n prod`.",
    type="procedural", tags=["deploy", "ops"], importance=0.9,
    source="ops-runbook",
)
store.remember(
    "To add a new endpoint: 1) define Pydantic schema, "
    "2) add route to router.py, 3) register in main.py, 4) write test.",
    type="procedural", tags=["fastapi", "workflow"], importance=0.75,
)
store.remember(
    "2026-04-22: Met Alice & Bob to scope the ingest rewrite. "
    "Decision: move from Kafka to NATS JetStream.",
    type="episodic", tags=["meeting", "ingest", "architecture"], importance=0.65,
)
store.remember(
    "2026-04-28: Deployed v2.3.1 to prod. P95 latency dropped from "
    "380 ms to 140 ms after enabling response caching.",
    type="episodic", tags=["deploy", "performance"], importance=0.7,
)
store.remember(
    "User prefers concise bullet answers. Never add a summary paragraph.",
    type="feedback", tags=["style"], importance=0.95, source="user",
)
store.remember(
    "User works in Python 3.12 and uses Ruff + mypy for linting.",
    type="feedback", tags=["env", "tooling"], importance=0.85, source="user",
)

print(f"  Stored {store.count()} memories across 4 types.\n")
for m in store.list_recent(limit=9):
    marker = {"semantic": "📘", "procedural": "⚙️ ", "episodic": "📅",
              "feedback": "💬"}.get(m.type, "  ")
    print(f"  {marker} [{m.importance:.2f}] {m.text[:65]}")


hr("2 · SEMANTIC RECALL — free-text queries")

show(store.recall("how do I release to production?", k=3),
     "query: 'how do I release to production?'")

show(store.recall("Python concurrency and threading", k=3),
     "query: 'Python concurrency and threading'")

show(store.recall("what did we decide about the ingest system?", k=2),
     "query: 'what did we decide about the ingest system?'")


hr("3 · FILTERED RECALL")

show(store.recall("workflow steps", k=5, type="procedural"),
     "type=procedural  (all how-to memories)")

show(store.recall("anything", k=5, type="feedback"),
     "type=feedback  (all user preferences)")

show(store.recall("deployment", k=5, tag="deploy", min_importance=0.8),
     "tag=deploy + min_importance=0.8")

show(store.recall("fastapi endpoint", k=5, tag="fastapi"),
     "tag=fastapi")


hr("4 · ACCESS TRACKING")
# Every recall hit increments last_accessed + access_count.
# This enables future decay / reinforcement policies.

queries = [
    "deployment procedure",
    "how to deploy to kubernetes",
    "release checklist",
]
for q in queries:
    store.recall(q, k=2, tag="deploy")

print("\n  Top memories by access_count after 3 deployment queries:")
for m in sorted(store.list_recent(limit=20), key=lambda m: m.access_count, reverse=True)[:4]:
    print(f"    {m.access_count:2d}x  [{m.type}]  {m.text[:60]}")


hr("5 · UPDATE VIA FORGET + REMEMBER")
# There's no in-place update: forget the old, write the corrected version.

old_hits = store.recall("release procedure", k=1, type="procedural")
if old_hits:
    old_mem, _ = old_hits[0]
    print(f"\n  Old: {old_mem.text}")
    store.forget(old_mem.id)
    new_mem = store.remember(
        "To release: run `make test && make release` then "
        "`helm upgrade api ./chart --set image.tag=$(git rev-parse --short HEAD)`.",
        type="procedural", tags=["deploy", "ops", "helm"], importance=0.95,
        source="ops-runbook-v2",
    )
    print(f"  New: {new_mem.text}")
    print(f"  count unchanged: {store.count()}")


hr("6 · BULK FORGET — clear a topic")

style_hits = store.recall("response style preferences", k=5, type="feedback")
print(f"\n  Removing {len(style_hits)} feedback memories...")
for m, _ in style_hits:
    store.forget(m.id)
print(f"  Remaining memories: {store.count()}")


hr()
print(f"  Done.  Cleaned up {tmp}\n")
shutil.rmtree(tmp, ignore_errors=True)
