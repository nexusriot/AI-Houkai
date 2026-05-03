"""
AI-Houkai · Example 05 · Decay + Reflection
             ║
Demonstrates the two cognitive maintenance features:

  1. DECAY   — old, unimportant memories fade and are pruned
  2. REFLECTION — clusters of related episodic memories are condensed
               into a single semantic "summary" memory

Run:
    python examples/05_decay_reflection.py
"""

from __future__ import annotations

import math
import shutil
import tempfile
import time

from ai_houkai.memory_system import DecayEngine, MemoryStore, ReflectionEngine

W = 65

def hr(title: str = "") -> None:
    pad  = f"  {title}  " if title else ""
    side = "─" * ((W - len(pad)) // 2)
    print(f"\n{side}{pad}{side}")

def bar(score: float, width: int = 30) -> str:
    filled = round(score * width)
    return "█" * filled + "░" * (width - filled)


tmp   = tempfile.mkdtemp(prefix="ai_houkai_05_")
store = MemoryStore(path=tmp)
print(f"\nAI-Houkai  ·  Decay + Reflection demo  (db: {tmp})")


def age(mem, days: float) -> None:
    """Backdate a memory's timestamps so it looks N days old."""
    t = time.time() - days * 86_400
    mem.last_accessed = t
    mem.created_at    = t
    store.collection.update(ids=[mem.id], metadatas=[mem.to_metadata()])


hr("PART 1 — DECAY")

# 1-A  seed memories at various ages and importances
print("""
  Formula:  score = importance × exp(-λ × days_since_last_access)
  Default:  λ = 0.1  →  half-life ≈ 7 days for a 0.5-importance memory
  Pruned:   score < 0.05
""")

seeds = [
    # (text,                              type,         importance, days_old)
    ("Python GIL blocks CPU parallelism", "semantic",   0.80,   1),
    ("FastAPI docs live at /docs",        "semantic",   0.60,   7),
    ("Slack bot API token expires 2027",  "semantic",   0.50,  14),
    ("npm audit warning 2026-01-05",      "episodic",   0.30,  30),
    ("Reviewed PR #441 (trivial fix)",    "episodic",   0.20,  45),
    ("Met Bob for coffee — no actions",   "episodic",   0.15,  60),
    ("Deployed hotfix v1.0.1 to prod",    "procedural", 0.85,  90),  # protected
    ("User likes concise answers",        "feedback",   0.95,   2),
]

memories = {}
for text, typ, imp, days in seeds:
    m = store.remember(text, type=typ, importance=imp)
    age(m, days)
    memories[text] = m

engine = DecayEngine(store, decay_rate=0.1, min_score=0.05,
                     protect_types=("procedural",))

# 1-B  show scores
hr("1-B · Decay scores")
print(f"\n  {'SCORE':6s}  {'BAR':32s}  {'IMP':5s}  {'AGE':6s}  MEMORY")
print(f"  {'─'*6}  {'─'*32}  {'─'*5}  {'─'*6}  {'─'*38}")

pairs = engine.score_all()
for mem, score in pairs:
    _, imp, days = next(
        (t[2], t[2], t[3]) for t in seeds if t[0] == mem.text
    )
    # retrieve original age from metadata
    days_real = (time.time() - mem.last_accessed) / 86_400
    protected = "🔒" if mem.type in engine.protect_types else "  "
    print(
        f"  {score:6.3f}  {bar(min(score, 1.0)):32s}  "
        f"{mem.importance:.2f}  {days_real:5.1f}d  "
        f"{protected} {mem.text[:40]}"
    )

# 1-C  dry run
hr("1-C · Dry run — what WOULD be pruned")
candidates = engine.prune(dry_run=True)
if candidates:
    for m in candidates:
        days_real = (time.time() - m.last_accessed) / 86_400
        s = engine.score(m)
        print(f"  💀 score={s:.4f}  age={days_real:.0f}d  [{m.type}] {m.text}")
else:
    print("  (nothing to prune)")

# 1-D  live prune
hr("1-D · Live prune")
before = store.count()
removed = engine.prune()
after  = store.count()

print(f"\n  Removed {len(removed)} memories  ({before} → {after})")
for m in removed:
    print(f"  ✗ [{m.type:9s}]  importance={m.importance:.2f}  {m.text}")

print(f"\n  Kept:")
for m in store.list_recent(limit=10):
    s = engine.score(m)
    protected = "🔒" if m.type in engine.protect_types else "  "
    print(f"  ✓ {protected} [{m.type:9s}]  score={s:.3f}  {m.text}")


hr("PART 2 — REFLECTION")

print("""
  Algorithm:
  1. Fetch all episodic memories with their stored embeddings.
  2. Greedy single-linkage clustering by cosine similarity.
  3. Each cluster ≥ min_cluster_size → new semantic 'reflection' memory.
""")

# 2-A  seed episodic memories in two distinct themes
# Cluster A — deployment events (semantically similar)
dep = [
    ("Deployed API v2.1 to production on 2026-04-10 — no issues.",
     ["deploy", "api"], 0.70),
    ("Released API v2.2 to prod on 2026-04-17, cut latency by 20 %%.",
     ["deploy", "api"], 0.75),
    ("Rolled out API v2.3 to production environment on 2026-04-24.",
     ["deploy", "api"], 0.65),
]
# Cluster B — meeting events (semantically similar among themselves)
mtg = [
    ("Met Alice & team on 2026-04-08 to kick off the ingest rewrite.",
     ["meeting", "ingest"], 0.60),
    ("Sprint planning with Alice on 2026-04-15: ingest rewrite scope locked.",
     ["meeting", "ingest"], 0.55),
    ("Retro with Alice on 2026-04-22: ingest rewrite on track, no blockers.",
     ["meeting", "ingest"], 0.58),
]

for text, tags, imp in dep + mtg:
    store.remember(text, type="episodic", tags=tags, importance=imp)

print(f"  Added {len(dep) + len(mtg)} episodic memories ({len(dep)} deploy, {len(mtg)} meeting)")
print(f"  Total store: {store.count()}\n")

engine_ref = ReflectionEngine(store, similarity_threshold=0.72, min_cluster_size=2)

# 2-B  inspect clusters
hr("2-B · Detected clusters")
clusters = engine_ref.clusters()
print(f"\n  Found {len(clusters)} cluster(s):\n")
for i, cluster in enumerate(clusters, 1):
    avg_imp = sum(m.importance for m in cluster) / len(cluster)
    all_tags = sorted({t for m in cluster for t in m.tags})
    print(f"  Cluster {i}  ({len(cluster)} memories, avg_importance={avg_imp:.2f}, tags={all_tags})")
    for m in cluster:
        print(f"    • {m.text[:70]}")

# 2-C  dry run reflection
hr("2-C · Dry run — preview reflections")
previews = engine_ref.reflect(dry_run=True)
print(f"\n  {len(previews)} reflection(s) would be created:\n")
for p in previews:
    print(f"  📝 [{p.importance:.2f}]  tags={p.tags}")
    print(f"     {p.text[:120]}\n")

# 2-D  live reflection (no consolidate)
hr("2-D · Live reflection — keep episodics")
count_before = store.count()
created = engine_ref.reflect(consolidate=False)
count_after = store.count()

print(f"\n  Created {len(created)} semantic memory(ies)  ({count_before} → {count_after})\n")
for mem in created:
    print(f"  ✨ semantic  [importance={mem.importance}]  tags={mem.tags}")
    print(f"     {mem.text[:120]}\n")

# 2-E  reflect + consolidate
hr("2-E · Reflection WITH consolidation — episodics removed")
# Re-seed fresh episodics for this demo
store2_tmp = tempfile.mkdtemp(prefix="ai_houkai_05b_")
store2 = MemoryStore(path=store2_tmp)
for text, tags, imp in dep:
    store2.remember(text, type="episodic", tags=tags, importance=imp)

engine2 = ReflectionEngine(store2, similarity_threshold=0.72, min_cluster_size=2)

ep_before = store2.count()
created2  = engine2.reflect(consolidate=True)
ep_after  = store2.count()

print(f"\n  Before: {ep_before} episodics")
print(f"  After:  {ep_after} memories (episodics consolidated → {len(created2)} semantic)")
for m in store2.list_recent():
    icon = "✨" if m.type == "semantic" else "📅"
    print(f"  {icon}  [{m.type}]  {m.text[:80]}")

shutil.rmtree(store2_tmp, ignore_errors=True)


hr("PART 3 — CUSTOM LLM SUMMARIZER (stub)")


print("""
  Replace the default extractive summarizer with any callable,
  e.g. an LLM API call:

    def my_summarizer(memories: list[Memory]) -> str:
        prompt = "\\n".join(m.text for m in memories)
        return openai_client.chat(f"Summarise these events: {prompt}")

    engine = ReflectionEngine(store, summarizer=my_summarizer)

  Below we stub it to verify the hook is called:
""")

called: list = []
def stub_summarizer(memories):
    called.append(len(memories))
    return f"LLM-summary of {len(memories)} deployment events."

# Re-seed
for text, tags, imp in dep:
    store.remember(text, type="episodic", tags=tags, importance=imp)

engine_custom = ReflectionEngine(store, similarity_threshold=0.72,
                                 min_cluster_size=2, summarizer=stub_summarizer)
results = engine_custom.reflect()
print(f"  Summarizer called {len(called)} time(s), cluster sizes: {called}")
for r in results:
    if r.source == "reflection":
        print(f"  → {r.text}")


# teardown
hr()
print(f"  Done.  Cleaned up {tmp}\n")
shutil.rmtree(tmp, ignore_errors=True)
