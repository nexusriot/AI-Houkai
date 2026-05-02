"""Standalone memory system demo — no LLM required.

Shows the full lifecycle:
  store → recall with filters → access-count tracking → forget
"""

from __future__ import annotations
import shutil, tempfile, textwrap

from memory_system import MemoryStore


BANNER = "─" * 60


def section(title: str) -> None:
    print(f"\n{BANNER}\n  {title}\n{BANNER}")


def show_hits(hits):
    if not hits:
        print("  (no results)")
    for mem, score in hits:
        tags = ", ".join(mem.tags) if mem.tags else "—"
        print(
            f"  [{score:.3f}] {mem.type:11s}  tags={tags:20s}  "
            f'"{textwrap.shorten(mem.text, 60)}"'
        )


def main() -> None:
    tmp = tempfile.mkdtemp(prefix="ai_houkai_demo_")
    store = MemoryStore(path=tmp)

    section("1 · Seeding memories")
    memories = [
        store.remember(
            "User strongly prefers terse, bullet-style answers.",
            type="feedback",
            tags=["style", "tone"],
            importance=0.95,
            source="user",
        ),
        store.remember(
            "The API service lives at api.internal:8080 and requires mTLS.",
            type="semantic",
            tags=["api", "infra"],
            importance=0.8,
        ),
        store.remember(
            "Deploy with `make release` then `kubectl rollout restart deploy/api`.",
            type="procedural",
            tags=["deploy", "api"],
            importance=0.85,
            source="ops-runbook",
        ),
        store.remember(
            "Met Alice on 2026-04-22 to scope the ingest pipeline rewrite.",
            type="episodic",
            tags=["meeting", "ingest"],
            importance=0.5,
        ),
        store.remember(
            "ChromaDB cosine distance is in [0, 2]; similarity = 1 − distance.",
            type="semantic",
            tags=["chromadb", "embeddings"],
            importance=0.4,
        ),
    ]
    print(f"  Stored {store.count()} memories.")
    for m in memories:
        print(f"  [{m.type:11s}] {m.text[:70]}")

    section("2 · Semantic recall: 'how do I release the API?'")
    show_hits(store.recall("how do I release the API?", k=3))

    section("3 · Semantic recall: 'how should I write responses?'")
    show_hits(store.recall("how should I write responses?", k=3))

    section("4 · Filter by type=procedural")
    show_hits(store.recall("deployment", k=5, type="procedural"))

    section("5 · Filter by tag='api' + min_importance=0.8")
    show_hits(store.recall("api endpoint", k=5, tag="api", min_importance=0.8))

    section("6 · Access-count tracking")
    mid = memories[0].id
    store.recall("response style", k=3)
    store.recall("tone of replies", k=3)
    recent = store.list_recent()
    for m in recent:
        if m.id == mid:
            print(f"  Feedback memory accessed {m.access_count} time(s).")

    section("7 · list_recent(limit=3)")
    for m in store.list_recent(limit=3):
        print(f"  {m.type:11s}  {m.text[:60]}")

    section("8 · Forget a memory")
    target = memories[-1]
    print(f"  Forgetting: '{target.text[:60]}'")
    ok = store.forget(target.id)
    print(f"  Deleted={ok}, count now={store.count()}")

    print(f"\n{BANNER}\nDone.  Temp dir: {tmp}\n")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
