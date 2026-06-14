"""MCP server exposing the AI-Houkai memory store.

Run with:
    ai-houkai-mcp
    # or: python -m ai_houkai.mcp_server.server

Tools exposed (15):
    remember(text, type?, tags?, importance?, source?, on_conflict?, polarity?)
    recall(query, k?, type?, tag?, min_importance?, source?, since?, until?, mode?, overfetch?)
    recall_pack(query, token_budget?, type?, tag?, min_importance?, source?, since?, until?, mode?, max_items?)
    forget(memory_id)
    list_recent(limit?, include_superseded?)
    stats()
    link(src_id, dst_id, rel?)
    unlink(src_id, dst_id, rel?)
    neighbors(memory_id, rel?, direction?, depth?)
    find_conflicts(memory_id?, threshold?)
    supersede(old_id, new_id)
    maintenance_tick(reflect_apply?)
    journal_tail(n?, op?, since_seconds?)
    export(path, include_vectors?, include_superseded?, type?, tag?, since?)
    import(path, on_conflict?, regenerate_vectors?, dry_run?)
"""

from __future__ import annotations

import os
import time
from typing import Any

from mcp.server.fastmcp import FastMCP

from ai_houkai.maintenance.scheduler import MaintenanceScheduler
from ai_houkai.cli.config import load_maintenance
from ai_houkai.memory_system import MemoryStore
from ai_houkai.memory_system.importance import score_importance
from ai_houkai.memory_system.store import ConflictError, HybridWeights
from ai_houkai.memory_system.summarizers import build_summarizer
from ai_houkai.timeparse import parse_timestamp

CHROMA_PATH = os.environ.get("AI_HOUKAI_PATH", "./.chroma")
COLLECTION  = os.environ.get("AI_HOUKAI_COLLECTION", "ai_houkai")
AUTO_IMPORTANCE = os.environ.get("AI_HOUKAI_AUTO_IMPORTANCE", "").lower() in (
    "1", "true", "yes", "on",
)

store = MemoryStore(
    path=CHROMA_PATH,
    collection=COLLECTION,
    actor="mcp",
    importance_fn=score_importance if AUTO_IMPORTANCE else None,
)
mcp   = FastMCP("AI-Houkai")


@mcp.tool()
def remember(
    text: str,
    type: str = "semantic",
    tags: list[str] | None = None,
    importance: float | None = None,
    source: str | None = None,
    on_conflict: str | None = None,
    polarity: int = 0,
) -> dict[str, Any]:
    """Store a new memory.
    type: episodic | semantic | procedural | feedback.
    importance: 0..1; omit for the default (0.5, or a heuristic score when
    the server runs with AI_HOUKAI_AUTO_IMPORTANCE=1).
    on_conflict: ignore | warn | supersede | raise (default: store policy).
    polarity: -1 (negative) | 0 (neutral) | +1 (positive).
    """

    policy = on_conflict  # type: ignore[assignment]
    try:
        mem = store.remember(
            text=text,
            type=type,           # type: ignore[arg-type]
            tags=tags or [],
            importance=importance,
            source=source,
            polarity=polarity,
            on_conflict=policy,  # type: ignore[arg-type]
        )
    except ConflictError as e:
        return {
            "stored": False,
            "conflicts": [
                {"kind": c.kind, "similarity": c.similarity,
                 "other_id": c.b.id, "other_text": c.b.text[:100]}
                for c in e.conflicts
            ],
        }
    return {"id": mem.id, "stored": True, "importance": mem.importance}


@mcp.tool()
def recall(
    query: str,
    k: int = 5,
    type: str | None = None,
    tag: str | None = None,
    min_importance: float | None = None,
    source: str | None = None,
    since: str | None = None,
    until: str | None = None,
    mode: str = "semantic",
    overfetch: int = 4,
    include_superseded: bool = False,
) -> list[dict[str, Any]]:
    """Semantic (or hybrid) search across stored memories.
    mode: "semantic" (default) | "hybrid" (cosine + BM25 + recency + importance).
    source: keep only memories with this exact provenance string.
    since/until: bound created_at — epoch seconds, an ISO-8601 date/datetime,
    or a relative span like "7d" / "24h" (since="7d" → last 7 days).
    """
    hits = store.recall(
        query=query,
        k=k,
        type=type,            # type: ignore[arg-type]
        tag=tag,
        min_importance=min_importance,
        source=source,
        since=parse_timestamp(since),
        until=parse_timestamp(until),
        mode=mode,            # type: ignore[arg-type]
        overfetch=overfetch,
        include_superseded=include_superseded,
    )
    return [
        {
            "id": m.id,
            "text": m.text,
            "type": m.type,
            "tags": m.tags,
            "importance": m.importance,
            "score": round(score, 4),
            "created_at": m.created_at,
            "superseded_by": m.superseded_by or None,
        }
        for m, score in hits
    ]


@mcp.tool()
def recall_pack(
    query: str,
    token_budget: int = 800,
    type: str | None = None,
    tag: str | None = None,
    min_importance: float | None = None,
    source: str | None = None,
    since: str | None = None,
    until: str | None = None,
    mode: str = "hybrid",
    max_items: int = 50,
    include_superseded: bool = False,
) -> dict[str, Any]:
    """Assemble the most relevant memories into a token-budgeted context block.

    Ranks with hybrid scoring (cosine + BM25 + recency + importance) by default,
    then greedily packs results until token_budget is reached. Returns a ready-to-
    inject `text` block plus the packed items. token_budget is a soft ceiling
    (estimated at ~4 chars/token) covering the memory lines, not the header.
    source/since/until filter candidates exactly as in `recall`.
    """
    pack = store.recall_pack(
        query=query,
        token_budget=token_budget,
        type=type,             # type: ignore[arg-type]
        tag=tag,
        min_importance=min_importance,
        source=source,
        since=parse_timestamp(since),
        until=parse_timestamp(until),
        mode=mode,             # type: ignore[arg-type]
        max_items=max_items,
        include_superseded=include_superseded,
    )
    return {
        "text": pack.text,
        "used_tokens": pack.used_tokens,
        "budget": pack.budget,
        "truncated": pack.truncated,
        "items": [
            {
                "id": p.memory.id,
                "text": p.memory.text,
                "type": p.memory.type,
                "tags": p.memory.tags,
                "importance": p.memory.importance,
                "score": round(p.score, 4),
                "tokens": p.tokens,
            }
            for p in pack.items
        ],
    }


@mcp.tool()
def forget(memory_id: str) -> dict[str, Any]:
    """Permanently delete a memory by id."""
    return {"deleted": store.forget(memory_id)}


@mcp.tool()
def list_recent(
    limit: int = 20,
    include_superseded: bool = False,
) -> list[dict[str, Any]]:
    """List the most recently created memories."""
    return [
        {
            "id": m.id,
            "text": m.text,
            "type": m.type,
            "tags": m.tags,
            "created_at": m.created_at,
            "superseded_by": m.superseded_by or None,
        }
        for m in store.list_recent(limit=limit, include_superseded=include_superseded)
    ]


@mcp.tool()
def stats() -> dict[str, Any]:
    """Return basic store statistics."""
    return {"count": store.count(), "path": CHROMA_PATH, "collection": COLLECTION}


@mcp.tool()
def link(
    src_id: str,
    dst_id: str,
    rel: str = "related",
) -> dict[str, Any]:
    """Add a directed link src → dst.
    rel: supersedes | refines | derived_from | example_of | contradicts | related.
    Idempotent — calling twice with the same arguments is safe.
    """
    try:
        store.link(src_id=src_id, dst_id=dst_id, rel=rel)
        return {"ok": True, "src_id": src_id, "dst_id": dst_id, "rel": rel}
    except (KeyError, ValueError) as e:
        return {"ok": False, "error": str(e)}


@mcp.tool()
def unlink(
    src_id: str,
    dst_id: str,
    rel: str | None = None,
) -> dict[str, Any]:
    """Remove link(s) from src to dst. If rel is omitted, all rels are removed."""
    removed = store.unlink(src_id=src_id, dst_id=dst_id, rel=rel)
    return {"removed": removed}


@mcp.tool()
def neighbors(
    memory_id: str,
    rel: str | None = None,
    direction: str = "both",
    depth: int = 1,
) -> list[dict[str, Any]]:
    """Return memories reachable from memory_id via links.
    direction: out | in | both (default).
    """
    hits = store.neighbors(
        memory_id,
        rel=rel,
        direction=direction,  # type: ignore[arg-type]
        depth=depth,
    )
    return [
        {
            "id": m.id,
            "text": m.text,
            "type": m.type,
            "tags": m.tags,
            "importance": m.importance,
            "rel": r,
        }
        for m, r in hits
    ]


@mcp.tool()
def find_conflicts(
    memory_id: str | None = None,
    threshold: float | None = None,
) -> list[dict[str, Any]]:
    """Detect duplicate or contradicting memories.
    memory_id: check one specific memory; omit for a full-store scan.
    threshold: cosine similarity cutoff (default: 0.80).
    """
    conflicts = store.find_conflicts(memory_id=memory_id, threshold=threshold)
    return [
        {
            "kind":       c.kind,
            "reason":     c.reason,
            "similarity": c.similarity,
            "a": {"id": c.a.id, "text": c.a.text[:120], "type": c.a.type},
            "b": {"id": c.b.id, "text": c.b.text[:120], "type": c.b.type},
        }
        for c in conflicts
    ]


@mcp.tool()
def supersede(old_id: str, new_id: str) -> dict[str, Any]:
    """Mark old_id as superseded by new_id (soft-delete old, add 'supersedes' link).
    The old memory stays in the store but is hidden from default queries.
    Use restore() / forget() to undo or hard-delete.
    """
    try:
        store.supersede(old_id=old_id, new_id=new_id)
        return {"ok": True, "old_id": old_id, "new_id": new_id}
    except (KeyError, ValueError) as e:
        return {"ok": False, "error": str(e)}


@mcp.tool()
def maintenance_tick(
    reflect_apply: bool = False,
) -> dict[str, Any]:
    """Run one maintenance tick: prune stale memories via decay and optionally
    consolidate episodic clusters via reflection.

    Uses the schedule configured in ~/.config/ai_houkai/config.toml
    [maintenance] section (or built-in defaults).  Jobs only run when their
    interval has elapsed since the last run — safe to call frequently.

    reflect_apply
        If True, reflection summaries are written to the store.
        If False (default), reflection runs in dry-run mode (no writes).

    Returns a summary dict with counts and any errors.
    """
    mcfg = load_maintenance()
    sched = MaintenanceScheduler(
        store=store,
        decay_every=mcfg.decay_every,
        reflect_every=mcfg.reflect_every,
        tick_interval=mcfg.tick_interval,
        state_path=mcfg.state_path,
        decay_rate=mcfg.decay_rate,
        min_score=mcfg.min_score,
        protect_types=mcfg.protect_types,
        frequency_weight=mcfg.frequency_weight,
        min_cluster_size=mcfg.min_cluster_size,
        reflect_apply=reflect_apply,
        summarizer=build_summarizer(mcfg.summarizer),
    )
    result = sched.tick()
    return {
        "summary": result.summary(),
        "ran_decay": result.ran_decay,
        "ran_reflect": result.ran_reflect,
        "decayed": result.decayed,
        "reflected": result.reflected,
        "decay_error": result.decay_error,
        "reflect_error": result.reflect_error,
    }


@mcp.tool()
def journal_tail(
    n: int = 20,
    op: str | None = None,
    since_seconds: float | None = None,
) -> list[dict[str, Any]]:
    """Return the most recent audit-journal entries (newest first).

    op: filter by remember|forget|supersede|restore|link|unlink|reflect|decay|...
    since_seconds: limit to entries within the last N seconds.
    """
    since = (time.time() - since_seconds) if since_seconds else None
    entries = list(store.journal.read(since=since, op=op))
    entries = entries[-n:][::-1]
    return [
        {"ts": e.ts, "op": e.op, "actor": e.actor,
         "id": e.id, "summary": e.summary(), "meta": e.meta}
        for e in entries
    ]


@mcp.tool(name="export")
def export(
    path: str,
    include_vectors: bool = True,
    include_superseded: bool = False,
    type: str | None = None,
    tag: str | None = None,
    since: float | None = None,
) -> dict[str, Any]:
    """Export memories to a portable .ahkai file (gzipped JSONL).

    The path is server-local. Returns summary counts.
    """
    summary = store.export(
        path,
        include_vectors=include_vectors,
        include_superseded=include_superseded,
        types=[type] if type else None,
        tags=[tag] if tag else None,
        since=since,
    )
    return {
        "path":    str(summary.path),
        "count":   summary.count,
        "bytes":   summary.bytes,
        "elapsed": summary.elapsed,
    }


@mcp.tool(name="import")
def import_(
    path: str,
    on_conflict: str = "skip",
    regenerate_vectors: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Import memories from a portable .ahkai file.

    on_conflict: skip | overwrite | rename | error.
    """
    try:
        summary = store.import_(
            path,
            on_conflict=on_conflict,            # type: ignore[arg-type]
            regenerate_vectors=regenerate_vectors,
            dry_run=dry_run,
        )
    except (ImportError, FileNotFoundError) as e:
        return {"ok": False, "error": str(e)}
    return {
        "ok": True,
        "imported":    summary.imported,
        "skipped":     summary.skipped,
        "overwritten": summary.overwritten,
        "renamed":     summary.renamed,
        "errors":      summary.errors,
        "vectors_regenerated": summary.vectors_regenerated,
    }


def run() -> None:
    """Console-script entry point: ``ai-houkai-mcp``."""
    mcp.run()


if __name__ == "__main__":
    run()
