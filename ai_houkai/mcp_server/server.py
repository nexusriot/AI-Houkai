"""MCP server exposing the AI-Houkai memory store.

Run with:
    ai-houkai-mcp
    # or: python -m ai_houkai.mcp_server.server

Tools exposed (22):
    remember(text, type?, tags?, importance?, source?, on_conflict?, polarity?, expires_at?, ttl_seconds?)
    edit(memory_id, text?, type?, tags?, importance?, polarity?, source?, clear_source?, expires_at?)
    recall(query, k?, type?, tag?, min_importance?, source?, since?, until?, mode?, overfetch?, include_superseded?, include_expired?, explain?)
    recall_pack(query, token_budget?, type?, tag?, min_importance?, source?, since?, until?, mode?, max_items?, compress?, compress_threshold?, compress_min_group?)
    auto_context(task, token_budget?, max_phrases?, mode?)
    forget(memory_id)
    purge_expired(dry_run?)
    list_recent(limit?, include_superseded?, include_expired?)
    stats()
    metrics()
    history(memory_id)
    state_at(ts)
    get_at(memory_id, ts)
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
from ai_houkai.memory_system.store import (
    ConflictError,
    ImportConflictError,
    extract_key_phrases,
)
from ai_houkai.memory_system.summarizers import build_summarizer
from ai_houkai.timeparse import parse_timestamp

mcp = FastMCP("AI-Houkai")

_store: MemoryStore | None = None


def get_store() -> MemoryStore:
    """Return the process-wide store, creating it on first use.

    Importing this module is side-effect-free: no ChromaDB directory is
    materialised and no embedding model is loaded until a tool actually
    runs. The installers rely on this to inspect the tool surface, and env
    configuration (AI_HOUKAI_PATH / AI_HOUKAI_COLLECTION /
    AI_HOUKAI_AUTO_IMPORTANCE) is read here, at first use, not at import.
    """
    global _store
    if _store is None:
        auto_importance = os.environ.get(
            "AI_HOUKAI_AUTO_IMPORTANCE", "").lower() in ("1", "true", "yes", "on")
        _store = MemoryStore(
            path=os.path.expanduser(os.environ.get("AI_HOUKAI_PATH", "./.chroma")),
            collection=os.environ.get("AI_HOUKAI_COLLECTION", "ai_houkai"),
            actor="mcp",
            importance_fn=score_importance if auto_importance else None,
        )
    return _store


@mcp.tool()
def remember(
    text: str,
    type: str = "semantic",
    tags: list[str] | None = None,
    importance: float | None = None,
    source: str | None = None,
    on_conflict: str | None = None,
    polarity: int = 0,
    expires_at: float | None = None,
    ttl_seconds: float | None = None,
) -> dict[str, Any]:
    """Store a new memory.
    type: episodic | semantic | procedural | feedback.
    importance: 0..1; omit for the default (0.5, or a heuristic score when
    the server runs with AI_HOUKAI_AUTO_IMPORTANCE=1).
    on_conflict: ignore | warn | supersede | raise (default: store policy).
    polarity: -1 (negative) | 0 (neutral) | +1 (positive).
    expires_at / ttl_seconds: optional TTL — an absolute epoch (expires_at) or
    a relative lifetime in seconds (ttl_seconds). Expired memories are hidden
    from recall and reclaimable by purge_expired. Pass at most one.
    """

    policy = on_conflict  # type: ignore[assignment]
    try:
        mem = get_store().remember(
            text=text,
            type=type,           # type: ignore[arg-type]
            tags=tags or [],
            importance=importance,
            source=source,
            polarity=polarity,
            expires_at=expires_at,
            ttl_seconds=ttl_seconds,
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
    return {"id": mem.id, "stored": True, "importance": mem.importance,
            "expires_at": mem.expires_at or None}


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
    include_expired: bool = False,
    explain: bool = False,
) -> list[dict[str, Any]]:
    """Semantic (or hybrid) search across stored memories.
    mode: "semantic" (default) | "hybrid" (cosine + BM25 + recency + importance).
    source: keep only memories with this exact provenance string.
    since/until: bound created_at — epoch seconds, an ISO-8601 date/datetime,
    or a relative span like "7d" / "24h" (since="7d" → last 7 days).
    include_expired: also return memories whose TTL has passed (hidden by default).
    explain: attach a per-signal score breakdown to each hit under "explain".
    """
    hits = get_store().recall(
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
        include_expired=include_expired,
        explain=explain,
    )

    def _hit(m: Any, score: float) -> dict[str, Any]:
        return {
            "id": m.id,
            "text": m.text,
            "type": m.type,
            "tags": m.tags,
            "importance": m.importance,
            "score": round(score, 4),
            "created_at": m.created_at,
            "superseded_by": m.superseded_by or None,
            "expires_at": m.expires_at or None,
        }

    if explain:
        return [{**_hit(m, score), "explain": expl} for m, score, expl in hits]
    return [_hit(m, score) for m, score in hits]


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
    compress: bool = False,
    compress_threshold: float = 0.30,
    compress_min_group: int = 2,
) -> dict[str, Any]:
    """Assemble the most relevant memories into a token-budgeted context block.

    Ranks with hybrid scoring (cosine + BM25 + recency + importance + polarity)
    by default, then greedily packs results until token_budget is reached.
    Returns a ready-to-inject `text` block plus the packed items.
    token_budget is a soft ceiling (~4 chars/token) covering memory lines only.

    compress=True: when candidates are dropped for exceeding the budget, similar
    ones are clustered by token-Jaccard and folded into a single summary line
    (marked "compressed") that may fit in the remaining space.
    compress_threshold: Jaccard similarity threshold for grouping (default 0.30).
    compress_min_group: minimum cluster size to produce a compressed line (default 2).
    """
    pack = get_store().recall_pack(
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
        compress=compress,
        compress_threshold=compress_threshold,
        compress_min_group=compress_min_group,
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
        "compressed_groups": [
            {
                "ids": [m.id for m in cg.memories],
                "text": cg.text,
                "tokens": cg.tokens,
                "count": len(cg.memories),
            }
            for cg in pack.compressed_groups
        ],
    }


@mcp.tool()
def auto_context(
    task: str,
    token_budget: int = 800,
    max_phrases: int = 3,
    mode: str = "hybrid",
) -> dict[str, Any]:
    """Build a ready-to-inject context block by fanning out over multiple recall angles.

    Extracts up to max_phrases key bigram/keyword phrases from the task description,
    recalls memories for each angle independently, deduplicates (keeping the highest
    score per memory), then packs greedily within token_budget.

    More thorough than a single recall_pack call for tasks with compound concepts
    (e.g. "deploy the API to production" → also searches "deploy api", "api production").
    Returns the same structure as recall_pack plus the queries that were used.
    """
    queries = [task] + extract_key_phrases(task, max_phrases)
    pack = get_store().auto_context_pack(
        task=task,
        token_budget=token_budget,
        max_phrases=max_phrases,
        mode=mode,             # type: ignore[arg-type]
    )
    return {
        "text": pack.text,
        "queries": queries,
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
    return {"deleted": get_store().forget(memory_id)}


@mcp.tool()
def edit(
    memory_id: str,
    text: str | None = None,
    type: str | None = None,
    tags: list[str] | None = None,
    importance: float | None = None,
    polarity: int | None = None,
    source: str | None = None,
    clear_source: bool = False,
    expires_at: float | None = None,
) -> dict[str, Any]:
    """Update fields of an existing memory in place, keeping its id.

    Omitted fields stay unchanged. Text changes are re-embedded; links,
    supersede state, and access tracking are preserved (do NOT forget+remember
    to fix a typo — that loses them). The change is journaled and undoable.
    type: episodic | semantic | procedural | feedback.
    clear_source: true removes the provenance string (an omitted `source`
    otherwise means "leave unchanged"); do not combine with `source`.
    expires_at: set the TTL to this absolute epoch; pass 0 to clear it (omit to
    leave unchanged).
    """
    if clear_source and source is not None:
        return {"ok": False,
                "error": "pass either source or clear_source, not both"}
    kwargs: dict[str, Any] = {}
    if text is not None:
        kwargs["text"] = text
    if type is not None:
        kwargs["type"] = type
    if tags is not None:
        kwargs["tags"] = tags
    if importance is not None:
        kwargs["importance"] = importance
    if polarity is not None:
        kwargs["polarity"] = polarity
    if expires_at is not None:
        kwargs["expires_at"] = expires_at
    if source is not None:
        kwargs["source"] = source
    if clear_source:
        kwargs["source"] = None
    try:
        mem = get_store().edit(memory_id, **kwargs)
    except (KeyError, ValueError) as e:
        return {"ok": False, "error": e.args[0] if e.args else str(e)}
    return {
        "ok": True,
        "id": mem.id,
        "text": mem.text,
        "type": mem.type,
        "tags": mem.tags,
        "importance": mem.importance,
        "polarity": mem.polarity,
        "source": mem.source,
        "expires_at": mem.expires_at or None,
    }


@mcp.tool()
def list_recent(
    limit: int = 20,
    include_superseded: bool = False,
    include_expired: bool = False,
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
            "expires_at": m.expires_at or None,
        }
        for m in get_store().list_recent(
            limit=limit, include_superseded=include_superseded,
            include_expired=include_expired)
    ]


@mcp.tool()
def purge_expired(dry_run: bool = False) -> dict[str, Any]:
    """Hard-delete memories whose TTL has passed (reclaims storage).

    Expired memories are already hidden from recall; this removes them for
    good. dry_run=True reports what would be purged without deleting.
    """
    purged = get_store().purge_expired(dry_run=dry_run)
    return {"purged": len(purged), "dry_run": dry_run,
            "ids": [p.id for p in purged]}


@mcp.tool()
def metrics() -> dict[str, Any]:
    """Runtime metrics: op counters + recall latency since server start.

    Process-local and in-memory (reset on restart). Complements `stats`
    (content aggregates) with operational counts.
    """
    return get_store().metrics()


@mcp.tool()
def history(memory_id: str) -> list[dict[str, Any]]:
    """Full journaled timeline of one memory, oldest first.

    Every event touching the memory — creation, edits, supersede/restore,
    links pointing at it, forget — with before/after snapshots. Bounded by
    journal retention.
    """
    return [
        {"ts": e.ts, "op": e.op, "actor": e.actor, "id": e.id,
         "before": e.before, "after": e.after, "meta": e.meta,
         "summary": e.summary()}
        for e in get_store().history(memory_id)
    ]


@mcp.tool()
def state_at(ts: str) -> dict[str, Any]:
    """Reconstruct the store's live memories as of a past time.

    ts: epoch seconds, an ISO-8601 date/datetime, or a relative span like "7d".
    Best-effort replay of the journal (see the store's state_at docs for its
    limits: retention window, nuke resets, journaling-disabled gaps).
    """
    t = parse_timestamp(ts)
    if t is None:
        return {"error": "ts is required"}
    mems = get_store().state_at(t)
    return {
        "ts": t,
        "count": len(mems),
        "memories": [
            {"id": m.id, "text": m.text, "type": m.type, "tags": m.tags,
             "importance": m.importance, "created_at": m.created_at}
            for m in mems
        ],
    }


@mcp.tool()
def get_at(memory_id: str, ts: str) -> dict[str, Any]:
    """Reconstruct a single memory as it was at a past time (see state_at)."""
    t = parse_timestamp(ts)
    if t is None:
        return {"error": "ts is required"}
    mem = get_store().get_at(memory_id, t)
    if mem is None:
        return {"ok": False, "error": "memory did not exist at that time"}
    return {"ok": True, "ts": t, **{
        "id": mem.id, "text": mem.text, "type": mem.type, "tags": mem.tags,
        "importance": mem.importance, "created_at": mem.created_at,
        "superseded_by": mem.superseded_by or None,
        "expires_at": mem.expires_at or None,
    }}


@mcp.tool()
def stats() -> dict[str, Any]:
    """Return basic store statistics."""
    store = get_store()
    return {"count": store.count(), "path": store.path,
            "collection": store.collection_name}


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
        get_store().link(src_id=src_id, dst_id=dst_id, rel=rel)
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
    removed = get_store().unlink(src_id=src_id, dst_id=dst_id, rel=rel)
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
    hits = get_store().neighbors(
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
    conflicts = get_store().find_conflicts(memory_id=memory_id, threshold=threshold)
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
        get_store().supersede(old_id=old_id, new_id=new_id)
        return {"ok": True, "old_id": old_id, "new_id": new_id}
    except (KeyError, ValueError) as e:
        return {"ok": False, "error": str(e)}


@mcp.tool()
def maintenance_tick(
    reflect_apply: bool | None = None,
) -> dict[str, Any]:
    """Run one maintenance tick: prune stale memories via decay and optionally
    consolidate episodic clusters via reflection.

    Uses the schedule configured in ~/.config/ai_houkai/config.toml
    [maintenance] section (or built-in defaults).  Jobs only run when their
    interval has elapsed since the last run — safe to call frequently.

    reflect_apply
        If True, reflection summaries are written to the store.
        If False, reflection runs in dry-run mode (reports what it would
        create, writes nothing). Omit to use the config file's
        [maintenance.reflect] apply setting (default False).

    When [maintenance].enabled = false in the config, nothing runs and the
    result reports enabled=false.

    Returns a summary dict with counts and any errors.
    """
    mcfg = load_maintenance()
    if not mcfg.enabled:
        return {
            "enabled": False,
            "summary": "maintenance disabled ([maintenance].enabled = false)",
            "ran_decay": False,
            "ran_reflect": False,
            "ran_purge": False,
            "decayed": 0,
            "reflected": 0,
            "purged": 0,
            "reflect_applied": False,
            "decay_error": None,
            "reflect_error": None,
            "purge_error": None,
        }
    if reflect_apply is None:
        reflect_apply = mcfg.reflect_apply
    sched = MaintenanceScheduler(
        store=get_store(),
        decay_every=mcfg.decay_every,
        reflect_every=mcfg.reflect_every,
        purge_every=mcfg.purge_every,
        tick_interval=mcfg.tick_interval,
        state_path=mcfg.state_path,
        decay_rate=mcfg.decay_rate,
        min_score=mcfg.min_score,
        protect_types=mcfg.protect_types,
        frequency_weight=mcfg.frequency_weight,
        min_cluster_size=mcfg.min_cluster_size,
        reflect_apply=reflect_apply,
        reflect_consolidate=mcfg.reflect_consolidate,
        summarizer=build_summarizer(mcfg.summarizer),
    )
    result = sched.tick()
    return {
        "enabled": True,
        "summary": result.summary(),
        "ran_decay": result.ran_decay,
        "ran_reflect": result.ran_reflect,
        "ran_purge": result.ran_purge,
        "decayed": result.decayed,
        "reflected": result.reflected,
        "purged": result.purged,
        "reflect_applied": result.reflect_applied,
        "decay_error": result.decay_error,
        "reflect_error": result.reflect_error,
        "purge_error": result.purge_error,
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
    if n <= 0:  # entries[-0:] would be the WHOLE journal, not none of it
        return []
    since = (time.time() - since_seconds) if since_seconds is not None else None
    entries = list(get_store().journal.read(since=since, op=op))
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
    since: str | float | None = None,
) -> dict[str, Any]:
    """Export memories to a portable .ahkai file (gzipped JSONL).

    The path is server-local. Returns summary counts.
    since: epoch float, ISO date, or relative like "7d" — same as recall.
    """
    summary = get_store().export(
        path,
        include_vectors=include_vectors,
        include_superseded=include_superseded,
        types=[type] if type else None,
        tags=[tag] if tag else None,
        since=parse_timestamp(since),
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
        summary = get_store().import_(
            path,
            on_conflict=on_conflict,            # type: ignore[arg-type]
            regenerate_vectors=regenerate_vectors,
            dry_run=dry_run,
        )
    except (ImportError, ImportConflictError, FileNotFoundError) as e:
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
