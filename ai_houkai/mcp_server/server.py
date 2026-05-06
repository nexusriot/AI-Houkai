"""MCP server exposing the AI-Houkai memory store.

Run with:
    ai-houkai-mcp
    # or: python -m ai_houkai.mcp_server.server

Tools exposed:
    remember(text, type?, tags?, importance?, source?, on_conflict?, polarity?)
    recall(query, k?, type?, tag?, min_importance?, mode?, overfetch?)
    forget(memory_id)
    list_recent(limit?, include_superseded?)
    stats()
    link(src_id, dst_id, rel?)
    unlink(src_id, dst_id, rel?)
    neighbors(memory_id, rel?, direction?, depth?)
    find_conflicts(memory_id?, threshold?)
    supersede(old_id, new_id)
"""

from __future__ import annotations

import os
from typing import Any

from mcp.server.fastmcp import FastMCP

from ai_houkai.memory_system import MemoryStore
from ai_houkai.memory_system.store import ConflictError

CHROMA_PATH = os.environ.get("AI_HOUKAI_PATH", "./.chroma")
COLLECTION  = os.environ.get("AI_HOUKAI_COLLECTION", "ai_houkai")

store = MemoryStore(path=CHROMA_PATH, collection=COLLECTION)
mcp   = FastMCP("AI-Houkai")


@mcp.tool()
def remember(
    text: str,
    type: str = "semantic",
    tags: list[str] | None = None,
    importance: float = 0.5,
    source: str | None = None,
    on_conflict: str | None = None,
    polarity: int = 0,
) -> dict[str, Any]:
    """Store a new memory.
    type: episodic | semantic | procedural | feedback.
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
    return {"id": mem.id, "stored": True}


@mcp.tool()
def recall(
    query: str,
    k: int = 5,
    type: str | None = None,
    tag: str | None = None,
    min_importance: float | None = None,
    mode: str = "semantic",
    overfetch: int = 4,
    include_superseded: bool = False,
) -> list[dict[str, Any]]:
    """Semantic (or hybrid) search across stored memories.
    mode: "semantic" (default) | "hybrid" (cosine + BM25 + recency + importance).
    """
    from ai_houkai.memory_system.store import HybridWeights
    hits = store.recall(
        query=query,
        k=k,
        type=type,            # type: ignore[arg-type]
        tag=tag,
        min_importance=min_importance,
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


def run() -> None:
    """Console-script entry point: ``ai-houkai-mcp``."""
    mcp.run()


if __name__ == "__main__":
    run()
