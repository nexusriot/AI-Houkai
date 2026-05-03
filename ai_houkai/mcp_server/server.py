"""MCP server exposing the AI-Houkai memory store.

Run with:
    ai-houkai-mcp
    # or: python -m ai_houkai.mcp_server.server

Tools exposed:
    remember(text, type?, tags?, importance?, source?)
    recall(query, k?, type?, tag?, min_importance?)
    forget(memory_id)
    list_recent(limit?)
    stats()
"""

from __future__ import annotations

import os
from typing import Any

from mcp.server.fastmcp import FastMCP

from ai_houkai.memory_system import MemoryStore

CHROMA_PATH = os.environ.get("AI_HOUKAI_PATH", "./.chroma")
COLLECTION = os.environ.get("AI_HOUKAI_COLLECTION", "ai_houkai")

store = MemoryStore(path=CHROMA_PATH, collection=COLLECTION)
mcp = FastMCP("AI-Houkai")


@mcp.tool()
def remember(
    text: str,
    type: str = "semantic",
    tags: list[str] | None = None,
    importance: float = 0.5,
    source: str | None = None,
) -> dict[str, Any]:
    """Store a new memory. `type` is one of episodic|semantic|procedural|feedback."""
    mem = store.remember(
        text=text,
        type=type,  # type: ignore[arg-type]
        tags=tags or [],
        importance=importance,
        source=source,
    )
    return {"id": mem.id, "stored": True}


@mcp.tool()
def recall(
    query: str,
    k: int = 5,
    type: str | None = None,
    tag: str | None = None,
    min_importance: float | None = None,
) -> list[dict[str, Any]]:
    """Semantic search across stored memories with optional metadata filters."""
    hits = store.recall(
        query=query,
        k=k,
        type=type,  # type: ignore[arg-type]
        tag=tag,
        min_importance=min_importance,
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
        }
        for m, score in hits
    ]


@mcp.tool()
def forget(memory_id: str) -> dict[str, Any]:
    """Delete a memory by id."""
    return {"deleted": store.forget(memory_id)}


@mcp.tool()
def list_recent(limit: int = 20) -> list[dict[str, Any]]:
    """List the most recently created memories."""
    return [
        {
            "id": m.id,
            "text": m.text,
            "type": m.type,
            "tags": m.tags,
            "created_at": m.created_at,
        }
        for m in store.list_recent(limit=limit)
    ]


@mcp.tool()
def stats() -> dict[str, Any]:
    """Return basic store statistics."""
    return {"count": store.count(), "path": CHROMA_PATH, "collection": COLLECTION}


def run() -> None:
    """Console-script entry point: ``ai-houkai-mcp``."""
    mcp.run()


if __name__ == "__main__":
    run()
