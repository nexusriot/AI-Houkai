from __future__ import annotations

from typing import Optional

import typer

from ai_houkai.cli import output as out
from ai_houkai.timeparse import parse_timestamp


def _parse_since(since: str) -> float:
    """Same grammar as `recall --since` (ISO date in UTC, epoch, or '7d').
    A private local-time parser here used to make the same date string
    filter a different memory set than recall's."""
    if not since:
        return 0.0
    try:
        ts = parse_timestamp(since)
    except ValueError as exc:
        raise typer.BadParameter(str(exc))
    return ts if ts is not None else 0.0


def list_memories(
    ctx: typer.Context,
    n: int = typer.Option(20, "-n", "--limit"),
    type: Optional[str] = typer.Option(None, "-t", "--type"),
    tag: Optional[str] = typer.Option(None, "-g", "--tag"),
    since: Optional[str] = typer.Option(None, "--since", help="e.g. 7d, 2h, 2026-01-01"),
    sort: str = typer.Option("created", "--sort", help="created|importance"),
    include_superseded: bool = typer.Option(False, "--include-superseded"),
    fmt: str = typer.Option("auto", "--format", "-f", help="auto|rich|tsv|json"),
) -> None:
    """List most recently created memories."""
    store = ctx.obj["store"]
    # Fetch everything: type/tag/since filter below, so a fixed fetch cap
    # would silently drop older matches.
    memories = store.list_recent(
        limit=max(store.count(), 1), include_superseded=include_superseded)

    if type:
        memories = [m for m in memories if m.type == type]
    if tag:
        memories = [m for m in memories if tag in m.tags]
    if since:
        cutoff = _parse_since(since)
        memories = [m for m in memories if m.created_at >= cutoff]

    if sort == "importance":
        memories.sort(key=lambda m: m.importance, reverse=True)
    else:
        memories.sort(key=lambda m: m.created_at, reverse=True)

    memories = memories[:n]

    if not memories:
        typer.echo("No memories found.", err=True)
        return
    out.print_memories_table(memories, show_score=False, fmt=fmt)
