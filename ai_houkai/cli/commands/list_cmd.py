from __future__ import annotations

import time
from typing import Optional

import typer

from ai_houkai.cli import output as out

_DURATION_MAP = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


def _parse_since(since: str) -> float:
    import datetime
    if not since:
        return 0.0
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.datetime.strptime(since, fmt)
            return dt.timestamp()
        except ValueError:
            pass
    unit = since[-1]
    if unit in _DURATION_MAP:
        try:
            n = float(since[:-1])
            return time.time() - n * _DURATION_MAP[unit]
        except ValueError:
            pass
    raise typer.BadParameter(f"Unrecognised --since value: {since!r}")


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
    memories = store.list_recent(limit=9999, include_superseded=include_superseded)

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
