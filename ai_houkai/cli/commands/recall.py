from __future__ import annotations

from typing import Optional

import typer

from ai_houkai.cli import output as out
from ai_houkai.timeparse import parse_timestamp


def recall(
    ctx: typer.Context,
    query: str = typer.Argument(..., help="Semantic search query"),
    k: int = typer.Option(5, "-k", "--limit"),
    type: Optional[str] = typer.Option(None, "-t", "--type"),
    tag: Optional[str] = typer.Option(None, "-g", "--tag"),
    min_importance: Optional[float] = typer.Option(None, "--min-importance"),
    source: Optional[str] = typer.Option(None, "--source", help="Filter by exact provenance string"),
    since: Optional[str] = typer.Option(None, "--since", help="Only memories created at/after (ISO date, epoch, or '7d')"),
    until: Optional[str] = typer.Option(None, "--until", help="Only memories created at/before (ISO date, epoch, or '7d')"),
    mode: str = typer.Option("semantic", "--mode", help="semantic|hybrid"),
    include_superseded: bool = typer.Option(False, "--include-superseded"),
    include_expired: bool = typer.Option(False, "--include-expired",
                                         help="Also return memories whose TTL has passed"),
    fmt: str = typer.Option("auto", "--format", "-f", help="auto|rich|tsv|json"),
) -> None:
    """Semantic search across memories."""
    store = ctx.obj["store"]
    try:
        since_ts = parse_timestamp(since)
        until_ts = parse_timestamp(until)
    except ValueError as exc:
        raise typer.BadParameter(str(exc))
    with out.friendly_errors():
        results = store.recall(
            query=query,
            k=k,
            type=type,
            tag=tag,
            min_importance=min_importance,
            source=source,
            since=since_ts,
            until=until_ts,
            mode=mode,
            include_superseded=include_superseded,
            include_expired=include_expired,
        )
    if not results:
        typer.echo("No memories found.", err=True)
        return
    out.print_memories_table(results, show_score=True, fmt=fmt)
