from __future__ import annotations

from typing import Optional

import typer

from ai_houkai.cli import output as out


def recall(
    ctx: typer.Context,
    query: str = typer.Argument(..., help="Semantic search query"),
    k: int = typer.Option(5, "-k", "--limit"),
    type: Optional[str] = typer.Option(None, "-t", "--type"),
    tag: Optional[str] = typer.Option(None, "-g", "--tag"),
    min_importance: Optional[float] = typer.Option(None, "--min-importance"),
    mode: str = typer.Option("semantic", "--mode", help="semantic|hybrid"),
    include_superseded: bool = typer.Option(False, "--include-superseded"),
    fmt: str = typer.Option("auto", "--format", "-f", help="auto|rich|tsv|json"),
) -> None:
    """Semantic search across memories."""
    store = ctx.obj["store"]
    results = store.recall(
        query=query,
        k=k,
        type=type,
        tag=tag,
        min_importance=min_importance,
        mode=mode,
        include_superseded=include_superseded,
    )
    if not results:
        typer.echo("No memories found.", err=True)
        return
    out.print_memories_table(results, show_score=True, fmt=fmt)
