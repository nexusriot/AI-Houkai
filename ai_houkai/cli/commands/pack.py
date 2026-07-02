from __future__ import annotations

import json
from typing import Optional

import typer

from ai_houkai.cli import output as out
from ai_houkai.timeparse import parse_timestamp


def pack(
    ctx: typer.Context,
    query: str = typer.Argument(..., help="Semantic search query"),
    budget: int = typer.Option(800, "-b", "--budget", help="Token budget for the packed block"),
    type: Optional[str] = typer.Option(None, "-t", "--type"),
    tag: Optional[str] = typer.Option(None, "-g", "--tag"),
    min_importance: Optional[float] = typer.Option(None, "--min-importance"),
    source: Optional[str] = typer.Option(None, "--source", help="Filter by exact provenance string"),
    since: Optional[str] = typer.Option(None, "--since", help="Only memories created at/after (ISO date, epoch, or '7d')"),
    until: Optional[str] = typer.Option(None, "--until", help="Only memories created at/before (ISO date, epoch, or '7d')"),
    mode: str = typer.Option("hybrid", "--mode", help="semantic|hybrid"),
    max_items: int = typer.Option(50, "--max-items", help="Ranked candidates to consider"),
    include_superseded: bool = typer.Option(False, "--include-superseded"),
    header: str = typer.Option("## Relevant memory", "--header", help="Block header (empty string to omit)"),
    fmt: str = typer.Option("text", "--format", "-f", help="text|json"),
) -> None:
    """Assemble the most relevant memories into a token-budgeted context block.

    Ranks with hybrid scoring by default, then greedily packs results until the
    token budget is reached. The block is printed to stdout (pipe it into a
    prompt); a one-line summary goes to stderr.
    """
    store = ctx.obj["store"]
    try:
        since_ts = parse_timestamp(since)
        until_ts = parse_timestamp(until)
    except ValueError as exc:
        raise typer.BadParameter(str(exc))
    with out.friendly_errors():
        result = store.recall_pack(
            query=query,
            token_budget=budget,
            type=type,
            tag=tag,
            min_importance=min_importance,
            source=source,
            since=since_ts,
            until=until_ts,
            mode=mode,
            max_items=max_items,
            include_superseded=include_superseded,
            header=header,
        )

    if fmt == "json":
        typer.echo(json.dumps(
            {
                "text": result.text,
                "used_tokens": result.used_tokens,
                "budget": result.budget,
                "truncated": result.truncated,
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
                    for p in result.items
                ],
            },
            indent=2,
        ))
        return

    if not result.items:
        typer.echo("No memories found.", err=True)
        return

    typer.echo(result.text)
    summary = (
        f"[{len(result.items)} memories · {result.used_tokens}/{result.budget} tokens"
        f"{' · truncated' if result.truncated else ''}]"
    )
    typer.echo(summary, err=True)
