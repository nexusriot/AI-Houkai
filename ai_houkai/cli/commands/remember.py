from __future__ import annotations

import sys
from typing import List, Optional

import typer

from ai_houkai.memory_system.importance import score_importance


def remember(
    ctx: typer.Context,
    text: Optional[str] = typer.Argument(
        None,
        help="Memory text. Use '-' or omit to read from stdin.",
    ),
    type: str = typer.Option("", "-t", "--type", help="episodic|semantic|procedural|feedback"),
    tags: List[str] = typer.Option([], "-g", "--tag", help="Tag (repeatable)"),
    importance: Optional[float] = typer.Option(None, "-i", "--importance", min=0.0, max=1.0),
    source: Optional[str] = typer.Option(None, "-s", "--source"),
    on_conflict: str = typer.Option("ignore", "--on-conflict", help="ignore|warn|supersede|raise"),
    polarity: int = typer.Option(0, "--polarity", help="-1, 0, or 1"),
    stdin: bool = typer.Option(False, "--stdin", help="Force reading text from stdin"),
    auto_importance: bool = typer.Option(
        False, "--auto-importance",
        help="Score importance heuristically from the text "
             "(also: default_importance = \"auto\" in config.toml)",
    ),
) -> None:
    """Store a new memory."""
    store = ctx.obj["store"]
    cfg = ctx.obj["config"]

    if stdin or text == "-" or text is None:
        body = sys.stdin.read().strip()
        if not body:
            typer.echo("Error: no text provided", err=True)
            raise typer.Exit(1)
    else:
        body = text

    mem_type = type or cfg.default_type
    if importance is not None:
        imp = importance
    elif auto_importance or cfg.default_importance == "auto":
        imp = score_importance(body, mem_type, list(tags))
    else:
        imp = cfg.default_importance

    mem = store.remember(
        text=body,
        type=mem_type,
        tags=list(tags),
        importance=imp,
        source=source,
        polarity=polarity,
        on_conflict=on_conflict,
    )
    typer.echo(mem.id)
