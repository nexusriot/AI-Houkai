from __future__ import annotations

import sys
from typing import List, Optional

import typer

from ai_houkai.cli import output as out
from ai_houkai.memory_system.importance import score_importance
from ai_houkai.memory_system.store import ConflictError


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
    pinned: bool = typer.Option(
        False, "--pin",
        help="Standing instruction: always offered to `pack --include-pinned`, "
             "never pruned by decay."),
    trust: str = typer.Option(
        "trusted", "--trust",
        help="trusted|reported|untrusted — how much the memory's ORIGIN is "
             "trusted. Use untrusted for anything read from content you did "
             "not author."),
    idempotent: bool = typer.Option(
        False, "--idempotent",
        help="No-op if a live memory already has the same normalised text."),
    ttl: Optional[float] = typer.Option(
        None, "--ttl", help="Time-to-live in seconds; the memory expires after this."),
    expires_at: Optional[float] = typer.Option(
        None, "--expires-at", help="Absolute expiry as a Unix timestamp."),
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

    try:
        with out.friendly_errors():
            mem = store.remember(
                text=body,
                type=mem_type,
                tags=list(tags),
                importance=imp,
                source=source,
                polarity=polarity,
                expires_at=expires_at,
                ttl_seconds=ttl,
                on_conflict=on_conflict,
                pinned=pinned,
                trust=trust,
                idempotent=idempotent,
            )
    except ConflictError as e:
        # --on-conflict raise doing its job: the memory was NOT stored.
        typer.echo(f"Not stored: {len(e.conflicts)} conflict(s) detected", err=True)
        for c in e.conflicts:
            typer.echo(
                f"  {c.kind} (sim={c.similarity:.3f}) with "
                f"{out.short_id(c.b.id)}: {c.b.text[:60]}",
                err=True,
            )
        raise typer.Exit(1)
    typer.echo(mem.id)
