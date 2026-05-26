"""Audit-journal CLI commands."""

from __future__ import annotations

import json as json
from datetime import datetime
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from ai_houkai.cli import output as out


def tail(
    ctx: typer.Context,
    n: int = typer.Option(20, "-n", "--number", help="Number of entries"),
    op: Optional[str] = typer.Option(None, "--op", help="Filter by operation"),
    actor: Optional[str] = typer.Option(None, "--actor", help="Filter by actor"),
    memory_id: Optional[str] = typer.Option(None, "--id", help="Filter by memory id"),
    include_archives: bool = typer.Option(False, "--all", help="Include rotated archives"),
) -> None:
    """Show the most recent journal entries (newest first)."""
    store = ctx.obj["store"]
    entries = list(store.journal.read(
        op=op, actor=actor, memory_id=memory_id,
        include_archives=include_archives,
    ))
    entries = entries[-n:][::-1]
    if not entries:
        typer.echo("(no journal entries)")
        return

    table = Table(title=f"Audit journal — {len(entries)} entries")
    table.add_column("time", style="dim")
    table.add_column("op")
    table.add_column("actor", style="cyan")
    table.add_column("summary")
    for e in entries:
        ts = datetime.fromtimestamp(e.ts).strftime("%Y-%m-%d %H:%M:%S")
        table.add_row(ts, e.op, e.actor, e.summary())
    Console().print(table)


def show(
    ctx: typer.Context,
    ts: float = typer.Argument(..., help="Entry timestamp (from `journal tail`)"),
) -> None:
    """Pretty-print one journal entry by timestamp."""
    store = ctx.obj["store"]
    entry = store.journal.find_by_ts(ts)
    if entry is None:
        typer.echo(f"No entry at ts={ts}", err=True)
        raise typer.Exit(1)
    typer.echo(json.dumps({
        "ts": entry.ts, "op": entry.op, "actor": entry.actor, "id": entry.id,
        "before": entry.before, "after": entry.after, "meta": entry.meta,
    }, indent=2))


def undo(
    ctx: typer.Context,
    ts: float = typer.Argument(..., help="Entry timestamp to reverse"),
    yes: bool = typer.Option(False, "--yes", "-y"),
) -> None:
    """Reverse a single journal entry."""
    store = ctx.obj["store"]
    entry = store.journal.find_by_ts(ts)
    if entry is None:
        typer.echo(f"No entry at ts={ts}", err=True)
        raise typer.Exit(1)
    if not out.confirm(f"Undo {entry.op} {entry.id[:8]}?", yes=yes):
        typer.echo("Aborted.")
        return
    ok = store.undo(entry)
    typer.echo("Undone." if ok else "Could not undo this entry.")
    if not ok:
        raise typer.Exit(1)


journal_app = typer.Typer(help="Inspect the audit journal.", no_args_is_help=True)
journal_app.command("tail")(tail)
journal_app.command("show")(show)
journal_app.command("undo")(undo)
