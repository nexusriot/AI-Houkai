"""Point-in-time and runtime-introspection commands.

`history` / `state-at` / `get-at` replay the audit journal ("what did I know as
of T?"); `metrics` reports the process-local op counters and recall latency.
All four wrap store methods that previously had no CLI surface at all.
"""

from __future__ import annotations

import json as jsonlib
import time
from datetime import datetime
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from ai_houkai.cli import output as out
from ai_houkai.timeparse import parse_timestamp


def _resolve(store, id_or_prefix: str) -> str:
    try:
        return out.resolve_id_prefix(store, id_or_prefix)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


def _require_ts(raw: str) -> float:
    """Resolve a CLI timestamp argument, or exit with a readable error.

    parse_timestamp raises on garbage rather than returning None, so catch it
    here — an operator typo must not surface as a traceback.
    """
    if raw.strip().lower() == "now":
        return time.time()
    try:
        ts = parse_timestamp(raw)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    if ts is None:
        typer.echo(f"Error: could not parse timestamp {raw!r}", err=True)
        raise typer.Exit(1)
    return ts


def history(
    ctx: typer.Context,
    id: str = typer.Argument(..., help="Memory id or 8-char prefix"),
    json: bool = typer.Option(False, "--json", help="Emit raw JSON"),
    all_archives: bool = typer.Option(
        True, "--archives/--no-archives",
        help="Include rotated journal segments (default: yes)"),
) -> None:
    """Show the full journaled timeline of one memory.

    Includes entries that only reference the memory indirectly — the link,
    supersede and undo counterparts recorded against the other side.
    """
    store = ctx.obj["store"]
    full_id = _resolve(store, id)
    entries = store.history(full_id, include_archives=all_archives)
    if not entries and store.get(full_id) is None:
        typer.echo(f"Error: memory {id!r} not found", err=True)
        raise typer.Exit(1)

    if json:
        typer.echo(jsonlib.dumps([
            {"ts": e.ts, "op": e.op, "actor": e.actor, "id": e.id,
             "before": e.before, "after": e.after, "meta": e.meta}
            for e in entries
        ], indent=2))
        return

    if not entries:
        typer.echo("(no journal history — journaling may have been disabled)")
        return

    table = Table(title=f"History — {out.short_id(full_id)} ({len(entries)} entries)")
    table.add_column("ts", style="dim")
    table.add_column("time", style="dim")
    table.add_column("op")
    table.add_column("actor", style="cyan")
    table.add_column("summary")
    for e in entries:
        when = datetime.fromtimestamp(e.ts).strftime("%Y-%m-%d %H:%M:%S")
        table.add_row(f"{e.ts:.3f}", when, e.op, e.actor, e.summary())
    Console().print(table)


def state_at(
    ctx: typer.Context,
    ts: str = typer.Argument(
        ..., help="'now', epoch seconds, ISO-8601, or a relative span like '7d'"),
    json: bool = typer.Option(False, "--json", help="Emit raw JSON"),
) -> None:
    """Reconstruct every live memory as it stood at a past time.

    Best-effort journal replay: only mutations still present in the journal
    (and its archives) can be reversed, and a `nuke` resets the reconstruction.
    """
    store = ctx.obj["store"]
    when = _require_ts(ts)
    mems = store.state_at(when)
    if json:
        typer.echo(jsonlib.dumps(
            {"ts": when, "count": len(mems), "memories": [m.to_dict() for m in mems]},
            indent=2))
        return
    stamp = datetime.fromtimestamp(when).strftime("%Y-%m-%d %H:%M:%S")
    typer.echo(f"{len(mems)} memories as of {stamp}")
    if mems:
        out.print_memories_table(mems)


def get_at(
    ctx: typer.Context,
    id: str = typer.Argument(..., help="Memory id or 8-char prefix"),
    ts: str = typer.Argument(
        ..., help="'now', epoch seconds, ISO-8601, or a relative span like '7d'"),
    json: bool = typer.Option(False, "--json", help="Emit raw JSON"),
) -> None:
    """Reconstruct one memory as it was at a past time (see `state-at`)."""
    store = ctx.obj["store"]
    full_id = _resolve(store, id)
    when = _require_ts(ts)
    mem = store.get_at(full_id, when)
    if mem is None:
        typer.echo("Memory did not exist at that time.", err=True)
        raise typer.Exit(1)
    if json:
        typer.echo(jsonlib.dumps(mem.to_dict(), indent=2))
        return
    out.print_memory_detail(mem)


def metrics(
    ctx: typer.Context,
    json: bool = typer.Option(False, "--json", help="Emit raw JSON"),
) -> None:
    """Show runtime op counters and recall latency for this process.

    Metrics are per-process and in-memory: a fresh CLI invocation starts from
    zero, so this is mostly useful against a long-lived `houkai serve` via
    `GET /metrics`. Shown here for parity and for scripted one-shot checks.
    """
    data = ctx.obj["store"].metrics()
    if json:
        typer.echo(jsonlib.dumps(data, indent=2))
        return

    table = Table(title="Runtime metrics")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("uptime_seconds", f"{data['uptime_seconds']:.1f}")
    table.add_row("count", str(data["count"]))
    for op, n in sorted(data["calls"].items()):
        table.add_row(f"calls.{op}", str(n))
    lat = data["recall_latency_ms"]
    for key in ("count", "avg", "max", "p50", "p95", "p99"):
        if key in lat:
            table.add_row(f"recall_latency_ms.{key}", f"{lat[key]}")
    Console().print(table)


def journal_undo_last(
    ctx: typer.Context,
    memory_id: Optional[str] = typer.Option(
        None, "--id", help="Undo the newest entry touching this memory"),
    yes: bool = typer.Option(False, "--yes", "-y"),
) -> None:
    """Reverse the most recent journaled mutation (optionally for one memory).

    `journal undo <ts>` targets an exact entry; this is the "undo my last
    change" shortcut, which is what an operator reaches for after a mistake.
    """
    store = ctx.obj["store"]
    entries = [
        e for e in store.journal.read()
        if memory_id is None or store._entry_touches(e, _resolve(store, memory_id))
    ]
    if not entries:
        typer.echo("No journal entry to undo.", err=True)
        raise typer.Exit(1)
    entry = entries[-1]
    if not out.confirm(f"Undo {entry.op} {entry.id[:8]}?", yes=yes):
        typer.echo("Aborted.")
        return
    ok = store.undo(entry)
    typer.echo("Undone." if ok else "Could not undo this entry.")
    if not ok:
        raise typer.Exit(1)
