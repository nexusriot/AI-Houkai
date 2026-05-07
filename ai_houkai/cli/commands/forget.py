from __future__ import annotations

from typing import List

import typer

from ai_houkai.cli import output as out


def forget(
    ctx: typer.Context,
    ids: List[str] = typer.Argument(..., help="Memory id(s) or 8-char prefixes"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
) -> None:
    """Delete one or more memories."""
    store = ctx.obj["store"]

    full_ids: list[str] = []
    for prefix in ids:
        try:
            full_ids.append(out.resolve_id_prefix(store, prefix))
        except ValueError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1)

    noun = "memory" if len(full_ids) == 1 else f"{len(full_ids)} memories"
    if not out.confirm(f"Delete {noun}?", yes=yes):
        typer.echo("Aborted.")
        return

    deleted = 0
    for fid in full_ids:
        if store.forget(fid):
            deleted += 1
            typer.echo(f"Deleted {out.short_id(fid)}")
        else:
            typer.echo(f"Warning: {out.short_id(fid)} not found", err=True)

    typer.echo(f"Deleted {deleted}/{len(full_ids)}.")
