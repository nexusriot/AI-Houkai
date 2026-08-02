from __future__ import annotations

import typer

from ai_houkai.cli import output as out


def show(
    ctx: typer.Context,
    id: str = typer.Argument(..., help="Memory id or 8-char prefix"),
) -> None:
    """Show full details of a memory including links and supersede chain."""
    store = ctx.obj["store"]
    try:
        full_id = out.resolve_id_prefix(store, id)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    mem = store.get(full_id)
    if mem is None:
        typer.echo(f"Error: memory {id!r} not found", err=True)
        raise typer.Exit(1)

    out.print_memory_detail(mem)

    if mem.superseded_by:
        typer.echo("\nSuperseded chain:")
        seen = {mem.id}
        current_id = mem.superseded_by
        while current_id and current_id not in seen:
            seen.add(current_id)
            parent = store.get(current_id)
            if parent is None:
                break
            typer.echo(f"  → {out.short_id(parent.id)}  {parent.text[:60]}")
            current_id = parent.superseded_by
