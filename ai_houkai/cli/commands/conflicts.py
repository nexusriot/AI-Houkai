"""Conflict scanning, supersede, and restore commands."""

from __future__ import annotations

from typing import Optional

import typer

from ai_houkai.cli import output as out


def conflicts(
    ctx: typer.Context,
    id: Optional[str] = typer.Option(None, "--id", help="Check one memory (default: full scan)"),
    threshold: Optional[float] = typer.Option(None, "--threshold", "-T", min=0.0, max=1.0),
    resolve: Optional[str] = typer.Option(None, "--resolve", help="interactive"),
) -> None:
    """Find duplicate or contradictory memories."""
    store = ctx.obj["store"]

    full_id = None
    if id:
        try:
            full_id = out.resolve_id_prefix(store, id)
        except ValueError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1)

    typer.echo("Scanning for conflicts…", err=True)
    found = store.find_conflicts(memory_id=full_id, threshold=threshold)

    if not found:
        typer.echo("No conflicts found.")
        return

    typer.echo(f"Found {len(found)} conflict(s):\n")
    # Memories deleted or superseded in an earlier pair must not be touched
    # again when they reappear in a later pair — supersede()/forget() on a
    # missing id raises and would abort the whole scan.
    resolved: set[str] = set()
    for i, c in enumerate(found, 1):
        typer.echo(
            f"[{i}] {c.kind.upper()}  sim={c.similarity:.3f}  reason={c.reason}\n"
            f"  A [{out.short_id(c.a.id)}] {c.a.text[:70]}\n"
            f"  B [{out.short_id(c.b.id)}] {c.b.text[:70]}\n"
        )
        if resolve == "interactive":
            if c.a.id in resolved or c.b.id in resolved:
                typer.echo("  Skipped (a memory in this pair was already resolved).")
                continue
            action = typer.prompt(
                "  Action: (k)eep both / (s)upersede A by B / (S)upersede B by A / "
                "(d)elete A / (D)elete B / (skip)",
                default="skip",
            )
            if action == "s":
                store.supersede(c.a.id, c.b.id)
                resolved.add(c.a.id)
                typer.echo("  A superseded by B.")
            elif action == "S":
                store.supersede(c.b.id, c.a.id)
                resolved.add(c.b.id)
                typer.echo("  B superseded by A.")
            elif action == "d":
                store.forget(c.a.id)
                resolved.add(c.a.id)
                typer.echo("  Deleted A.")
            elif action == "D":
                store.forget(c.b.id)
                resolved.add(c.b.id)
                typer.echo("  Deleted B.")
            else:
                typer.echo("  Skipped.")


def supersede(
    ctx: typer.Context,
    old_id: str = typer.Argument(..., help="Memory to mark as superseded (id or prefix)"),
    new_id: str = typer.Argument(..., help="Superseding memory (id or prefix)"),
) -> None:
    """Mark old memory as superseded by new memory."""
    store = ctx.obj["store"]
    try:
        old = out.resolve_id_prefix(store, old_id)
        new = out.resolve_id_prefix(store, new_id)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    with out.friendly_errors():
        store.supersede(old, new)
    typer.echo(f"{out.short_id(old)} superseded by {out.short_id(new)}")


def restore(
    ctx: typer.Context,
    id: str = typer.Argument(..., help="Memory id or prefix to restore"),
) -> None:
    """Undo a supersede (clear the superseded_by marker)."""
    store = ctx.obj["store"]
    try:
        full_id = out.resolve_id_prefix(store, id)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    if store.restore(full_id):
        typer.echo(f"{out.short_id(full_id)} restored.")
    else:
        typer.echo(f"{out.short_id(full_id)} was not superseded.", err=True)
