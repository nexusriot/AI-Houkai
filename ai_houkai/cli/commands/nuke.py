from __future__ import annotations

import typer

from ai_houkai.cli import output as out


def nuke(
    ctx: typer.Context,
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
) -> None:
    """Delete EVERY memory in the current collection. Irreversible."""
    store = ctx.obj["store"]
    count = store.collection.count()

    if count == 0:
        typer.echo("Collection is already empty.")
        return

    if not out.confirm(
        f"Destroy all {count} memories in '{store.collection_name}'?", yes=yes
    ):
        typer.echo("Aborted.")
        return

    deleted = store.nuke()
    typer.echo(f"Nuked {deleted} memories.")
