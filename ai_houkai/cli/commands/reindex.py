"""Rebuild the derived SQLite sidecar index."""

from __future__ import annotations

import json as jsonlib

import typer

from ai_houkai.memory_system import MemoryStore


def reindex(
    ctx: typer.Context,
    json: bool = typer.Option(False, "--json", help="Emit raw JSON"),
) -> None:
    """Rebuild the sidecar index from the Chroma collection.

    Needed when enabling `index = "sqlite"` on an existing store (nothing has
    been indexed yet), and the only way back after the index has been disabled
    — which happens whenever it disagrees with Chroma, so that a stale index
    makes reads slower rather than wrong.
    """
    cfg = ctx.obj["config"]
    store: MemoryStore = ctx.obj["store"]

    if store.index is None:
        # The CLI callback builds the store from config, so an unconfigured
        # index means there is nothing to rebuild — but the user clearly wants
        # one, so build it here rather than making them edit config first.
        store = MemoryStore(
            path=cfg.store_path, collection=cfg.collection, actor="cli",
            index="sqlite",
        )
        ctx.obj["store"] = store

    result = store.reindex()
    if json:
        typer.echo(jsonlib.dumps(result, indent=2))
    else:
        if not result.get("enabled"):
            typer.echo(f"Error: {result.get('error')}", err=True)
            raise typer.Exit(1)
        typer.echo(f"Indexed {result['indexed']} memories → {result['path']}")
        typer.echo(f"  full-text search: {'yes' if result['fts'] else 'no (SQLite built without FTS5)'}")
        if not result["healthy"]:
            typer.echo(f"  WARNING: index still unhealthy — {result['error']}",
                       err=True)
    if not result.get("healthy"):
        raise typer.Exit(1)
