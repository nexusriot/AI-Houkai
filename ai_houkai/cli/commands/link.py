"""Link, unlink, neighbors, graph commands."""

from __future__ import annotations

import json
from typing import List, Optional

import typer

from ai_houkai.cli import output as out

_VALID_RELS = ("related", "refines", "example_of", "derived_from", "supersedes", "contradicts")


def link(
    ctx: typer.Context,
    src: str = typer.Argument(..., help="Source memory id or prefix"),
    dst: str = typer.Argument(..., help="Destination memory id or prefix"),
    rel: str = typer.Option("related", "--rel", "-r", help="|".join(_VALID_RELS)),
) -> None:
    """Add a directed link src → dst."""
    store = ctx.obj["store"]
    try:
        src_id = out.resolve_id_prefix(store, src)
        dst_id = out.resolve_id_prefix(store, dst)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    store.link(src_id, dst_id, rel)
    typer.echo(f"Linked {out.short_id(src_id)} --[{rel}]--> {out.short_id(dst_id)}")


def unlink(
    ctx: typer.Context,
    src: str = typer.Argument(..., help="Source memory id or prefix"),
    dst: str = typer.Argument(..., help="Destination memory id or prefix"),
    rel: Optional[str] = typer.Option(None, "--rel", "-r", help="Relation to remove (all if omitted)"),
) -> None:
    """Remove link(s) from src to dst."""
    store = ctx.obj["store"]
    try:
        src_id = out.resolve_id_prefix(store, src)
        dst_id = out.resolve_id_prefix(store, dst)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    removed = store.unlink(src_id, dst_id, rel)
    typer.echo(f"Removed {removed} link(s).")


def neighbors(
    ctx: typer.Context,
    id: str = typer.Argument(..., help="Memory id or prefix"),
    rel: Optional[str] = typer.Option(None, "--rel", "-r"),
    direction: str = typer.Option("both", "--direction", "-d", help="out|in|both"),
    depth: int = typer.Option(1, "--depth", "-D"),
    fmt: str = typer.Option("auto", "--format", "-f", help="auto|rich|tsv|json"),
) -> None:
    """Show memories linked to/from a memory."""
    store = ctx.obj["store"]
    try:
        full_id = out.resolve_id_prefix(store, id)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    results = store.neighbors(full_id, rel=rel, direction=direction, depth=depth)
    if not results:
        typer.echo("No neighbors found.")
        return

    if fmt == "json":
        print(json.dumps([{"id": m.id, "rel": r, "text": m.text} for m, r in results], indent=2))
        return

    try:
        from rich.console import Console
        from rich.table import Table
        console = Console()
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("REL", width=14)
        table.add_column("ID", width=10)
        table.add_column("TYPE", width=10)
        table.add_column("TEXT")
        for mem, r in results:
            table.add_row(r, out.short_id(mem.id), mem.type, out._truncate(mem.text, 65))
        console.print(table)
    except ImportError:
        for mem, r in results:
            typer.echo(f"{r}\t{out.short_id(mem.id)}\t{mem.text[:60]}")


def graph(
    ctx: typer.Context,
    ids: List[str] = typer.Argument(..., help="Seed memory ids or prefixes"),
    depth: int = typer.Option(1, "--depth", "-D"),
    fmt: str = typer.Option("ascii", "--format", "-f", help="ascii|dot|json"),
) -> None:
    """Show a subgraph of memories reachable from seed ids."""
    store = ctx.obj["store"]
    full_ids = []
    for prefix in ids:
        try:
            full_ids.append(out.resolve_id_prefix(store, prefix))
        except ValueError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1)

    g = store.subgraph(full_ids, depth=depth)

    if fmt == "json":
        print(json.dumps({
            "nodes": [out._mem_to_dict(m) for m in g.nodes.values()],
            "edges": [{"src": s[:8], "dst": d[:8], "rel": r} for s, d, r in g.edges],
        }, indent=2))
        return

    if fmt == "dot":
        print("digraph houkai {")
        for mid, mem in g.nodes.items():
            label = mem.text[:30].replace('"', "'")
            print(f'  "{mid[:8]}" [label="{label}"];')
        for src, dst, rel in g.edges:
            print(f'  "{src[:8]}" -> "{dst[:8]}" [label="{rel}"];')
        print("}")
        return

    for mid, mem in g.nodes.items():
        typer.echo(f"[{out.short_id(mid)}] {mem.text[:60]}")
    for src, dst, rel in g.edges:
        typer.echo(f"  {out.short_id(src)} --[{rel}]--> {out.short_id(dst)}")
