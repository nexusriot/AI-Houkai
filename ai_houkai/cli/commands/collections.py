"""Collections sub-commands — manage namespaces inside one Chroma store.

Commands
    houkai collections list           List collections with memory counts.
    houkai collections create NAME    Create an empty collection.
    houkai collections delete NAME    Delete a collection and its memories.
    houkai collections copy SRC DST   Copy memories (with embeddings) SRC → DST.
"""

from __future__ import annotations

import json
import sys

import typer
from rich.console import Console
from rich.table import Table

from ai_houkai.cli import output as out
from ai_houkai.memory_system.store import _get_embed_fn

collections_app = typer.Typer(
    name="collections",
    help="Manage collections (namespaces) in the store.",
    no_args_is_help=True,
)

_BATCH = 256


def _client(ctx: typer.Context):
    return ctx.obj["store"].client


def _open(ctx: typer.Context, name: str):
    """Open a collection the same way MemoryStore does (cosine + same EF)."""
    store = ctx.obj["store"]
    return store.client.get_or_create_collection(
        name=name,
        embedding_function=_get_embed_fn(store.embedding_model),
        metadata={"hnsw:space": "cosine"},
    )


@collections_app.command("list")
def list_cmd(
    ctx: typer.Context,
    fmt: str = typer.Option("auto", "--format", "-f", help="auto|json"),
) -> None:
    """List all collections in the store with their memory counts."""
    client = _client(ctx)
    active = ctx.obj["config"].collection
    rows = [
        {
            "name": col.name,
            "count": client.get_collection(col.name).count(),
            "active": col.name == active,
        }
        for col in sorted(client.list_collections(), key=lambda c: c.name)
    ]

    if fmt == "json":
        print(json.dumps(rows, indent=2))
        return
    if fmt == "auto" and not sys.stdout.isatty():
        for r in rows:
            print(f"{r['name']}\t{r['count']}\t{'*' if r['active'] else ''}")
        return
    t = Table(show_header=True, header_style="bold cyan")
    t.add_column("COLLECTION")
    t.add_column("MEMORIES", justify="right")
    t.add_column("", justify="center")
    for r in rows:
        t.add_row(r["name"], str(r["count"]), "*" if r["active"] else "")
    Console().print(t)


@collections_app.command("create")
def create_cmd(ctx: typer.Context, name: str) -> None:
    """Create an empty collection (cosine space, store's embedding model)."""
    client = _client(ctx)
    if any(c.name == name for c in client.list_collections()):
        typer.echo(f"Collection {name!r} already exists.")
        raise typer.Exit(1)
    _open(ctx, name)
    typer.echo(f"Created collection {name!r}.")


@collections_app.command("delete")
def delete_cmd(
    ctx: typer.Context,
    name: str,
    yes: bool = typer.Option(False, "--yes", "-y"),
) -> None:
    """Delete a collection and every memory in it. Irreversible."""
    client = _client(ctx)
    if not any(c.name == name for c in client.list_collections()):
        typer.echo(f"Collection {name!r} not found.", err=True)
        raise typer.Exit(1)
    if name == ctx.obj["config"].collection:
        typer.echo(
            f"Refusing to delete the active collection {name!r} — "
            "switch with --collection first.", err=True,
        )
        raise typer.Exit(1)
    count = client.get_collection(name).count()
    if not out.confirm(f"Delete collection {name!r} ({count} memories)?", yes=yes):
        typer.echo("Aborted.")
        return
    client.delete_collection(name)
    typer.echo(f"Deleted collection {name!r} ({count} memories).")


@collections_app.command("copy")
def copy_cmd(
    ctx: typer.Context,
    src: str,
    dst: str,
    yes: bool = typer.Option(False, "--yes", "-y"),
) -> None:
    """Copy all memories (embeddings included — no re-embedding) SRC → DST.

    DST is created if missing; existing DST ids are overwritten.
    """
    client = _client(ctx)
    if not any(c.name == src for c in client.list_collections()):
        typer.echo(f"Collection {src!r} not found.", err=True)
        raise typer.Exit(1)
    if src == dst:
        typer.echo("SRC and DST are the same collection.", err=True)
        raise typer.Exit(1)

    src_col = client.get_collection(src)
    total = src_col.count()
    if total == 0:
        typer.echo(f"Collection {src!r} is empty — nothing to copy.")
        return
    if not out.confirm(f"Copy {total} memories {src!r} → {dst!r}?", yes=yes):
        typer.echo("Aborted.")
        return

    dst_col = _open(ctx, dst)
    copied = 0
    for offset in range(0, total, _BATCH):
        res = src_col.get(
            include=["documents", "metadatas", "embeddings"],
            limit=_BATCH,
            offset=offset,
        )
        ids = res.get("ids") or []
        if not ids:
            break
        embs = res.get("embeddings")
        dst_col.upsert(
            ids=ids,
            documents=res.get("documents"),
            metadatas=res.get("metadatas"),
            embeddings=None if embs is None else [list(e) for e in embs],
        )
        copied += len(ids)
    typer.echo(f"Copied {copied} memories {src!r} → {dst!r}.")
