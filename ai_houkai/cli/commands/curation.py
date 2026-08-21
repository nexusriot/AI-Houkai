"""Curation commands: merge, versions, tags, path, trash."""

from __future__ import annotations

import json as jsonlib
from datetime import datetime
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

from ai_houkai.cli import output as out
from ai_houkai.memory_system.curation import MergeError

tags_app = typer.Typer(help="Curate tags across the collection.",
                       no_args_is_help=True)
trash_app = typer.Typer(help="Soft-deleted memories (recoverable).",
                        no_args_is_help=True)


def _resolve(store, prefix: str) -> str:
    try:
        return out.resolve_id_prefix(store, prefix)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


def merge(
    ctx: typer.Context,
    target: str = typer.Argument(..., help="Memory to keep"),
    other: str = typer.Argument(..., help="Memory to fold in and delete"),
    separator: str = typer.Option("\n\n", "--separator",
                                  help="Text joined between the two"),
    yes: bool = typer.Option(False, "--yes", "-y"),
) -> None:
    """Fold one memory into another and delete the absorbed one.

    Transfers the absorbed memory's outgoing links and re-points every INCOMING
    link at the target — a plain forget would strand those relationships.
    """
    store = ctx.obj["store"]
    target_id = _resolve(store, target)
    other_id = _resolve(store, other)
    if not out.confirm(
            f"Merge {out.short_id(other_id)} into {out.short_id(target_id)} "
            f"and delete it?", yes=yes):
        typer.echo("Aborted.")
        return
    try:
        mem = store.merge(target_id, other_id, separator=separator)
    except MergeError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    typer.echo(f"Merged. {out.short_id(mem.id)} now has "
               f"{len(mem.text)} chars and {len(mem.links)} links.")


def versions(
    ctx: typer.Context,
    id: str = typer.Argument(..., help="Memory id or 8-char prefix"),
    json: bool = typer.Option(False, "--json", help="Emit raw JSON"),
) -> None:
    """Show past text states of a memory, oldest first.

    Each entry is the state BEFORE an edit; the live text is excluded — use
    `houkai show` for that.
    """
    store = ctx.obj["store"]
    full_id = _resolve(store, id)
    history = store.versions(full_id)
    if json:
        typer.echo(jsonlib.dumps([
            {"ts": v.ts, "text": v.text, "tags": v.tags,
             "importance": v.importance, "source": v.source, "type": v.type}
            for v in history
        ], indent=2))
        return
    if not history:
        typer.echo("(no earlier versions — this memory has never been edited)")
        return
    table = Table(title=f"Versions — {out.short_id(full_id)} ({len(history)})")
    table.add_column("when", style="dim")
    table.add_column("importance", justify="right")
    table.add_column("text")
    for v in history:
        table.add_row(datetime.fromtimestamp(v.ts).strftime("%Y-%m-%d %H:%M:%S"),
                      f"{v.importance:.2f}", v.text[:70])
    Console().print(table)


def tags_list(
    ctx: typer.Context,
    include_superseded: bool = typer.Option(
        False, "--all", help="Count superseded memories too"),
    json: bool = typer.Option(False, "--json", help="Emit raw JSON"),
) -> None:
    """List every tag with its usage count."""
    pairs = ctx.obj["store"].list_tags(include_superseded=include_superseded)
    if json:
        typer.echo(jsonlib.dumps(
            [{"tag": t, "count": n} for t, n in pairs], indent=2))
        return
    if not pairs:
        typer.echo("(no tags)")
        return
    table = Table(title=f"Tags ({len(pairs)})")
    table.add_column("tag")
    table.add_column("count", justify="right")
    for tag, count in pairs:
        table.add_row(tag, str(count))
    Console().print(table)


def tags_rename(
    ctx: typer.Context,
    old: str = typer.Argument(..., help="Existing tag"),
    new: str = typer.Argument(..., help="Replacement tag"),
) -> None:
    """Rename a tag across the collection (de-duplicating on collision)."""
    try:
        res = ctx.obj["store"].rename_tag(old, new)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    typer.echo(f"Renamed {old!r} → {res.tag!r} on {res.changed} memories.")


def tags_merge(
    ctx: typer.Context,
    into: str = typer.Argument(..., help="Tag to keep"),
    sources: List[str] = typer.Argument(..., help="Tags to fold in"),
) -> None:
    """Fold several tags into one across the collection."""
    try:
        res = ctx.obj["store"].merge_tags(sources, into)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    typer.echo(f"Merged {', '.join(sources)} → {res.tag!r} "
               f"on {res.changed} memories.")


def tags_delete(
    ctx: typer.Context,
    tag: str = typer.Argument(..., help="Tag to strip"),
    yes: bool = typer.Option(False, "--yes", "-y"),
) -> None:
    """Strip a tag from every memory that carries it."""
    if not out.confirm(f"Remove tag {tag!r} from every memory?", yes=yes):
        typer.echo("Aborted.")
        return
    res = ctx.obj["store"].delete_tag(tag)
    typer.echo(f"Removed {tag!r} from {res.changed} memories.")


def path(
    ctx: typer.Context,
    from_id: str = typer.Argument(..., help="Start memory"),
    to_id: str = typer.Argument(..., help="End memory"),
    max_depth: int = typer.Option(6, "--max-depth", help="Hop limit"),
    json: bool = typer.Option(False, "--json", help="Emit raw JSON"),
) -> None:
    """Find the shortest link path between two memories.

    Undirected: "how are these related?" does not care which way the author
    happened to draw the arrow.
    """
    store = ctx.obj["store"]
    src = _resolve(store, from_id)
    dst = _resolve(store, to_id)
    hops = store.find_path(src, dst, max_depth=max_depth)
    if json:
        typer.echo(jsonlib.dumps({
            "found": bool(hops), "length": max(0, len(hops) - 1),
            "path": [{"id": mid, "rel": rel} for mid, rel in hops],
        }, indent=2))
        return
    if not hops:
        typer.echo(f"No path within {max_depth} hops.")
        raise typer.Exit(1)
    for i, (mid, rel) in enumerate(hops):
        mem = store.get(mid)
        arrow = "   " if i == 0 else f"─{rel}→"
        typer.echo(f"{arrow} {out.short_id(mid)}  "
                   f"{(mem.text[:60] if mem else '(missing)')}")


def trash_put(
    ctx: typer.Context,
    id: str = typer.Argument(..., help="Memory id or 8-char prefix"),
) -> None:
    """Soft-delete a memory — recoverable, unlike `forget`."""
    store = ctx.obj["store"]
    full_id = _resolve(store, id)
    if not store.trash(full_id):
        typer.echo(f"Error: memory {id!r} not found", err=True)
        raise typer.Exit(1)
    typer.echo(f"Trashed {out.short_id(full_id)}. "
               f"Restore with: houkai trash restore {full_id[:8]}")


def trash_list_cmd(
    ctx: typer.Context,
    json: bool = typer.Option(False, "--json", help="Emit raw JSON"),
) -> None:
    """List soft-deleted memories, oldest first."""
    entries = ctx.obj["store"].trash_list()
    if json:
        typer.echo(jsonlib.dumps([e.to_dict() for e in entries], indent=2))
        return
    if not entries:
        typer.echo("(trash is empty)")
        return
    table = Table(title=f"Trash ({len(entries)})")
    table.add_column("id", style="dim")
    table.add_column("deleted", style="dim")
    table.add_column("actor", style="cyan")
    table.add_column("text")
    for e in entries:
        table.add_row(
            out.short_id(e.memory_id),
            datetime.fromtimestamp(e.deleted_at).strftime("%Y-%m-%d %H:%M:%S"),
            e.actor, (e.memory.get("text") or "")[:60])
    Console().print(table)


def trash_restore_cmd(
    ctx: typer.Context,
    id: str = typer.Argument(..., help="Memory id (from `houkai trash list`)"),
) -> None:
    """Bring a trashed memory back with its id, tags and links intact."""
    store = ctx.obj["store"]
    resolved = _resolve_trash_id(store, id)
    mem = store.trash_restore(resolved)
    if mem is None:
        typer.echo(
            f"Error: {out.short_id(resolved)} could not be restored — a "
            "memory with this id is live again; forget or trash it first",
            err=True)
        raise typer.Exit(1)
    typer.echo(f"Restored {out.short_id(mem.id)}.")


def _resolve_trash_id(store, prefix: str) -> str:
    """Resolve an id prefix against the trash, not the store — a trashed
    memory is no longer in the store. Several entries may share one id
    (trash → restore → trash), so dedupe before the ambiguity check."""
    matches = sorted({e.memory_id for e in store.trash_list()
                      if e.memory_id.startswith(prefix)})
    if not matches:
        typer.echo(f"Error: {prefix!r} is not in the trash", err=True)
        raise typer.Exit(1)
    if len(matches) > 1:
        typer.echo(f"Error: {prefix!r} is ambiguous ({len(matches)} matches)",
                   err=True)
        raise typer.Exit(1)
    return matches[0]


def trash_purge_cmd(
    ctx: typer.Context,
    id: Optional[str] = typer.Argument(
        None, help="Memory id; omit to empty the whole trash"),
    older_than: Optional[float] = typer.Option(
        None, "--older-than",
        help="Purge only entries trashed more than this many days ago"),
    yes: bool = typer.Option(False, "--yes", "-y"),
) -> None:
    """Permanently drop trashed memories. Irreversible."""
    store = ctx.obj["store"]
    if older_than is not None:
        if id is not None:
            typer.echo("Error: pass either an id or --older-than, not both",
                       err=True)
            raise typer.Exit(1)
        what = f"everything trashed over {older_than} day(s) ago"
    else:
        # Resolve before confirming: `trash list` shows 8-char ids, and an
        # unresolved prefix would confirm destructively, then purge nothing.
        if id is not None:
            id = _resolve_trash_id(store, id)
        what = f"memory {id}" if id else "the ENTIRE trash"
    if not out.confirm(f"Permanently delete {what}? This cannot be undone.",
                       yes=yes):
        typer.echo("Aborted.")
        return
    if older_than is not None:
        purged = store.trash_purge_expired(older_than)
    else:
        purged = store.trash_purge(id)
    typer.echo(f"Purged {purged} entries.")


tags_app.command("list")(tags_list)
tags_app.command("rename")(tags_rename)
tags_app.command("merge")(tags_merge)
tags_app.command("delete")(tags_delete)

trash_app.command("put")(trash_put)
trash_app.command("list")(trash_list_cmd)
trash_app.command("restore")(trash_restore_cmd)
trash_app.command("purge")(trash_purge_cmd)
