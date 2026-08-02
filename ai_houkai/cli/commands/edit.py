"""Edit, tag, and bump commands."""

from __future__ import annotations
import subprocess
import tempfile
from typing import List

import typer

from ai_houkai.cli import output as out


def edit(
    ctx: typer.Context,
    id: str = typer.Argument(..., help="Memory id or 8-char prefix"),
) -> None:
    """Open a memory in $EDITOR. Re-embeds if text changes."""
    store = ctx.obj["store"]
    cfg = ctx.obj["config"]

    try:
        full_id = out.resolve_id_prefix(store, id)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    mem = store.get(full_id)
    if mem is None:
        typer.echo(f"Error: memory {id!r} not found", err=True)
        raise typer.Exit(1)

    yaml_block = (
        f"# Edit memory — save and close to apply. Lines starting with # are ignored.\n"
        f"id: {mem.id}\n"
        f"type: {mem.type}\n"
        f"importance: {mem.importance}\n"
        f"tags: {', '.join(mem.tags)}\n"
        f"source: {mem.source or ''}\n"
        f"polarity: {mem.polarity}\n"
        f"---\n"
        f"{mem.text}\n"
    )

    with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False, prefix="houkai_") as tf:
        tf.write(yaml_block)
        tmp_path = tf.name

    result = subprocess.run([cfg.editor, tmp_path])
    if result.returncode != 0:
        typer.echo("Editor exited with error.", err=True)
        raise typer.Exit(1)

    with open(tmp_path) as f:
        raw = f.read()

    if "---" not in raw:
        typer.echo("Error: missing '---' separator in edited file.", err=True)
        raise typer.Exit(1)

    front, _, body = raw.partition("---")
    # Strip '#' comment lines from the front matter ONLY — the body is the
    # memory text and may legitimately contain markdown headings.
    front = "\n".join(l for l in front.splitlines() if not l.startswith("#"))
    new_text = body.strip()

    new_type = mem.type
    new_importance = mem.importance
    new_tags = mem.tags
    new_source = mem.source
    new_polarity = mem.polarity

    for line in front.splitlines():
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        k, v = k.strip(), v.strip()
        if k == "type" and v:
            new_type = v
        elif k == "importance" and v:
            try:
                new_importance = max(0.0, min(1.0, float(v)))
            except ValueError:
                pass
        elif k == "tags":
            new_tags = [t.strip() for t in v.split(",") if t.strip()]
        elif k == "source":
            new_source = v or None
        elif k == "polarity" and v:
            try:
                new_polarity = int(v)
            except ValueError:
                pass

    text_changed = new_text != mem.text

    # store.edit() keeps the same id, re-embeds when the text changed,
    # preserves links / superseded_by / access tracking, and journals the
    # change so `journal tail` shows it and `journal undo` can reverse it.
    with out.friendly_errors():
        store.edit(
            full_id,
            text=new_text,
            type=new_type,
            tags=new_tags,
            importance=new_importance,
            polarity=new_polarity,
            source=new_source,
        )

    if text_changed:
        typer.echo(f"Updated (re-embedded) → {out.short_id(full_id)}")
    else:
        typer.echo(f"Updated metadata for {out.short_id(full_id)}")


def tag(
    ctx: typer.Context,
    id: str = typer.Argument(..., help="Memory id or prefix"),
    add: List[str] = typer.Option([], "--add", "-a", help="Tag to add (repeatable)"),
    remove: List[str] = typer.Option([], "--remove", "-r", help="Tag to remove (repeatable)"),
) -> None:
    """Add or remove tags. Example: houkai tag abc123 --add hardware --remove old"""
    store = ctx.obj["store"]

    try:
        full_id = out.resolve_id_prefix(store, id)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    mem = store.get(full_id)
    if mem is None:
        typer.echo("Error: not found", err=True)
        raise typer.Exit(1)

    tag_set = set(mem.tags)
    for t in add:
        tag_set.add(t)
    for t in remove:
        tag_set.discard(t)

    with out.friendly_errors():
        mem = store.edit(full_id, tags=sorted(tag_set))
    typer.echo(f"{out.short_id(full_id)} tags: {', '.join(mem.tags) or '(none)'}")


def bump(
    ctx: typer.Context,
    id: str = typer.Argument(..., help="Memory id or prefix"),
    delta: str = typer.Argument(..., help="+0.2, -0.1, or =0.9"),
) -> None:
    """Adjust importance. Examples: +0.2  -0.1  =0.9"""
    store = ctx.obj["store"]

    try:
        full_id = out.resolve_id_prefix(store, id)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    mem = store.get(full_id)
    if mem is None:
        typer.echo("Error: not found", err=True)
        raise typer.Exit(1)

    old = mem.importance
    try:
        if delta.startswith("="):
            new_val = float(delta[1:])
        elif delta.startswith(("+", "-")):
            new_val = old + float(delta)
        else:
            raise ValueError
    except ValueError:
        typer.echo(
            f"Error: delta must be +N, -N, or =N (e.g. +0.2) — got {delta!r}",
            err=True,
        )
        raise typer.Exit(1)

    with out.friendly_errors():
        mem = store.edit(full_id, importance=new_val)
    typer.echo(f"{out.short_id(full_id)} importance: {old:.2f} → {mem.importance:.2f}")
