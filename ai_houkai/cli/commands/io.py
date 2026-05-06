"""Export, import, and backup commands."""

from __future__ import annotations

import json
import hashlib
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer

from ai_houkai.cli import output as out


def export_cmd(
    ctx: typer.Context,
    path: Optional[str] = typer.Argument(None, help="Output file (default: stdout)"),
    type: Optional[str] = typer.Option(None, "-t", "--type"),
    tag: Optional[str] = typer.Option(None, "-g", "--tag"),
    include_superseded: bool = typer.Option(False, "--include-superseded"),
) -> None:
    """Export memories to JSONL (stdout or file)."""
    store = ctx.obj["store"]
    memories = store.list_recent(limit=999_999, include_superseded=include_superseded)

    if type:
        memories = [m for m in memories if m.type == type]
    if tag:
        memories = [m for m in memories if tag in m.tags]

    lines = [json.dumps(out._mem_to_dict(m)) for m in memories]
    output = "\n".join(lines)

    if path:
        Path(path).write_text(output + "\n")
        typer.echo(f"Exported {len(memories)} memories to {path}")
    else:
        typer.echo(output)


def import_cmd(
    ctx: typer.Context,
    path: str = typer.Argument(..., help="JSONL file to import"),
    dedupe: str = typer.Option("text", "--dedupe-by", help="text|id|none"),
    yes: bool = typer.Option(False, "--yes", "-y"),
) -> None:
    """Import memories from a JSONL file."""
    store = ctx.obj["store"]
    records = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                typer.echo(f"Warning: line {lineno} invalid JSON — skipping ({e})", err=True)

    if not records:
        typer.echo("No records to import.")
        return

    if dedupe != "none":
        existing = store.list_recent(limit=999_999, include_superseded=True)
        if dedupe == "text":
            existing_hashes = {hashlib.md5(m.text.encode()).hexdigest() for m in existing}
            records = [
                r for r in records
                if hashlib.md5(r.get("text", "").encode()).hexdigest() not in existing_hashes
            ]
        elif dedupe == "id":
            existing_ids = {m.id for m in existing}
            records = [r for r in records if r.get("id") not in existing_ids]

    if not records:
        typer.echo("All records already present — nothing to import.")
        return

    if not out.confirm(f"Import {len(records)} memories?", yes=yes):
        typer.echo("Aborted.")
        return

    imported = 0
    for r in records:
        try:
            mem = store.remember(
                text=r["text"],
                type=r.get("type", "semantic"),
                tags=r.get("tags", []),
                importance=r.get("importance", 0.5),
                source=r.get("source"),
                polarity=r.get("polarity", 0),
            )
            if "created_at" in r:
                mem.created_at = r["created_at"]
                store.collection.update(ids=[mem.id], metadatas=[mem.to_metadata()])
            imported += 1
        except Exception as e:
            typer.echo(f"Warning: failed to import record — {e}", err=True)

    typer.echo(f"Imported {imported}/{len(records)} memories.")


def backup(
    ctx: typer.Context,
) -> None:
    """Snapshot the Chroma store to ~/.ai_houkai/backups/<ISO timestamp>/."""
    cfg = ctx.obj["config"]
    store_path = Path(cfg.store_path)
    if not store_path.exists():
        typer.echo(f"Error: store path {store_path} does not exist.", err=True)
        raise typer.Exit(1)

    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    backup_dir = Path.home() / ".ai_houkai" / "backups" / ts
    backup_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(store_path, backup_dir)
    typer.echo(f"Backup written to {backup_dir}")
