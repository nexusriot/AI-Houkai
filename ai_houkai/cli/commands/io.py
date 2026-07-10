"""Export, import, and backup commands.

Uses the portable ``.ahkai`` format — gzipped JSONL with a header line
on line 1. See PROPOSALS.md §2 for the schema.
"""

from __future__ import annotations

import gzip
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer

from ai_houkai.cli import output as out
from ai_houkai.memory_system.store import ImportConflictError


def export_cmd(
    ctx: typer.Context,
    path: str = typer.Argument(..., help="Output .ahkai file"),
    type: Optional[str] = typer.Option(None, "-t", "--type"),
    tag: Optional[str] = typer.Option(None, "-g", "--tag"),
    include_superseded: bool = typer.Option(False, "--include-superseded"),
    no_vectors: bool = typer.Option(False, "--no-vectors",
                                    help="Omit embeddings — smaller file"),
) -> None:
    """Export memories to a portable .ahkai file (gzipped JSONL)."""
    store = ctx.obj["store"]
    summary = store.export(
        path,
        include_vectors=not no_vectors,
        include_superseded=include_superseded,
        types=[type] if type else None,
        tags=[tag] if tag else None,
    )
    typer.echo(
        f"Exported {summary.count} memories → {summary.path} "
        f"({summary.bytes:,} bytes, {summary.elapsed:.2f}s)"
    )


def import_cmd(
    ctx: typer.Context,
    path: str = typer.Argument(..., help=".ahkai file to import"),
    on_conflict: str = typer.Option(
        "skip", "--on-conflict",
        help="skip | overwrite | rename | error",
    ),
    regenerate_vectors: bool = typer.Option(
        False, "--regenerate-vectors",
        help="Re-embed text on import (required if model differs)",
    ),
    dry_run: bool = typer.Option(False, "--dry-run"),
    yes: bool = typer.Option(False, "--yes", "-y"),
) -> None:
    """Import memories from an .ahkai file."""
    store = ctx.obj["store"]
    if not dry_run and not out.confirm(f"Import from {path}?", yes=yes):
        typer.echo("Aborted.")
        return
    try:
        summary = store.import_(
            path,
            on_conflict=on_conflict,        # type: ignore[arg-type]
            regenerate_vectors=regenerate_vectors,
            dry_run=dry_run,
        )
    except ImportConflictError as e:
        # Not an ImportError subclass — without this clause `--on-conflict
        # error` hitting a real collision died with a raw traceback.
        typer.echo(f"Error: {e}", err=True)
        for cid, reason in e.collisions[:10]:
            typer.echo(f"  ! {cid}: {reason}", err=True)
        raise typer.Exit(1)
    except (ImportError, FileNotFoundError, ValueError) as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    prefix = "[dry-run] " if dry_run else ""
    typer.echo(
        f"{prefix}imported={summary.imported} skipped={summary.skipped} "
        f"overwritten={summary.overwritten} renamed={summary.renamed} "
        f"errors={len(summary.errors)}"
    )
    if summary.vectors_regenerated:
        typer.echo("(embeddings were re-generated for the local model)")
    for mid, msg in summary.errors[:5]:
        typer.echo(f"  ! {mid}: {msg}", err=True)


def info_cmd(
    path: str = typer.Argument(..., help=".ahkai file to inspect"),
) -> None:
    """Print the header of an .ahkai file without touching the store."""
    fp = Path(path)
    if not fp.exists():
        typer.echo(f"Error: no such file: {fp}", err=True)
        raise typer.Exit(1)
    try:
        with gzip.open(fp, "rt", encoding="utf-8") as f:
            header_line = f.readline()
            first_data = f.readline()
            # consume remaining to get total — cheap on small/medium files
            count = 1 if first_data.strip() else 0
            for line in f:
                if line.strip():
                    count += 1
    except OSError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    try:
        header = json.loads(header_line)
    except json.JSONDecodeError:
        typer.echo("Error: not an ai-houkai export (bad header).", err=True)
        raise typer.Exit(1)
    typer.echo(json.dumps(header, indent=2))
    typer.echo(f"\nmemories on disk: {count}")


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
