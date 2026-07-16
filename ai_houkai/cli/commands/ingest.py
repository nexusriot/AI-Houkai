"""Ingest command — chunk files (or stdin) into memories."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import typer

from ai_houkai.cli import output as out
from ai_houkai.memory_system.importance import score_importance
from ai_houkai.memory_system.ingest import chunk_text
from ai_houkai.memory_system.store import RememberItem


def ingest(
    ctx: typer.Context,
    files: Optional[List[Path]] = typer.Argument(
        None,
        help="Text/markdown files to ingest. Omit (or pass '-') to read stdin.",
    ),
    type: str = typer.Option("episodic", "-t", "--type", help="episodic|semantic|procedural|feedback"),
    tags: List[str] = typer.Option([], "-g", "--tag", help="Tag (repeatable)"),
    source: Optional[str] = typer.Option(
        None, "-s", "--source",
        help="Source label; default ingest:<filename> (or ingest:stdin)",
    ),
    importance: Optional[float] = typer.Option(None, "-i", "--importance", min=0.0, max=1.0),
    auto_importance: bool = typer.Option(
        False, "--auto-importance", help="Score each chunk heuristically"
    ),
    max_chars: int = typer.Option(500, "--max-chars", help="Max chunk size"),
    min_chars: int = typer.Option(30, "--min-chars", help="Drop chunks shorter than this"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show chunks without writing"),
    yes: bool = typer.Option(False, "--yes", "-y"),
) -> None:
    """Split files into chunks and store each chunk as a memory.

    Splits on blank lines, keeps markdown headings attached to their
    paragraph, re-packs long paragraphs on sentence boundaries.
    """
    store = ctx.obj["store"]
    cfg = ctx.obj["config"]

    # (label, text) per input
    inputs: list[tuple[str, str]] = []
    from_stdin = not files or [str(f) for f in files] == ["-"]
    if from_stdin:
        body = sys.stdin.read()
        if not body.strip():
            typer.echo("Error: no input provided", err=True)
            raise typer.Exit(1)
        inputs.append(("stdin", body))
    else:
        for f in files:
            if not f.exists():
                typer.echo(f"Error: {f} not found", err=True)
                raise typer.Exit(1)
            inputs.append((f.name, f.read_text(encoding="utf-8", errors="replace")))

    auto = auto_importance or (
        importance is None and cfg.default_importance == "auto"
    )

    plan: list[tuple[str, str, float]] = []   # (label, chunk, importance)
    for label, body in inputs:
        for chunk in chunk_text(body, max_chars=max_chars, min_chars=min_chars):
            if importance is not None:
                imp = importance
            elif auto:
                imp = score_importance(chunk, type, list(tags))
            else:
                imp = (
                    cfg.default_importance
                    if isinstance(cfg.default_importance, float)
                    else 0.5
                )
            plan.append((label, chunk, imp))

    if not plan:
        typer.echo("Nothing to ingest (all chunks below --min-chars?).")
        return

    for i, (label, chunk, imp) in enumerate(plan, 1):
        first_line = chunk.splitlines()[0][:70]
        typer.echo(f"  [{i:3}] {imp:.2f}  {label}: {first_line}")
    typer.echo(f"\n{len(plan)} chunk(s) from {len(inputs)} input(s).")

    if dry_run:
        typer.echo("Dry-run — nothing written.")
        return

    if not out.confirm(f"Store {len(plan)} memories?", yes=yes, use_tty=from_stdin):
        typer.echo("Aborted.")
        return

    # friendly_errors: a bad --type/--tag surfaces as a one-line error, not a
    # traceback. Journal as "ingest" — "import" is the .ahkai importer's actor.
    # One batched write — collapses N per-chunk encodes into ceil(N/batch)
    # (see MemoryStore.remember_many). on_conflict="ignore": ingesting raw
    # document chunks shouldn't trigger conflict management (near-duplicate
    # chunks — shared headings/boilerplate — are normal source material).
    with store.as_actor("ingest"), out.friendly_errors():
        store.remember_many(
            [
                RememberItem(
                    text=chunk,
                    type=type,
                    tags=tuple(tags),
                    importance=imp,
                    source=source or f"ingest:{label}",
                )
                for label, chunk, imp in plan
            ],
            on_conflict="ignore",
        )
    typer.echo(f"Stored {len(plan)} memories.")
