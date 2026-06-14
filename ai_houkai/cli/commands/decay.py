"""Prune command — wraps DecayEngine."""

from __future__ import annotations

from typing import List

import typer

from ai_houkai.memory_system.decay import DecayEngine
from ai_houkai.cli import output as out



def prune(
    ctx: typer.Context,
    decay_rate: float = typer.Option(0.1, "--decay-rate"),
    min_score: float = typer.Option(0.05, "--min-score"),
    protect_type: List[str] = typer.Option(["procedural"], "--protect-type"),
    frequency_weight: float = typer.Option(
        0.0, "--frequency-weight",
        help="Reinforcement: how strongly recall count resists decay (0 = off)",
    ),
    apply: bool = typer.Option(False, "--apply", help="Actually delete (default is dry-run)"),
    yes: bool = typer.Option(False, "--yes", "-y"),
) -> None:
    """Prune low-score memories via exponential decay. Dry-run by default.

    With --frequency-weight > 0, frequently-recalled memories age out more
    slowly than untouched ones of equal importance and age.
    """
    store = ctx.obj["store"]
    engine = DecayEngine(
        store,
        decay_rate=decay_rate,
        min_score=min_score,
        protect_types=tuple(protect_type),
        frequency_weight=frequency_weight,
    )

    candidates = engine.prune(dry_run=True)
    if not candidates:
        typer.echo("Nothing to prune.")
        return

    scored = [(m, engine.score(m)) for m in candidates]
    typer.echo(f"Prune candidates ({len(candidates)}):")
    out.print_memories_table(scored, show_score=True, fmt="auto")

    if not apply:
        typer.echo("\nDry-run — pass --apply to delete.")
        return

    if not out.confirm(f"Delete {len(candidates)} memories?", yes=yes):
        typer.echo("Aborted.")
        return

    removed = engine.prune(dry_run=False)
    typer.echo(f"Pruned {len(removed)} memories.")
