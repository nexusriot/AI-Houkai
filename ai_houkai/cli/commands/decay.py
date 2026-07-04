"""Prune command — wraps DecayEngine."""

from __future__ import annotations

from typing import List, Optional

import typer

from ai_houkai.cli.config import load_maintenance
from ai_houkai.memory_system.decay import DecayEngine
from ai_houkai.cli import output as out



def prune(
    ctx: typer.Context,
    decay_rate: Optional[float] = typer.Option(
        None, "--decay-rate",
        help="Decay λ. Defaults to [maintenance.decay].decay_rate from config.",
    ),
    min_score: Optional[float] = typer.Option(
        None, "--min-score",
        help="Prune below this score. Defaults to [maintenance.decay].min_score.",
    ),
    protect_type: Optional[List[str]] = typer.Option(
        None, "--protect-type",
        help="Type never pruned (repeatable). "
             "Defaults to [maintenance.decay].protect_types.",
    ),
    frequency_weight: Optional[float] = typer.Option(
        None, "--frequency-weight",
        help="Reinforcement: how strongly recall count resists decay (0 = off). "
             "Defaults to [maintenance.decay].frequency_weight.",
    ),
    apply: bool = typer.Option(False, "--apply", help="Actually delete (default is dry-run)"),
    yes: bool = typer.Option(False, "--yes", "-y"),
) -> None:
    """Prune low-score memories via exponential decay. Dry-run by default.

    With --frequency-weight > 0, frequently-recalled memories age out more
    slowly than untouched ones of equal importance and age.

    Parameters not given on the command line come from the [maintenance.decay]
    config section, so a plain `houkai prune` previews exactly what the
    maintenance daemon would remove.
    """
    store = ctx.obj["store"]
    mcfg = load_maintenance()
    engine = DecayEngine(
        store,
        decay_rate=decay_rate if decay_rate is not None else mcfg.decay_rate,
        min_score=min_score if min_score is not None else mcfg.min_score,
        protect_types=(tuple(protect_type) if protect_type is not None
                       else tuple(mcfg.protect_types)),
        frequency_weight=(frequency_weight if frequency_weight is not None
                          else mcfg.frequency_weight),
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
