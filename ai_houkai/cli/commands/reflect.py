"""Reflect command — wraps ReflectionEngine."""

from __future__ import annotations

import typer

from ai_houkai.cli import output as out


def reflect(
    ctx: typer.Context,
    threshold: float = typer.Option(0.75, "--threshold"),
    min_cluster_size: int = typer.Option(3, "--min-cluster-size"),
    consolidate: str = typer.Option("none", "--consolidate", help="none|soft|hard"),
    apply: bool = typer.Option(False, "--apply", help="Actually write (default is dry-run)"),
    yes: bool = typer.Option(False, "--yes", "-y"),
) -> None:
    """Cluster episodic memories and synthesise semantic summaries. Dry-run by default."""
    from ai_houkai.memory_system.reflection import ReflectionEngine

    store = ctx.obj["store"]
    engine = ReflectionEngine(
        store,
        similarity_threshold=threshold,
        min_cluster_size=min_cluster_size,
    )

    consolidate_arg: bool | str
    if consolidate == "soft":
        consolidate_arg = True
    elif consolidate == "hard":
        consolidate_arg = "hard"
    else:
        consolidate_arg = False

    plan = engine.reflect(dry_run=True, consolidate=False)
    if not plan:
        typer.echo("No clusters found to reflect on.")
        return

    n = len(plan)
    typer.echo(f"Reflection plan: {n} new semantic memor{'y' if n == 1 else 'ies'} would be created.")
    for i, mem in enumerate(plan, 1):
        typer.echo(f"  [{i}] {mem.text[:80]}")

    if not apply:
        typer.echo("\nDry-run — pass --apply to write.")
        return

    if not out.confirm(f"Create {n} semantic memories?", yes=yes):
        typer.echo("Aborted.")
        return

    created = engine.reflect(dry_run=False, consolidate=consolidate_arg)
    typer.echo(f"Created {len(created)} memory(-ies).")
    for mem in created:
        typer.echo(f"  {out.short_id(mem.id)}  {mem.text[:70]}")
