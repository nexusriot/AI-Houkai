"""TUI command — launch the Textual memory browser."""

from __future__ import annotations

import typer

try:
    from ai_houkai.tui.app import HoukaiTui
except ImportError:
    HoukaiTui = None  # type: ignore


def tui(ctx: typer.Context) -> None:
    """Browse memories interactively (search, detail, link-graph walking)."""
    if HoukaiTui is None:
        typer.echo(
            'The TUI needs textual — pip install "ai-houkai[tui]"', err=True
        )
        raise typer.Exit(1)

    cfg = ctx.obj["config"]
    HoukaiTui(store=ctx.obj["store"], collection=cfg.collection).run()
