"""Maintenance sub-commands — scheduled decay + reflection daemon.

Commands
    houkai maintenance tick     Run one tick synchronously (cron-friendly).
    houkai maintenance run      Foreground blocking loop (Ctrl-C to exit).
    houkai maintenance start    Detach daemon into the background.
    houkai maintenance stop     Send SIGTERM to the background daemon.
    houkai maintenance status   Show daemon state, last runs, and next schedules.
"""

from __future__ import annotations

import logging
import signal
import threading
import time
from datetime import datetime

import typer

from ai_houkai.cli.config import load_maintenance
from ai_houkai.maintenance.daemon import (
    get_pid,
    is_alive,
    remove_pid,
    spawn_detached,
    stop_daemon,
)
from ai_houkai.maintenance.durations import format_duration
from ai_houkai.maintenance.scheduler import MaintenanceScheduler
from ai_houkai.maintenance.state import MaintenanceState
from ai_houkai.memory_system.summarizers import build_summarizer

maintenance_app = typer.Typer(
    name="maintenance",
    help="Scheduled decay + reflection daemon.",
    no_args_is_help=True,
)


def _make_scheduler(store, mcfg) -> MaintenanceScheduler:
    try:
        summarizer = build_summarizer(mcfg.summarizer)
    except (ValueError, ImportError) as e:
        typer.echo(f"Error in [maintenance.reflect].summarizer: {e}", err=True)
        raise typer.Exit(1)
    return MaintenanceScheduler(
        store=store,
        decay_every=mcfg.decay_every,
        reflect_every=mcfg.reflect_every,
        purge_every=mcfg.purge_every,
        trash_ttl_days=mcfg.trash_ttl_days,
        tick_interval=mcfg.tick_interval,
        state_path=mcfg.state_path,
        decay_rate=mcfg.decay_rate,
        min_score=mcfg.min_score,
        protect_types=mcfg.protect_types,
        frequency_weight=mcfg.frequency_weight,
        min_cluster_size=mcfg.min_cluster_size,
        reflect_apply=mcfg.reflect_apply,
        reflect_consolidate=mcfg.reflect_consolidate,
        summarizer=summarizer,
    )


_DISABLED_MSG = (
    "Maintenance is disabled ([maintenance].enabled = false in "
    "~/.config/ai_houkai/config.toml) — nothing to do."
)


def _fmt_ts(ts: float | None) -> str:
    if ts is None:
        return "never"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _fmt_next(last_at: float | None, interval: int | None) -> str:
    if interval is None:
        return "disabled"
    if last_at is None:
        return "now (never ran)"
    next_ts = last_at + interval
    diff = next_ts - time.time()
    direction = "in" if diff > 0 else "overdue by"
    return f"{_fmt_ts(next_ts)}  ({direction} {format_duration(abs(diff))})"


@maintenance_app.command("tick")
def tick_cmd(ctx: typer.Context) -> None:
    """Run one maintenance tick synchronously and print results.

    Suitable for use from cron:  houkai maintenance tick
    """
    store = ctx.obj["store"]
    mcfg = load_maintenance()
    if not mcfg.enabled:
        typer.echo(_DISABLED_MSG)
        return
    sched = _make_scheduler(store, mcfg)

    typer.echo("Running maintenance tick…")
    result = sched.tick()
    typer.echo(f"Result: {result.summary()}")

    if result.decay_error:
        typer.echo(f"  Decay error: {result.decay_error}", err=True)
    if result.reflect_error:
        typer.echo(f"  Reflect error: {result.reflect_error}", err=True)

    if result.decay_error or result.reflect_error:
        raise typer.Exit(1)


@maintenance_app.command("run")
def run_cmd(ctx: typer.Context) -> None:
    """Run the maintenance scheduler in the foreground (Ctrl-C to stop).

    Used internally by 'start'; also useful for testing before daemonizing.
    """
    store = ctx.obj["store"]
    mcfg = load_maintenance()
    if not mcfg.enabled:
        typer.echo(_DISABLED_MSG)
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stop = threading.Event()

    def _handle_term(signum, frame):
        logging.getLogger(__name__).info("Received signal %d — stopping.", signum)
        stop.set()

    signal.signal(signal.SIGTERM, _handle_term)
    signal.signal(signal.SIGINT, _handle_term)

    sched = _make_scheduler(store, mcfg)
    sched.run_forever(stop)


@maintenance_app.command("start")
def start_cmd(ctx: typer.Context) -> None:
    """Detach the maintenance daemon into the background."""
    cfg = ctx.obj["config"]
    mcfg = load_maintenance()
    if not mcfg.enabled:
        typer.echo(_DISABLED_MSG, err=True)
        raise typer.Exit(1)

    if is_alive(mcfg.pid_path):
        pid = get_pid(mcfg.pid_path)
        typer.echo(f"Daemon is already running (pid {pid}).")
        raise typer.Exit(1)

    pid = spawn_detached(
        store_path=cfg.store_path,
        collection=cfg.collection,
        log_path=mcfg.log_path,
        pid_path=mcfg.pid_path,
    )
    typer.echo(f"Maintenance daemon started (pid {pid}).")
    typer.echo(f"Logs → {mcfg.log_path}")
    typer.echo(f"Stop → houkai maintenance stop")


@maintenance_app.command("stop")
def stop_cmd(ctx: typer.Context) -> None:
    """Send SIGTERM to the background daemon."""
    mcfg = load_maintenance()

    if not is_alive(mcfg.pid_path):
        pid = get_pid(mcfg.pid_path)
        if pid:
            typer.echo(f"Daemon (pid {pid}) is not running. Cleaning up stale pid file.")
            remove_pid(mcfg.pid_path)
        else:
            typer.echo("No daemon is running.")
        return

    pid = get_pid(mcfg.pid_path)
    sent = stop_daemon(mcfg.pid_path)
    if sent:
        typer.echo(f"SIGTERM sent to daemon (pid {pid}).")
        remove_pid(mcfg.pid_path)
    else:
        typer.echo("Failed to send signal — daemon may have already exited.")


@maintenance_app.command("status")
def status_cmd(ctx: typer.Context) -> None:
    """Show daemon state, last run times, and next scheduled runs."""
    mcfg = load_maintenance()
    state = MaintenanceState.load(mcfg.state_path)

    pid = get_pid(mcfg.pid_path)
    alive = is_alive(mcfg.pid_path)
    daemon_line = (
        f"running (pid {pid})" if alive
        else ("stopped (stale pid file)" if pid else "stopped")
    )

    width = 60
    typer.echo("─" * width)
    if not mcfg.enabled:
        typer.echo("  Enabled:        false ([maintenance].enabled — "
                   "tick/run/start are no-ops)")
    typer.echo(f"  Daemon:         {daemon_line}")
    typer.echo("─" * width)
    typer.echo(f"  Last decay:     {_fmt_ts(state.last_decay_at)}")
    typer.echo(f"  Last reflect:   {_fmt_ts(state.last_reflect_at)}")
    typer.echo(f"  Total decayed:  {state.total_decayed}")
    typer.echo(f"  Total reflected:{state.total_reflected}")
    typer.echo("─" * width)
    typer.echo(f"  Next decay:     {_fmt_next(state.last_decay_at, mcfg.decay_every)}")
    typer.echo(f"  Next reflect:   {_fmt_next(state.last_reflect_at, mcfg.reflect_every)}")
    typer.echo("─" * width)
    typer.echo(f"  State file:     {mcfg.state_path}")
    typer.echo(f"  Log file:       {mcfg.log_path}")
    typer.echo(f"  reflect_apply:  {mcfg.reflect_apply}")
    consolidate = {False: "none", True: "soft"}.get(
        mcfg.reflect_consolidate, mcfg.reflect_consolidate)
    typer.echo(f"  consolidate:    {consolidate}")
    typer.echo(f"  summarizer:     {mcfg.summarizer or 'extractive (built-in)'}")
    typer.echo(
        f"  reinforcement:  "
        + (f"on (frequency_weight={mcfg.frequency_weight})"
           if mcfg.frequency_weight else "off")
    )
