"""Example 07 — Scheduled Maintenance Daemon.

Demonstrates all three usage modes for the maintenance scheduler:

  Mode A — one-shot tick (cron-friendly)
  Mode B — foreground loop (observe before daemonizing)
  Mode C — background daemon lifecycle (start / status / stop)

Run:
    python examples/07_maintenance_daemon.py
    python examples/07_maintenance_daemon.py --mode foreground  # Ctrl-C to stop
    python examples/07_maintenance_daemon.py --mode daemon      # start/stop demo
"""

from __future__ import annotations

import argparse
import threading
import datetime
import time
import tempfile

from ai_houkai.maintenance.scheduler import MaintenanceScheduler
from ai_houkai.maintenance.state import MaintenanceState
from ai_houkai.maintenance.durations import format_duration
from ai_houkai.memory_system import MemoryStore

from ai_houkai.maintenance.daemon import (
    is_alive, get_pid, spawn_detached, stop_daemon, remove_pid,
)


def _seed_store(store: MemoryStore) -> None:
    """Populate the store with a mix of fresh and stale memories."""
    fresh = [
        ("Python's GIL blocks CPU-bound parallelism.", 0.85, "semantic"),
        ("Use asyncio for IO-bound concurrency.", 0.80, "semantic"),
        ("The project uses Poetry for dependency management.", 0.70, "procedural"),
        ("User prefers concise commit messages.", 0.75, "feedback"),
    ]
    stale_texts = [
        "Temporary debug note from last week.",
        "Old experiment — no longer relevant.",
        "Scratch notes from the failed refactor.",
    ]

    for text, importance, mtype in fresh:
        store.remember(text, type=mtype, importance=importance)   # type: ignore

    now = time.time()
    for text in stale_texts:
        mem = store.remember(text, type="episodic", importance=0.1)
        # Backdate by 60 days so decay will prune them
        mem.last_accessed = now - 60 * 86_400
        mem.created_at    = now - 60 * 86_400
        store.collection.update(ids=[mem.id], metadatas=[mem.to_metadata()])

    print(f"  Seeded {store.count()} memories ({len(stale_texts)} stale).")


def _print_state(state_path: str) -> None:
    state = MaintenanceState.load(state_path)

    def _ts(t):
        if t is None:
            return "never"
        return datetime.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S")

    print(f"  last_decay_at:   {_ts(state.last_decay_at)}")
    print(f"  last_reflect_at: {_ts(state.last_reflect_at)}")
    print(f"  total_decayed:   {state.total_decayed}")
    print(f"  total_reflected: {state.total_reflected}")


def demo_tick(tmp_path: str) -> None:
    print("\n=== Mode A: one-shot tick ===")
    print("Ideal for cron:  houkai maintenance tick")

    store = MemoryStore(path=f"{tmp_path}/chroma", collection="demo_tick")
    _seed_store(store)

    state_path = f"{tmp_path}/state.json"
    sched = MaintenanceScheduler(
        store=store,
        decay_every=1,          # 1 second → always overdue in a demo
        reflect_every=1,
        state_path=state_path,
        min_score=0.05,
        min_cluster_size=2,
        reflect_apply=False,    # dry-run: observe without writing
    )

    print(f"\n  Store size before tick: {store.count()} memories")
    result = sched.tick()
    print(f"  Tick result: {result.summary()}")
    print(f"  Store size after tick:  {store.count()} memories")
    print("\n  State after tick:")
    _print_state(state_path)
    store.client.close()


def demo_foreground(tmp_path: str) -> None:
    print("\n=== Mode B: foreground loop (runs for 5 seconds) ===")
    print("CLI equivalent:  houkai maintenance run")

    store = MemoryStore(path=f"{tmp_path}/chroma", collection="demo_run")
    _seed_store(store)

    state_path = f"{tmp_path}/state.json"
    sched = MaintenanceScheduler(
        store=store,
        decay_every=2,          # every 2 seconds for demo speed
        reflect_every=4,
        tick_interval=2,
        state_path=state_path,
        min_score=0.05,
        min_cluster_size=2,
        reflect_apply=False,
    )

    stop = threading.Event()

    def _loop():
        sched.run_forever(stop)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()

    time.sleep(5)
    stop.set()
    t.join(timeout=3)

    print("\n  State after 5-second run:")
    _print_state(state_path)
    store.client.close()


def demo_daemon(tmp_path: str) -> None:
    print("\n=== Mode C: background daemon lifecycle ===")
    print("CLI equivalents:")
    print("  houkai maintenance start")
    print("  houkai maintenance status")
    print("  houkai maintenance stop")
    print()

    pid_path   = f"{tmp_path}/daemon.pid"
    log_path   = f"{tmp_path}/maintenance.log"
    state_path = f"{tmp_path}/state.json"

    print("  → Starting daemon…")
    pid = spawn_detached(
        store_path=f"{tmp_path}/chroma",
        collection="demo_daemon",
        log_path=log_path,
        pid_path=pid_path,
    )
    print(f"  Daemon pid: {pid}")
    time.sleep(1)   # let it boot

    print(f"  is_alive: {is_alive(pid_path)}")

    print("\n  → Stopping daemon…")
    stop_daemon(pid_path)
    remove_pid(pid_path)
    time.sleep(0.5)
    print(f"  is_alive after stop: {is_alive(pid_path)}")

    print(f"\n  Log written to: {log_path}")
    try:
        lines = open(log_path).readlines()[-5:]
        for line in lines:
            print("  " + line.rstrip())
    except FileNotFoundError:
        print("  (log file not yet created)")


TOML_EXAMPLE = """
# ~/.config/ai_houkai/config.toml

[maintenance]
enabled       = true
decay_every   = "24h"     # or "off" to disable
reflect_every = "7d"      # or "off" to disable
tick_interval = "5m"      # how often the loop checks schedules

[maintenance.decay]
decay_rate    = 0.1
min_score     = 0.05
protect_types = ["procedural"]

[maintenance.reflect]
min_cluster_size = 3
apply            = false   # set true to actually write reflection summaries
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["tick", "foreground", "daemon", "all"],
        default="all",
        help="Which demo to run (default: all)",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="houkai_demo_") as tmp:
        if args.mode in ("tick", "all"):
            demo_tick(tmp)
        if args.mode in ("foreground", "all"):
            demo_foreground(tmp)
        if args.mode in ("daemon", "all"):
            demo_daemon(tmp)

    print()
    print("─" * 60)
    print("Config TOML example:")
    print(TOML_EXAMPLE)
    print("CLI quick-start:")
    print("  houkai maintenance tick        # one-shot (great for cron)")
    print("  houkai maintenance run         # foreground loop")
    print("  houkai maintenance start       # background daemon")
    print("  houkai maintenance status      # show state")
    print("  houkai maintenance stop        # send SIGTERM")


if __name__ == "__main__":
    main()
