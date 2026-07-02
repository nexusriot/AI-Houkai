"""MaintenanceScheduler — orchestrates decay and reflection on a schedule.

Usage (programmatic)
    from ai_houkai.maintenance import MaintenanceScheduler
    import threading

    stop = threading.Event()
    sched = MaintenanceScheduler(store, decay_every=86_400, reflect_every=604_800)

    # One-shot tick (great for cron):
    result = sched.tick()
    print(result.summary())

    # Blocking loop (runs until stop is set):
    sched.run_forever(stop)
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

from ai_houkai.memory_system import DecayEngine, ReflectionEngine
from ai_houkai.maintenance.state import MaintenanceState

logger = logging.getLogger(__name__)


@dataclass
class TickResult:
    ran_decay: bool = False
    ran_reflect: bool = False
    decayed: int = 0
    reflected: int = 0
    decay_error: str | None = None
    reflect_error: str | None = None
    # False when reflection ran in dry-run mode: `reflected` is then the
    # number of summaries that WOULD be created, not that were created.
    reflect_applied: bool = True

    def summary(self) -> str:
        parts: list[str] = []
        if self.ran_decay:
            if self.decay_error:
                parts.append(f"decay FAILED: {self.decay_error}")
            else:
                parts.append(f"decay pruned {self.decayed}")
        if self.ran_reflect:
            if self.reflect_error:
                parts.append(f"reflect FAILED: {self.reflect_error}")
            elif not self.reflected:
                parts.append("reflect nothing to reflect")
            elif self.reflect_applied:
                parts.append(f"reflect created {self.reflected}")
            else:
                parts.append(f"reflect would create {self.reflected} (dry-run)")
        return " | ".join(parts) if parts else "nothing to do"


class MaintenanceScheduler:
    """Runs DecayEngine and ReflectionEngine on configurable intervals.

    Parameters
    ----------
    store
        A MemoryStore instance to operate on.
    decay_every
        Seconds between decay runs.  None disables decay.
    reflect_every
        Seconds between reflection runs.  None disables reflection.
    tick_interval
        Seconds the run_forever loop sleeps between ticks.
    state_path
        Path to the JSON state file (tracks last-run timestamps and totals).
    decay_rate, min_score, protect_types, frequency_weight
        Forwarded to DecayEngine. ``frequency_weight`` > 0 makes
        frequently-recalled memories resist decay (0 = off).
    min_cluster_size
        Forwarded to ReflectionEngine.
    reflect_apply
        If False (default), reflection runs in dry-run mode (no writes).
        If True, reflection summaries are written to the store.
    summarizer
        Optional ``Callable[[list[Memory]], str]`` forwarded to
        ReflectionEngine (e.g. from ``build_summarizer("ollama:llama3.1")``).
        None → the built-in extractive summarizer.
    """

    def __init__(
        self,
        store,
        decay_every: int | None = 86_400,
        reflect_every: int | None = 604_800,
        tick_interval: int = 300,
        state_path: str = "~/.ai_houkai/maintenance.state.json",
        decay_rate: float = 0.1,
        min_score: float = 0.05,
        protect_types: tuple[str, ...] = ("procedural",),
        frequency_weight: float = 0.0,
        min_cluster_size: int = 3,
        reflect_apply: bool = False,
        summarizer=None,
    ) -> None:
        self.store = store
        self.decay_every = decay_every
        self.reflect_every = reflect_every
        self.tick_interval = tick_interval
        self.state_path = state_path
        self.decay_rate = decay_rate
        self.min_score = min_score
        self.protect_types = protect_types
        self.frequency_weight = frequency_weight
        self.min_cluster_size = min_cluster_size
        self.reflect_apply = reflect_apply
        self.summarizer = summarizer

    def tick(self, now: float | None = None) -> TickResult:
        """Run any overdue jobs and persist the updated state.

        Accepts an optional ``now`` timestamp so tests can step time without
        sleeping.  Each job is wrapped in try/except so one failure never
        prevents the other from running.
        """
        t = now if now is not None else time.time()
        state = MaintenanceState.load(self.state_path)
        result = TickResult()

        if self.decay_every is not None:
            next_at = state.next_run_at(state.last_decay_at, self.decay_every, now=t)
            if t >= next_at:
                result.ran_decay = True
                try:
                    engine = DecayEngine(
                        self.store,
                        decay_rate=self.decay_rate,
                        min_score=self.min_score,
                        protect_types=self.protect_types,
                        frequency_weight=self.frequency_weight,
                    )
                    pruned = engine.prune(now=t)
                    result.decayed = len(pruned)
                    state.last_decay_at = t
                    state.total_decayed += result.decayed
                    logger.info("Decay: pruned %d memories", result.decayed)
                except Exception as exc:
                    result.decay_error = str(exc)
                    logger.exception("Decay run failed: %s", exc)

        if self.reflect_every is not None:
            next_at = state.next_run_at(state.last_reflect_at, self.reflect_every, now=t)
            if t >= next_at:
                result.ran_reflect = True
                result.reflect_applied = self.reflect_apply
                try:
                    engine = ReflectionEngine(
                        self.store,
                        min_cluster_size=self.min_cluster_size,
                        summarizer=self.summarizer,
                    )
                    created = engine.reflect(dry_run=not self.reflect_apply)
                    result.reflected = len(created)
                    # The schedule gates the WORK, not the writes: clustering
                    # is O(n²) and the summarizer may call an LLM, and both
                    # happen on a dry-run too. Stamp last_reflect_at whenever
                    # the job ran, or a dry-run-configured caller (daemon,
                    # MCP maintenance_tick) re-pays that cost on every tick.
                    # Totals still count persisted summaries only.
                    state.last_reflect_at = t
                    if self.reflect_apply:
                        state.total_reflected += result.reflected
                    logger.info(
                        "Reflect (%s): %d summaries",
                        "apply" if self.reflect_apply else "dry-run",
                        result.reflected,
                    )
                except Exception as exc:
                    result.reflect_error = str(exc)
                    logger.exception("Reflect run failed: %s", exc)

        state.save(self.state_path)
        return result

    def run_forever(self, stop: threading.Event) -> None:
        """Block and tick every tick_interval seconds until stop is set.

        Install a SIGTERM handler before calling this in a daemon process.
        """
        logger.info(
            "Maintenance scheduler started (decay_every=%s, reflect_every=%s, tick=%ss)",
            f"{self.decay_every}s" if self.decay_every else "off",
            f"{self.reflect_every}s" if self.reflect_every else "off",
            self.tick_interval,
        )
        while not stop.is_set():
            try:
                result = self.tick()
                if result.ran_decay or result.ran_reflect:
                    logger.info("Tick: %s", result.summary())
            except Exception as exc:
                logger.exception("Unexpected error in tick: %s", exc)
            stop.wait(timeout=self.tick_interval)
        logger.info("Maintenance scheduler stopped.")
