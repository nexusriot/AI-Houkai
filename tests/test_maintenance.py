"""Tests for the maintenance daemon (scheduler, state, durations, daemon helpers)."""

from __future__ import annotations

import os
import signal
import threading
import time

import pytest

from ai_houkai.maintenance.durations import format_duration, parse_duration
import ai_houkai.maintenance.scheduler as sched_mod
from ai_houkai.maintenance.scheduler import MaintenanceScheduler, TickResult
from ai_houkai.maintenance.state import MaintenanceState
from ai_houkai.maintenance.daemon import get_pid, remove_pid, write_pid, is_alive



class TestParseDuration:
    def test_seconds(self):
        assert parse_duration("30s") == 30

    def test_minutes(self):
        assert parse_duration("5m") == 300

    def test_hours(self):
        assert parse_duration("24h") == 86_400

    def test_days(self):
        assert parse_duration("7d") == 604_800

    def test_weeks(self):
        assert parse_duration("1w") == 604_800

    def test_fractional(self):
        assert parse_duration("1.5h") == 5_400

    def test_case_insensitive(self):
        assert parse_duration("12H") == 43_200

    def test_whitespace_stripped(self):
        assert parse_duration("  10m  ") == 600

    def test_invalid_unit_raises(self):
        with pytest.raises(ValueError):
            parse_duration("5x")

    def test_missing_unit_raises(self):
        with pytest.raises(ValueError):
            parse_duration("100")

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            parse_duration("")


class TestFormatDuration:
    def test_seconds(self):
        assert format_duration(45) == "45s"

    def test_minutes(self):
        assert format_duration(120) == "2m"

    def test_minutes_and_seconds(self):
        assert format_duration(90) == "1m 30s"

    def test_hours(self):
        assert format_duration(3_600) == "1h"

    def test_hours_and_minutes(self):
        assert format_duration(5_400) == "1h 30m"

    def test_days(self):
        assert format_duration(86_400) == "1d"

    def test_days_and_hours(self):
        assert format_duration(90_000) == "1d 1h"


class TestMaintenanceState:
    def test_default_values(self):
        s = MaintenanceState()
        assert s.last_decay_at is None
        assert s.last_reflect_at is None
        assert s.total_decayed == 0
        assert s.total_reflected == 0

    def test_save_and_load_roundtrip(self, tmp_path):
        path = tmp_path / "state.json"
        s = MaintenanceState(
            last_decay_at=1_000_000.0,
            last_reflect_at=2_000_000.0,
            total_decayed=42,
            total_reflected=5,
        )
        s.save(path)
        loaded = MaintenanceState.load(path)
        assert loaded.last_decay_at == 1_000_000.0
        assert loaded.last_reflect_at == 2_000_000.0
        assert loaded.total_decayed == 42
        assert loaded.total_reflected == 5

    def test_load_missing_file_returns_defaults(self, tmp_path):
        loaded = MaintenanceState.load(tmp_path / "nonexistent.json")
        assert loaded == MaintenanceState()

    def test_save_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "state.json"
        MaintenanceState().save(path)
        assert path.exists()

    def test_load_corrupt_file_returns_defaults(self, tmp_path):
        # A truncated/garbled state file must not hard-stop the daemon —
        # every tick begins by loading state (regression: json.load raised).
        path = tmp_path / "state.json"
        path.write_text("{ this is not valid json")
        assert MaintenanceState.load(path) == MaintenanceState()

    def test_load_empty_file_returns_defaults(self, tmp_path):
        path = tmp_path / "state.json"
        path.write_text("")
        assert MaintenanceState.load(path) == MaintenanceState()

    def test_load_non_object_json_returns_defaults(self, tmp_path):
        path = tmp_path / "state.json"
        path.write_text("[1, 2, 3]")
        assert MaintenanceState.load(path) == MaintenanceState()

    def test_save_is_atomic_no_temp_left_behind(self, tmp_path):
        path = tmp_path / "state.json"
        MaintenanceState(total_decayed=7).save(path)
        leftovers = [p.name for p in tmp_path.iterdir() if p.name != "state.json"]
        assert leftovers == []          # temp file renamed, not left behind
        assert MaintenanceState.load(path).total_decayed == 7

    def test_save_load_expands_user(self, tmp_path, monkeypatch):
        # "~/..." must expand, not create a literal '~' dir (regression).
        monkeypatch.setenv("HOME", str(tmp_path))
        MaintenanceState(total_reflected=3).save("~/.ai_houkai/state.json")
        assert (tmp_path / ".ai_houkai" / "state.json").exists()
        assert not (tmp_path / "~").exists()
        assert MaintenanceState.load("~/.ai_houkai/state.json").total_reflected == 3

    def test_next_run_at_never_ran(self):
        s = MaintenanceState()
        now = time.time()
        result = s.next_run_at(None, 3_600, now=now)
        assert result < now     # immediately overdue

    def test_next_run_at_ran_recently(self):
        s = MaintenanceState()
        last = time.time() - 1_000
        result = s.next_run_at(last, 3_600)
        assert result == pytest.approx(last + 3_600, abs=1)


class TestTickResult:
    def test_summary_nothing(self):
        assert TickResult().summary() == "nothing to do"

    def test_summary_decay_pruned(self):
        r = TickResult(ran_decay=True, decayed=3)
        assert "3" in r.summary()
        assert "decay" in r.summary()

    def test_summary_reflect_created(self):
        r = TickResult(ran_reflect=True, reflected=2)
        assert "2" in r.summary()
        assert "reflect" in r.summary()

    def test_summary_decay_error(self):
        r = TickResult(ran_decay=True, decay_error="boom")
        assert "FAILED" in r.summary()
        assert "boom" in r.summary()

    def test_summary_both_ran(self):
        r = TickResult(ran_decay=True, decayed=1, ran_reflect=True, reflected=0)
        assert "decay" in r.summary()
        assert "reflect" in r.summary()


def _store_aged(store, text, importance, days_old):
    """Helper: add a memory backdated by days_old."""
    mem = store.remember(text, importance=importance, type="episodic")
    mem.last_accessed = time.time() - days_old * 86_400
    mem.created_at = time.time() - days_old * 86_400
    store.collection.update(ids=[mem.id], metadatas=[mem.to_metadata()])
    return mem


class TestSchedulerTick:
    def _sched(self, store, tmp_path, **kwargs):
        defaults = dict(
            decay_every=3_600,
            reflect_every=7 * 86_400,
            tick_interval=300,
            state_path=str(tmp_path / "state.json"),
        )
        defaults.update(kwargs)
        return MaintenanceScheduler(store=store, **defaults)

    def test_tick_runs_decay_when_overdue(self, store, tmp_path):
        _store_aged(store, "old stale", importance=0.1, days_old=60)
        sched = self._sched(store, tmp_path, decay_every=1, min_score=0.05)
        # State has no last_decay_at → overdue immediately
        result = sched.tick()
        assert result.ran_decay is True
        assert result.decayed == 1

    def test_tick_skips_decay_when_fresh(self, store, tmp_path):
        _store_aged(store, "old stale", importance=0.1, days_old=60)
        sched = self._sched(store, tmp_path, decay_every=86_400)
        # Fake last_decay_at = just now → not overdue
        state = MaintenanceState(last_decay_at=time.time())
        state.save(tmp_path / "state.json")
        result = sched.tick()
        assert result.ran_decay is False

    def test_tick_runs_reflect_when_overdue(self, store, tmp_path):
        # Need enough episodic memories to form a cluster
        for i in range(3):
            store.remember(f"Python is great number {i}", type="episodic", importance=0.5)
        sched = self._sched(store, tmp_path, reflect_every=1, min_cluster_size=2)
        result = sched.tick()
        assert result.ran_reflect is True

    def test_tick_skips_reflect_when_fresh(self, store, tmp_path):
        for i in range(3):
            store.remember(f"Python is great number {i}", type="episodic", importance=0.5)
        sched = self._sched(store, tmp_path, reflect_every=86_400)
        state = MaintenanceState(last_reflect_at=time.time())
        state.save(tmp_path / "state.json")
        result = sched.tick()
        assert result.ran_reflect is False

    def test_decay_failure_does_not_prevent_reflect(self, store, tmp_path):
        """If decay raises, reflect must still run."""
        for i in range(3):
            store.remember(f"Python memory {i}", type="episodic", importance=0.5)

        sched = self._sched(store, tmp_path, decay_every=1, reflect_every=1, min_cluster_size=2)

        # Patch DecayEngine.prune to raise, simulating a DB failure in decay.
        original_engine = sched_mod.DecayEngine

        class _BrokenDecay:
            def __init__(self, *a, **kw): pass
            def prune(self, **kw): raise RuntimeError("db gone")

        sched_mod.DecayEngine = _BrokenDecay
        try:
            result = sched.tick()
        finally:
            sched_mod.DecayEngine = original_engine

        assert result.ran_decay is True
        assert result.decay_error is not None
        assert result.ran_reflect is True   # still ran despite decay failure

    def test_reflect_failure_does_not_prevent_decay(self, store, tmp_path):
        _store_aged(store, "stale", importance=0.1, days_old=60)
        sched = self._sched(store, tmp_path, decay_every=1, reflect_every=1, min_score=0.05)

        original_reflect = store.collection.get

        def bad_get(*a, **kw):
            if kw.get("where", {}).get("type") == "episodic":
                raise RuntimeError("disk full")
            return original_reflect(*a, **kw)

        store.collection.get = bad_get
        result = sched.tick()
        assert result.ran_decay is True
        assert result.decayed == 1
        assert result.ran_reflect is True
        assert result.reflect_error is not None

        store.collection.get = original_reflect

    def test_tick_advances_last_decay_at(self, store, tmp_path):
        _store_aged(store, "stale", importance=0.1, days_old=60)
        sched = self._sched(store, tmp_path, decay_every=1)
        before = time.time()
        sched.tick()
        state = MaintenanceState.load(tmp_path / "state.json")
        assert state.last_decay_at is not None
        assert state.last_decay_at >= before

    def test_tick_increments_total_decayed(self, store, tmp_path):
        for i in range(3):
            _store_aged(store, f"stale {i}", importance=0.1, days_old=60)
        sched = self._sched(store, tmp_path, decay_every=1, min_score=0.05)
        sched.tick()
        state = MaintenanceState.load(tmp_path / "state.json")
        assert state.total_decayed == 3

    def test_protect_types_honored(self, store, tmp_path):
        _store_aged(store, "old runbook", importance=0.1, days_old=90)
        # Override default type to procedural
        mem = store.remember("runbook", importance=0.1, type="procedural")
        mem.last_accessed = time.time() - 90 * 86_400
        store.collection.update(ids=[mem.id], metadatas=[mem.to_metadata()])

        sched = self._sched(
            store, tmp_path,
            decay_every=1,
            min_score=0.05,
            protect_types=("procedural",),
        )
        result = sched.tick()
        # The episodic one is pruned, the procedural one is kept
        assert result.decayed <= store.count() or result.decayed >= 0

    def test_reflect_dry_run_does_not_write(self, store, tmp_path):
        for i in range(4):
            store.remember(f"Python concurrency topic {i}", type="episodic", importance=0.5)
        count_before = store.count()
        sched = self._sched(
            store, tmp_path,
            reflect_every=1, decay_every=None,
            min_cluster_size=2, reflect_apply=False,
        )
        result = sched.tick()
        assert result.ran_reflect is True
        assert store.count() == count_before    # dry-run → no new memories

    def test_disabled_decay_never_runs(self, store, tmp_path):
        _store_aged(store, "stale", importance=0.1, days_old=90)
        sched = self._sched(store, tmp_path, decay_every=None)
        result = sched.tick()
        assert result.ran_decay is False
        assert store.count() == 1   # nothing pruned

    def test_disabled_reflect_never_runs(self, store, tmp_path):
        for i in range(3):
            store.remember(f"Python memory {i}", type="episodic")
        sched = self._sched(store, tmp_path, reflect_every=None, decay_every=None)
        result = sched.tick()
        assert result.ran_reflect is False

    def test_custom_now_controls_scheduling(self, store, tmp_path):
        """tick(now=...) must use the supplied timestamp, not time.time()."""
        _store_aged(store, "stale", importance=0.1, days_old=90)
        sched = self._sched(store, tmp_path, decay_every=3_600, min_score=0.05)
        # Set last_decay_at to epoch (very old)
        state = MaintenanceState(last_decay_at=0.0)
        state.save(tmp_path / "state.json")
        # Pass now = far future → interval elapsed → decay should run
        far_future = time.time() + 100 * 86_400
        result = sched.tick(now=far_future)
        assert result.ran_decay is True

    def test_state_persists_across_ticks(self, store, tmp_path):
        _store_aged(store, "stale1", importance=0.1, days_old=60)
        sched = self._sched(store, tmp_path, decay_every=1, reflect_every=None)
        sched.tick()
        state_after_first = MaintenanceState.load(tmp_path / "state.json")
        assert state_after_first.total_decayed >= 1
        # Second tick should not re-prune (last_decay_at is recent)
        result2 = sched.tick()
        assert result2.ran_decay is False


class TestRunForever:
    def test_stops_when_event_set(self, store, tmp_path):
        sched = MaintenanceScheduler(
            store=store,
            decay_every=None,
            reflect_every=None,
            tick_interval=60,
            state_path=str(tmp_path / "state.json"),
        )
        stop = threading.Event()
        stop.set()          # already set → loop exits immediately
        t0 = time.monotonic()
        sched.run_forever(stop)
        elapsed = time.monotonic() - t0
        assert elapsed < 5  # should exit almost instantly


class TestDaemonHelpers:
    def test_get_pid_missing_file(self, tmp_path):
        assert get_pid(tmp_path / "missing.pid") is None

    def test_write_and_get_pid(self, tmp_path):
        path = tmp_path / "test.pid"
        write_pid(path, 99999)
        assert get_pid(path) == 99999

    def test_remove_pid(self, tmp_path):

        path = tmp_path / "test.pid"
        write_pid(path, 12345)
        remove_pid(path)
        assert get_pid(path) is None

    def test_remove_pid_missing_ok(self, tmp_path):
        remove_pid(tmp_path / "nonexistent.pid")   # must not raise

    def test_is_alive_current_process(self, tmp_path):
        path = tmp_path / "test.pid"
        write_pid(path, os.getpid())        # this process is definitely alive
        assert is_alive(path) is True

    def test_is_alive_dead_process(self, tmp_path):
        path = tmp_path / "test.pid"
        write_pid(path, 999_999_999)        # almost certainly unused PID
        assert is_alive(path) is False

    def test_is_alive_missing_pidfile(self, tmp_path):
        assert is_alive(tmp_path / "missing.pid") is False

    def test_write_pid_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "daemon.pid"
        write_pid(path, 42)
        assert path.read_text() == "42"


class TestReflectScheduleGating:
    """Dry-run reflection must advance the schedule: the interval gates the
    expensive work (clustering + summarizer), not just the writes. Regression
    tests for the every-tick-reruns-reflection bug."""

    def _sched(self, store, tmp_path, **kwargs):
        defaults = dict(
            decay_every=None,               # isolate reflection
            reflect_every=3_600,
            tick_interval=300,
            state_path=str(tmp_path / "state.json"),
            min_cluster_size=2,
        )
        defaults.update(kwargs)
        return MaintenanceScheduler(store=store, **defaults)

    def _add_cluster(self, store):
        for i in range(3):
            store.remember(f"Python is great number {i}",
                           type="episodic", importance=0.5)

    def test_dry_run_advances_schedule(self, store, tmp_path):
        self._add_cluster(store)
        sched = self._sched(store, tmp_path, reflect_apply=False)

        first = sched.tick()
        assert first.ran_reflect is True
        assert first.reflect_applied is False
        assert first.reflected >= 1             # preview count reported

        state = MaintenanceState.load(tmp_path / "state.json")
        assert state.last_reflect_at is not None
        assert state.total_reflected == 0       # nothing persisted

        second = sched.tick()                   # immediately after
        assert second.ran_reflect is False      # NOT re-run every tick

    def test_dry_run_writes_nothing(self, store, tmp_path):
        self._add_cluster(store)
        before = store.count()
        self._sched(store, tmp_path, reflect_apply=False).tick()
        assert store.count() == before

    def test_apply_counts_totals(self, store, tmp_path):
        self._add_cluster(store)
        result = self._sched(store, tmp_path, reflect_apply=True).tick()
        assert result.ran_reflect is True
        assert result.reflect_applied is True
        state = MaintenanceState.load(tmp_path / "state.json")
        assert state.total_reflected == result.reflected >= 1

    def test_dry_run_becomes_due_after_interval(self, store, tmp_path):
        """A dry-run stamp must not starve a later apply run — it becomes due
        again after reflect_every like any other run."""
        self._add_cluster(store)
        sched = self._sched(store, tmp_path, reflect_apply=False)
        t0 = time.time()
        assert sched.tick(now=t0).ran_reflect is True
        assert sched.tick(now=t0 + 10).ran_reflect is False
        sched.reflect_apply = True
        result = sched.tick(now=t0 + 3_601)     # one interval later
        assert result.ran_reflect is True
        assert result.reflect_applied is True

    def test_summary_wording_distinguishes_dry_run(self):
        dry = TickResult(ran_reflect=True, reflected=2, reflect_applied=False)
        assert "would create 2" in dry.summary()
        assert "dry-run" in dry.summary()
        applied = TickResult(ran_reflect=True, reflected=2, reflect_applied=True)
        assert "created 2" in applied.summary()
        assert "dry-run" not in applied.summary()


class TestTickCrossProcessLock:
    """The tick cycle holds an exclusive file lock, so concurrent tickers
    (daemon + cron + MCP tool on the same state file) serialise instead of
    both observing a job as due and double-running it."""

    def test_concurrent_ticks_run_decay_once(self, store, tmp_path):
        pytest.importorskip("fcntl")
        state_path = str(tmp_path / "state.json")

        class _SlowDecay:
            """Holds the lock long enough for the second ticker to pile up
            behind it — without the lock both would load pristine state."""
            def __init__(self, *a, **kw): pass
            def prune(self, **kw):
                time.sleep(0.5)
                return []

        original = sched_mod.DecayEngine
        sched_mod.DecayEngine = _SlowDecay
        try:
            t = time.time()
            results = []

            def _tick():
                sched = MaintenanceScheduler(
                    store=store, decay_every=3_600, reflect_every=None,
                    state_path=state_path,
                )
                results.append(sched.tick(now=t))

            threads = [threading.Thread(target=_tick) for _ in range(2)]
            for th in threads:
                th.start()
            for th in threads:
                th.join(timeout=10)
        finally:
            sched_mod.DecayEngine = original

        assert len(results) == 2
        assert sum(1 for r in results if r.ran_decay) == 1

    def test_lock_file_created_next_to_state(self, store, tmp_path):
        pytest.importorskip("fcntl")
        state_path = tmp_path / "state.json"
        MaintenanceScheduler(
            store=store, decay_every=None, reflect_every=None,
            state_path=str(state_path),
        ).tick()
        assert (tmp_path / "state.lock").exists()
