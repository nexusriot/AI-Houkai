"""Tests for `houkai stats --health`."""

from __future__ import annotations

import json
import math
import time

import pytest
from typer.testing import CliRunner

from ai_houkai.cli.commands.stats import _compute_health, _decay_score
from ai_houkai.cli.main import app
from ai_houkai.memory_system import Memory, MemoryStore


runner = CliRunner()

_COLLECTION = "test_health"


@pytest.fixture()
def seeded(tmp_path):
    """Seed a MemoryStore that the CLI will also hit (same path + collection)."""
    store_path = str(tmp_path / "chroma")
    store = MemoryStore(path=store_path, collection=_COLLECTION)

    # High-importance procedural (stays healthy)
    store.remember("Deploy with make release", type="procedural", importance=0.9, tags=["ops"])
    # Medium semantic memories
    store.remember("Python uses GIL", type="semantic", importance=0.5)
    store.remember("ChromaDB persists to disk", type="semantic", importance=0.6, tags=["infra"])
    # Episodic cluster candidates
    ep1 = store.remember("Fixed a flaky test in suite A", type="episodic", importance=0.4)
    ep2 = store.remember("Fixed another flaky test in suite A", type="episodic", importance=0.4)
    # Old (stale) memory — backdate last_accessed so it counts as stale (>30 days)
    old_mem = store.remember("Very old fact", type="semantic", importance=0.3)
    old_meta = old_mem.to_metadata()
    old_meta["last_accessed"] = time.time() - 35 * 86_400
    old_meta["created_at"]    = time.time() - 35 * 86_400
    store.collection.update(ids=[old_mem.id], metadatas=[old_meta])

    # Link two memories so link_density > 0
    store.link(ep1.id, ep2.id, "related")

    store.client.close()
    # Return just the path; CLI receives it via --store
    return store_path


def _invoke_health(store_path: str, extra_args: list[str] | None = None) -> dict:
    args = ["--store", store_path, "--collection", _COLLECTION,
            "stats", "--health", "--format", "json"] + (extra_args or [])
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    return json.loads(result.output)


class TestStatsHealthJson:
    def test_health_key_present(self, seeded):
        data = _invoke_health(seeded)
        assert "health" in data
        h = data["health"]
        for key in ("decay_histogram", "at_risk_count", "never_recalled_count",
                    "stale_count", "episodic_active_count", "link_density",
                    "avg_importance", "top_recalled"):
            assert key in h, f"missing key: {key}"

    def test_histogram_bands(self, seeded):
        data = _invoke_health(seeded)
        hist = data["health"]["decay_histogram"]
        expected = {"0.0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"}
        assert set(hist.keys()) == expected
        assert sum(hist.values()) == data["active"]

    def test_stale_count(self, seeded):
        data = _invoke_health(seeded, ["--stale-days", "30"])
        assert data["health"]["stale_count"] >= 1

    def test_episodic_active_count(self, seeded):
        data = _invoke_health(seeded)
        assert data["health"]["episodic_active_count"] == 2

    def test_link_density_positive(self, seeded):
        data = _invoke_health(seeded)
        # 1 link among 6 active memories → 1/6 > 0
        assert data["health"]["link_density"] > 0.0

    def test_never_recalled(self, seeded):
        data = _invoke_health(seeded)
        # No recall() called → all active memories have access_count == 0
        assert data["health"]["never_recalled_count"] == data["active"]

    def test_top_recalled_empty_when_no_recalls(self, seeded):
        data = _invoke_health(seeded)
        assert data["health"]["top_recalled"] == []

    def test_avg_importance_in_range(self, seeded):
        data = _invoke_health(seeded)
        avg = data["health"]["avg_importance"]
        assert 0.0 < avg <= 1.0


class TestStatsHealthRich:
    def test_rich_output_exits_cleanly(self, seeded):
        result = runner.invoke(app, [
            "--store", seeded, "--collection", _COLLECTION, "stats", "--health",
        ])
        assert result.exit_code == 0
        assert "Health Report" in result.output
        assert "Decay score distribution" in result.output

    def test_rich_shows_at_risk(self, seeded):
        result = runner.invoke(app, [
            "--store", seeded, "--collection", _COLLECTION, "stats", "--health",
        ])
        assert "At-risk" in result.output

    def test_rich_shows_stale(self, seeded):
        result = runner.invoke(app, [
            "--store", seeded, "--collection", _COLLECTION, "stats", "--health",
        ])
        assert "Stale" in result.output


class TestStatsHealthEmpty:
    def test_empty_store(self, tmp_path):
        result = runner.invoke(app, [
            "--store", str(tmp_path / "chroma"), "--collection", "test_empty",
            "stats", "--health", "--format", "json",
        ])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["active"] == 0
        h = data["health"]
        assert h["at_risk_count"] == 0
        assert h["link_density"] == 0.0
        assert h["avg_importance"] == 0.0


class TestStatsBasicUnchanged:
    def test_no_health_key_by_default(self, seeded):
        result = runner.invoke(app, [
            "--store", seeded, "--collection", _COLLECTION,
            "stats", "--format", "json",
        ])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "health" not in data
        assert data["active"] > 0


class TestStatsHealthAlignment:
    """The health report's decay maths mirror DecayEngine exactly (incl.
    frequency reinforcement) and honour protected types."""

    def test_decay_score_matches_engine_with_reinforcement(self):
        t_old = time.time() - 10 * 86_400.0
        base = _decay_score(0.5, t_old, 0.1, access_count=20, frequency_weight=0.0)
        reinforced = _decay_score(0.5, t_old, 0.1, access_count=20, frequency_weight=0.3)
        assert reinforced > base
        # matches DecayEngine.score formula exactly
        expected = 0.5 * math.exp(-0.1 * 10) * (1 + 0.3 * math.log1p(20))
        assert reinforced == pytest.approx(expected)

    def test_protect_types_excluded_from_at_risk(self):
        old = time.time() - 100 * 86_400.0
        proc = Memory(id="p", text="runbook", type="procedural", importance=0.1,
                      created_at=old, last_accessed=old)
        sem = Memory(id="s", text="fact", type="semantic", importance=0.1,
                     created_at=old, last_accessed=old)
        h = _compute_health([proc, sem], stale_days=30, decay_rate=0.1,
                            min_score=0.05, protect_types=("procedural",),
                            frequency_weight=0.0)
        # Both decayed below 0.05, but procedural is protected → only semantic at risk.
        assert h["at_risk_count"] == 1
