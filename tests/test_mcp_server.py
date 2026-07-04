"""In-process tests for the MCP server tools (lazy store + edit +
maintenance_tick config default)."""

from __future__ import annotations

import pytest

import ai_houkai.mcp_server.server as srv
from ai_houkai.cli.config import MaintenanceConfig


@pytest.fixture()
def mcp_store(tmp_path, monkeypatch):
    """Point the lazily-created server store at a per-test directory."""
    monkeypatch.setenv("AI_HOUKAI_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("AI_HOUKAI_COLLECTION", "mcp_test")
    monkeypatch.setattr(srv, "_store", None)
    yield
    if srv._store is not None:
        srv._store.client.close()
        srv._store = None


def _mcfg(
    tmp_path, *, reflect_apply: bool, enabled: bool = True,
    reflect_consolidate: bool | str = True,
) -> MaintenanceConfig:
    return MaintenanceConfig(
        enabled=enabled, decay_every=None, reflect_every=3_600, tick_interval=300,
        log_path=str(tmp_path / "m.log"), state_path=str(tmp_path / "m.state.json"),
        pid_path=str(tmp_path / "m.pid"), decay_rate=0.1, min_score=0.05,
        protect_types=("procedural",), frequency_weight=0.0,
        min_cluster_size=2, reflect_apply=reflect_apply, summarizer=None,
        reflect_consolidate=reflect_consolidate,
    )


class TestMcpTools:
    def test_lazy_store_honours_env(self, mcp_store, tmp_path):
        out = srv.stats()
        assert out["path"] == str(tmp_path / "chroma")
        assert out["collection"] == "mcp_test"

    def test_edit_tool_roundtrip(self, mcp_store):
        created = srv.remember(text="mcp editable fact", type="semantic")
        assert created["stored"] is True
        out = srv.edit(memory_id=created["id"], importance=0.9, tags=["t"])
        assert out["ok"] is True
        assert out["importance"] == 0.9
        assert out["tags"] == ["t"]

    def test_edit_tool_error_dict_on_missing_id(self, mcp_store):
        out = srv.edit(memory_id="00000000-0000-4000-8000-000000000000",
                       text="nope")
        assert out["ok"] is False
        assert "not found" in out["error"]

    def test_edit_tool_error_dict_on_bad_type(self, mcp_store):
        created = srv.remember(text="mcp typed fact")
        out = srv.edit(memory_id=created["id"], type="sematic")
        assert out["ok"] is False
        assert "type must be one of" in out["error"]

    def test_maintenance_tick_defaults_reflect_apply_from_config(
        self, mcp_store, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(
            srv, "load_maintenance", lambda: _mcfg(tmp_path, reflect_apply=False))
        out = srv.maintenance_tick()          # omit → config value (False)
        assert out["ran_reflect"] is True
        assert out["reflect_applied"] is False
        assert "dry-run" in out["summary"] or "nothing" in out["summary"]

    def test_maintenance_tick_explicit_overrides_config(
        self, mcp_store, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(
            srv, "load_maintenance", lambda: _mcfg(tmp_path, reflect_apply=False))
        out = srv.maintenance_tick(reflect_apply=True)
        assert out["ran_reflect"] is True
        assert out["reflect_applied"] is True

    def test_maintenance_tick_disabled_reports_and_runs_nothing(
        self, mcp_store, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(
            srv, "load_maintenance",
            lambda: _mcfg(tmp_path, reflect_apply=True, enabled=False))
        out = srv.maintenance_tick()
        assert out["enabled"] is False
        assert out["ran_decay"] is False and out["ran_reflect"] is False
        assert "disabled" in out["summary"]

    def test_maintenance_tick_reports_enabled_true(
        self, mcp_store, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(
            srv, "load_maintenance", lambda: _mcfg(tmp_path, reflect_apply=False))
        out = srv.maintenance_tick()
        assert out["enabled"] is True

    def test_maintenance_tick_threads_consolidate_from_config(
        self, mcp_store, tmp_path, monkeypatch
    ):
        from ai_houkai.maintenance.scheduler import TickResult

        monkeypatch.setattr(
            srv, "load_maintenance",
            lambda: _mcfg(tmp_path, reflect_apply=True,
                          reflect_consolidate="hard"))
        captured: dict = {}

        class FakeScheduler:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def tick(self):
                return TickResult()

        monkeypatch.setattr(srv, "MaintenanceScheduler", FakeScheduler)
        srv.maintenance_tick()
        assert captured["reflect_consolidate"] == "hard"

    def test_import_tool_reports_conflict_not_crash(self, mcp_store, tmp_path):
        created = srv.remember(text="conflicting row")
        assert created["stored"] is True
        path = str(tmp_path / "conflict.ahkai")
        exported = srv.export(path=path)
        assert exported["count"] == 1
        out = srv.import_(path=path, on_conflict="error")
        assert out["ok"] is False
        assert "collision" in out["error"]

    def test_edit_clear_source(self, mcp_store):
        created = srv.remember(text="sourced fact", source="unit-test")
        out = srv.edit(memory_id=created["id"], clear_source=True)
        assert out["ok"] is True
        assert out["source"] is None

    def test_edit_source_and_clear_source_conflict(self, mcp_store):
        created = srv.remember(text="sourced fact", source="unit-test")
        out = srv.edit(memory_id=created["id"], source="new",
                       clear_source=True)
        assert out["ok"] is False
        assert "clear_source" in out["error"]

    def test_edit_omitted_source_left_unchanged(self, mcp_store):
        created = srv.remember(text="sourced fact", source="keep-me")
        out = srv.edit(memory_id=created["id"], importance=0.7)
        assert out["ok"] is True
        assert out["source"] == "keep-me"
