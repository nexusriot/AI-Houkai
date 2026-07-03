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


def _mcfg(tmp_path, *, reflect_apply: bool) -> MaintenanceConfig:
    return MaintenanceConfig(
        enabled=True, decay_every=None, reflect_every=3_600, tick_interval=300,
        log_path=str(tmp_path / "m.log"), state_path=str(tmp_path / "m.state.json"),
        pid_path=str(tmp_path / "m.pid"), decay_rate=0.1, min_score=0.05,
        protect_types=("procedural",), frequency_weight=0.0,
        min_cluster_size=2, reflect_apply=reflect_apply, summarizer=None,
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
