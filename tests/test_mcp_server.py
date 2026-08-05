"""In-process tests for the MCP server tools (lazy store + edit +
maintenance_tick config default)."""

from __future__ import annotations

import time

import pytest

import ai_houkai.mcp_server.server as srv
from ai_houkai.cli.config import MaintenanceConfig
from ai_houkai.maintenance.scheduler import TickResult


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
        enabled=enabled, decay_every=None, reflect_every=3_600, purge_every=None,
        tick_interval=300,
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

    def test_remember_many_tool(self, mcp_store):
        out = srv.remember_many(items=[
            {"text": "mcp batch one"},
            {"text": "mcp batch two", "type": "procedural", "tags": ["t"]},
        ])
        assert out["stored"] == 2 and len(out["ids"]) == 2
        assert srv.stats()["count"] == 2

    def test_remember_many_raise_rejected(self, mcp_store):
        out = srv.remember_many(items=[{"text": "x"}], on_conflict="raise")
        assert out["stored"] == 0 and "error" in out

    def test_remember_many_bad_item(self, mcp_store):
        out = srv.remember_many(items=[{"bogus": 1}])
        assert out["stored"] == 0 and "error" in out

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


class TestMcpNewFeatures:
    def test_metrics_tool(self, mcp_store):
        srv.remember(text="counted memory")
        srv.recall(query="counted", k=2)
        out = srv.metrics()
        assert out["calls"]["remember"] == 1
        assert out["calls"]["recall"] == 1
        assert out["recall_latency_ms"]["count"] == 1

    def test_remember_ttl_and_purge(self, mcp_store):
        created = srv.remember(text="ephemeral mcp", expires_at=1.0)  # long past
        assert created["expires_at"] is not None
        # Hidden from recall.
        hits = srv.recall(query="ephemeral", k=5)
        assert all(h["id"] != created["id"] for h in hits)
        # But visible with include_expired.
        hits2 = srv.recall(query="ephemeral", k=5, include_expired=True)
        assert any(h["id"] == created["id"] for h in hits2)
        # And purgeable.
        purged = srv.purge_expired()
        assert created["id"] in purged["ids"]

    def test_recall_explain(self, mcp_store):
        srv.remember(text="explain this via mcp")
        hits = srv.recall(query="explain", k=1, mode="hybrid", explain=True)
        assert "explain" in hits[0]
        assert hits[0]["explain"]["mode"] == "hybrid"

    def test_history_tool(self, mcp_store):
        created = srv.remember(text="mcp v1")
        srv.edit(memory_id=created["id"], text="mcp v2")
        hist = srv.history(memory_id=created["id"])
        assert [e["op"] for e in hist] == ["remember", "edit"]

    def test_state_at_and_get_at(self, mcp_store):
        created = srv.remember(text="mcp point in time")
        time.sleep(0.02)
        ts = str(time.time())
        state = srv.state_at(ts=ts)
        assert any(m["id"] == created["id"] for m in state["memories"])
        one = srv.get_at(memory_id=created["id"], ts=ts)
        assert one["ok"] is True and one["text"] == "mcp point in time"

    def test_get_at_before_creation(self, mcp_store):
        ts = str(time.time())
        time.sleep(0.02)
        created = srv.remember(text="mcp later")
        out = srv.get_at(memory_id=created["id"], ts=ts)
        assert out["ok"] is False


class TestIdempotentRememberReportsNoNewRow:
    """`stored: true` means "I created a row". An idempotent repeat creates
    nothing — it finds the existing memory and bumps its access count — so an
    agent re-asserting known facts every session was told each one was newly
    stored. The Go port reported `{stored: false, conflicts: []}`, which reads
    as a rejected write with no reason given; both now return the row with
    `stored: false`."""

    def test_first_write_is_stored(self, mcp_store):
        out = srv.remember(text="repeat me", idempotent=True)
        assert out["stored"] is True
        assert out["id"]

    def test_the_repeat_returns_the_existing_row(self, mcp_store):
        first = srv.remember(text="repeat me", idempotent=True)
        second = srv.remember(text="repeat me", idempotent=True)
        assert second["stored"] is False
        assert second["id"] == first["id"]
        assert "conflicts" not in second

    def test_without_the_flag_a_repeat_is_a_new_row(self, mcp_store):
        first = srv.remember(text="no flag here")
        second = srv.remember(text="no flag here")
        assert second["stored"] is True
        assert second["id"] != first["id"]


class TestBatchStoredCountsOnlyNewRows:
    """Same contract as the single write: `stored` is rows created, not items
    submitted. A replayed idempotent batch creates nothing."""

    def test_a_replayed_batch_reports_no_new_rows(self, mcp_store):
        items = [{"text": "batch fact one"}, {"text": "batch fact two"}]
        first = srv.remember_many(items=items, idempotent=True)
        assert first["stored"] == 2

        again = srv.remember_many(items=items, idempotent=True)
        assert again["stored"] == 0
        assert again["ids"] == first["ids"], "every input still maps to an id"

    def test_intra_batch_duplicates_count_once(self, mcp_store):
        out = srv.remember_many(
            items=[{"text": "same text"}, {"text": "same text"}],
            idempotent=True)
        assert out["stored"] == 1
        assert len(out["ids"]) == 2 and out["ids"][0] == out["ids"][1]

    def test_without_the_flag_every_item_is_a_new_row(self, mcp_store):
        out = srv.remember_many(items=[{"text": "dup"}, {"text": "dup"}])
        assert out["stored"] == 2
