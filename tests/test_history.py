"""Tests for point-in-time / history queries (Feature 2)."""

from __future__ import annotations

import time

from ai_houkai.memory_system import MemoryStore


class TestHistory:
    def test_timeline_is_oldest_first(self, store: MemoryStore):
        m = store.remember("draft")
        store.edit(m.id, text="draft v2")
        store.forget(m.id)
        ops = [e.op for e in store.history(m.id)]
        assert ops == ["remember", "edit", "forget"]

    def test_history_includes_snapshots(self, store: MemoryStore):
        m = store.remember("with snapshot")
        entry = store.history(m.id)[0]
        assert entry.op == "remember"
        assert entry.after is not None and entry.after["text"] == "with snapshot"

    def test_history_includes_edge_where_memory_is_link_target(self, store: MemoryStore):
        a = store.remember("source memory")
        b = store.remember("target memory")
        store.link(a.id, b.id, rel="refines")
        # The link entry is filed under src (a); b appears only via meta.dst_id.
        assert any(e.op == "link" for e in store.history(b.id))

    def test_history_includes_supersede_where_memory_is_new_id(self, store: MemoryStore):
        old = store.remember("old fact")
        new = store.remember("new fact")
        store.supersede(old_id=old.id, new_id=new.id)
        # supersede is filed under old; new appears only via meta.new_id.
        assert any(e.op == "supersede" for e in store.history(new.id))

    def test_unknown_id_has_empty_history(self, store: MemoryStore):
        assert store.history("does-not-exist") == []


class TestStateAt:
    def test_reconstructs_lifecycle(self, store: MemoryStore):
        t_before = time.time()
        time.sleep(0.02)
        m = store.remember("original text")
        time.sleep(0.02)
        t_after_create = time.time()
        time.sleep(0.02)
        store.edit(m.id, text="edited text")
        time.sleep(0.02)
        t_after_edit = time.time()
        time.sleep(0.02)
        store.forget(m.id)
        time.sleep(0.02)
        t_after_forget = time.time()

        assert store.get_at(m.id, t_before) is None
        assert store.get_at(m.id, t_after_create).text == "original text"
        assert store.get_at(m.id, t_after_edit).text == "edited text"
        assert store.get_at(m.id, t_after_forget) is None

    def test_state_at_returns_all_live_memories(self, store: MemoryStore):
        store.remember("one")
        store.remember("two")
        time.sleep(0.02)
        t = time.time()
        mems = store.state_at(t)
        assert {m.text for m in mems} == {"one", "two"}

    def test_nuke_resets_reconstructed_state(self, store: MemoryStore):
        store.remember("doomed one")
        store.remember("doomed two")
        time.sleep(0.02)
        store.nuke()
        time.sleep(0.02)
        assert store.state_at(time.time()) == []

    def test_link_delta_replayed(self, store: MemoryStore):
        a = store.remember("has links")
        b = store.remember("neighbour")
        store.link(a.id, b.id, rel="related")
        time.sleep(0.02)
        reconstructed = store.get_at(a.id, time.time())
        assert any(l.to == b.id and l.rel == "related"
                   for l in reconstructed.links)

    def test_get_at_before_creation_is_none(self, store: MemoryStore):
        t = time.time()
        time.sleep(0.02)
        m = store.remember("later")
        assert store.get_at(m.id, t) is None
