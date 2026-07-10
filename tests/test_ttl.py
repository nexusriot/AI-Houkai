"""Tests for explicit TTL / expiry (Feature 3)."""

from __future__ import annotations

import time

import pytest

from ai_houkai.memory_system import Memory, MemoryStore


class TestRememberTTL:
    def test_ttl_seconds_sets_future_expiry(self, store: MemoryStore):
        before = time.time()
        m = store.remember("milk", ttl_seconds=100)
        assert before + 99 <= m.expires_at <= before + 101

    def test_expires_at_absolute(self, store: MemoryStore):
        ts = time.time() + 500
        m = store.remember("bread", expires_at=ts)
        assert m.expires_at == ts

    def test_no_ttl_means_never_expires(self, store: MemoryStore):
        assert store.remember("forever").expires_at == 0.0

    def test_both_ttl_and_expires_at_rejected(self, store: MemoryStore):
        with pytest.raises(ValueError):
            store.remember("x", expires_at=time.time() + 10, ttl_seconds=10)

    def test_nonpositive_ttl_rejected(self, store: MemoryStore):
        with pytest.raises(ValueError):
            store.remember("x", ttl_seconds=0)
        with pytest.raises(ValueError):
            store.remember("x", ttl_seconds=-5)

    def test_negative_expires_at_rejected(self, store: MemoryStore):
        with pytest.raises(ValueError):
            store.remember("x", expires_at=-1)


class TestRecallHidesExpired:
    def _seed(self, store: MemoryStore):
        live = store.remember("the deployment pipeline runbook", ttl_seconds=1000)
        exp = store.remember("the deployment pipeline hotfix note",
                             expires_at=time.time() - 10)  # already expired
        return live, exp

    def test_recall_excludes_expired(self, store: MemoryStore):
        live, exp = self._seed(store)
        ids = {m.id for m, _ in store.recall("deployment pipeline", k=10)}
        assert live.id in ids
        assert exp.id not in ids

    def test_include_expired_returns_them(self, store: MemoryStore):
        live, exp = self._seed(store)
        ids = {m.id for m, _ in store.recall("deployment pipeline", k=10,
                                             include_expired=True)}
        assert exp.id in ids and live.id in ids

    def test_hybrid_mode_also_hides_expired(self, store: MemoryStore):
        _, exp = self._seed(store)
        ids = {m.id for m, _ in store.recall("deployment pipeline", k=10,
                                             mode="hybrid")}
        assert exp.id not in ids

    def test_expired_not_underfetched_on_fast_path(self, store: MemoryStore):
        # include_superseded=True would trip the semantic fast path (fetch
        # exactly k); expiry filtering must still force the overfetch pool so a
        # live memory ranked just behind an expired one is not lost.
        store.remember("alpha topic note", expires_at=time.time() - 10)
        live = store.remember("alpha topic note two")
        hits = store.recall("alpha topic note", k=1, include_superseded=True)
        assert [m.id for m, _ in hits] == [live.id]


class TestListRecentHidesExpired:
    def test_list_recent_hides_expired(self, store: MemoryStore):
        store.remember("live one")
        exp = store.remember("expired one", expires_at=time.time() - 5)
        ids = {m.id for m in store.list_recent(limit=50)}
        assert exp.id not in ids
        ids_inc = {m.id for m in store.list_recent(limit=50, include_expired=True)}
        assert exp.id in ids_inc


class TestEditTTL:
    def test_edit_sets_expiry(self, store: MemoryStore):
        m = store.remember("editable")
        store.edit(m.id, expires_at=time.time() - 1)  # expire it now
        assert store._get_by_id(m.id).expires_at > 0
        assert m.id not in {x.id for x, _ in store.recall("editable", k=5)}

    def test_edit_clears_expiry_with_zero(self, store: MemoryStore):
        m = store.remember("clearable", ttl_seconds=50)
        store.edit(m.id, expires_at=0.0)
        assert store._get_by_id(m.id).expires_at == 0.0

    def test_edit_negative_expiry_rejected(self, store: MemoryStore):
        m = store.remember("x")
        with pytest.raises(ValueError):
            store.edit(m.id, expires_at=-3)


class TestPurgeExpired:
    def test_purge_deletes_only_expired(self, store: MemoryStore):
        live = store.remember("keep me", ttl_seconds=1000)
        exp = store.remember("drop me", expires_at=time.time() - 10)
        purged = store.purge_expired()
        assert [p.id for p in purged] == [exp.id]
        assert store._get_by_id(exp.id) is None
        assert store._get_by_id(live.id) is not None

    def test_purge_dry_run_deletes_nothing(self, store: MemoryStore):
        exp = store.remember("drop me", expires_at=time.time() - 10)
        purged = store.purge_expired(dry_run=True)
        assert [p.id for p in purged] == [exp.id]
        assert store._get_by_id(exp.id) is not None  # still there

    def test_purge_honors_custom_now(self, store: MemoryStore):
        # expires in 100s; not expired "now", but expired at now+200.
        m = store.remember("future", ttl_seconds=100)
        assert store.purge_expired() == []
        purged = store.purge_expired(now=time.time() + 200)
        assert [p.id for p in purged] == [m.id]

    def test_purge_is_journaled_with_purge_actor(self, store: MemoryStore):
        exp = store.remember("drop me", expires_at=time.time() - 10)
        store.purge_expired()
        forgets = [e for e in store.journal.read()
                   if e.op == "forget" and e.id == exp.id]
        assert forgets and forgets[-1].actor == "purge"


class TestSerializationRoundTrip:
    def test_metadata_roundtrip_preserves_expiry(self):
        ts = time.time() + 42
        m = Memory(id="i", text="t", type="semantic", expires_at=ts)
        restored = Memory.from_record(m.id, m.text, m.to_metadata())
        assert restored.expires_at == ts

    def test_dict_roundtrip_preserves_expiry(self):
        ts = time.time() + 42
        m = Memory(id="i", text="t", type="semantic", expires_at=ts)
        assert Memory.from_dict(m.to_dict()).expires_at == ts

    def test_migration_missing_key_defaults_to_zero(self):
        # A pre-TTL Chroma row has no "expires_at" key.
        m = Memory.from_record("x", "text", {"type": "semantic"})
        assert m.expires_at == 0.0
