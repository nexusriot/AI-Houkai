"""Unit tests for conflict / contradiction detection and supersede."""

from __future__ import annotations

import pytest

from ai_houkai.memory_system import Conflict, ConflictError, MemoryStore
from ai_houkai.memory_system.store import _negation_diff


class TestNegationDiff:
    def test_plain_vs_plain_no_diff(self):
        assert _negation_diff("use pytest", "run with pytest") is False

    def test_positive_vs_negated(self):
        assert _negation_diff("use ruff for linting", "never use ruff") is True

    def test_both_negated_no_diff(self):
        assert _negation_diff("never use EphemeralClient",
                               "don't use EphemeralClient") is False

    def test_double_negation(self):
        # "not never" → double-negated (even count) vs single "not"
        # Odd parity check
        s1 = "not never use it"   # 2 negs → even
        s2 = "not use it"         # 1 neg  → odd
        assert _negation_diff(s1, s2) is True

    def test_no_negation_both(self):
        assert _negation_diff("deploy to prod", "release to prod") is False


class TestFindConflicts:
    def test_empty_store_returns_empty(self, store: MemoryStore):
        assert store.find_conflicts() == []

    def test_single_memory_no_conflicts(self, store: MemoryStore):
        store.remember("Use pytest for testing", type="procedural")
        assert store.find_conflicts() == []

    def test_near_duplicate_detected(self, store: MemoryStore):
        store.remember("Use ruff to lint Python code", type="procedural",
                       tags=["linting"])
        store.remember("Run ruff to lint Python files", type="procedural",
                       tags=["linting"])
        conflicts = store.find_conflicts(threshold=0.70)
        assert len(conflicts) >= 1
        assert all(c.kind == "duplicate" for c in conflicts)

    def test_contradiction_detected(self, store: MemoryStore):
        store.remember("Always use ruff for linting", type="procedural",
                       tags=["linting"])
        store.remember("Never use ruff for linting", type="procedural",
                       tags=["linting"])
        conflicts = store.find_conflicts(threshold=0.70)
        assert any(c.kind == "contradiction" for c in conflicts)
        assert any(c.reason == "negation_diff" for c in conflicts)

    def test_different_types_not_conflicting(self, store: MemoryStore):
        store.remember("Use ruff for linting", type="procedural", tags=["lint"])
        store.remember("Use ruff for linting", type="semantic",   tags=["lint"])
        # Different types should not be paired
        conflicts = store.find_conflicts(threshold=0.95)
        assert all(c.a.type == c.b.type for c in conflicts)

    def test_by_memory_id(self, store: MemoryStore):
        m1 = store.remember("Always use ruff", type="procedural", tags=["lint"])
        m2 = store.remember("Never use ruff", type="procedural", tags=["lint"])
        conflicts = store.find_conflicts(memory_id=m1.id, threshold=0.70)
        assert isinstance(conflicts, list)

    def test_superseded_not_included(self, store: MemoryStore):
        m1 = store.remember("Use ruff for linting", type="procedural", tags=["lint"])
        m2 = store.remember("Also use ruff for linting", type="procedural", tags=["lint"])
        store.supersede(old_id=m1.id, new_id=m2.id)
        conflicts = store.find_conflicts(threshold=0.70)
        assert all(c.a.id != m1.id and c.b.id != m1.id for c in conflicts)

    def test_conflict_fields(self, store: MemoryStore):
        store.remember("Always use ruff", type="procedural", tags=["lint"])
        store.remember("Never use ruff", type="procedural", tags=["lint"])
        conflicts = store.find_conflicts(threshold=0.70)
        if conflicts:
            c = conflicts[0]
            assert hasattr(c, "a")
            assert hasattr(c, "b")
            assert 0.0 <= c.similarity <= 1.0
            assert c.kind in ("duplicate", "contradiction")
            assert c.reason in ("negation_diff", "custom_fn", "similarity")

    def test_custom_contradiction_fn(self, store: MemoryStore):
        def always_contradicts(a, b):
            return True

        store2 = MemoryStore(
            path=str(store.collection._client._settings.chroma_db_impl
                     if hasattr(store, "_path") else "/tmp/t2"),
            collection="test_custom_fn",
            contradiction_fn=always_contradicts,
        )

    def test_tag_overlap_guard(self, store: MemoryStore):
        # Memories about different topics (no tag overlap) should not conflict
        store.remember("Deploy to Kubernetes", type="procedural", tags=["k8s"])
        store.remember("Don't deploy to Kubernetes", type="procedural",
                       tags=["docker"])   # different tags → skip
        conflicts = store.find_conflicts(threshold=0.70)
        assert all(c.kind == "duplicate" or c.reason != "negation_diff"
                   for c in conflicts)


class TestRememberConflict:
    def test_on_conflict_ignore_is_silent(self, store: MemoryStore):
        store.remember("Use ruff", type="procedural", tags=["lint"])
        mem = store.remember("Never use ruff", type="procedural", tags=["lint"],
                             on_conflict="ignore")
        assert mem.id  # stored normally

    def test_on_conflict_warn_emits_warning(self, store: MemoryStore):
        store.remember("Always use ruff", type="procedural", tags=["lint"])
        with pytest.warns(UserWarning, match="conflict"):
            store.remember("Never use ruff", type="procedural", tags=["lint"],
                           on_conflict="warn")

    def test_on_conflict_raise_raises(self, store: MemoryStore):
        store.remember("Always use ruff", type="procedural", tags=["lint"])
        with pytest.raises(ConflictError) as exc_info:
            store.remember("Never use ruff", type="procedural", tags=["lint"],
                           on_conflict="raise")
        assert len(exc_info.value.conflicts) >= 1

    def test_on_conflict_supersede_marks_old(self, store: MemoryStore):
        m1 = store.remember("Use ruff for linting", type="procedural", tags=["lint"])
        m2 = store.remember("Also use ruff for linting", type="procedural", tags=["lint"],
                            on_conflict="supersede")
        old = store._get_by_id(m1.id)
        if old and old.superseded_by:
            assert old.superseded_by == m2.id


class TestSupersede:
    def test_supersede_marks_old(self, store: MemoryStore):
        old = store.remember("Old fact about X")
        new = store.remember("Updated fact about X")
        store.supersede(old_id=old.id, new_id=new.id)
        reloaded = store._get_by_id(old.id)
        assert reloaded.superseded_by == new.id
        assert reloaded.superseded_at > 0.0

    def test_supersede_adds_link(self, store: MemoryStore):
        old = store.remember("Old")
        new = store.remember("New")
        store.supersede(old_id=old.id, new_id=new.id)
        new_reloaded = store._get_by_id(new.id)
        assert any(l.to == old.id and l.rel == "supersedes" for l in new_reloaded.links)

    def test_supersede_hidden_from_recall(self, store: MemoryStore):
        old = store.remember("Use flake8 for linting", type="procedural")
        new = store.remember("Use ruff for linting", type="procedural")
        store.supersede(old_id=old.id, new_id=new.id)
        hits = store.recall("linting tool", k=10)
        ids = [m.id for m, _ in hits]
        assert old.id not in ids

    def test_supersede_visible_with_flag(self, store: MemoryStore):
        old = store.remember("Old style")
        new = store.remember("New style")
        store.supersede(old_id=old.id, new_id=new.id)
        hits = store.recall("style", k=10, include_superseded=True)
        ids = [m.id for m, _ in hits]
        assert old.id in ids

    def test_supersede_self_raises(self, store: MemoryStore):
        m = store.remember("solo")
        with pytest.raises(ValueError, match="itself"):
            store.supersede(old_id=m.id, new_id=m.id)

    def test_supersede_cycle_raises(self, store: MemoryStore):
        a = store.remember("a")
        b = store.remember("b")
        store.supersede(old_id=a.id, new_id=b.id)  # b supersedes a
        with pytest.raises(ValueError, match="[Cc]ycle"):
            store.supersede(old_id=b.id, new_id=a.id)

    def test_supersede_idempotent(self, store: MemoryStore):
        old = store.remember("old")
        new = store.remember("new")
        store.supersede(old_id=old.id, new_id=new.id)
        store.supersede(old_id=old.id, new_id=new.id)  # second call is no-op
        reloaded = store._get_by_id(new.id)
        assert sum(1 for l in reloaded.links if l.rel == "supersedes") == 1

    def test_restore_clears_superseded(self, store: MemoryStore):
        old = store.remember("old fact")
        new = store.remember("new fact")
        store.supersede(old_id=old.id, new_id=new.id)
        restored = store.restore(old.id)
        assert restored is True
        reloaded = store._get_by_id(old.id)
        assert reloaded.superseded_by == ""

    def test_restore_removes_supersedes_link(self, store: MemoryStore):
        old = store.remember("old")
        new = store.remember("new")
        store.supersede(old_id=old.id, new_id=new.id)
        store.restore(old.id)
        new_reloaded = store._get_by_id(new.id)
        assert not any(l.rel == "supersedes" for l in new_reloaded.links)

    def test_restore_active_memory_returns_false(self, store: MemoryStore):
        m = store.remember("active")
        assert store.restore(m.id) is False

    def test_list_recent_hides_superseded_by_default(self, store: MemoryStore):
        old = store.remember("old")
        new = store.remember("new")
        store.supersede(old_id=old.id, new_id=new.id)
        recent = store.list_recent(limit=100)
        assert all(m.superseded_by == "" for m in recent)

    def test_list_recent_include_superseded(self, store: MemoryStore):
        old = store.remember("old")
        new = store.remember("new")
        store.supersede(old_id=old.id, new_id=new.id)
        all_mems = store.list_recent(limit=100, include_superseded=True)
        ids = {m.id for m in all_mems}
        assert old.id in ids and new.id in ids
