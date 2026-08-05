"""Tests for the audit journal."""

from __future__ import annotations

import gzip
import json
import time
from pathlib import Path

import pytest

import ai_houkai.memory_system.journal as journal_mod
from ai_houkai.memory_system import Journal, JournalEntry, MemoryStore


def test_remember_writes_one_entry(store: MemoryStore) -> None:
    before = list(store.journal.read())
    assert before == []
    mem = store.remember(text="hello")
    entries = list(store.journal.read())
    assert len(entries) == 1
    e = entries[0]
    assert e.op == "remember"
    assert e.id == mem.id
    assert e.after and e.after["text"] == "hello"
    assert e.before is None


def test_forget_captures_before(store: MemoryStore) -> None:
    mem = store.remember(text="erase me")
    store.forget(mem.id)
    entries = [e for e in store.journal.read() if e.op == "forget"]
    assert len(entries) == 1
    assert entries[0].before["text"] == "erase me"


def test_supersede_records_old_and_new(store: MemoryStore) -> None:
    a = store.remember(text="use ruff")
    b = store.remember(text="use flake8")
    store.supersede(old_id=a.id, new_id=b.id)
    sup = [e for e in store.journal.read() if e.op == "supersede"]
    assert len(sup) == 1
    assert sup[0].meta["old_id"] == a.id
    assert sup[0].meta["new_id"] == b.id


def test_link_and_unlink_are_journaled(store: MemoryStore) -> None:
    a = store.remember(text="rule")
    b = store.remember(text="example")
    store.link(b.id, a.id, rel="example_of")
    store.unlink(b.id, a.id, rel="example_of")
    ops = [e.op for e in store.journal.read() if e.op in ("link", "unlink")]
    assert ops == ["link", "unlink"]


def test_actor_propagation(store: MemoryStore) -> None:
    with store.as_actor("reflection"):
        store.remember(text="from reflection")
    actors = [e.actor for e in store.journal.read() if e.op == "remember"]
    assert "reflection" in actors
    # Outside the context manager actor reverts
    store.remember(text="from default")
    assert actors[-1] == "reflection"
    new_actors = [e.actor for e in store.journal.read() if e.op == "remember"]
    assert new_actors[-1] != "reflection"


def test_disabled_journal_writes_nothing(tmp_path: Path) -> None:
    s = MemoryStore(
        path=str(tmp_path / "chroma"), collection="test_disabled",
        journal_enabled=False,
    )
    try:
        s.remember(text="silent")
        assert not s.journal.path.exists()
    finally:
        s.client.close()


def test_corrupt_line_is_skipped(tmp_path: Path) -> None:
    j = Journal(tmp_path / "j.log")
    j.append(JournalEntry(
        ts=1.0, op="remember", actor="t", id="abc",
        before=None, after={"text": "x"}, meta={},
    ))
    # Append a malformed (truncated) line manually
    with open(j.path, "a") as f:
        f.write('{"ts":2.0,"op":"forge')   # truncated, no newline
    entries = list(j.read())
    assert len(entries) == 1
    assert entries[0].op == "remember"


def test_undo_remember_deletes_memory(store: MemoryStore) -> None:
    mem = store.remember(text="undo me")
    entry = next(e for e in store.journal.read() if e.op == "remember" and e.id == mem.id)
    assert store.undo(entry) is True
    assert store._get_by_id(mem.id) is None
    # An "undo" record should have been written
    assert any(e.op == "undo" for e in store.journal.read())


def test_undo_forget_restores_memory(store: MemoryStore) -> None:
    mem = store.remember(text="bring me back")
    mid = mem.id
    store.forget(mid)
    forget_entry = next(e for e in store.journal.read() if e.op == "forget")
    assert store.undo(forget_entry) is True
    restored = store._get_by_id(mid)
    assert restored is not None
    assert restored.text == "bring me back"


def test_undo_supersede(store: MemoryStore) -> None:
    a = store.remember(text="alpha")
    b = store.remember(text="alpha v2")
    store.supersede(old_id=a.id, new_id=b.id)
    entry = next(e for e in store.journal.read() if e.op == "supersede")
    assert store.undo(entry) is True
    assert (store._get_by_id(a.id)).superseded_by == ""


def test_undo_restore_after_forget_returns_false(store: MemoryStore) -> None:
    # supersede → restore → forget: undoing the restore would re-supersede a
    # memory that no longer exists. Must return False, not raise KeyError.
    a = store.remember(text="old fact")
    b = store.remember(text="new fact")
    store.supersede(old_id=a.id, new_id=b.id)
    store.restore(a.id)
    store.forget(a.id)
    entry = next(e for e in store.journal.read() if e.op == "restore")
    assert store.undo(entry) is False


def test_undo_link_removes_edge(store: MemoryStore) -> None:
    a = store.remember(text="a")
    b = store.remember(text="b")
    store.link(a.id, b.id, rel="related")
    entry = next(e for e in store.journal.read() if e.op == "link")
    assert store.undo(entry) is True
    src = store._get_by_id(a.id)
    assert all(not (l.to == b.id and l.rel == "related") for l in src.links)


def test_rotate_when_size_exceeded(tmp_path: Path) -> None:
    j = Journal(tmp_path / "j.log", rotate_mb=0.001)   # ~1 KB
    for i in range(100):
        j.append(JournalEntry(
            ts=time.time(), op="remember", actor="t",
            id=f"id{i:08d}", before=None,
            after={"text": "x" * 200}, meta={},
        ))
    # Force the rotation check by simulating reaching the interval
    j._writes_since_check = j._ROTATE_CHECK_EVERY
    j.append(JournalEntry(
        ts=time.time(), op="remember", actor="t", id="trigger",
        before=None, after={"text": "x"}, meta={},
    ))
    archives = list(tmp_path.glob("j-*.log.gz"))
    assert archives, "expected at least one rotated archive"
    with gzip.open(archives[0], "rt") as f:
        first_line = f.readline()
    assert json.loads(first_line)["op"] == "remember"


def test_read_filters(store: MemoryStore) -> None:
    a = store.remember(text="alpha")
    b = store.remember(text="beta")
    store.forget(b.id)
    # by op
    rems = list(store.journal.read(op="remember"))
    assert len(rems) == 2
    forgets = list(store.journal.read(op="forget"))
    assert len(forgets) == 1
    # by memory_id
    by_id = list(store.journal.read(memory_id=a.id))
    assert len(by_id) == 1
    assert by_id[0].op == "remember"


def test_undo_unlink_restores_all_parallel_rels(store: MemoryStore) -> None:
    """Regression: unlink(rel=None) removing several differently-typed edges
    used to undo as a single 'related' link."""
    a = store.remember("undo src")
    b = store.remember("undo dst")
    store.link(a.id, b.id, rel="related")
    store.link(a.id, b.id, rel="example_of")
    assert store.unlink(a.id, b.id, rel=None) == 2

    entry = list(store.journal.read(op="unlink"))[-1]
    assert entry.meta["removed_rels"] == ["related", "example_of"]
    assert store.undo(entry) is True

    restored = sorted(l.rel for l in store._get_by_id(a.id).links if l.to == b.id)
    assert restored == ["example_of", "related"]


def test_undo_unlink_legacy_entry_without_removed_rels(store: MemoryStore) -> None:
    """Entries journaled before removed_rels existed fall back to meta.rel."""
    a = store.remember("legacy src")
    b = store.remember("legacy dst")
    store.link(a.id, b.id, rel="refines")
    store.unlink(a.id, b.id, rel="refines")
    legacy = JournalEntry(
        ts=1.0, op="unlink", actor="lib", id=a.id, before=None, after=None,
        meta={"src_id": a.id, "dst_id": b.id, "rel": "refines", "removed": 1},
    )
    assert store.undo(legacy) is True
    assert [l.rel for l in store._get_by_id(a.id).links if l.to == b.id] == ["refines"]


def test_undo_edit_restores_previous_state(store: MemoryStore) -> None:
    mem = store.remember(text="version A", importance=0.4)
    store.edit(mem.id, text="version B")
    entry = next(e for e in store.journal.read() if e.op == "edit")
    assert store.undo(entry) is True
    assert store._get_by_id(mem.id).text == "version A"


def test_undo_edit_refuses_after_later_edit(store: MemoryStore) -> None:
    """Undoing an edit whose target has since been edited again would clobber
    the newer state — it must return False and change nothing."""
    mem = store.remember(text="state A")
    store.edit(mem.id, text="state B")
    store.edit(mem.id, text="state C")
    first_edit = next(e for e in store.journal.read()
                      if e.op == "edit" and (e.after or {}).get("text") == "state B")
    assert store.undo(first_edit) is False
    assert store._get_by_id(mem.id).text == "state C"


def test_undo_edit_refuses_after_supersede(store: MemoryStore) -> None:
    """undo(edit) must not resurrect a memory that was superseded after the
    edit — the restore would silently clear superseded_by."""
    old = store.remember(text="fact v1")
    store.edit(old.id, text="fact v1 (fixed)")
    new = store.remember(text="fact v2")
    store.supersede(old_id=old.id, new_id=new.id)
    entry = next(e for e in store.journal.read() if e.op == "edit")
    assert store.undo(entry) is False
    assert store._get_by_id(old.id).superseded_by == new.id


def test_undo_edit_tolerates_access_bumps(store: MemoryStore) -> None:
    """A recall touch between edit and undo only changes volatile access
    tracking — that must not block the undo."""
    mem = store.remember(text="often recalled fact")
    store.edit(mem.id, text="often recalled fact (edited)")
    store.recall("often recalled", k=1)   # bumps access_count/last_accessed
    entry = next(e for e in store.journal.read() if e.op == "edit")
    assert store.undo(entry) is True
    assert store._get_by_id(mem.id).text == "often recalled fact"


def _entry(i: int) -> JournalEntry:
    return JournalEntry(ts=1000.0 + i, op="remember", actor="t", id=f"id-{i}",
                        before=None, after={"text": "x" * 200}, meta={})


def test_rotate_renames_then_compresses(tmp_path: Path) -> None:
    j = Journal(tmp_path / "j.log", rotate_mb=64)
    for i in range(10):
        j.append(_entry(i))
    j._rotate()
    # Active file was renamed away — nothing left to lose to a truncate.
    assert not (tmp_path / "j.log").exists()
    archives = list(tmp_path.glob("j-*.log.gz"))
    assert len(archives) == 1
    assert not list(tmp_path.glob("j-*.log"))       # rotated file cleaned up
    # New appends land in a fresh active file.
    j.append(_entry(99))
    assert (tmp_path / "j.log").exists()
    ids = [e.id for e in j.read(include_archives=True)]
    assert ids == [f"id-{i}" for i in range(10)] + ["id-99"]


def test_rotate_crash_mid_compress_loses_nothing(
    tmp_path: Path, monkeypatch
) -> None:
    """If compression dies after the rename, the plain rotated file remains
    and read(include_archives=True) still returns every entry."""
    j = Journal(tmp_path / "j.log", rotate_mb=64)
    for i in range(5):
        j.append(_entry(i))

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(journal_mod.gzip, "open", boom)
    with pytest.raises(OSError):
        j._rotate()
    monkeypatch.undo()

    orphans = list(tmp_path.glob("j-*.log"))
    assert len(orphans) == 1                        # rename happened
    assert not (tmp_path / "j.log").exists()
    ids = [e.id for e in j.read(include_archives=True)]
    assert ids == [f"id-{i}" for i in range(5)]
    # And appends keep working on a fresh active file.
    j.append(_entry(9))
    assert [e.id for e in j.read(include_archives=True)][-1] == "id-9"


def test_undo_of_a_trash_clears_the_trash_entry(store: MemoryStore) -> None:
    """`trash` deletes through `forget`, so undoing it resurrects the row — but
    the trash entry stayed behind, listing a memory that is live again. The
    stale entry then makes `trash_restore` a silent no-op on the newer row, and
    `trash_list` shows a recovery point that recovers nothing."""
    mem = store.remember(text="binned then reinstated")
    store.trash(mem.id)
    entry = next(e for e in store.journal.read()
                 if e.op == "forget" and e.id == mem.id)

    assert store.undo(entry) is True
    assert store.get(mem.id) is not None
    assert store.trash_list() == []


def test_undo_of_a_plain_forget_leaves_the_trash_alone(store: MemoryStore) -> None:
    """Only the entry for the undone memory goes — anything else in the bin
    stays recoverable."""
    kept = store.remember(text="still in the bin")
    store.trash(kept.id)
    gone = store.remember(text="plainly forgotten")
    store.forget(gone.id)
    entry = next(e for e in store.journal.read()
                 if e.op == "forget" and e.id == gone.id)

    assert store.undo(entry) is True
    assert [e.memory_id for e in store.trash_list()] == [kept.id]


def test_trash_restore_does_not_clobber_a_live_memory(store: MemoryStore) -> None:
    """Belt and braces: even if a stale entry appears some other way (an import
    that recreates a trashed id, say), restoring it must not roll a live memory
    back to an older snapshot."""
    mem = store.remember(text="original text", tags=["before"])
    store.trash(mem.id)
    entry = next(e for e in store.journal.read()
                 if e.op == "forget" and e.id == mem.id)
    store.undo(entry)
    store.edit(mem.id, tags=["after"])

    store.trash_restore(mem.id)
    assert store.get(mem.id).tags == ["after"]
