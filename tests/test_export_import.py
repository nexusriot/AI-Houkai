"""Tests for portable export / import (.ahkai format)."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from ai_houkai.memory_system import MemoryStore
from ai_houkai.memory_system import ImportConflictError

def _new_store(tmp_path: Path, name: str = "t") -> MemoryStore:
    return MemoryStore(path=str(tmp_path / name), collection=f"col_{name}")


def test_export_roundtrip(store: MemoryStore, tmp_path: Path) -> None:
    a = store.remember(text="first thing", tags=["x"], importance=0.7)
    b = store.remember(text="second thing", tags=["y"], importance=0.3)
    out = tmp_path / "dump.ahkai"
    summary = store.export(out)
    assert summary.count == 2
    assert summary.path == out
    assert out.exists()

    target = _new_store(tmp_path, "t2")
    try:
        res = target.import_(out)
        assert res.imported == 2
        assert res.skipped == 0
        assert target.count() == 2
        # Verify text and tags survived
        mems = {m.id: m for m in target.list_recent(limit=10)}
        assert mems[a.id].text == "first thing"
        assert "x" in mems[a.id].tags
    finally:
        target.client.close()


def test_export_filters(store: MemoryStore, tmp_path: Path) -> None:
    store.remember(text="episodic one", type="episodic")
    store.remember(text="semantic one", type="semantic", tags=["keep"])
    store.remember(text="semantic two", type="semantic", tags=["drop"])
    out = tmp_path / "filtered.ahkai"
    summary = store.export(out, types=["semantic"], tags=["keep"])
    assert summary.count == 1


def test_export_omits_superseded_by_default(store: MemoryStore, tmp_path: Path) -> None:
    a = store.remember(text="old")
    b = store.remember(text="new")
    store.supersede(old_id=a.id, new_id=b.id)
    out = tmp_path / "x.ahkai"
    summary = store.export(out)
    assert summary.count == 1
    out2 = tmp_path / "y.ahkai"
    s2 = store.export(out2, include_superseded=True)
    assert s2.count == 2


def test_header_present_and_valid(store: MemoryStore, tmp_path: Path) -> None:
    store.remember(text="hi")
    out = tmp_path / "h.ahkai"
    store.export(out)
    with gzip.open(out, "rt") as f:
        header = json.loads(f.readline())
    assert header["format"] == "ai-houkai/export"
    assert header["version"] == 1
    assert header["source"]["embedding_model"] == store.embedding_model
    assert header["source"]["embedding_dim"] > 0
    assert header["source"]["count"] == 1


def test_import_skip_existing(store: MemoryStore, tmp_path: Path) -> None:
    mem = store.remember(text="dupe")
    out = tmp_path / "d.ahkai"
    store.export(out)
    res = store.import_(out, on_conflict="skip")
    assert res.imported == 0
    assert res.skipped == 1
    assert store.count() == 1


def test_import_overwrite(store: MemoryStore, tmp_path: Path) -> None:
    mem = store.remember(text="original", importance=0.2)
    out = tmp_path / "ov.ahkai"
    store.export(out)
    # Tamper with the stored version
    mem2 = store._get_by_id(mem.id)
    mem2.text = "modified locally"
    store.collection.update(ids=[mem.id], documents=["modified locally"],
                            metadatas=[mem2.to_metadata()])
    res = store.import_(out, on_conflict="overwrite")
    assert res.overwritten == 1
    assert store._get_by_id(mem.id).text == "original"


def test_import_rename(store: MemoryStore, tmp_path: Path) -> None:
    mem = store.remember(text="copy me")
    out = tmp_path / "r.ahkai"
    store.export(out)
    res = store.import_(out, on_conflict="rename")
    assert res.renamed == 1
    assert store.count() == 2


def test_import_error_policy(store: MemoryStore, tmp_path: Path) -> None:
    store.remember(text="collide")
    out = tmp_path / "e.ahkai"
    store.export(out)
    with pytest.raises(ImportConflictError):
        store.import_(out, on_conflict="error")


def test_import_idempotent(store: MemoryStore, tmp_path: Path) -> None:
    store.remember(text="a")
    store.remember(text="b")
    out = tmp_path / "i.ahkai"
    store.export(out)
    target = _new_store(tmp_path, "tt")
    try:
        r1 = target.import_(out)
        assert r1.imported == 2
        r2 = target.import_(out)
        assert r2.imported == 0
        assert r2.skipped == 2
        assert target.count() == 2
    finally:
        target.client.close()


def test_import_rejects_model_mismatch(store: MemoryStore, tmp_path: Path) -> None:
    store.remember(text="hello")
    out = tmp_path / "m.ahkai"
    store.export(out)
    # Forge a header with a different model
    with gzip.open(out, "rt") as f:
        lines = f.readlines()
    header = json.loads(lines[0])
    header["source"]["embedding_model"] = "some-other-model"
    lines[0] = json.dumps(header) + "\n"
    with gzip.open(out, "wt") as f:
        f.writelines(lines)
    target = _new_store(tmp_path, "x")
    try:
        with pytest.raises(ImportError):
            target.import_(out)
        # With regenerate_vectors, succeeds
        res = target.import_(out, regenerate_vectors=True)
        assert res.imported == 1
        assert res.vectors_regenerated
    finally:
        target.client.close()


def test_import_dry_run(store: MemoryStore, tmp_path: Path) -> None:
    store.remember(text="dry one")
    store.remember(text="dry two")
    out = tmp_path / "dr.ahkai"
    store.export(out)
    target = _new_store(tmp_path, "dr")
    try:
        res = target.import_(out, dry_run=True)
        assert res.imported == 2
        assert target.count() == 0
    finally:
        target.client.close()


def test_export_journal_entry(store: MemoryStore, tmp_path: Path) -> None:
    store.remember(text="x")
    out = tmp_path / "j.ahkai"
    store.export(out)
    exports = [e for e in store.journal.read() if e.op == "export"]
    assert len(exports) == 1
    assert exports[0].meta["count"] == 1


def test_import_journals_each_row(store: MemoryStore, tmp_path: Path) -> None:
    store.remember(text="row a")
    store.remember(text="row b")
    out = tmp_path / "ji.ahkai"
    store.export(out)
    target = _new_store(tmp_path, "ji")
    try:
        target.import_(out)
        imports = [e for e in target.journal.read() if e.op == "import"]
        assert len(imports) == 2
        for e in imports:
            assert e.actor == "import"
    finally:
        target.client.close()


def test_import_error_policy_is_atomic(store: MemoryStore, tmp_path: Path) -> None:
    """on_conflict="error" must pre-scan: a raised conflict leaves the target
    store completely untouched (no partial import of the non-colliding rows)."""
    kept = store.remember(text="already here")
    out = tmp_path / "atomic.ahkai"
    store.export(out)
    store.remember(text="brand new row")
    out2 = tmp_path / "atomic2.ahkai"
    store.export(out2)  # colliding row (kept) + one new row

    target = _new_store(tmp_path, "atomic")
    try:
        # Seed the collision, then snapshot the state.
        target.import_(out)
        assert target.count() == 1
        with pytest.raises(ImportConflictError):
            target.import_(out2, on_conflict="error")
        assert target.count() == 1              # the new row was NOT written
        assert target._get_by_id(kept.id) is not None
        imports = [e for e in target.journal.read() if e.op == "import"]
        assert len(imports) == 1                # only the seeding import
    finally:
        target.client.close()


def test_import_error_policy_detects_duplicate_ids_within_file(
    store: MemoryStore, tmp_path: Path
) -> None:
    mem = store.remember(text="dup me")
    out = tmp_path / "dup.ahkai"
    store.export(out)
    # Append a second row with the same id.
    with gzip.open(out, "rt") as f:
        lines = f.readlines()
    lines.append(lines[1])
    with gzip.open(out, "wt") as f:
        f.writelines(lines)

    target = _new_store(tmp_path, "dup")
    try:
        with pytest.raises(ImportConflictError):
            target.import_(out, on_conflict="error")
        assert target.count() == 0              # nothing written at all
    finally:
        target.client.close()
    assert mem.id  # silence unused warning


def test_vectorless_export_importable_across_models(
    store: MemoryStore, tmp_path: Path
) -> None:
    """A vectorless file re-embeds on import anyway, so a model mismatch must
    not require regenerate_vectors."""
    store.remember(text="portable fact")
    out = tmp_path / "nv.ahkai"
    store.export(out, include_vectors=False)
    # Forge a different source model, as if exported elsewhere.
    with gzip.open(out, "rt") as f:
        lines = f.readlines()
    header = json.loads(lines[0])
    header["source"]["embedding_model"] = "some-other-model"
    lines[0] = json.dumps(header) + "\n"
    with gzip.open(out, "wt") as f:
        f.writelines(lines)

    target = _new_store(tmp_path, "nv")
    try:
        res = target.import_(out)               # no regenerate_vectors needed
        assert res.imported == 1
        assert res.vectors_regenerated is True
        assert target.count() == 1
        hits = target.recall("portable", k=1)
        assert hits and hits[0][0].text == "portable fact"
    finally:
        target.client.close()


def test_vectorless_import_same_model_reports_regenerated(
    store: MemoryStore, tmp_path: Path
) -> None:
    store.remember(text="no vectors here")
    out = tmp_path / "nv2.ahkai"
    store.export(out, include_vectors=False)
    target = _new_store(tmp_path, "nv2")
    try:
        res = target.import_(out)
        assert res.imported == 1
        assert res.vectors_regenerated is True  # rows carried no vectors
    finally:
        target.client.close()


class _BrokenVectorSegment:
    """Wraps a Chroma collection so any get() asking for embeddings raises the
    way a missing/corrupt HNSW segment does in production."""

    def __init__(self, inner) -> None:
        self._inner = inner

    def get(self, *args, **kwargs):
        if "embeddings" in (kwargs.get("include") or []):
            raise RuntimeError(
                "Error executing plan: Internal error: Error creating hnsw "
                "segment reader: Nothing found on disk")
        return self._inner.get(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def test_export_degrades_to_vectorless_on_broken_vector_segment(
    store: MemoryStore, tmp_path: Path
) -> None:
    """Chroma's HNSW files can go missing on disk, making every embeddings
    read raise. Export must fall back to an honest include_vectors=False
    archive (which import_ re-embeds) instead of failing the whole backup."""
    store.remember(text="survives a broken vector segment")
    orig = store.collection
    store.collection = _BrokenVectorSegment(orig)
    out = tmp_path / "broken.ahkai"
    try:
        with pytest.warns(UserWarning, match="exporting without vectors"):
            summary = store.export(out, include_superseded=True)
    finally:
        store.collection = orig
    assert summary.count == 1
    assert summary.vectors_included is False

    with gzip.open(out, "rt") as f:
        header = json.loads(f.readline())
        row = json.loads(f.readline())
    assert header["options"]["include_vectors"] is False
    assert header["source"]["embedding_dim"] == 0
    assert "vector" not in row

    target = _new_store(tmp_path, "brk")
    try:
        res = target.import_(out)               # no regenerate_vectors needed
        assert res.imported == 1
        assert res.vectors_regenerated is True
        hits = target.recall("broken vector segment", k=1)
        assert hits and hits[0][0].text == "survives a broken vector segment"
    finally:
        target.client.close()


def test_export_vectorless_request_still_raises_on_get_failure(
    store: MemoryStore, tmp_path: Path
) -> None:
    """The fallback only covers the embeddings read: a store whose metadata
    segment is broken too must surface the error, not write an empty file."""
    store.remember(text="anything")
    orig = store.collection

    class _AllBroken(_BrokenVectorSegment):
        def get(self, *args, **kwargs):
            raise RuntimeError("Nothing found on disk")

    store.collection = _AllBroken(orig)
    try:
        with pytest.raises(RuntimeError):
            store.export(tmp_path / "nope.ahkai", include_vectors=False)
    finally:
        store.collection = orig


def test_rename_repoints_links_between_imported_rows(
        store: MemoryStore, tmp_path: Path) -> None:
    """on_conflict="rename" gives a colliding row a fresh id — but the rest
    of the file still references the old one, which now resolves to the
    unrelated pre-existing memory. Those references must follow the rename."""
    src = _new_store(tmp_path, "src")
    tgt = _new_store(tmp_path, "tgt")

    hub = src.remember(text="the hub")
    spoke = src.remember(text="the spoke")
    src.link(spoke.id, hub.id, rel="refines")
    out = tmp_path / "linked.ahkai"
    src.export(out)

    # Squat the hub's id in the target with an unrelated memory.
    squatter = tgt.remember(text="unrelated squatter")
    tgt.collection.delete(ids=[squatter.id])
    tgt.collection.add(ids=[hub.id], documents=["unrelated squatter"],
                       metadatas=[{**squatter.to_metadata()}])

    summary = tgt.import_(out, on_conflict="rename")
    assert summary.renamed == 1 and summary.imported == 1

    imported_spoke = tgt.get(spoke.id)
    (link,) = imported_spoke.links
    assert link.to != hub.id, "link must not point at the squatter"
    renamed_hub = tgt.get(link.to)
    assert renamed_hub is not None and renamed_hub.text == "the hub"
    src.client.close()
    tgt.client.close()


def test_rename_repoints_superseded_by(store: MemoryStore,
                                       tmp_path: Path) -> None:
    src = _new_store(tmp_path, "src2")
    tgt = _new_store(tmp_path, "tgt2")

    old = src.remember(text="old belief")
    new = src.remember(text="new belief")
    src.supersede(old.id, new.id)
    out = tmp_path / "superseded.ahkai"
    src.export(out, include_superseded=True)

    squatter = tgt.remember(text="squats the new id")
    tgt.collection.delete(ids=[squatter.id])
    tgt.collection.add(ids=[new.id], documents=["squats the new id"],
                       metadatas=[{**squatter.to_metadata()}])

    tgt.import_(out, on_conflict="rename")
    imported_old = tgt.get(old.id)
    assert imported_old.superseded_by != new.id
    assert tgt.get(imported_old.superseded_by).text == "new belief"
    src.client.close()
    tgt.client.close()
