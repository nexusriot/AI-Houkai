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
