"""A collection can lose its vectors while keeping every word of its text.

Chroma splits a collection across two segments: documents and metadata in
sqlite, the embeddings in HNSW index files on disk. Lose only the second —
deleted files, an interrupted flush, a data directory copied while the service
was writing to it — and the collection still counts, lists and exports its
memories while every vector read either fails or answers with nothing. These
tests damage a real store that way and pin both halves of the response:
recall refuses to answer instead of reporting a false empty, and
``rebuild_vectors`` puts the index back from the surviving text.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ai_houkai.cli.main import app
from ai_houkai.memory_system import MemoryStore, VectorIndexError
from ai_houkai.testing import FakeEmbedder


def _lose_the_vector_index(root: Path) -> None:
    """Delete the HNSW segment files and compact the write-ahead log.

    Deleting the files alone proves nothing: Chroma replays the WAL on open and
    rebuilds the index behind your back. It is only once the queue has been
    purged — which Chroma itself does after a flush — that the segment files
    are the sole copy of the vectors and losing them actually loses them.
    """
    db = sqlite3.connect(f"{root}/chroma.sqlite3")
    try:
        segments = [r[0] for r in
                    db.execute("select id from segments where scope='VECTOR'")]
        db.execute("delete from embeddings_queue")
        db.commit()
    finally:
        db.close()
    for seg in segments:
        shutil.rmtree(root / seg, ignore_errors=True)


@pytest.fixture()
def damaged(tmp_path):
    """A store holding six memories whose vector index has been lost."""
    root = tmp_path / "chroma"
    s = MemoryStore(path=str(root), collection="dmg")
    ids = [s.remember(text=f"note number {i}", tags=[f"t{i}"]).id
           for i in range(6)]
    s.link(ids[1], ids[0], rel="refines")
    s.supersede(old_id=ids[4], new_id=ids[5])
    s.client.close()

    _lose_the_vector_index(root)
    s = MemoryStore(path=str(root), collection="dmg")
    yield s, ids
    s.client.close()


def test_text_survives_the_loss(damaged) -> None:
    """The premise the repair rests on: everything but the vectors is intact."""
    s, _ = damaged
    assert s.count() == 6
    assert len(s.list_recent(limit=10)) == 5      # the superseded row is hidden
    assert s.collection.get(include=["documents", "metadatas"])["ids"]


def test_vector_index_ok_answers_without_running_a_query(damaged, store) -> None:
    """What a sweep over many collections asks, to repair only the broken ones
    instead of re-embedding everything."""
    broken, _ = damaged
    assert broken.vector_index_ok() is False

    assert store.vector_index_ok() is True          # empty: nothing to index
    store.remember(text="a healthy note")
    assert store.vector_index_ok() is True


def test_vector_index_ok_is_true_again_after_a_rebuild(damaged) -> None:
    s, _ = damaged
    s.rebuild_vectors()
    assert s.vector_index_ok() is True


def test_recall_refuses_rather_than_reporting_nothing(damaged) -> None:
    s, _ = damaged
    with pytest.raises(VectorIndexError, match="rebuild_vectors"):
        s.recall("note", k=3)


def test_find_conflicts_reports_the_index_not_a_chroma_internal_error(
        damaged) -> None:
    s, _ = damaged
    with pytest.raises(VectorIndexError, match="rebuild_vectors"):
        s.find_conflicts()


def test_a_filtered_query_that_matches_nothing_is_not_a_broken_index(
        store: MemoryStore) -> None:
    """The guard must not fire on a healthy store whose filter excludes
    everything — that is an empty answer, not a missing index."""
    store.remember(text="a semantic note", type="semantic")
    assert store.recall("note", k=3, type="procedural") == []
    assert store.recall("note", k=3, tag="nonexistent") == []
    assert store.recall("note", k=3, min_importance=0.99) == []


def test_an_empty_collection_is_not_a_broken_index(store: MemoryStore) -> None:
    assert store.recall("anything", k=3) == []


def test_the_empty_result_check_does_not_re_embed_the_query(tmp_path) -> None:
    """The check runs on every empty result, so it must not cost a second
    embedding — that would tax the ordinary empty case (a filter matching
    nothing on a perfectly healthy collection) with the slowest thing recall
    does."""
    inner = FakeEmbedder()
    calls: list[list[str]] = []

    def counting(texts):
        calls.append(list(texts))
        return inner(texts)

    s = MemoryStore(path=str(tmp_path / "chroma"), collection="probe",
                    embedding_function=counting)
    try:
        s.remember(text="a semantic note")
        calls.clear()
        assert s.recall("note", k=3, type="procedural") == []
        assert len(calls) == 1, f"query embedded {len(calls)} times, expected 1"
    finally:
        s.client.close()


def test_rebuild_restores_recall_and_keeps_every_field(damaged) -> None:
    s, ids = damaged
    summary = s.rebuild_vectors()
    assert summary.count == 6
    assert summary.elapsed >= 0

    assert len(s.recall("note", k=6)) == 5        # superseded stays excluded
    assert s.find_conflicts() is not None         # vectors readable again
    assert s.count() == 6

    mem = s.get(ids[1])
    assert [(link.to, link.rel) for link in mem.links] == [(ids[0], "refines")]
    assert mem.tags == ["t1"]
    assert mem.created_at > 0
    assert s.get(ids[4]).superseded_by == ids[5]


def test_rebuild_removes_its_archive_only_on_success(damaged) -> None:
    s, _ = damaged
    summary = s.rebuild_vectors()
    assert summary.backup is not None
    assert not summary.backup.exists(), "archive kept after a clean rebuild"


def test_rebuild_keeps_the_archive_when_it_cannot_finish(damaged, tmp_path,
                                                        monkeypatch) -> None:
    """The rebuild replaces the collection, so a failure part-way through must
    leave the text recoverable from somewhere."""
    s, _ = damaged
    archive = tmp_path / "safety.ahkai"

    def boom(*_a, **_k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(s.client, "delete_collection", boom)
    with pytest.raises(VectorIndexError, match="archive was kept"):
        s.rebuild_vectors(backup_path=archive)
    assert archive.exists(), "the only copy of the text was discarded"


def test_rebuild_is_journaled_without_rewriting_history(damaged) -> None:
    """Content is preserved, so the rebuild must not journal a wipe: a `nuke`
    entry would make state_at() clear everything it had reconstructed."""
    s, _ = damaged
    before = len(s.state_at(9e9))
    s.rebuild_vectors()
    ops = [e.op for e in s.journal.read()]
    assert "rebuild_vectors" in ops
    assert "nuke" not in ops
    assert len(s.state_at(9e9)) == before


def test_export_degrades_to_vectorless_rather_than_failing(damaged, tmp_path) -> None:
    """Backing up a collection whose vectors are gone must still work: the text
    is what a backup is for, and import re-embeds it. (test_export_import
    covers the same fallback at the seam; this proves it against real damage.)"""
    s, _ = damaged
    out = tmp_path / "dmg.ahkai"
    with pytest.warns(UserWarning, match="exporting without vectors"):
        summary = s.export(out, include_superseded=True)
    assert summary.count == 6
    assert summary.vectors_included is False
    assert out.exists()


def test_doctor_reports_a_lost_index_instead_of_ready(damaged, tmp_path) -> None:
    """The diagnosis has to name this: doctor read the same row's embedding to
    check the embedding dimension, swallowed the failure, skipped the check and
    printed "OK — ready" for a collection that could not answer one query."""
    s, _ = damaged
    s.client.close()                      # the CLI opens its own client
    result = CliRunner().invoke(
        app, ["--store", str(tmp_path / "chroma"), "--collection", "dmg",
              "doctor", "--json"])
    assert result.exit_code == 1, result.output
    report = json.loads(result.output[result.output.index("{"):])
    index = next(c for c in report["checks"] if c["name"] == "vector_index")
    assert index["ok"] is False
    assert "rebuild-vectors" in index["error"]
    assert report["ok"] is False


def test_rebuild_vectors_cli_repairs_the_store(damaged, tmp_path) -> None:
    s, _ = damaged
    s.client.close()
    result = CliRunner().invoke(
        app, ["--store", str(tmp_path / "chroma"), "--collection", "dmg",
              "rebuild-vectors", "--yes"])
    assert result.exit_code == 0, result.output
    assert "6 memories re-embedded" in result.output

    repaired = MemoryStore(path=str(tmp_path / "chroma"), collection="dmg")
    try:
        assert len(repaired.recall("note", k=6)) == 5
    finally:
        repaired.client.close()


def test_rebuild_works_on_a_healthy_store(store: MemoryStore) -> None:
    """Also the way to re-embed a collection after changing embedders, so it
    has to be safe to run when nothing is wrong."""
    store.remember(text="first fact")
    store.remember(text="second fact")
    assert store.rebuild_vectors().count == 2
    assert len(store.recall("fact", k=2)) == 2
