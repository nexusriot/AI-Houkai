"""The derived SQLite sidecar index (E).

Two invariants matter more than any speedup:

  1. **Off by default.** An existing store has no index; enabling it silently
     would make list/neighbors read an empty table.
  2. **A cache, never a source of truth.** Every read has a scan fallback, and
     an index that disagrees with Chroma is disabled rather than trusted — a
     stale index must degrade to "slower", never to "wrong".
"""

from __future__ import annotations

import json
import sqlite3

import pytest
from typer.testing import CliRunner

from ai_houkai.cli.main import app
from ai_houkai.memory_system import HybridWeights, MemoryStore
from ai_houkai.memory_system.sidecar import (
    SidecarIndex,
    _fts_query,
    fts5_available,
)
from ai_houkai.testing import FakeEmbedder

pytestmark = pytest.mark.skipif(
    not fts5_available(),
    reason="this SQLite build lacks FTS5; the lexical channel is unavailable",
)


@pytest.fixture()
def indexed(tmp_path):
    """A store with the sidecar enabled."""
    store = MemoryStore(
        path=str(tmp_path / "chroma"), collection="sidecar",
        embedding_function=FakeEmbedder(), index="sqlite",
    )
    yield store
    store.client.close()


@pytest.fixture()
def plain(tmp_path):
    """A store without the sidecar — the fallback path."""
    store = MemoryStore(
        path=str(tmp_path / "chroma2"), collection="plain",
        embedding_function=FakeEmbedder(),
    )
    yield store
    store.client.close()


class TestOptIn:
    def test_off_by_default(self, plain):
        assert plain.index is None

    def test_rejects_an_unknown_mode(self, tmp_path):
        with pytest.raises(ValueError, match="index must be 'sqlite' or None"):
            MemoryStore(path=str(tmp_path / "c"), collection="unknown_mode",
                        embedding_function=FakeEmbedder(), index="postgres")

    def test_env_var_enables_it(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AI_HOUKAI_INDEX", "sqlite")
        store = MemoryStore(path=str(tmp_path / "c"), collection="envidx",
                            embedding_function=FakeEmbedder())
        try:
            assert store.index is not None
        finally:
            store.client.close()

    def test_index_file_lands_beside_the_store(self, indexed, tmp_path):
        assert indexed.index.path.exists()
        assert indexed.index.path.name == "sidecar.index.sqlite3"


class TestWriteThrough:
    def test_remember_indexes_the_row(self, indexed):
        mem = indexed.remember("indexed subject", tags=["a", "b"],
                               type="procedural", importance=0.8)
        rows = indexed.index._conn.execute(
            "SELECT * FROM memories WHERE id = ?", (mem.id,)).fetchall()
        assert len(rows) == 1
        assert rows[0]["text"] == "indexed subject"
        assert rows[0]["type"] == "procedural"
        assert rows[0]["importance"] == pytest.approx(0.8)
        assert indexed.index.tag_counts() == {"a": 1, "b": 1}

    def test_remember_many_indexes_every_row(self, indexed):
        indexed.remember_many(["batch one", "batch two", "batch three"])
        assert indexed.index.count() == 3

    def test_edit_updates_the_index(self, indexed):
        mem = indexed.remember("before the edit", tags=["old"])
        indexed.edit(mem.id, text="after the edit", tags=["new"])
        row = indexed.index._conn.execute(
            "SELECT text FROM memories WHERE id = ?", (mem.id,)).fetchone()
        assert row["text"] == "after the edit"
        assert indexed.index.tag_counts() == {"new": 1}

    def test_edit_does_not_duplicate_fts_rows(self, indexed):
        """An external-content FTS table needs the old row deleted by hand."""
        mem = indexed.remember("original wording here")
        for i in range(3):
            indexed.edit(mem.id, text=f"revision number {i} wording")
        hits = indexed.index.search_lexical("wording", limit=50)
        assert hits.count(mem.id) == 1

    def test_forget_removes_the_row_and_its_edges(self, indexed):
        a = indexed.remember("edge source")
        b = indexed.remember("edge target")
        indexed.link(a.id, b.id, rel="refines")
        assert indexed.index.incoming(b.id) == [(a.id, "refines")]

        indexed.forget(a.id)
        assert indexed.index.count() == 1
        # A dangling dst would make reverse lookups report a memory that is gone.
        assert indexed.index.incoming(b.id) == []

    def test_link_and_unlink_track_edges(self, indexed):
        a = indexed.remember("link a")
        b = indexed.remember("link b")
        indexed.link(a.id, b.id, rel="refines")
        indexed.link(a.id, b.id, rel="related")
        assert sorted(indexed.index.incoming(b.id)) == [
            (a.id, "refines"), (a.id, "related")]
        assert indexed.index.incoming(b.id, rel="refines") == [(a.id, "refines")]

        indexed.unlink(a.id, b.id, rel="refines")
        assert indexed.index.incoming(b.id) == [(a.id, "related")]

    def test_nuke_empties_the_index(self, indexed):
        indexed.remember_many(["one", "two"])
        indexed.nuke()
        assert indexed.index.count() == 0

    def test_index_count_tracks_the_collection(self, indexed):
        for i in range(5):
            indexed.remember(f"tracking subject {i}")
        assert indexed.index.count() == indexed.collection.count() == 5


class TestReverseLinks:
    def test_neighbors_in_uses_the_index(self, indexed):
        hub = indexed.remember("the hub")
        a = indexed.remember("points at the hub, a")
        b = indexed.remember("points at the hub, b")
        indexed.remember("unrelated bystander")
        indexed.link(a.id, hub.id, rel="refines")
        indexed.link(b.id, hub.id, rel="refines")

        got = indexed.neighbors(hub.id, direction="in")
        assert {m.id for m, _ in got} == {a.id, b.id}

    def test_matches_the_scan_fallback(self, indexed, plain):
        """The index must not change the answer, only how it is found."""
        for store in (indexed, plain):
            hub = store.remember("parity hub")
            src = store.remember("parity source")
            other = store.remember("parity bystander")
            store.link(src.id, hub.id, rel="refines")
            store.link(hub.id, other.id, rel="related")
            both = {(m.text, rel) for m, rel in
                    store.neighbors(hub.id, direction="both")}
            assert both == {("parity source", "refines"),
                            ("parity bystander", "related")}


class TestListRecentPagination:
    def test_cursor_walks_the_store(self, indexed):
        made = [indexed.remember(f"page subject {i}") for i in range(6)]
        newest_first = list(reversed(made))

        page1 = indexed.list_recent(limit=2)
        assert [m.id for m in page1] == [m.id for m in newest_first[:2]]

        page2 = indexed.list_recent(limit=2, before=page1[-1].created_at)
        assert [m.id for m in page2] == [m.id for m in newest_first[2:4]]
        assert not set(m.id for m in page1) & set(m.id for m in page2)

    def test_cursor_also_works_without_the_index(self, plain):
        made = [plain.remember(f"fallback page {i}") for i in range(4)]
        page1 = plain.list_recent(limit=2)
        page2 = plain.list_recent(limit=2, before=page1[-1].created_at)
        assert len(page1) == len(page2) == 2
        assert not set(m.id for m in page1) & set(m.id for m in page2)
        assert {m.id for m in page1 + page2} == {m.id for m in made}

    def test_hides_superseded_and_expired_like_the_scan(self, indexed):
        live = indexed.remember("still live")
        old = indexed.remember("about to be superseded")
        new = indexed.remember("the replacement")
        indexed.supersede(old_id=old.id, new_id=new.id)
        gone = indexed.remember("already expired",
                                expires_at=live.created_at - 1)

        ids = {m.id for m in indexed.list_recent(limit=100)}
        assert live.id in ids and new.id in ids
        assert old.id not in ids and gone.id not in ids

        everything = {m.id for m in indexed.list_recent(
            limit=100, include_superseded=True, include_expired=True)}
        assert {old.id, gone.id} <= everything


class TestFullCorpusLexical:
    def test_reaches_a_match_outside_the_vector_pool(self, indexed):
        """The Tier-1 claim: an exact-token match is reachable at all.

        `overfetch=1, k=1` makes the vector pool exactly one row wide, so in a
        61-memory corpus the only way this memory can be scored is the lexical
        channel unioning it in. The weights favour lexical because that is the
        configuration the feature exists for — a query typed as literal tokens.
        """
        target = indexed.remember("the quetzalcoatlus deployment checklist")
        for i in range(60):
            indexed.remember(f"unrelated filler memory number {i}")

        lexical_first = HybridWeights(
            cosine=0.2, lexical=0.6, recency=0.1, importance=0.1)
        hits = indexed.recall(
            "quetzalcoatlus", k=1, overfetch=1, mode="hybrid",
            weights=lexical_first, lexical_index="fts", explain=True)

        assert [m.id for m, _, _ in hits] == [target.id]
        # Full BM25 credit: it is the only memory carrying the token.
        assert hits[0][2]["lexical"] == 1.0

    def test_the_unioned_candidate_keeps_its_real_cosine(self, indexed):
        """Not a fabricated distance.

        A neutral value would invent vector evidence the candidate never
        earned; a worst-case one (-1 similarity x the 0.55 cosine weight)
        would bury it below anything the 0.20 lexical weight could recover,
        making the channel decorative.
        """
        indexed.remember("the pterodactyl migration procedure")
        for i in range(20):
            indexed.remember(f"filler number {i}")

        hits = indexed.recall("pterodactyl", k=1, overfetch=1, mode="hybrid",
                              weights=HybridWeights(cosine=0.2, lexical=0.6,
                                                    recency=0.1, importance=0.1),
                              lexical_index="fts", explain=True)
        cosine = hits[0][2]["cosine"]
        assert -1.0 <= cosine <= 1.0
        # The two fabricated values this must never be.
        assert cosine not in (0.0, -1.0)

    def test_respects_metadata_filters(self, indexed):
        """A lexical hit must obey type/source filters like a vector hit."""
        indexed.remember("filtered pterodactyl note", type="episodic")
        keep = indexed.remember("kept pterodactyl note", type="procedural")
        hits = {m.id for m, _ in indexed.recall(
            "pterodactyl", k=10, mode="hybrid", lexical_index="fts",
            type="procedural")}
        assert hits == {keep.id}

    def test_is_a_noop_without_an_index(self, plain):
        plain.remember("no index here")
        got = plain.recall("index", k=3, mode="hybrid", lexical_index="fts")
        assert isinstance(got, list)  # no error, just the ordinary pool

    def test_default_is_pool_only(self, indexed, monkeypatch):
        called = []
        monkeypatch.setattr(indexed.index, "search_lexical",
                            lambda *a, **kw: called.append(a) or [])
        indexed.remember("default lexical subject")
        indexed.recall("default lexical subject", k=1, mode="hybrid")
        assert called == [], "lexical_index defaults to 'pool'"


class TestExpiryAndAggregates:
    def test_purge_uses_the_index(self, indexed):
        live = indexed.remember("outlives the purge")
        dead = indexed.remember("expires immediately",
                                expires_at=live.created_at - 1)
        purged = indexed.purge_expired()
        assert [m.id for m in purged] == [dead.id]
        assert indexed.get(dead.id) is None
        assert indexed.get(live.id) is not None

    def test_tag_and_type_counts(self, indexed):
        indexed.remember("counted a", tags=["x"], type="semantic")
        indexed.remember("counted b", tags=["x", "y"], type="procedural")
        assert indexed.index.tag_counts() == {"x": 2, "y": 1}
        assert indexed.index.type_counts() == {"procedural": 1, "semantic": 1}

    def test_superseded_rows_leave_the_counts(self, indexed):
        a = indexed.remember("count me", tags=["t"])
        b = indexed.remember("replacement", tags=["t"])
        assert indexed.index.tag_counts() == {"t": 2}
        indexed.supersede(old_id=a.id, new_id=b.id)
        assert indexed.index.tag_counts() == {"t": 1}
        assert indexed.index.tag_counts(include_superseded=True) == {"t": 2}


class TestHealthAndFallback:
    def test_count_mismatch_disables_the_index(self, tmp_path):
        path = str(tmp_path / "chroma")
        store = MemoryStore(path=path, collection="health",
                            embedding_function=FakeEmbedder(), index="sqlite")
        store.remember("indexed row")
        index_path = store.index.path
        store.client.close()

        # Simulate an index that fell behind (writes while it was missing).
        conn = sqlite3.connect(index_path)
        conn.execute("DELETE FROM memories")
        conn.commit()
        conn.close()

        reopened = MemoryStore(path=path, collection="health",
                               embedding_function=FakeEmbedder(), index="sqlite")
        try:
            assert reopened.index.healthy is False
            assert "row count" in reopened.index.disabled_reason
            # Reads must still be correct — via the scan fallback.
            assert len(reopened.list_recent(limit=10)) == 1
        finally:
            reopened.client.close()

    def test_a_disabled_index_still_serves_reverse_links(self, indexed):
        a = indexed.remember("disabled-path source")
        b = indexed.remember("disabled-path target")
        indexed.link(a.id, b.id, rel="refines")
        indexed.index.disable("test")
        got = indexed.neighbors(b.id, direction="in")
        assert {m.id for m, _ in got} == {a.id}

    def test_reindex_restores_health(self, indexed):
        indexed.remember_many(["restore one", "restore two"])
        indexed.index.disable("test")
        result = indexed.reindex()
        assert result["enabled"] is True
        assert result["indexed"] == 2
        assert result["healthy"] is True
        assert indexed.index.count() == 2

    def test_reindex_on_a_store_without_an_index(self, plain):
        result = plain.reindex()
        assert result["enabled"] is False and "no sidecar index" in result["error"]

    def test_write_failure_disables_rather_than_raises(self, indexed):
        """A broken cache must never take a write down with it."""
        indexed.index._conn.close()  # every later statement now raises
        mem = indexed.remember("survives a broken index")
        assert indexed.index.healthy is False
        assert indexed.get(mem.id) is not None


class TestFtsQueryBuilding:
    @pytest.mark.parametrize("raw,expected", [
        ("deploy runbook", '"deploy" OR "runbook"'),
        ("", ""),
        ("   ", ""),
        # Bare punctuation would be read as FTS5 operators (or a syntax error).
        ("foo-bar", '"foo" OR "bar"'),
        ('say "hi"', '"say" OR "hi"'),
        ("NEAR AND OR", '"NEAR" OR "AND" OR "OR"'),
    ])
    def test_tokenises_and_quotes(self, raw, expected):
        assert _fts_query(raw) == expected

    def test_operator_soup_does_not_raise(self, indexed):
        indexed.remember("harmless subject")
        assert indexed.index.search_lexical('*"^ NEAR( AND') == []


class TestReindexCli:
    def _run(self, tmp_path, *args):
        return CliRunner().invoke(app, ["--store", str(tmp_path / "chroma"), *args])

    def test_builds_an_index_for_an_existing_store(self, tmp_path):
        assert self._run(tmp_path, "remember", "pre-existing memory").exit_code == 0
        res = self._run(tmp_path, "reindex", "--json")
        assert res.exit_code == 0, res.stdout
        out = json.loads(res.stdout)
        assert out["enabled"] is True and out["indexed"] == 1
        assert out["healthy"] is True
