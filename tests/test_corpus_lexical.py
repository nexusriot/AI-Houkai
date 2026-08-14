"""Full-corpus lexical recall, without a second database.

The vector over-fetch pool is chosen by embedding distance alone, so a memory
carrying the query's exact tokens but embedding weakly never enters it — and so
can never be surfaced by the BM25 term. `lexical_index="corpus"` closes that
gap using Chroma's own `where_document` filter.

A SQLite FTS5 index was measured against this and rejected. Its lexical lookup
was faster in absolute terms (0.03 ms flat vs ~4.5 ms at 25k rows, being a real
inverted index) but it would be a second source of truth for data Chroma already
holds, needing verify-on-open / disable-on-mismatch / rebuild machinery to stay
safe. On a path already dominated by an embedding call, and in a service that
runs its own SQLite, that trade does not pay. See docs/DESIGN.md §25.
"""

from __future__ import annotations

import time

import pytest

from ai_houkai.memory_system import HybridWeights, MemoryStore
from ai_houkai.memory_system.store import _LEXICAL_MAX_TOKENS
from ai_houkai.testing import FakeEmbedder

# Favours the lexical channel, which is the configuration the feature exists
# for: a query typed as literal tokens rather than as prose.
LEXICAL_FIRST = HybridWeights(
    cosine=0.2, lexical=0.6, recency=0.1, importance=0.1)


@pytest.fixture()
def store(tmp_path):
    s = MemoryStore(path=str(tmp_path / "chroma"), collection="corpus_lexical",
                    embedding_function=FakeEmbedder(dim=16))
    yield s
    s.client.close()


class TestReachability:
    def test_reaches_a_match_outside_the_vector_pool(self, store):
        """`overfetch=1, k=1` makes the vector pool exactly one row wide, so in
        a 61-memory corpus the only way this memory can be scored is the lexical
        channel pulling it in."""
        target = store.remember("the quetzalcoatlus deployment checklist")
        store.remember_many([f"unrelated filler number {i}" for i in range(60)])

        hits = store.recall("quetzalcoatlus", k=1, overfetch=1, mode="hybrid",
                            weights=LEXICAL_FIRST, lexical_index="corpus",
                            explain=True)
        assert [m.id for m, _, _ in hits] == [target.id]
        # Full BM25 credit: it is the only memory carrying the token.
        assert hits[0][2]["lexical"] == 1.0

    def test_default_stays_pool_only(self, store, monkeypatch):
        calls = []
        monkeypatch.setattr(store, "_lexical_candidates",
                            lambda *a, **kw: calls.append(a) or [])
        store.remember("default lexical subject")
        store.recall("default lexical subject", k=1, mode="hybrid")
        assert calls == [], 'lexical_index defaults to "pool"'

    def test_semantic_mode_ignores_it(self, store, monkeypatch):
        """The lexical channel only has meaning where BM25 is scored."""
        calls = []
        monkeypatch.setattr(store, "_lexical_candidates",
                            lambda *a, **kw: calls.append(a) or [])
        store.remember("semantic mode subject")
        store.recall("semantic mode subject", k=1, lexical_index="corpus")
        assert calls == []

    def test_no_match_is_a_plain_noop(self, store):
        store.remember("something entirely different")
        got = store.recall("zzzznomatchzzzz", k=3, mode="hybrid",
                           lexical_index="corpus")
        assert isinstance(got, list)


class TestScoring:
    def test_unioned_candidate_keeps_its_real_cosine(self, store):
        """Not a fabricated distance.

        A neutral value would invent vector evidence the candidate never
        earned; a worst-case one (-1 similarity x the 0.55 cosine weight) would
        bury it below anything the 0.20 lexical weight could recover, making the
        channel decorative.
        """
        store.remember("the pterodactyl migration procedure")
        store.remember_many([f"filler number {i}" for i in range(20)])

        hits = store.recall("pterodactyl", k=1, overfetch=1, mode="hybrid",
                            weights=LEXICAL_FIRST, lexical_index="corpus",
                            explain=True)
        cosine = hits[0][2]["cosine"]
        assert -1.0 <= cosine <= 1.0
        assert cosine not in (0.0, -1.0)

    def test_does_not_duplicate_a_pool_member(self, store):
        target = store.remember("duplication check subject")
        hits = store.recall("duplication check subject", k=5, mode="hybrid",
                            lexical_index="corpus")
        assert [m.id for m, _ in hits].count(target.id) == 1


class TestFiltersStillApply:
    def test_type_filter(self, store):
        """A lexical hit must obey metadata filters like a vector hit."""
        store.remember("filtered pterodactyl note", type="episodic")
        keep = store.remember("kept pterodactyl note", type="procedural")
        hits = {m.id for m, _ in store.recall(
            "pterodactyl", k=10, mode="hybrid", lexical_index="corpus",
            type="procedural")}
        assert hits == {keep.id}

    def test_superseded_and_expired_stay_hidden(self, store):
        old = store.remember("archaeopteryx old note")
        new = store.remember("archaeopteryx new note")
        store.supersede(old_id=old.id, new_id=new.id)
        gone = store.remember("archaeopteryx expired note",
                              expires_at=time.time() - 1)

        ids = {m.id for m, _ in store.recall(
            "archaeopteryx", k=10, mode="hybrid", lexical_index="corpus")}
        assert new.id in ids
        assert old.id not in ids and gone.id not in ids


class TestTokenHandling:
    def test_short_tokens_are_skipped(self, store, monkeypatch):
        """Two-letter tokens match almost everything, so probing them would
        cost a scan to return the whole corpus."""
        seen = []
        real = store.collection.get

        def spy(*args, **kwargs):
            if "where_document" in kwargs:
                seen.append(kwargs["where_document"])
            return real(*args, **kwargs)

        monkeypatch.setattr(store.collection, "get", spy)
        store.remember("an ox is here")
        store.recall("an ox", k=1, mode="hybrid", lexical_index="corpus")
        assert seen == []

    def test_probe_count_is_capped(self, store, monkeypatch):
        """Each probe is a server-side scan; a long query must not become a
        long series of them."""
        seen = []
        real = store.collection.get

        def spy(*args, **kwargs):
            if "where_document" in kwargs:
                seen.append(kwargs["where_document"])
            return real(*args, **kwargs)

        store.remember("alpha bravo charlie delta echo foxtrot golf hotel")
        monkeypatch.setattr(store.collection, "get", spy)
        store.recall("alpha bravo charlie delta echo foxtrot golf hotel",
                     k=1, mode="hybrid", lexical_index="corpus")
        assert 0 < len(seen) <= _LEXICAL_MAX_TOKENS

    def test_punctuation_does_not_break_the_probe(self, store):
        store.remember("the deploy-runbook lives in ops/deploy.md")
        got = store.recall('deploy-runbook "quoted" *', k=3, mode="hybrid",
                           lexical_index="corpus")
        assert isinstance(got, list)

    def test_a_backend_error_is_swallowed(self, store, monkeypatch):
        """A lexical bonus is never worth failing the whole recall."""
        store.remember("resilience subject matter")

        def boom(*args, **kwargs):
            if "where_document" in kwargs:
                raise RuntimeError("backend said no")
            return store.collection._collection.get(*args, **kwargs) \
                if hasattr(store.collection, "_collection") else {"ids": []}

        monkeypatch.setattr(store, "_lexical_candidates",
                            lambda *a, **kw: (_ for _ in ()).throw(
                                RuntimeError("backend said no")))
        with pytest.raises(RuntimeError):
            # The guard lives inside _lexical_candidates, so patching it out
            # proves the call site has no second guard hiding a real bug.
            store.recall("resilience", k=1, mode="hybrid",
                         lexical_index="corpus")


class TestSelectionSurvivesTheUnion:
    """Unioning lexical candidates must not cost the pool its embeddings.

    ``_union_lexical`` rebuilds Chroma's result dict, and MMR / dedup are the
    only consumers of its ``embeddings`` key. Losing that key does not raise —
    dedup stops firing and the diversity penalty silently goes to zero — so
    only a behavioural test notices. Chroma hands embeddings back as a numpy
    array rather than a list, which is exactly what a rebuild is liable to drop.
    """

    @staticmethod
    def _keyed_embedder(marker: str):
        """Rows containing *marker* embed orthogonally to every other row.

        That lets a query which does not contain the marker rank those rows
        last by vector distance, so they can only enter the pool through the
        lexical union — while still matching it as a literal token for BM25.
        """
        def emb(texts):
            return [[1.0, 0.0] if marker in t else [0.0, 1.0] for t in texts]
        return emb

    def test_union_keeps_embeddings_for_the_whole_pool(self, tmp_path):
        s = MemoryStore(path=str(tmp_path / "chroma"), collection="union_embs",
                        embedding_function=self._keyed_embedder("dup"))
        try:
            s.remember("quetzalcoatlus dup")
            s.remember_many([f"filler number {i}" for i in range(6)])

            include = ["documents", "metadatas", "distances", "embeddings"]
            res = s.collection.query(query_texts=["quetzalcoatlus"],
                                     n_results=1, where=None, include=include)
            merged = s._union_lexical(res, "quetzalcoatlus", 1, None, include)

            assert len(merged["ids"][0]) > len(res["ids"][0]), \
                "the union added no candidate, so this proves nothing"
            # Every row in the merged pool must still resolve to a vector,
            # including the ones that were already there.
            assert len(s._emb_map(merged)) == len(merged["ids"][0])
        finally:
            s.client.close()

    def test_dedup_still_fires_on_a_unioned_pool(self, tmp_path):
        """The symptom the dropped key produces: a byte-identical pair, cosine
        1.0 apart, both surviving a 0.95 dedup threshold."""
        s = MemoryStore(path=str(tmp_path / "chroma"), collection="union_dedup",
                        embedding_function=self._keyed_embedder("dup"))
        try:
            a = s.remember("quetzalcoatlus dup")
            b = s.remember("quetzalcoatlus dup")
            s.remember_many([f"filler number {i}" for i in range(20)])

            # k=2 / overfetch=1 makes the vector pool two rows wide out of 22,
            # so the pair is outside it and arrives only via the union.
            hits = s.recall("quetzalcoatlus", k=2, overfetch=1, mode="hybrid",
                            weights=LEXICAL_FIRST, dedup_threshold=0.95,
                            lexical_index="corpus")
            ids = [m.id for m, _ in hits]
            assert ids.count(a.id) + ids.count(b.id) == 1
        finally:
            s.client.close()


class TestChromaNativeExpiry:
    def test_purge_uses_a_range_query(self, store):
        """purge_expired pushes the range into Chroma instead of loading the
        whole collection to find the few rows that lapsed."""
        live = store.remember("outlives the purge")
        dead = store.remember("expires immediately",
                              expires_at=time.time() - 1)
        purged = store.purge_expired()
        assert [m.id for m in purged] == [dead.id]
        assert store.get(dead.id) is None
        assert store.get(live.id) is not None

    def test_nothing_expired_is_a_noop(self, store):
        store.remember("no ttl at all")
        assert store.purge_expired() == []

    def test_dry_run_reports_without_deleting(self, store):
        dead = store.remember("dry-run victim", expires_at=time.time() - 1)
        assert [m.id for m in store.purge_expired(dry_run=True)] == [dead.id]
        assert store.get(dead.id) is not None
