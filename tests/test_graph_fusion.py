"""Unit tests for graph-proximity fusion (HybridWeights.graph) and the
gated graph expansion (ExpandSpec.rerank)."""

from __future__ import annotations

import pytest

from ai_houkai.memory_system import (
    ExpandSpec, HybridWeights, Link, Memory, MemoryStore,
)


class TestGraphSpread:
    """The pure PPR-lite helper — deterministic, no embeddings needed."""

    def _mem(self, mid: str, links=()):
        m = Memory(id=mid, text=mid, type="semantic")
        for to, rel in links:
            m.links.append(Link(to=to, rel=rel))
        return m

    def test_empty_pool(self, store: MemoryStore):
        assert store._graph_spread([]) == {}

    def test_no_internal_edges_returns_empty(self, store: MemoryStore):
        pool = [(self._mem("a"), 0.9), (self._mem("b"), 0.5)]
        assert store._graph_spread(pool) == {}

    def test_star_hub_gets_max_spread(self, store: MemoryStore):
        # center linked to three leaves; an isolated node sits outside.
        center = self._mem("center", links=[("l1", "related"),
                                             ("l2", "related"),
                                             ("l3", "related")])
        pool = [
            (center, 0.5),
            (self._mem("l1"), 0.5),
            (self._mem("l2"), 0.5),
            (self._mem("l3"), 0.5),
            (self._mem("lonely"), 0.5),
        ]
        spread = store._graph_spread(pool)
        assert spread  # has edges → non-empty
        # The hub of the star accumulates the most activation.
        assert spread["center"] == pytest.approx(1.0)
        assert spread["center"] > spread["l1"]
        # The isolated node gains nothing from neighbours, so it ranks below
        # the hub (it is not necessarily the pool minimum — the leaves are).
        assert spread["center"] > spread["lonely"]

    def test_reverse_edges_are_followed(self, store: MemoryStore):
        # center stores only OUTGOING links to its two leaves, yet ends up with
        # the most activation — only possible if reverse (leaf -> center) edges
        # are followed too, i.e. the spread graph is undirected.
        center = self._mem("center", links=[("l1", "related"),
                                             ("l2", "related")])
        pool = [(center, 0.3), (self._mem("l1"), 0.3), (self._mem("l2"), 0.3)]
        spread = store._graph_spread(pool)
        assert spread["center"] == pytest.approx(1.0)
        assert spread["center"] > spread["l1"]


class TestGraphFusionWeighted:
    def _seed_star(self, store: MemoryStore):
        center = store.remember("kubernetes networking overview", type="semantic")
        l1 = store.remember("kubernetes ingress setup", type="semantic")
        l2 = store.remember("kubernetes service mesh", type="semantic")
        l3 = store.remember("kubernetes dns resolution", type="semantic")
        lonely = store.remember("kubernetes storage volumes", type="semantic")
        for leaf in (l1, l2, l3):
            store.link(center.id, leaf.id, rel="related")
        return center, lonely

    def test_graph_zero_is_noop(self, store: MemoryStore):
        self._seed_star(store)
        base = store.recall("kubernetes", k=5, mode="hybrid",
                            weights=HybridWeights(), touch=False)
        withw = store.recall("kubernetes", k=5, mode="hybrid",
                             weights=HybridWeights(graph=0.0), touch=False)
        assert [(m.id, round(s, 6)) for m, s in base] == \
               [(m.id, round(s, 6)) for m, s in withw]

    def test_graph_lifts_connected_node(self, store: MemoryStore):
        center, lonely = self._seed_star(store)
        base = dict(
            (m.id, s) for m, s in store.recall(
                "kubernetes", k=5, mode="hybrid",
                weights=HybridWeights(graph=0.0), touch=False))
        boosted = dict(
            (m.id, s) for m, s in store.recall(
                "kubernetes", k=5, mode="hybrid",
                weights=HybridWeights(graph=0.5), touch=False))
        # The star hub gains more than the isolated node.
        hub_delta = boosted[center.id] - base[center.id]
        lonely_delta = boosted[lonely.id] - base[lonely.id]
        assert hub_delta > 0
        assert hub_delta > lonely_delta

    def test_explain_records_graph_term(self, store: MemoryStore):
        center, _ = self._seed_star(store)
        hits = store.recall("kubernetes", k=5, mode="hybrid",
                            weights=HybridWeights(graph=0.5),
                            explain=True, touch=False)
        by_id = {m.id: expl for m, _, expl in hits}
        assert "graph" in by_id[center.id]
        assert "graph" in by_id[center.id]["weights"]


class TestGraphFusionRRF:
    def test_graph_zero_is_noop_rrf(self, store: MemoryStore):
        a = store.remember("postgres replication tuning", type="semantic")
        b = store.remember("postgres vacuum settings", type="semantic")
        store.link(a.id, b.id, rel="related")
        base = store.recall("postgres", k=5, mode="hybrid", fusion="rrf",
                            weights=HybridWeights(), touch=False)
        withw = store.recall("postgres", k=5, mode="hybrid", fusion="rrf",
                             weights=HybridWeights(graph=0.0), touch=False)
        assert [m.id for m, _ in base] == [m.id for m, _ in withw]

    def test_graph_signal_in_rrf_explain(self, store: MemoryStore):
        a = store.remember("postgres replication tuning", type="semantic")
        b = store.remember("postgres vacuum settings", type="semantic")
        store.link(a.id, b.id, rel="related")
        hits = store.recall("postgres", k=5, mode="hybrid", fusion="rrf",
                            weights=HybridWeights(graph=0.3),
                            explain=True, touch=False)
        # At least one connected node exposes a graph rank in its RRF signals.
        assert any("graph" in expl.get("signals", {}) for _, _, expl in hits)


class TestExpansionGating:
    def _seed_hub(self, store: MemoryStore, n_children: int = 5):
        hub = store.remember("microservice deployment runbook",
                             type="procedural")
        children = []
        for i in range(n_children):
            c = store.remember(f"deployment note number {i}", type="episodic")
            store.link(hub.id, c.id, rel="refines")
            children.append(c)
        return hub, children

    def test_legacy_append_can_exceed_k(self, store: MemoryStore):
        self._seed_hub(store, n_children=5)
        hits = store.recall(
            "microservice deployment", k=2,
            expand=ExpandSpec(rels=("refines",), cap=5, score=0.6,
                              rerank=False),
            touch=False)
        # rerank=False appends expanded nodes after the top-k cut.
        assert len(hits) > 2

    def test_rerank_respects_k(self, store: MemoryStore):
        self._seed_hub(store, n_children=5)
        hits = store.recall(
            "microservice deployment", k=2,
            expand=ExpandSpec(rels=("refines",), cap=5, score=0.6,
                              rerank=True),
            touch=False)
        # rerank=True makes expanded nodes compete for the k slots — never over.
        assert len(hits) <= 2

    def test_rerank_dedups_duplicate_expansion(self, store: MemoryStore):
        hub = store.remember("cache invalidation policy", type="procedural")
        dup = store.remember("cache invalidation policy", type="episodic")
        store.link(hub.id, dup.id, rel="refines")
        hits = store.recall(
            "cache invalidation policy", k=5,
            expand=ExpandSpec(rels=("refines",), cap=5, score=0.95,
                              rerank=True),
            dedup_threshold=0.9, touch=False)
        ids = [m.id for m, _ in hits]
        assert hub.id in ids
        # The near-duplicate expanded node is dropped by dedup in rerank mode.
        assert dup.id not in ids


class TestRerankHardening:
    """Regression guards for the rerank-path fixes."""

    def test_rrf_rerank_does_not_bury_primary(self, store: MemoryStore):
        # RRF scores are ~1/rrf_k (tiny); a raw hop score of 0.9 would sort the
        # expanded node above the real hit. The fix rescales the hop score into
        # the pool's own range, so the strongest primary stays rank 0.
        hub = store.remember("distributed tracing spans", type="procedural")
        child = store.remember("totally unrelated weather almanac", type="episodic")
        store.link(hub.id, child.id, rel="refines")
        hits = store.recall(
            "distributed tracing spans", k=5, mode="hybrid", fusion="rrf",
            expand=ExpandSpec(rels=("refines",), cap=5, score=0.9, rerank=True),
            min_cosine=0.99, touch=False)
        assert hits, "expected at least the primary hit"
        assert hits[0][0].id == hub.id

    def test_rerank_drops_unembeddable_expansion(self, store: MemoryStore):
        # When dedup is on (need_emb) but an expanded node's embedding can't be
        # fetched, it must be dropped — not admitted with a free novelty pass.
        # overfetch=1 keeps `child` out of the query pool, so its only embedding
        # source is _emb_for_ids (which we make miss).
        hub = store.remember("cache invalidation policy", type="procedural")
        child = store.remember("totally unrelated weather almanac", type="episodic")
        store.link(hub.id, child.id, rel="refines")
        store._emb_for_ids = lambda ids: {}  # simulate an embedding-fetch miss
        hits = store.recall(
            "cache invalidation policy", k=1, mode="hybrid", overfetch=1,
            expand=ExpandSpec(rels=("refines",), cap=5, score=0.95, rerank=True),
            dedup_threshold=0.9, min_cosine=0.99, touch=False)
        assert child.id not in [m.id for m, _ in hits]

    def test_collect_expansion_shields_seen_ids(self, store: MemoryStore):
        # A node in seen_ids is neither re-emitted nor has its explain clobbered.
        a = store.remember("alpha root fact", type="semantic")
        b = store.remember("beta child fact", type="semantic")
        store.link(a.id, b.id, rel="refines")
        expl = {b.id: {"mode": "hybrid", "cosine": 0.42, "score": 0.42}}
        extra = store._collect_expansion(
            [(a, 0.9)], ExpandSpec(rels=("refines",), cap=5),
            False, expl, seen_ids={a.id, b.id})
        assert all(m.id != b.id for m, _ in extra)
        assert expl[b.id] == {"mode": "hybrid", "cosine": 0.42, "score": 0.42}
