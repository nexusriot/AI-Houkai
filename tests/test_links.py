"""Unit tests for memory linking (link / unlink / neighbors / subgraph)."""

from __future__ import annotations

import pytest

from ai_houkai.memory_system import Link, Graph, MemoryStore


def _mem(store: MemoryStore, text: str, **kw) -> str:
    return store.remember(text, **kw).id


class TestLink:
    def test_link_creates_edge(self, store: MemoryStore):
        a = _mem(store, "procedural rule A")
        b = _mem(store, "example of rule A")
        store.link(a, b, rel="example_of")
        src = store._get_by_id(a)
        assert any(l.to == b and l.rel == "example_of" for l in src.links)

    def test_link_is_idempotent(self, store: MemoryStore):
        a = _mem(store, "foo")
        b = _mem(store, "bar")
        store.link(a, b, rel="related")
        store.link(a, b, rel="related")
        src = store._get_by_id(a)
        assert sum(1 for l in src.links if l.to == b and l.rel == "related") == 1

    def test_link_self_raises(self, store: MemoryStore):
        a = _mem(store, "solo")
        with pytest.raises(ValueError, match="itself"):
            store.link(a, a)

    def test_link_unknown_src_raises(self, store: MemoryStore):
        b = _mem(store, "exists")
        with pytest.raises(KeyError):
            store.link("00000000-0000-0000-0000-000000000001", b)

    def test_multiple_rels_on_same_pair(self, store: MemoryStore):
        a = _mem(store, "root")
        b = _mem(store, "child")
        store.link(a, b, rel="refines")
        store.link(a, b, rel="example_of")
        src = store._get_by_id(a)
        rels = {l.rel for l in src.links if l.to == b}
        assert rels == {"refines", "example_of"}

    def test_link_default_rel_is_related(self, store: MemoryStore):
        a = _mem(store, "a")
        b = _mem(store, "b")
        store.link(a, b)
        src = store._get_by_id(a)
        assert any(l.rel == "related" for l in src.links)


class TestUnlink:
    def test_unlink_removes_specific_rel(self, store: MemoryStore):
        a = _mem(store, "a")
        b = _mem(store, "b")
        store.link(a, b, rel="refines")
        store.link(a, b, rel="related")
        removed = store.unlink(a, b, rel="refines")
        assert removed == 1
        src = store._get_by_id(a)
        assert not any(l.rel == "refines" for l in src.links)
        assert any(l.rel == "related" for l in src.links)

    def test_unlink_all_rels(self, store: MemoryStore):
        a = _mem(store, "a")
        b = _mem(store, "b")
        store.link(a, b, rel="refines")
        store.link(a, b, rel="related")
        removed = store.unlink(a, b, rel=None)
        assert removed == 2
        src = store._get_by_id(a)
        assert not any(l.to == b for l in src.links)

    def test_unlink_nonexistent_returns_zero(self, store: MemoryStore):
        a = _mem(store, "a")
        b = _mem(store, "b")
        assert store.unlink(a, b, rel="refines") == 0

    def test_unlink_unknown_src_returns_zero(self, store: MemoryStore):
        assert store.unlink("no-such-id", "no-such-id") == 0


class TestNeighbors:
    def test_outgoing_neighbors(self, store: MemoryStore):
        a = _mem(store, "parent rule")
        b = _mem(store, "example instance")
        store.link(a, b, rel="example_of")
        hits = store.neighbors(a, direction="out")
        ids  = [m.id for m, _ in hits]
        rels = [r for _, r in hits]
        assert b in ids
        assert "example_of" in rels

    def test_incoming_neighbors(self, store: MemoryStore):
        a = _mem(store, "parent")
        b = _mem(store, "child")
        store.link(a, b, rel="refines")
        hits = store.neighbors(b, direction="in")
        assert any(m.id == a for m, _ in hits)

    def test_rel_filter(self, store: MemoryStore):
        a = _mem(store, "center")
        b = _mem(store, "node b")
        c = _mem(store, "node c")
        store.link(a, b, rel="refines")
        store.link(a, c, rel="example_of")
        hits = store.neighbors(a, rel="refines", direction="out")
        assert len(hits) == 1
        assert hits[0][0].id == b

    def test_depth_two(self, store: MemoryStore):
        a = _mem(store, "a")
        b = _mem(store, "b")
        c = _mem(store, "c")
        store.link(a, b, rel="refines")
        store.link(b, c, rel="refines")
        hits = store.neighbors(a, direction="out", depth=2)
        ids = [m.id for m, _ in hits]
        assert b in ids
        assert c in ids

    def test_cycle_terminates(self, store: MemoryStore):
        a = _mem(store, "a")
        b = _mem(store, "b")
        store.link(a, b, rel="related")
        store.link(b, a, rel="related")
        # Should not loop forever
        hits = store.neighbors(a, direction="out", depth=5)
        assert isinstance(hits, list)

    def test_direction_both(self, store: MemoryStore):
        a = _mem(store, "hub")
        b = _mem(store, "inbound")
        c = _mem(store, "outbound")
        store.link(b, a, rel="refines")   # b → a (incoming to a)
        store.link(a, c, rel="refines")   # a → c (outgoing from a)
        hits = store.neighbors(a, direction="both")
        ids  = [m.id for m, _ in hits]
        assert b in ids
        assert c in ids


class TestSubgraph:
    def test_single_seed_no_links(self, store: MemoryStore):
        a = _mem(store, "isolated")
        g = store.subgraph([a])
        assert a in g.nodes
        assert g.edges == []

    def test_subgraph_includes_linked_nodes(self, store: MemoryStore):
        a = _mem(store, "root")
        b = _mem(store, "child")
        store.link(a, b, rel="refines")
        g = store.subgraph([a], depth=1)
        assert a in g.nodes
        assert b in g.nodes
        assert (a, b, "refines") in g.edges

    def test_subgraph_depth_zero(self, store: MemoryStore):
        a = _mem(store, "root")
        b = _mem(store, "child")
        store.link(a, b, rel="refines")
        g = store.subgraph([a], depth=0)
        assert a in g.nodes
        assert b not in g.nodes


class TestLinkMetadataRoundtrip:
    def test_links_survive_store_roundtrip(self, store: MemoryStore):
        a = _mem(store, "parent")
        b = _mem(store, "child")
        store.link(a, b, rel="derived_from")
        # reload from DB
        reloaded = store._get_by_id(a)
        assert len(reloaded.links) == 1
        assert reloaded.links[0].to == b
        assert reloaded.links[0].rel == "derived_from"

    def test_memory_with_no_links_loads_empty_list(self, store: MemoryStore):
        a = _mem(store, "plain memory")
        reloaded = store._get_by_id(a)
        assert reloaded.links == []

    def test_multiple_links_roundtrip(self, store: MemoryStore):
        a = _mem(store, "hub")
        b = _mem(store, "b")
        c = _mem(store, "c")
        store.link(a, b, rel="refines")
        store.link(a, c, rel="example_of")
        reloaded = store._get_by_id(a)
        link_map = {l.to: l.rel for l in reloaded.links}
        assert link_map[b] == "refines"
        assert link_map[c] == "example_of"


class TestSubgraphDiamond:
    """Regression: a plain visited set truncated diamond shapes — a node
    first reached with 0 remaining hops was never expanded when a shorter
    path later reached it with budget to spare."""

    def test_diamond_expands_short_path(self, store: MemoryStore):
        a = _mem(store, "node a")
        b = _mem(store, "node b")
        c = _mem(store, "node c")
        d = _mem(store, "node d")
        store.link(a, b, rel="refines")   # long path a→b→c visits c first
        store.link(b, c, rel="refines")   # with 0 remaining budget…
        store.link(a, c, rel="refines")   # …but the direct edge leaves 1 hop
        store.link(c, d, rel="refines")
        g = store.subgraph([a], depth=2)
        assert d in g.nodes
        assert (c, d, "refines") in g.edges

    def test_cycle_still_terminates(self, store: MemoryStore):
        a = _mem(store, "cyc a")
        b = _mem(store, "cyc b")
        store.link(a, b, rel="related")
        store.link(b, a, rel="related")
        g = store.subgraph([a], depth=5)
        assert set(g.nodes) == {a, b}
        assert len(g.edges) == 2


class TestNeighborsParallelEdges:
    """Regression: two differently-typed edges between the same pair were
    reported as a single (memory, rel) pair in both directions."""

    def test_outgoing_reports_all_rels(self, store: MemoryStore):
        a = _mem(store, "par out a")
        b = _mem(store, "par out b")
        store.link(a, b, rel="related")
        store.link(a, b, rel="refines")
        pairs = store.neighbors(a, direction="out")
        assert sorted(r for m, r in pairs if m.id == b) == ["refines", "related"]

    def test_incoming_reports_all_rels(self, store: MemoryStore):
        a = _mem(store, "par in a")
        b = _mem(store, "par in b")
        store.link(a, b, rel="related")
        store.link(a, b, rel="refines")
        pairs = store.neighbors(b, direction="in")
        assert sorted(r for m, r in pairs if m.id == a) == ["refines", "related"]

    def test_rel_filter_still_narrows(self, store: MemoryStore):
        a = _mem(store, "flt a")
        b = _mem(store, "flt b")
        store.link(a, b, rel="related")
        store.link(a, b, rel="refines")
        pairs = store.neighbors(a, direction="out", rel="refines")
        assert [(m.id, r) for m, r in pairs] == [(b, "refines")]

    def test_node_still_expanded_once(self, store: MemoryStore):
        a = _mem(store, "exp a")
        b = _mem(store, "exp b")
        c = _mem(store, "exp c")
        store.link(a, b, rel="related")
        store.link(a, b, rel="refines")
        store.link(b, c, rel="related")
        pairs = store.neighbors(a, direction="out", depth=2)
        ids = [m.id for m, _ in pairs]
        assert ids.count(c) == 1          # c reached once via b
