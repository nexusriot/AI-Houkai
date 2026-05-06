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
