"""The MCP recall / recall_pack / auto_context tools must reach the same
retrieval controls as the store and the HTTP surface (A1).

Before this, an MCP client could not use RRF fusion, MMR diversity, dedup, the
min_cosine gate, graph-proximity fusion or graph-walk expansion at all — every
one of those shipped features was invisible on the surface agents actually use.
"""

from __future__ import annotations

import inspect

import pytest

import ai_houkai.mcp_server.server as srv
from ai_houkai.memory_system.store import ExpandSpec, HybridWeights, PackResult


@pytest.fixture()
def mcp_store(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_HOUKAI_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("AI_HOUKAI_COLLECTION", "mcp_knobs")
    monkeypatch.setattr(srv, "_store", None)
    yield
    if srv._store is not None:
        srv._store.client.close()
        srv._store = None


class TestKnobPlumbing:
    """The tool signatures expose the knobs, and they reach store.recall."""

    @pytest.mark.parametrize("knob", [
        "fusion", "diversity", "dedup_threshold", "min_cosine", "graph", "touch",
        "expand_rels", "expand_depth", "expand_cap", "expand_score",
        "expand_decay", "expand_rerank",
    ])
    def test_recall_exposes_knob(self, knob):
        assert knob in inspect.signature(srv.recall).parameters

    @pytest.mark.parametrize("knob", [
        "fusion", "diversity", "dedup_threshold", "min_cosine", "graph", "touch",
        "header", "expand_rels", "expand_rerank",
    ])
    def test_recall_pack_exposes_knob(self, knob):
        assert knob in inspect.signature(srv.recall_pack).parameters

    @pytest.mark.parametrize("knob", ["min_cosine", "touch", "header", "compress"])
    def test_auto_context_exposes_knob(self, knob):
        assert knob in inspect.signature(srv.auto_context).parameters

    def test_recall_forwards_every_knob(self, mcp_store, monkeypatch):
        seen = {}

        def fake_recall(**kwargs):
            seen.update(kwargs)
            return []

        monkeypatch.setattr(srv.get_store(), "recall", fake_recall)
        srv.recall(
            query="q", mode="hybrid", fusion="rrf", diversity=0.6,
            dedup_threshold=0.9, min_cosine=0.2, graph=0.15, touch=False,
            expand_rels=["refines"], expand_depth=2, expand_cap=7,
            expand_score=0.5, expand_decay=0.8, expand_rerank=True,
        )
        assert seen["fusion"] == "rrf"
        assert seen["diversity"] == 0.6
        assert seen["dedup_threshold"] == 0.9
        assert seen["min_cosine"] == 0.2
        assert seen["touch"] is False
        assert seen["weights"] == HybridWeights(graph=0.15)
        assert seen["expand"] == ExpandSpec(
            rels=("refines",), depth=2, cap=7, score=0.5, decay=0.8, rerank=True,
        )

    def test_recall_pack_forwards_every_knob(self, mcp_store, monkeypatch):
        seen = {}

        def fake_pack(**kwargs):
            seen.update(kwargs)
            return PackResult(items=[], text="", used_tokens=0, budget=0,
                              truncated=False)

        monkeypatch.setattr(srv.get_store(), "recall_pack", fake_pack)
        srv.recall_pack(
            query="q", fusion="rrf", diversity=0.4, dedup_threshold=0.95,
            min_cosine=0.1, graph=0.2, touch=False, header="# H",
            expand_rerank=True,
        )
        assert seen["fusion"] == "rrf"
        assert seen["diversity"] == 0.4
        assert seen["dedup_threshold"] == 0.95
        assert seen["min_cosine"] == 0.1
        assert seen["touch"] is False
        assert seen["header"] == "# H"
        assert seen["weights"] == HybridWeights(graph=0.2)
        assert seen["expand"] is not None and seen["expand"].rerank is True


class TestKnobDefaults:
    """Omitting the knobs must be byte-for-byte the previous behaviour."""

    def test_no_weights_and_no_expand_by_default(self, mcp_store, monkeypatch):
        seen = {}
        monkeypatch.setattr(srv.get_store(), "recall",
                            lambda **kw: seen.update(kw) or [])
        srv.recall(query="q")
        assert seen["weights"] is None
        assert seen["expand"] is None
        assert seen["fusion"] == "weighted"
        assert seen["touch"] is True

    def test_expand_helper_needs_at_least_one_field(self):
        assert srv._expand(None, None, None, None, None, None) is None
        assert srv._expand(None, None, None, None, None, False) == ExpandSpec(rerank=False)

    def test_weights_helper_is_none_without_graph(self):
        assert srv._weights(None) is None
        assert srv._weights(0.0) == HybridWeights(graph=0.0)


class TestKnobsEndToEnd:
    def test_min_cosine_gate_returns_nothing_for_off_topic(self, mcp_store):
        srv.remember(text="the deployment pipeline runs on friday")
        assert srv.recall(query="the deployment pipeline", k=3)
        # An absolute floor of 0.99 admits nothing short of an exact restatement.
        assert srv.recall(query="unrelated marine biology", k=3, min_cosine=0.99) == []

    def test_touch_false_does_not_bump_access_count(self, mcp_store):
        created = srv.remember(text="read-only recall subject")
        srv.recall(query="read-only recall subject", k=1, touch=False)
        mem = srv.get_store().get(created["id"])
        assert mem is not None and mem.access_count == 0
        srv.recall(query="read-only recall subject", k=1)
        assert srv.get_store().get(created["id"]).access_count == 1

    def test_expansion_pulls_in_a_linked_neighbour(self, mcp_store):
        a = srv.remember(text="alpha topic about compilers")
        b = srv.remember(text="an entirely separate note on gardening")
        srv.link(src_id=a["id"], dst_id=b["id"], rel="refines")
        plain = {h["id"] for h in srv.recall(query="compilers", k=1)}
        assert b["id"] not in plain
        expanded = {h["id"] for h in
                    srv.recall(query="compilers", k=1, expand_rels=["refines"])}
        assert b["id"] in expanded

    @pytest.mark.needs_model
    def test_rrf_fusion_runs_and_ranks(self, mcp_store):
        srv.remember(text="kubernetes ingress controller notes")
        srv.remember(text="postgres vacuum tuning notes")
        hits = srv.recall(query="kubernetes ingress", k=2, mode="hybrid", fusion="rrf")
        assert hits and "kubernetes" in hits[0]["text"]

    def test_pack_header_override(self, mcp_store):
        srv.remember(text="packing header subject matter")
        out = srv.recall_pack(query="packing header subject matter", header="# Custom")
        assert out["text"].startswith("# Custom")
