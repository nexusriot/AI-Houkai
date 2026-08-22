"""The Python remote surface must match parity.json (C).

parity.json is the single source of truth shared with the Go port, which
asserts against the same file in go/internal/parity. Neither CI job needs the
other toolchain, and a surface added to only one port fails that port's build.

This is the guard for the A1 class of bug: the Python MCP `recall` had silently
fallen behind the Go one by five ranking knobs, and nothing noticed because
nothing compared them.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
from pathlib import Path

import pytest

import ai_houkai.http_server.server as http_srv
import ai_houkai.mcp_server.server as mcp_srv

PARITY_PATH = Path(__file__).resolve().parents[1] / "parity.json"


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(PARITY_PATH.read_text())


def _python_mcp_tools() -> list[str]:
    tools = asyncio.new_event_loop().run_until_complete(mcp_srv.mcp.list_tools())
    return sorted(t.name for t in tools)


def _python_http_routes() -> list[str]:
    """Render the regex routing table in the manifest's `METHOD /path` form."""
    out = []
    for method, pattern, _fn, _needs_body in http_srv._ROUTES:
        path = pattern.pattern.lstrip("^").rstrip("$")
        # Any named group renders as {name}, so a new path parameter does not
        # need this helper updated alongside it.
        path = re.sub(r"\(\?P<(\w+)>[^)]*\)", r"{\1}", path)
        out.append(f"{method} {path}")
    return sorted(out)


class TestManifestItself:
    def test_lists_are_sorted_and_unique(self, manifest):
        """Sorted + unique keeps diffs readable and makes duplicates impossible."""
        for key in ("mcp_tools", "http_routes", "recall_knobs",
                    "recall_expand_knobs"):
            values = manifest[key]
            assert values == sorted(values), f"{key} is not sorted"
            assert len(values) == len(set(values)), f"{key} has duplicates"


class TestMcpToolParity:
    def test_tool_names_match_the_manifest(self, manifest):
        assert _python_mcp_tools() == sorted(manifest["mcp_tools"])

    def test_no_tool_is_undocumented(self):
        """Every tool appears in the module docstring's inventory."""
        doc = mcp_srv.__doc__ or ""
        missing = [n for n in _python_mcp_tools() if f"\n    {n}(" not in doc]
        assert not missing, f"tools missing from the module docstring: {missing}"

    def test_docstring_count_matches(self):
        doc = mcp_srv.__doc__ or ""
        assert f"Tools exposed ({len(_python_mcp_tools())})" in doc


class TestHttpRouteParity:
    def test_routes_match_the_manifest(self, manifest):
        assert _python_http_routes() == sorted(manifest["http_routes"])

    def test_every_route_is_documented(self):
        doc = http_srv.__doc__ or ""
        missing = [
            entry for entry in _python_http_routes()
            # The docstring lists paths in the same {id} form but may append a
            # query string, so match the path prefix rather than the whole line.
            if entry.split(" ", 1)[1] not in doc
        ]
        assert not missing, f"routes missing from the module docstring: {missing}"


class TestRecallKnobParity:
    """The A1 regression, pinned: MCP recall must expose every ranking knob."""

    def test_recall_signature_matches_the_manifest(self, manifest):
        params = set(inspect.signature(mcp_srv.recall).parameters)
        expected = set(manifest["recall_knobs"]) | set(manifest["recall_expand_knobs"])
        assert expected <= params, f"recall is missing {sorted(expected - params)}"

    def test_recall_pack_carries_the_same_ranking_knobs(self, manifest):
        params = set(inspect.signature(mcp_srv.recall_pack).parameters)
        # recall_pack packs rather than paginates, so `k`/`overfetch`/`explain`
        # do not apply; everything that shapes *ranking* must be there.
        ranking = set(manifest["recall_knobs"]) - {"k", "overfetch", "explain",
                                                   "include_expired"}
        expected = ranking | set(manifest["recall_expand_knobs"])
        assert expected <= params, \
            f"recall_pack is missing {sorted(expected - params)}"

    def test_auto_context_carries_the_provenance_and_lexical_knobs(self):
        """auto_context is the fan-out an agent calls WITHOUT choosing a query.

        It cannot take the full ranking set — `fusion` is meaningless across a
        fan-out (RRF scores are rank-relative to each query's own pool), and the
        filters that scope a single query do not generalise. But the two knobs
        that decide *what may enter the context block at all* must be here, or
        the safest-by-default entry point is the one with no trust floor.
        """
        params = set(inspect.signature(mcp_srv.auto_context).parameters)
        assert {"min_trust", "lexical_index"} <= params, \
            f"auto_context is missing {sorted({'min_trust', 'lexical_index'} - params)}"
