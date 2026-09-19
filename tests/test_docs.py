"""Numbers in the documentation must match the thing they describe.

`docs/DESIGN.md` carries a per-file table of what each test module covers, and
both it and the README quote a headline total. Every one of those numbers is a
hand-maintained copy of something pytest already knows, and an audit found
thirteen of them stale, two modules missing from the table entirely, and a
"needs_model" count off by one — none of which any reader could have caught
without re-running the suite.

The same went for the size of the remote surface: `parity.json` is the source
of truth and both ports assert against it, but nothing checked the counts
*written about* it, and three of `go/README.md`'s had sat at 22 tools through
the growth to 41. This module covers the Python docs;
`go/internal/parity/docs_test.go` covers the Go ones, because parity.json's
rule is that each port asserts against the manifest in its own suite.

The prose is the part worth writing by hand; the counts are not. These tests
hold the counts, so a new test module, a new case or a new tool fails here —
pointing at the line to edit — instead of quietly making the documentation
wrong.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DESIGN = REPO / "docs" / "DESIGN.md"
README = REPO / "README.md"

# `| `test_foo.py` | 12 | …` — the table rows in DESIGN.md's test inventory.
_ROW = re.compile(r"^\| `(test_\w+\.py)` \| (\d+) \|", re.MULTILINE)
# `### 1356 tests across 47 files`, and the same pair in the README's comments.
_HEADLINE = re.compile(r"(\d+) tests across (\d+) files")


def _collect(*extra: str) -> Counter:
    """Test counts per module, straight from pytest's own collection.

    A subprocess rather than an in-process collect: nesting pytest inside a
    running session shares plugin state, and the point here is to ask the same
    question a developer asks from the shell.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "--collect-only", "-q",
         "-p", "no:randomly", "-p", "no:cacheprovider", *extra],
        cwd=REPO, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    counts: Counter = Counter()
    for line in proc.stdout.splitlines():
        if "::" in line:
            counts[line.split("::")[0].split("/")[-1]] += 1
    assert counts, f"collected nothing:\n{proc.stdout[-2000:]}"
    return counts


@pytest.fixture(scope="module")
def collected() -> Counter:
    return _collect()


def test_design_table_covers_every_test_module(collected: Counter) -> None:
    rows = {m.group(1): int(m.group(2)) for m in _ROW.finditer(DESIGN.read_text())}
    missing = sorted(set(collected) - set(rows))
    phantom = sorted(set(rows) - set(collected))
    assert not missing, (
        "test modules with no row in the docs/DESIGN.md table — add one "
        f"saying what they cover: {missing}")
    assert not phantom, (
        f"docs/DESIGN.md lists test modules that no longer exist: {phantom}")


def test_design_table_counts_are_current(collected: Counter) -> None:
    rows = {m.group(1): int(m.group(2)) for m in _ROW.finditer(DESIGN.read_text())}
    stale = {f: (n, collected[f]) for f, n in rows.items()
             if f in collected and n != collected[f]}
    assert not stale, (
        "docs/DESIGN.md test counts are stale (file: documented -> actual): "
        f"{stale}")


@pytest.mark.parametrize("doc", [DESIGN, README])
def test_headline_totals_are_current(doc: Path, collected: Counter) -> None:
    """`N tests across M files` — quoted in DESIGN.md's heading and twice in
    the README's tree/commands."""
    found = _HEADLINE.findall(doc.read_text())
    assert found, f"{doc.name} no longer quotes a test total"
    want = (str(sum(collected.values())), str(len(collected)))
    wrong = [pair for pair in found if pair != want]
    assert not wrong, (
        f"{doc.name} claims {wrong}, actual {want} — "
        "'<total> tests across <files> files'")


def test_needs_model_count_is_current() -> None:
    """The fast suite's promise: everything else runs on FakeEmbedder, and
    exactly this many tests opt back into a real sentence-transformers model."""
    marked = sum(_collect("-m", "needs_model").values())
    for doc in (DESIGN, README):
        quoted = re.findall(r"\*{0,2}(\d+) tests?\*{0,2} (?:are )?marked "
                            r"`needs_model`", doc.read_text())
        assert quoted, f"{doc.name} no longer quotes the needs_model count"
        assert all(int(q) == marked for q in quoted), (
            f"{doc.name} says {quoted} tests are marked needs_model, "
            f"actual {marked}")


_PARITY = REPO / "parity.json"
# Any "<N> tools" / "<N> MCP tools" / "<N> HTTP routes" claim, however it is
# punctuated: `**41 MCP tools**`, `(41 tools)`, `— 41 tools`, `routes (41)`.
_TOOL_CLAIM = re.compile(r"(\d+)\s*(?:MCP\s+)?tools?\b")
_ROUTE_CLAIM = re.compile(r"(?:(\d+)\s*HTTP\s+routes?\b|HTTP\s+routes?\s*\((\d+)\))")


@pytest.mark.parametrize("doc", [DESIGN, README])
def test_documented_surface_sizes_match_parity(doc: Path) -> None:
    """Prose quotes the size of the remote surface in a dozen places.

    `parity.json` is the source of truth for the surface itself and both ports
    assert against it — but nothing checked the counts *written about* it, and
    three of go/README.md's had been left at 22 through the growth to 41.

    Only this port's docs. parity.json states the rule for itself: each port
    asserts against the manifest "in their own test suites, so neither CI job
    needs the other toolchain". The Go docs are covered by the same check in
    `go/internal/parity/docs_test.go`; reaching across from here also broke
    concretely, because the hermetic Python image excludes the whole Go tree.
    """
    parity = json.loads(_PARITY.read_text())
    text = doc.read_text()

    tools = str(len(parity["mcp_tools"]))
    wrong = [n for n in _TOOL_CLAIM.findall(text) if n != tools]
    assert not wrong, (
        f"{doc.name} quotes {wrong} tools; parity.json lists {tools}")

    routes = str(len(parity["http_routes"]))
    found = [a or b for a, b in _ROUTE_CLAIM.findall(text)]
    wrong = [n for n in found if n != routes]
    assert not wrong, (
        f"{doc.name} quotes {wrong} HTTP routes; parity.json lists {routes}")
