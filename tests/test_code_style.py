"""Two house rules, enforced mechanically rather than by review.

Both have been fixed by hand more than once, which is the signal that a test
should hold them:

  * **No banner comments.** Decorative ``# --- section ---`` dividers add a
    second, un-checked table of contents that drifts from the code under it.
    Module and function docstrings carry the same information and cannot go
    stale silently.
  * **Imports at module top.** A deferred ``import`` inside a function hides
    the real dependency graph, makes import cost unpredictable, and lets an
    optional dependency masquerade as a required one until the branch runs.
    The one legitimate reason — breaking a cycle — is better fixed by moving
    the shared code (see ``curation.py``, which reaches the store through a
    method rather than importing back into it).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# A run of 3+ divider characters in a comment, with or without a label.
_BANNER = re.compile(
    r"""^\s*(?:\#|//)\s*
        (?:(?:--|──|==)\s*\S.*?)?      # optional "-- label"
        [-=─—~*_]{3,}\s*$              # the rule itself
    """,
    re.VERBOSE,
)

_SKIP_DIRS = {".venv", "__pycache__", "build", "dist", ".git", "node_modules"}


def _sources(*suffixes: str) -> list[Path]:
    out = []
    for suffix in suffixes:
        for path in REPO.rglob(f"*{suffix}"):
            if not any(part in _SKIP_DIRS for part in path.parts):
                out.append(path)
    return sorted(out)


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO))


def test_no_banner_comments():
    offenders = []
    for path in _sources(".py", ".go"):
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if _BANNER.match(line):
                offenders.append(f"{_rel(path)}:{lineno}: {line.strip()}")
    assert not offenders, (
        "banner-style divider comments found — use a docstring instead:\n  "
        + "\n  ".join(offenders)
    )


def test_no_function_level_imports():
    offenders = []
    for path in _sources(".py"):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:  # pragma: no cover — a broken file fails elsewhere
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for sub in ast.walk(node):
                if isinstance(sub, (ast.Import, ast.ImportFrom)):
                    name = getattr(sub, "module", None) or ", ".join(
                        a.name for a in sub.names)
                    offenders.append(
                        f"{_rel(path)}:{sub.lineno}: import {name} "
                        f"inside {node.name}()")
    assert not offenders, (
        "imports must be at module top:\n  " + "\n  ".join(offenders)
    )


@pytest.mark.parametrize("line,is_banner", [
    ("# --- helpers ---", True),
    ("    # -- writes -----------------", True),
    ("// --- helpers ---", True),
    ("# ────────────────────", True),
    ("# =====================", True),
    # Prose that merely contains a dash must not trip the check.
    ("# Skip self-loops and edges whose destination is already gone.", False),
    ("# rel=None removes every relation - not just the first", False),
    ("#", False),
    ("x = 1  # --- not a full-line comment", False),
])
def test_banner_pattern_discriminates(line, is_banner):
    """The detector has to be sharp enough to be worth having."""
    assert bool(_BANNER.match(line)) is is_banner
