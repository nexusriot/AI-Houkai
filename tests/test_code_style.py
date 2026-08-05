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

Scope notes, both learned from banners that slipped through an earlier pass:

  * The banner check reads every hand-maintained file with ``#``/``//``
    comments, not just ``.py`` and ``.go``. The packaged
    ``etc/ai-houkai/config.toml`` grew eight banner blocks precisely because
    it sat outside the old ``.py``/``.go`` net. Markdown is deliberately
    excluded: there ``---`` is front matter or a real horizontal rule.
  * A banner is recognised by how it *ends*, so the label may come first
    (``# Storage ------``) or sit between two rules (``# --- Storage ---``).
"""

from __future__ import annotations

import ast
import re
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# A full-line comment that trails off into a run of 3+ divider characters.
# Keying on the trailing rule catches every ordering of label and rule, while
# leaving prose that merely contains a dash alone.
_BANNER = re.compile(
    r"""^\s*(?:\#|//)      # the line is nothing but a comment
        .*?                # whatever it says
        [-=─—~*_]{3,}\s*$  # trailing decorative rule
    """,
    re.VERBOSE,
)

_SKIP_DIRS = {".venv", "__pycache__", "build", "dist", ".git", "node_modules",
              ".chroma", ".idea", ".pytest_cache", "ai_houkai.egg-info"}

# Hand-maintained files that carry `#`- or `//`-style comments.
_COMMENTED = (".py", ".go", ".sh", ".yml", ".yaml", ".toml",
              "Makefile", "Dockerfile")


def _sources(*patterns: str) -> list[Path]:
    """Every tracked file whose name ends in one of ``patterns``.

    Accepts extensions (``".py"``) and bare filenames (``"Makefile"``) alike.
    """
    out = []
    for pattern in patterns:
        for path in REPO.rglob(f"*{pattern}"):
            if path.is_file() and not any(p in _SKIP_DIRS for p in path.parts):
                out.append(path)
    return sorted(set(out))


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO))


def test_no_banner_comments():
    offenders = []
    for path in _sources(*_COMMENTED):
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if _BANNER.match(line):
                offenders.append(f"{_rel(path)}:{lineno}: {line.strip()}")
    assert not offenders, (
        "banner-style divider comments found — use a plain comment or a "
        "docstring instead:\n  " + "\n  ".join(offenders)
    )


def _import_name(node: ast.Import | ast.ImportFrom) -> str:
    """Render an import for a failure message, leading dots and all.

    ``ast`` keeps the dots of a relative import in ``level`` rather than in
    ``module``, so ``from .store import Memory`` has ``module == "store"``.
    Printing that verbatim would point at the wrong module.
    """
    if isinstance(node, ast.ImportFrom):
        return "." * (node.level or 0) + (node.module or "")
    return ", ".join(alias.name for alias in node.names)


def _deferred_imports(tree: ast.Module) -> list[ast.stmt]:
    """Imports that sit below module top level.

    Allowed, and therefore excluded: direct children of the module, plus
    anything inside a module-level ``if`` or ``try``. Those are the guards
    that make an optional dependency optional (``try: import openai``), keep a
    platform module off the wrong platform (``try: import fcntl``), or confine
    a typing-only import to ``if TYPE_CHECKING:``. All of them still read as
    imports at the top of the file, which is the property the rule is after.

    Everything else — function bodies, class bodies, nested scopes — is a
    deferred import.
    """
    allowed = set()
    for node in tree.body:
        allowed.add(id(node))
        if isinstance(node, (ast.If, ast.Try)):
            allowed.update(id(sub) for sub in ast.walk(node))
    return [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and id(node) not in allowed
    ]


def test_imports_are_at_module_top():
    offenders = []
    for path in _sources(".py"):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:  # pragma: no cover — a broken file fails elsewhere
            continue
        for node in _deferred_imports(tree):
            offenders.append(
                f"{_rel(path)}:{node.lineno}: import {_import_name(node)}")
    assert not offenders, (
        "imports must be at module top:\n  " + "\n  ".join(offenders)
    )


@pytest.mark.parametrize("line,is_banner", [
    ("# --- helpers ---", True),
    ("    # -- writes -----------------", True),
    ("// --- helpers ---", True),
    ("# ────────────────────", True),
    ("# =====================", True),
    # Label first, rule second — the ordering that slipped past the old check.
    ("# Storage ------------------------", True),
    ('# --- Ollama (used when embed_provider = "ollama") -----', True),
    # Prose that merely contains a dash must not trip the check.
    ("# Skip self-loops and edges whose destination is already gone.", False),
    ("# rel=None removes every relation - not just the first", False),
    ("#", False),
    ("x = 1  # --- not a full-line comment", False),
    # Two dividers is punctuation, not decoration.
    ("# an aside -- like this one", False),
])
def test_banner_pattern_discriminates(line, is_banner):
    """The detector has to be sharp enough to be worth having."""
    assert bool(_BANNER.match(line)) is is_banner


@pytest.mark.parametrize("source,expected", [
    ("import os", []),
    ("from __future__ import annotations\nimport os", []),
    # Module-level guards read as top-of-file imports and stay allowed.
    ("try:\n    import fcntl\nexcept ImportError:\n    fcntl = None", []),
    ("from typing import TYPE_CHECKING\n"
     "if TYPE_CHECKING:\n    from .store import Memory", []),
    # The deferrals the rule exists to catch.
    ("def f():\n    import os", ["os"]),
    ("async def f():\n    from . import store", ["."]),
    ("class C:\n    import os", ["os"]),
    ("def f():\n    if True:\n        import os", ["os"]),
    ("class C:\n    def m(self):\n        from .store import Memory",
     [".store"]),
])
def test_deferred_import_detection(source, expected):
    """Function bodies, class bodies and nested scopes all count as deferred."""
    found = _deferred_imports(ast.parse(textwrap.dedent(source)))
    assert [_import_name(node) for node in found] == expected
