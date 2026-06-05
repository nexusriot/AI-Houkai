"""Shared helpers for the AI-Houkai client installers.

Each installer (Claude Code, Cursor, OpenCode, …) registers the same stdio
MCP server — `ai-houkai-mcp` — into a client-specific config file. The only
differences are the file location and the JSON schema the client expects.
This module collects the bits every installer needs.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import textwrap

CONSOLE_SCRIPT = "ai-houkai-mcp"


def resolve_mcp_command() -> str:
    """Return the absolute path to the `ai-houkai-mcp` console script if found,
    otherwise the bare name (resolved via PATH at runtime)."""
    candidate = os.path.join(os.path.dirname(sys.executable), CONSOLE_SCRIPT)
    if os.path.isfile(candidate):
        return candidate
    return CONSOLE_SCRIPT


def load_json(path: str, *, overwrite_unparseable: bool = True) -> dict:
    """Load a JSON config file, returning {} if missing or (optionally) invalid."""
    if not os.path.isfile(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError:
        if not overwrite_unparseable:
            raise
        return {}


def write_json(path: str, config: dict) -> str:
    """Write `config` to `path` (creating parent dirs). Returns the path."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")
    return path


def verify_server(server_name: str, *, stream=sys.stdout) -> bool:
    """Smoke-test shared across installers: the `ai-houkai-mcp` console script is
    reachable and the MCP server module imports with its core tools. Returns True
    on success."""
    ok = True
    cmd = resolve_mcp_command()
    found = shutil.which(cmd) or (os.path.isfile(cmd) and cmd)
    if found:
        print(f"  ok   console script: {found}", file=stream)
    else:
        print(f"  err  '{cmd}' not on PATH — run: pip install ai-houkai",
              file=stream)
        ok = False

    try:
        from ai_houkai.mcp_server import server as srv
        tools = ["remember", "recall", "forget", "list_recent", "stats"]
        missing = [t for t in tools if not hasattr(srv, t)]
        if missing:
            print(f"  err  missing tools: {missing}", file=stream)
            ok = False
        else:
            print(f"  ok   tools: {', '.join(tools)} | "
                  f"store count = {srv.store.count()}", file=stream)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"  err  import failed: {exc}", file=stream)
        ok = False

    return ok


# A client-agnostic description of when/how to use the memory tools. Each
# installer wraps this in its own instruction-file format (CLAUDE.md, AGENTS.md,
# .cursor/rules/*.mdc, …).
MEMORY_GUIDE = textwrap.dedent("""
    You have access to a persistent memory store via AI-Houkai MCP tools:

    - **remember(text, type, tags, importance)** — store a fact, decision, or preference
    - **recall(query, k)** — semantic search across stored memories
    - **forget(memory_id)** — remove a specific memory
    - **list_recent()** — see the most recently created memories

    ### When to use memory

    | Situation | Action |
    |---|---|
    | User states a preference or coding convention | `remember` with `type="feedback"` or `"procedural"` |
    | You learn something about the codebase | `remember` with `type="semantic"` |
    | Starting a new task | `recall` relevant context first |
    | User corrects you | `remember` the correction, `forget` the wrong fact |

    ### Memory types
    - `episodic` — time-stamped events ("Fixed auth bug in PR #441")
    - `semantic` — distilled facts ("API versioned at /api/v1/")
    - `procedural` — how-to rules ("Always use tmp_path in tests")
    - `feedback` — user preferences ("Prefers concise answers")
""").strip()
