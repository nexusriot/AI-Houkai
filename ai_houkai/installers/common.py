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
import tempfile
import textwrap

from ai_houkai.memory_system import MemoryStore
from ai_houkai.mcp_server import server as srv

CONSOLE_SCRIPT = "ai-houkai-mcp"


def resolve_mcp_command() -> str:
    """Return the absolute path to the `ai-houkai-mcp` console script if found,
    otherwise the bare name (resolved via PATH at runtime)."""
    candidate = os.path.join(os.path.dirname(sys.executable), CONSOLE_SCRIPT)
    if os.path.isfile(candidate):
        return candidate
    return CONSOLE_SCRIPT


def load_json(path: str, *, overwrite_unparseable: bool = False) -> dict:
    """Load a JSON config file, returning {} when it is missing.

    An existing-but-invalid (or non-object) file raises ValueError by
    default: these are the user's own client configs, and install() writes
    the merged result back — treating garbage as {} would silently replace
    the whole file with just our server block. overwrite_unparseable=True
    opts into exactly that, after parking a ``.bak`` copy of the original.
    """
    if not os.path.isfile(path):
        return {}
    try:
        with open(path) as f:
            loaded = json.load(f)
    except json.JSONDecodeError as e:
        if overwrite_unparseable:
            shutil.copy2(path, path + ".bak")
            return {}
        raise ValueError(
            f"{path} is not valid JSON ({e}) — fix or remove it, then re-run"
        ) from e
    if isinstance(loaded, dict):
        return loaded
    if overwrite_unparseable:
        shutil.copy2(path, path + ".bak")
        return {}
    raise ValueError(
        f"{path}: expected a JSON object at the top level, "
        f"got {type(loaded).__name__} — fix or remove it, then re-run"
    )


def write_json(path: str, config: dict) -> str:
    """Atomically write `config` to `path` (creating parent dirs). Returns the path.

    Write-to-temp + os.replace so a crash mid-write can never leave the
    user's client config truncated."""
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=parent, prefix=".ahk-", suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(config, f, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def verify_server(
    *,
    memory_path: str | None = None,
    collection: str | None = None,
    stream=sys.stdout,
) -> bool:
    """Smoke-test shared across installers: the `ai-houkai-mcp` console script is
    reachable and the MCP server module imports with its core tools. Returns True
    on success.

    When *memory_path*/*collection* are given (the values the installer wrote
    into the client config), the reported store count comes from THAT store —
    not from whatever env-default store the server would open."""
    ok = True
    cmd = resolve_mcp_command()
    found = shutil.which(cmd) or (os.path.isfile(cmd) and cmd)
    if found:
        print(f"  ok   console script: {found}", file=stream)
    else:
        print(f"  err  '{cmd}' not on PATH — run: pip install ai-houkai",
              file=stream)
        ok = False

    tools = ["remember", "recall", "forget", "list_recent", "stats"]
    missing = [t for t in tools if not hasattr(srv, t)]
    if missing:
        print(f"  err  missing tools: {missing}", file=stream)
        ok = False
    else:
        try:
            count = _target_store_count(memory_path, collection)
            print(f"  ok   tools: {', '.join(tools)} | "
                  f"store count = {count}", file=stream)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"  err  store check failed: {exc}", file=stream)
            ok = False

    return ok


def _target_store_count(memory_path: str | None, collection: str | None) -> int:
    """Count memories in the store the installer is configuring.

    Opens the target store explicitly — verify is a smoke test, and creating
    the directory the MCP server will use anyway is fine. Without a
    memory_path, falls back to the server's own (env-configured) store."""
    if memory_path is None:
        return srv.get_store().count()
    store = MemoryStore(path=os.path.expanduser(memory_path),
                        collection=collection or "ai_houkai")
    try:
        return store.count()
    finally:
        store.client.close()


# A client-agnostic description of when/how to use the memory tools. Each
# installer wraps this in its own instruction-file format (CLAUDE.md, AGENTS.md,
# .cursor/rules/*.mdc, …).
MEMORY_GUIDE = textwrap.dedent("""
    You have access to a persistent memory store via AI-Houkai MCP tools:

    - **remember(text, type, tags, importance)** — store a fact, decision, or preference
    - **recall(query, k)** — semantic search across stored memories
    - **edit(memory_id, …)** — update a memory in place (keeps id, links, history)
    - **forget(memory_id)** — remove a specific memory
    - **list_recent()** — see the most recently created memories

    ### When to use memory

    | Situation | Action |
    |---|---|
    | User states a preference or coding convention | `remember` with `type="feedback"` or `"procedural"` |
    | You learn something about the codebase | `remember` with `type="semantic"` |
    | Starting a new task | `recall` relevant context first |
    | A stored fact is outdated or has a typo | `edit` it in place — don't forget+remember |
    | User corrects you | `remember` the correction, `forget` the wrong fact |

    ### Memory types
    - `episodic` — time-stamped events ("Fixed auth bug in PR #441")
    - `semantic` — distilled facts ("API versioned at /api/v1/")
    - `procedural` — how-to rules ("Always use tmp_path in tests")
    - `feedback` — user preferences ("Prefers concise answers")
""").strip()
