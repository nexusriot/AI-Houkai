"""
AI-Houkai · Example 03 · Claude Desktop via MCP

Connects AI-Houkai's memory to the Claude Desktop application
using the Model Context Protocol (MCP).

How it works
────────────
  Claude Desktop  ──MCP──  ai_houkai/mcp_server/server.py  ──  ChromaDB
        │                         │
   (GUI chat)         exposes remember/recall/forget/stats

Claude Desktop reads a config file at startup that lists MCP servers.
This script generates / patches that config automatically, then
verifies the MCP server is importable and shows what Claude will see.

Run once to install:
    python examples/03_claude_desktop.py --install

Then restart Claude Desktop.  Claude will automatically call the
memory tools when relevant (no manual prompting needed).

Run without --install to just print the config block:
    python examples/03_claude_desktop.py
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import textwrap
import tempfile, shutil
import importlib, importlib.util, types as _types

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

from ai_houkai.memory_system import MemoryStore

DEFAULT_MEMORY_PATH = os.path.join(PROJECT_ROOT, ".chroma")


def _claude_config_path() -> str:
    """Return the platform-specific Claude Desktop config file path."""
    system = platform.system()
    if system == "Darwin":
        return os.path.expanduser(
            "~/Library/Application Support/Claude/claude_desktop_config.json"
        )
    if system == "Windows":
        appdata = os.environ.get("APPDATA", os.path.expanduser("~"))
        return os.path.join(appdata, "Claude", "claude_desktop_config.json")
    # Linux (unofficial Claude Desktop builds / Wine)
    return os.path.expanduser("~/.config/Claude/claude_desktop_config.json")


def _python_executable() -> str:
    """Prefer the venv Python if present, else the current interpreter."""
    venv_py = os.path.join(PROJECT_ROOT, ".venv", "bin", "python")
    if os.path.isfile(venv_py):
        return venv_py
    return sys.executable


def build_mcp_block(memory_path: str) -> dict:
    """Return the JSON block for claude_desktop_config.json."""
    return {
        "command": _python_executable(),
        "args": ["-m", "ai_houkai.mcp_server.server"],
        "cwd": PROJECT_ROOT,
        "env": {
            "AI_HOUKAI_PATH": memory_path,
            "AI_HOUKAI_COLLECTION": "ai_houkai",
        },
    }


def print_config(memory_path: str) -> None:
    block = {"mcpServers": {"ai-houkai": build_mcp_block(memory_path)}}
    print("\nAdd this to your claude_desktop_config.json:\n")
    print(json.dumps(block, indent=2))
    print(f"\nConfig file location on this machine:\n  {_claude_config_path()}\n")


def install(memory_path: str) -> None:
    cfg_path = _claude_config_path()
    os.makedirs(os.path.dirname(cfg_path), exist_ok=True)

    # Load existing config or start fresh
    config: dict = {}
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path) as f:
                config = json.load(f)
            print(f"  Found existing config: {cfg_path}")
        except json.JSONDecodeError:
            print(f"  ⚠  Could not parse {cfg_path} — will overwrite.")

    config.setdefault("mcpServers", {})
    config["mcpServers"]["ai-houkai"] = build_mcp_block(memory_path)

    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"  ✓  Written to {cfg_path}")
    print(f"  ✓  Memory path: {memory_path}")
    print("\n  ➜  Restart Claude Desktop to activate.\n")


def verify() -> None:
    """Smoke-test: import the MCP server and call stats()."""
    print("\nVerifying MCP server…")
    try:
        # Temporarily stub FastMCP so we don't need it installed for the check
        mcp_mod = _types.ModuleType("mcp")
        mcp_mod.server = _types.ModuleType("mcp.server")  # type: ignore
        fastmcp = _types.ModuleType("mcp.server.fastmcp")
        fastmcp.FastMCP = lambda name: _types.SimpleNamespace(  # type: ignore
            tool=lambda fn=None, **kw: (lambda f: f) if fn is None else fn
        )
        mcp_mod.server.fastmcp = fastmcp  # type: ignore
        sys.modules.setdefault("mcp", mcp_mod)
        sys.modules.setdefault("mcp.server", mcp_mod.server)
        sys.modules.setdefault("mcp.server.fastmcp", fastmcp)

        os.environ.setdefault("AI_HOUKAI_PATH", DEFAULT_MEMORY_PATH)
        spec = importlib.util.spec_from_file_location(
            "ai_houkai.mcp_server.server",
            os.path.join(PROJECT_ROOT, "ai_houkai", "mcp_server", "server.py"),
        )
        mod = importlib.util.module_from_spec(spec)  # type: ignore
        spec.loader.exec_module(mod)  # type: ignore

        count = mod.store.count()
        print(f"  ✓  MCP server importable  |  memory count = {count}")

        tools = ["remember", "recall", "forget", "list_recent", "stats"]
        for t in tools:
            assert hasattr(mod, t), f"missing tool: {t}"
        print(f"  ✓  Tools exposed: {', '.join(tools)}")

    except Exception as exc:
        print(f"  ✗  {exc}")
        sys.exit(1)


def demo_session() -> None:
    """
    Show a simulated Claude Desktop conversation to illustrate
    how the memory tools get called.  No network required.
    """


    tmp = tempfile.mkdtemp(prefix="ai_houkai_cd_demo_")
    store = MemoryStore(path=tmp)

    DEMO = textwrap.dedent("""
    ┌─────────────────────────────────────────────────────────┐
    │  Simulated Claude Desktop session with AI-Houkai memory │
    └─────────────────────────────────────────────────────────┘

    TURN 1 — User introduces themselves
    ──────────────────────────────────────────────────────────
    User: Hi! I'm a backend developer working with Python and FastAPI.
          I prefer short answers and I hate jargon.

    Claude → calls: remember("User is a backend developer, Python+FastAPI")
             calls: remember("User prefers short answers, dislikes jargon",
                             type="feedback", importance=0.95)

    Claude: Got it!  I'll keep things brief and practical.

    TURN 2 — Topic switch, memory recall in action
    ──────────────────────────────────────────────────────────
    User: How do I handle background tasks in FastAPI?

    Claude → calls: recall("FastAPI background tasks")
             (returns the user-profile memory with score 0.71)

    Claude: Use `BackgroundTasks`:

              @app.post("/send-email")
              async def send(bg: BackgroundTasks):
                  bg.add_task(send_email, to="...", body="...")

            Runs after the response is sent.  For heavier work use
            Celery or ARQ.

    TURN 3 — Preference recalled automatically
    ──────────────────────────────────────────────────────────
    User: Explain dependency injection in FastAPI.

    Claude → calls: recall("FastAPI dependency injection")
             (feedback memory: "short answers, no jargon" — score 0.68)

    Claude: `Depends()` injects shared logic into routes:

              def get_db(): yield SessionLocal()

              @app.get("/users")
              def users(db = Depends(get_db)): ...

            FastAPI resolves deps automatically; great for auth, DB, caching.

    ──────────────────────────────────────────────────────────
    Notice: Claude never asked the user to repeat their preferences.
    The memory system surfaced them automatically on every turn.
    """)
    print(DEMO)

    # Actually store + recall to show real scores
    m1 = store.remember("User is a backend developer, Python+FastAPI",
                        type="semantic", tags=["profile"], importance=0.8)
    m2 = store.remember("User prefers short answers, dislikes jargon",
                        type="feedback", tags=["style"], importance=0.95)

    print("  Real recall scores for 'FastAPI background tasks':")
    for m, s in store.recall("FastAPI background tasks", k=3):
        print(f"    {s:.3f}  {m.text[:60]}")

    shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="AI-Houkai Claude Desktop integration")
    ap.add_argument("--install", action="store_true",
                    help="Write config to Claude Desktop settings file")
    ap.add_argument("--memory-path", default=DEFAULT_MEMORY_PATH,
                    help=f"ChromaDB directory (default: {DEFAULT_MEMORY_PATH})")
    ap.add_argument("--verify", action="store_true",
                    help="Check MCP server is importable")
    ap.add_argument("--demo", action="store_true",
                    help="Show a simulated conversation")
    args = ap.parse_args()

    print(f"\nAI-Houkai · Claude Desktop integration")
    print(f"  Project root : {PROJECT_ROOT}")
    print(f"  Python       : {_python_executable()}")
    print(f"  Memory path  : {args.memory_path}")
    print(f"  Config file  : {_claude_config_path()}")

    if args.verify:
        verify()

    if args.demo:
        demo_session()

    if args.install:
        install(args.memory_path)
    else:
        print_config(args.memory_path)
        print("  Run with --install to write this automatically.\n")


if __name__ == "__main__":
    main()
