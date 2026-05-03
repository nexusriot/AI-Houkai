"""
AI-Houkai · Example 06 · Claude Code via MCP

Gives Claude Code a persistent memory so it remembers project conventions,
past debugging sessions, architectural decisions, and your preferences across
every coding session — without you repeating yourself.

How it works
────────────
  Claude Code CLI  ──MCP──  ai-houkai-mcp  ──  ChromaDB on disk
       │                          │
  ~/.claude/                 remember / recall / forget
  settings.json              list_recent / stats
  (or project .claude/)

Two ways to register the MCP server
  Option A — one-liner (recommended):

      claude mcp add ai-houkai -- ai-houkai-mcp

  Option B — this script (auto-patches ~/.claude/settings.json):

      python examples/06_claude_code.py --install

  Option C — manual (paste into ~/.claude/settings.json):

      python examples/06_claude_code.py          # prints the block

After registering, Claude Code calls the memory tools automatically
whenever relevant — no manual prompting needed.

Flags
─────
  --install             Patch ~/.claude/settings.json and exit
  --memory-path PATH    Override ChromaDB directory (default: ~/.ai_houkai)
  --verify              Smoke-test that the MCP server is importable
  --demo                Simulate a coding session with active memory
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap

from ai_houkai.memory_system import MemoryStore

# ── paths ─────────────────────────────────────────────────────────────────────

SETTINGS_PATH   = os.path.expanduser("~/.claude/settings.json")
DEFAULT_MEM_PATH = os.path.expanduser("~/.ai_houkai")


# ── helpers ───────────────────────────────────────────────────────────────────

def _mcp_command() -> str:
    """Return the absolute path to the ai-houkai-mcp console script."""
    # Prefer the same interpreter's Scripts/ or bin/ directory
    scripts = os.path.join(os.path.dirname(sys.executable), "ai-houkai-mcp")
    if os.path.isfile(scripts):
        return scripts
    return "ai-houkai-mcp"   # fall back to PATH lookup


def _build_mcp_block(memory_path: str) -> dict:
    return {
        "command": _mcp_command(),
        "env": {
            "AI_HOUKAI_PATH":       memory_path,
            "AI_HOUKAI_COLLECTION": "claude_code",
        },
    }


# ── sub-commands ───────────────────────────────────────────────────────────────

def cmd_print(memory_path: str) -> None:
    """Print the JSON block to paste into settings.json."""
    block = {"mcpServers": {"ai-houkai": _build_mcp_block(memory_path)}}
    print("\nPaste this into ~/.claude/settings.json:\n")
    print(json.dumps(block, indent=2))
    print(f"\nSettings file: {SETTINGS_PATH}")
    print("\nOr run once with `claude mcp add`:")
    print(f"    claude mcp add ai-houkai -- {_mcp_command()}")
    print("\nThen: claude  (no restart needed — Claude Code hot-reloads MCP)\n")


def cmd_install(memory_path: str) -> None:
    """Patch ~/.claude/settings.json with the MCP server block."""
    os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)

    config: dict = {}
    if os.path.isfile(SETTINGS_PATH):
        try:
            with open(SETTINGS_PATH) as f:
                config = json.load(f)
            print(f"  Found existing settings: {SETTINGS_PATH}")
        except json.JSONDecodeError:
            print(f"  ⚠  Could not parse {SETTINGS_PATH} — will overwrite.")

    config.setdefault("mcpServers", {})
    config["mcpServers"]["ai-houkai"] = _build_mcp_block(memory_path)

    with open(SETTINGS_PATH, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    print(f"  ✓  Written:      {SETTINGS_PATH}")
    print(f"  ✓  Memory path:  {memory_path}")
    print(f"  ✓  MCP command:  {_mcp_command()}")
    print("\n  Claude Code picks up MCP changes on the next invocation.")
    print("  Verify with:  claude mcp list\n")


def cmd_verify() -> None:
    """Smoke-test: import the server module and check all tools are present."""
    print("\nVerifying MCP server…")

    # Check console script
    cmd = _mcp_command()
    found = shutil.which(cmd) or (os.path.isfile(cmd) and cmd)
    if found:
        print(f"  ✓  Console script: {found}")
    else:
        print(f"  ✗  '{cmd}' not found in PATH — run: pip install ai-houkai")
        sys.exit(1)

    # Import the server module and check tools
    try:
        from ai_houkai.mcp_server import server as srv
        tools = ["remember", "recall", "forget", "list_recent", "stats"]
        for t in tools:
            assert hasattr(srv, t), f"missing tool: {t}"
        count = srv.store.count()
        print(f"  ✓  Module importable | store count = {count}")
        print(f"  ✓  Tools: {', '.join(tools)}")
    except Exception as exc:
        print(f"  ✗  {exc}")
        sys.exit(1)

    # Optionally check claude CLI
    if shutil.which("claude"):
        try:
            result = subprocess.run(
                ["claude", "mcp", "list"], capture_output=True, text=True, timeout=5
            )
            if "ai-houkai" in result.stdout:
                print("  ✓  Registered in `claude mcp list`")
            else:
                print("  ⚠  Not yet in `claude mcp list` — run --install first")
        except Exception:
            pass
    else:
        print("  ⚠  `claude` CLI not found — install from https://claude.ai/claude-code")

    print()


def cmd_demo(memory_path: str) -> None:
    """
    Simulate a multi-session coding workflow to show what memory enables.
    Seeds real memories and shows live recall scores.
    """
    tmp = tempfile.mkdtemp(prefix="ai_houkai_cc_demo_")
    store = MemoryStore(path=tmp)

    print(textwrap.dedent("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║   Simulated Claude Code sessions with AI-Houkai memory        ║
    ╚═══════════════════════════════════════════════════════════════╝

    SESSION 1 — project onboarding
    ───────────────────────────────────────────────────────────────
    $ claude

    User:  We use pytest with tmp_path fixtures for test isolation.
           Never use EphemeralClient — it leaks state across tests.
           Auth tokens go in env vars, never in code.

    Claude → remember("Use pytest tmp_path fixtures for test isolation",
                       type="procedural", importance=0.9, tags=["testing"])
           → remember("Never use EphemeralClient: leaks state across tests",
                       type="procedural", importance=0.95, tags=["testing","chromadb"])
           → remember("Auth tokens in env vars only, never hardcoded",
                       type="procedural", importance=0.95, tags=["security"])

    Claude: Noted — I'll apply these conventions to all code I write.

    SESSION 2 — a week later, new task
    ───────────────────────────────────────────────────────────────
    $ claude

    User:  Write a pytest fixture for our vector store.

    Claude → recall("pytest fixture vector store")
             ┌─ score 0.91: "Use pytest tmp_path fixtures for test isolation"
             └─ score 0.88: "Never use EphemeralClient: leaks state across tests"

    Claude: Here's the fixture, using tmp_path per your conventions:

              @pytest.fixture()
              def store(tmp_path):
                  return MemoryStore(
                      path=str(tmp_path / "chroma"),
                      collection="test",
                  )

    SESSION 3 — debugging, memory carries forward
    ───────────────────────────────────────────────────────────────
    $ claude

    User:  Our CI is failing — tests are stepping on each other.

    Claude → recall("test isolation CI failure")
             ┌─ score 0.87: "Never use EphemeralClient: leaks state across tests"

    Claude: Likely cause: EphemeralClient shares in-process state.
            Check if any fixture uses it and switch to PersistentClient
            + tmp_path.  That's the convention established in this project.
    ───────────────────────────────────────────────────────────────
    Claude Code never had to ask you to repeat the testing conventions.
    """))

    # Seed the demo memories
    store.remember("Use pytest tmp_path fixtures for test isolation",
                   type="procedural", importance=0.9, tags=["testing"])
    store.remember("Never use EphemeralClient: leaks state across tests",
                   type="procedural", importance=0.95, tags=["testing", "chromadb"])
    store.remember("Auth tokens in env vars only, never hardcoded",
                   type="procedural", importance=0.95, tags=["security"])
    store.remember("API follows REST conventions, versioned at /api/v1/",
                   type="semantic", importance=0.8, tags=["api", "architecture"])
    store.remember("Deployed on Kubernetes; use ConfigMaps for non-secret config",
                   type="procedural", importance=0.85, tags=["deploy", "k8s"])

    print(f"  Seeded {store.count()} project memories.")
    print()

    queries = [
        "pytest fixture vector store",
        "test isolation CI failure",
        "how to add a new API endpoint",
        "deploy configuration secret",
    ]
    for q in queries:
        hits = store.recall(q, k=2)
        print(f"  recall({q!r})")
        for m, s in hits:
            print(f"    {s:.3f}  [{m.type}]  {m.text[:65]}")
        print()

    shutil.rmtree(tmp, ignore_errors=True)


# ── CLAUDE.md note ─────────────────────────────────────────────────────────────

CLAUDEMD_SNIPPET = textwrap.dedent("""
    ## Memory (AI-Houkai MCP)

    You have access to a persistent memory store via MCP tools:

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


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="AI-Houkai · Claude Code MCP integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--install", action="store_true",
                    help=f"Write MCP block to {SETTINGS_PATH}")
    ap.add_argument("--memory-path", default=DEFAULT_MEM_PATH,
                    metavar="PATH",
                    help=f"ChromaDB directory (default: {DEFAULT_MEM_PATH})")
    ap.add_argument("--verify", action="store_true",
                    help="Smoke-test the MCP server and claude CLI")
    ap.add_argument("--demo", action="store_true",
                    help="Simulate a coding session with active memory")
    ap.add_argument("--claudemd", action="store_true",
                    help="Print a CLAUDE.md snippet that instructs Claude how to use memory")
    args = ap.parse_args()

    print(f"\nAI-Houkai · Claude Code integration")
    print(f"  Settings file : {SETTINGS_PATH}")
    print(f"  Memory path   : {args.memory_path}")
    print(f"  MCP command   : {_mcp_command()}")

    if args.verify:
        cmd_verify()

    if args.demo:
        cmd_demo(args.memory_path)

    if args.claudemd:
        print("\n── CLAUDE.md snippet ─────────────────────────────────────\n")
        print(CLAUDEMD_SNIPPET)
        print("\n──────────────────────────────────────────────────────────")
        print("Paste this into your project's CLAUDE.md to guide Claude Code.\n")

    if args.install:
        cmd_install(args.memory_path)
    else:
        cmd_print(args.memory_path)
        print("  Run with --install to write this automatically.")
        print("  Or: claude mcp add ai-houkai -- ai-houkai-mcp\n")


if __name__ == "__main__":
    main()
