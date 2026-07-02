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
  ~/.claude.json             list_recent / stats
  (or project .claude/)

The actual install/verify/print logic lives in
`ai_houkai.installers.claude_code` (a reusable library module).  This
example wires it up to a CLI plus an offline `--demo` that simulates a
multi-session coding workflow.

Three ways to register the MCP server
  Option A — one-liner (recommended):
      claude mcp add ai-houkai -- ai-houkai-mcp

  Option B — installer console script (after `pip install ai-houkai`):
      ai-houkai-install-claude-code --install

  Option C — this example script (registers via `claude mcp add` or ~/.claude.json):
      python examples/06_claude_code.py --install
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import textwrap

from ai_houkai.installers.claude_code import (
    ClaudeCodeInstaller,
    DEFAULT_MEMORY_PATH,
    DEFAULT_CONFIG_PATH,
)
from ai_houkai.memory_system import MemoryStore


def cmd_demo(memory_path: str) -> None:
    """Simulate a multi-session coding workflow with active memory."""
    tmp = tempfile.mkdtemp(prefix="ai_houkai_cc_demo_")
    store = MemoryStore(path=tmp)

    print(textwrap.dedent("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║   Simulated Claude Code sessions with AI-Houkai memory        ║
    ╚═══════════════════════════════════════════════════════════════╝

    SESSION 1 — project onboarding
    ───────────────────────────────────────────────────────────────
    User:  We use pytest with tmp_path fixtures for test isolation.
           Never use EphemeralClient — it leaks state across tests.

    Claude → remember(... type="procedural", importance=0.9 ...)

    SESSION 2 — a week later, new task
    ───────────────────────────────────────────────────────────────
    User:  Write a pytest fixture for our vector store.
    Claude → recall("pytest fixture vector store")
             ┌─ score 0.91: tmp_path fixtures for test isolation
             └─ score 0.88: never use EphemeralClient
    Claude: Here's the fixture, using tmp_path per your conventions.
    """))

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

    print(f"  Seeded {store.count()} project memories.\n")

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


def main() -> None:
    ap = argparse.ArgumentParser(
        description="AI-Houkai · Claude Code MCP integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--install", action="store_true",
                    help="Register the MCP server (claude mcp add, or ~/.claude.json directly)")
    ap.add_argument("--memory-path", default=DEFAULT_MEMORY_PATH,
                    metavar="PATH",
                    help=f"ChromaDB directory (default: {DEFAULT_MEMORY_PATH})")
    ap.add_argument("--verify", action="store_true",
                    help="Smoke-test the MCP server and claude CLI")
    ap.add_argument("--demo", action="store_true",
                    help="Simulate a coding session with active memory")
    ap.add_argument("--claudemd", action="store_true",
                    help="Print a CLAUDE.md snippet that instructs Claude how to use memory")
    args = ap.parse_args()

    inst = ClaudeCodeInstaller(memory_path=args.memory_path)

    print(f"\nAI-Houkai · Claude Code integration")
    print(f"  Config file   : {inst.config_path}")
    print(f"  Memory path   : {inst.memory_path}")
    print(f"  MCP command   : {inst.mcp_command}")

    if args.verify:
        inst.verify()

    if args.demo:
        cmd_demo(args.memory_path)

    if args.claudemd:
        print("\n── CLAUDE.md snippet ─────────────────────────────────────\n")
        print(inst.claudemd_snippet())
        print("\n──────────────────────────────────────────────────────────")
        print("Paste into your project's CLAUDE.md to guide Claude Code.\n")

    if args.install:
        written = inst.install()
        print(f"  ✓ registered via: {written}")
        print(f"  Verify with: claude mcp list\n")
    elif not (args.verify or args.demo or args.claudemd):
        inst.print_config()
        print("  Run with --install to write this automatically.")
        print(f"  Or: claude mcp add {inst.server_name} -- {inst.mcp_command}\n")


if __name__ == "__main__":
    main()
