"""
Claude Code installer for the AI-Houkai MCP server.

Registers `ai-houkai-mcp` in `~/.claude/settings.json` (or a project-level
`.claude/settings.json`) so the Claude Code CLI launches the memory MCP
server automatically.

Library use:

    from ai_houkai.installers import ClaudeCodeInstaller

    inst = ClaudeCodeInstaller(memory_path="~/.ai_houkai")
    inst.install()                # patch settings.json
    inst.print_config()           # preview the JSON block
    inst.verify()                 # smoke-test the server + CLI
    print(inst.claudemd_snippet())

CLI use (also exposed as the `ai-houkai-install-claude-code` console script):

    python -m ai_houkai.installers.claude_code --install
    python -m ai_houkai.installers.claude_code --verify
    python -m ai_houkai.installers.claude_code --claudemd
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from typing import Optional

from ai_houkai.mcp_server import server as srv


DEFAULT_SETTINGS_PATH = os.path.expanduser("~/.claude/settings.json")
DEFAULT_MEMORY_PATH   = os.path.expanduser("~/.ai_houkai")
DEFAULT_COLLECTION    = "claude_code"
SERVER_NAME           = "ai-houkai"
CONSOLE_SCRIPT        = "ai-houkai-mcp"


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


def _resolve_mcp_command() -> str:
    """Return the absolute path to the ai-houkai-mcp console script if found."""
    scripts = os.path.join(os.path.dirname(sys.executable), CONSOLE_SCRIPT)
    if os.path.isfile(scripts):
        return scripts
    return CONSOLE_SCRIPT  # PATH lookup at runtime


@dataclass
class ClaudeCodeInstaller:
    """Register the AI-Houkai MCP server with Claude Code."""

    memory_path:   str = DEFAULT_MEMORY_PATH
    collection:    str = DEFAULT_COLLECTION
    settings_path: str = DEFAULT_SETTINGS_PATH
    server_name:   str = SERVER_NAME
    extra_env:     dict = field(default_factory=dict)

    @property
    def mcp_command(self) -> str:
        return _resolve_mcp_command()

    def build_mcp_block(self) -> dict:
        env = {
            "AI_HOUKAI_PATH":       self.memory_path,
            "AI_HOUKAI_COLLECTION": self.collection,
            **self.extra_env,
        }
        return {"command": self.mcp_command, "env": env}

    def build_settings_block(self) -> dict:
        return {"mcpServers": {self.server_name: self.build_mcp_block()}}

    def install(self, *, overwrite_unparseable: bool = True) -> str:
        """Patch settings.json with the MCP server block. Returns the path written."""
        os.makedirs(os.path.dirname(self.settings_path), exist_ok=True)

        config: dict = {}
        if os.path.isfile(self.settings_path):
            try:
                with open(self.settings_path) as f:
                    config = json.load(f)
            except json.JSONDecodeError:
                if not overwrite_unparseable:
                    raise
                config = {}

        config.setdefault("mcpServers", {})
        config["mcpServers"][self.server_name] = self.build_mcp_block()

        with open(self.settings_path, "w") as f:
            json.dump(config, f, indent=2)
            f.write("\n")
        return self.settings_path

    def print_config(self, *, stream=sys.stdout) -> None:
        block = self.build_settings_block()
        print(f"\nPaste this into {self.settings_path}:\n", file=stream)
        print(json.dumps(block, indent=2), file=stream)
        print(f"\nOr run once with `claude mcp add`:", file=stream)
        print(f"    claude mcp add {self.server_name} -- {self.mcp_command}\n",
              file=stream)

    def verify(self, *, stream=sys.stdout) -> bool:
        """Smoke-test: console script present + server module importable.
        Returns True on success."""
        ok = True
        cmd = self.mcp_command
        found = shutil.which(cmd) or (os.path.isfile(cmd) and cmd)
        if found:
            print(f"  ok   console script: {found}", file=stream)
        else:
            print(f"  err  '{cmd}' not on PATH — run: pip install ai-houkai",
                  file=stream)
            ok = False

        try:
            tools = ["remember", "recall", "forget", "list_recent", "stats"]
            missing = [t for t in tools if not hasattr(srv, t)]
            if missing:
                print(f"  err  missing tools: {missing}", file=stream)
                ok = False
            else:
                print(f"  ok   tools: {', '.join(tools)} | "
                      f"store count = {srv.store.count()}", file=stream)
        except Exception as exc:
            print(f"  err  import failed: {exc}", file=stream)
            ok = False

        if shutil.which("claude"):
            try:
                result = subprocess.run(
                    ["claude", "mcp", "list"],
                    capture_output=True, text=True, timeout=5,
                )
                if self.server_name in result.stdout:
                    print(f"  ok   registered in `claude mcp list`", file=stream)
                else:
                    print(f"  warn not yet in `claude mcp list` — run install first",
                          file=stream)
            except Exception:
                pass
        else:
            print(f"  warn `claude` CLI not on PATH", file=stream)
        return ok

    @staticmethod
    def claudemd_snippet() -> str:
        return CLAUDEMD_SNIPPET


def _main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="ai-houkai-install-claude-code",
        description="Register the AI-Houkai MCP server with Claude Code.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--install", action="store_true",
                    help="Write MCP block to settings.json")
    ap.add_argument("--memory-path", default=DEFAULT_MEMORY_PATH, metavar="PATH",
                    help=f"ChromaDB directory (default: {DEFAULT_MEMORY_PATH})")
    ap.add_argument("--collection", default=DEFAULT_COLLECTION,
                    help=f"Collection name (default: {DEFAULT_COLLECTION})")
    ap.add_argument("--settings", default=DEFAULT_SETTINGS_PATH,
                    help=f"Path to settings.json (default: {DEFAULT_SETTINGS_PATH})")
    ap.add_argument("--verify", action="store_true",
                    help="Smoke-test the MCP server")
    ap.add_argument("--claudemd", action="store_true",
                    help="Print a CLAUDE.md memory-usage snippet")
    args = ap.parse_args(argv)

    inst = ClaudeCodeInstaller(
        memory_path=args.memory_path,
        collection=args.collection,
        settings_path=args.settings,
    )

    print(f"\nAI-Houkai · Claude Code installer")
    print(f"  Settings file : {inst.settings_path}")
    print(f"  Memory path   : {inst.memory_path}")
    print(f"  MCP command   : {inst.mcp_command}\n")

    if args.verify:
        ok = inst.verify()
        if not ok:
            return 1

    if args.claudemd:
        print("\n── CLAUDE.md snippet ─────────────────────────────────────\n")
        print(inst.claudemd_snippet())
        print("\n──────────────────────────────────────────────────────────\n")

    if args.install:
        path = inst.install()
        print(f"  written: {path}")
        print(f"  verify:  claude mcp list\n")
    elif not (args.verify or args.claudemd):
        inst.print_config()
        print("  Run with --install to write this automatically.")
        print(f"  Or: claude mcp add {inst.server_name} -- {inst.mcp_command}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
