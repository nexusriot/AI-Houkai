"""
Cursor installer for the AI-Houkai MCP server.

Registers `ai-houkai-mcp` in Cursor's MCP config so the editor's agent gains a
persistent memory. Cursor uses the same `mcpServers` schema as Claude Desktop /
Claude Code, but reads it from a different file:

    ~/.cursor/mcp.json            global (all projects)
    <project>/.cursor/mcp.json    project-scoped

Library use:

    from ai_houkai.installers import CursorInstaller

    inst = CursorInstaller(memory_path="~/.ai_houkai")
    inst.install()                # patch ~/.cursor/mcp.json
    inst.print_config()           # preview the JSON block
    inst.verify()                 # smoke-test the server
    print(inst.rule_snippet())    # .cursor/rules/*.mdc content

CLI use (also exposed as the `ai-houkai-install-cursor` console script):

    python -m ai_houkai.installers.cursor --install
    python -m ai_houkai.installers.cursor --project        # ./.cursor/mcp.json
    python -m ai_houkai.installers.cursor --verify
    python -m ai_houkai.installers.cursor --rule
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from ai_houkai.installers.common import (
    MEMORY_GUIDE,
    load_json,
    resolve_mcp_command,
    verify_server,
    write_json,
)

GLOBAL_CONFIG_PATH  = os.path.expanduser("~/.cursor/mcp.json")
PROJECT_CONFIG_PATH = os.path.join(".cursor", "mcp.json")
DEFAULT_MEMORY_PATH = os.path.expanduser("~/.ai_houkai")
DEFAULT_COLLECTION  = "cursor"
SERVER_NAME         = "ai-houkai"


# Cursor reads project rules from `.cursor/rules/*.mdc` — Markdown with a small
# YAML frontmatter. `alwaysApply: true` keeps the rule in context every request.
RULE_SNIPPET = (
    "---\n"
    "description: AI-Houkai persistent memory — when and how to use the MCP tools\n"
    "alwaysApply: true\n"
    "---\n\n"
    "# Memory (AI-Houkai MCP)\n\n"
    f"{MEMORY_GUIDE}"
)


@dataclass
class CursorInstaller:
    """Register the AI-Houkai MCP server with Cursor."""

    memory_path:   str = DEFAULT_MEMORY_PATH
    collection:    str = DEFAULT_COLLECTION
    settings_path: str = GLOBAL_CONFIG_PATH
    server_name:   str = SERVER_NAME
    extra_env:     dict = field(default_factory=dict)

    @property
    def mcp_command(self) -> str:
        return resolve_mcp_command()

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
        """Patch the Cursor mcp.json with the MCP server block. Returns the path."""
        config = load_json(self.settings_path,
                           overwrite_unparseable=overwrite_unparseable)
        config.setdefault("mcpServers", {})
        config["mcpServers"][self.server_name] = self.build_mcp_block()
        return write_json(self.settings_path, config)

    def print_config(self, *, stream=sys.stdout) -> None:
        block = self.build_settings_block()
        print(f"\nPaste this into {self.settings_path}:\n", file=stream)
        print(json.dumps(block, indent=2), file=stream)
        print("\nThen reload Cursor and open Settings → MCP to confirm "
              f"'{self.server_name}' is listed.\n", file=stream)

    def verify(self, *, stream=sys.stdout) -> bool:
        ok = verify_server(self.server_name, memory_path=self.memory_path,
                           collection=self.collection, stream=stream)
        if os.path.isfile(self.settings_path):
            cfg = load_json(self.settings_path)
            if self.server_name in cfg.get("mcpServers", {}):
                print(f"  ok   registered in {self.settings_path}", file=stream)
            else:
                print(f"  warn not yet in {self.settings_path} — run --install",
                      file=stream)
        else:
            print(f"  warn {self.settings_path} not found — run --install",
                  file=stream)
        return ok

    @staticmethod
    def rule_snippet() -> str:
        return RULE_SNIPPET


def _main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="ai-houkai-install-cursor",
        description="Register the AI-Houkai MCP server with Cursor.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--install", action="store_true",
                    help="Write the MCP block to Cursor's mcp.json")
    ap.add_argument("--project", action="store_true",
                    help=f"Target ./{PROJECT_CONFIG_PATH} instead of the global config")
    ap.add_argument("--memory-path", default=DEFAULT_MEMORY_PATH, metavar="PATH",
                    help=f"ChromaDB directory (default: {DEFAULT_MEMORY_PATH})")
    ap.add_argument("--collection", default=DEFAULT_COLLECTION,
                    help=f"Collection name (default: {DEFAULT_COLLECTION})")
    ap.add_argument("--settings", default=None,
                    help="Explicit path to mcp.json (overrides --project)")
    ap.add_argument("--verify", action="store_true",
                    help="Smoke-test the MCP server + check registration")
    ap.add_argument("--rule", action="store_true",
                    help="Print a .cursor/rules/*.mdc memory-usage snippet")
    args = ap.parse_args(argv)

    settings_path = (args.settings
                     or (PROJECT_CONFIG_PATH if args.project else GLOBAL_CONFIG_PATH))

    inst = CursorInstaller(
        memory_path=args.memory_path,
        collection=args.collection,
        settings_path=settings_path,
    )

    print("\nAI-Houkai · Cursor installer")
    print(f"  Config file : {inst.settings_path}")
    print(f"  Memory path : {inst.memory_path}")
    print(f"  MCP command : {inst.mcp_command}\n")

    if args.verify:
        if not inst.verify():
            return 1

    if args.rule:
        print("\n.cursor/rules/ai-houkai-memory.mdc\n")
        print(inst.rule_snippet())
        print("\n\n")

    if args.install:
        path = inst.install()
        print(f"  written: {path}")
        print("  Reload Cursor, then check Settings → MCP.\n")
    elif not (args.verify or args.rule):
        inst.print_config()
        print("  Run with --install to write this automatically.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
