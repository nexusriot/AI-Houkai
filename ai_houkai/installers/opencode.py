"""
OpenCode installer for the AI-Houkai MCP server.

Registers `ai-houkai-mcp` in OpenCode's config so the agent gains a persistent
memory. OpenCode (sst/opencode) uses its own `mcp` schema — distinct from the
`mcpServers` block used by Claude/Cursor: each server is an entry under `mcp`
with `type: "local"`, a `command` *array*, an `environment` object, and an
`enabled` flag.

    ~/.config/opencode/opencode.json   global (all projects)
    <project>/opencode.json            project-scoped

Library use:

    from ai_houkai.installers import OpenCodeInstaller

    inst = OpenCodeInstaller(memory_path="~/.ai_houkai")
    inst.install()                # patch ~/.config/opencode/opencode.json
    inst.print_config()           # preview the JSON block
    inst.verify()                 # smoke-test the server
    print(inst.agents_snippet())  # AGENTS.md content

CLI use (also exposed as the `ai-houkai-install-opencode` console script):

    python -m ai_houkai.installers.opencode --install
    python -m ai_houkai.installers.opencode --project       # ./opencode.json
    python -m ai_houkai.installers.opencode --verify
    python -m ai_houkai.installers.opencode --agents
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

GLOBAL_CONFIG_PATH  = os.path.expanduser("~/.config/opencode/opencode.json")
PROJECT_CONFIG_PATH = "opencode.json"
DEFAULT_MEMORY_PATH = os.path.expanduser("~/.ai_houkai")
DEFAULT_COLLECTION  = "opencode"
SERVER_NAME         = "ai-houkai"
CONFIG_SCHEMA_URL   = "https://opencode.ai/config.json"


# OpenCode reads project/global instructions from AGENTS.md.
AGENTS_SNIPPET = f"## Memory (AI-Houkai MCP)\n\n{MEMORY_GUIDE}"


@dataclass
class OpenCodeInstaller:
    """Register the AI-Houkai MCP server with OpenCode."""

    memory_path:   str = DEFAULT_MEMORY_PATH
    collection:    str = DEFAULT_COLLECTION
    settings_path: str = GLOBAL_CONFIG_PATH
    server_name:   str = SERVER_NAME
    extra_env:     dict = field(default_factory=dict)

    @property
    def mcp_command(self) -> str:
        return resolve_mcp_command()

    def build_mcp_block(self) -> dict:
        environment = {
            "AI_HOUKAI_PATH":       self.memory_path,
            "AI_HOUKAI_COLLECTION": self.collection,
            **self.extra_env,
        }
        return {
            "type":        "local",
            "command":     [self.mcp_command],
            "enabled":     True,
            "environment": environment,
        }

    def build_settings_block(self) -> dict:
        return {
            "$schema": CONFIG_SCHEMA_URL,
            "mcp": {self.server_name: self.build_mcp_block()},
        }

    def install(self, *, overwrite_unparseable: bool = True) -> str:
        """Patch opencode.json with the MCP server block. Returns the path."""
        config = load_json(self.settings_path,
                           overwrite_unparseable=overwrite_unparseable)
        config.setdefault("$schema", CONFIG_SCHEMA_URL)
        config.setdefault("mcp", {})
        config["mcp"][self.server_name] = self.build_mcp_block()
        return write_json(self.settings_path, config)

    def print_config(self, *, stream=sys.stdout) -> None:
        block = self.build_settings_block()
        print(f"\nPaste this into {self.settings_path}:\n", file=stream)
        print(json.dumps(block, indent=2), file=stream)
        print(f"\nThen restart OpenCode — '{self.server_name}' tools become "
              "available to the agent.\n", file=stream)

    def verify(self, *, stream=sys.stdout) -> bool:
        ok = verify_server(self.server_name, stream=stream)
        if os.path.isfile(self.settings_path):
            cfg = load_json(self.settings_path)
            if self.server_name in cfg.get("mcp", {}):
                print(f"  ok   registered in {self.settings_path}", file=stream)
            else:
                print(f"  warn not yet in {self.settings_path} — run --install",
                      file=stream)
        else:
            print(f"  warn {self.settings_path} not found — run --install",
                  file=stream)
        return ok

    @staticmethod
    def agents_snippet() -> str:
        return AGENTS_SNIPPET


def _main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="ai-houkai-install-opencode",
        description="Register the AI-Houkai MCP server with OpenCode.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--install", action="store_true",
                    help="Write the MCP block to opencode.json")
    ap.add_argument("--project", action="store_true",
                    help=f"Target ./{PROJECT_CONFIG_PATH} instead of the global config")
    ap.add_argument("--memory-path", default=DEFAULT_MEMORY_PATH, metavar="PATH",
                    help=f"ChromaDB directory (default: {DEFAULT_MEMORY_PATH})")
    ap.add_argument("--collection", default=DEFAULT_COLLECTION,
                    help=f"Collection name (default: {DEFAULT_COLLECTION})")
    ap.add_argument("--settings", default=None,
                    help="Explicit path to opencode.json (overrides --project)")
    ap.add_argument("--verify", action="store_true",
                    help="Smoke-test the MCP server + check registration")
    ap.add_argument("--agents", action="store_true",
                    help="Print an AGENTS.md memory-usage snippet")
    args = ap.parse_args(argv)

    settings_path = (args.settings
                     or (PROJECT_CONFIG_PATH if args.project else GLOBAL_CONFIG_PATH))

    inst = OpenCodeInstaller(
        memory_path=args.memory_path,
        collection=args.collection,
        settings_path=settings_path,
    )

    print("\nAI-Houkai · OpenCode installer")
    print(f"  Config file : {inst.settings_path}")
    print(f"  Memory path : {inst.memory_path}")
    print(f"  MCP command : {inst.mcp_command}\n")

    if args.verify:
        if not inst.verify():
            return 1

    if args.agents:
        print("\nAGENTS.md snippet\n")
        print(inst.agents_snippet())
        print("\n")

    if args.install:
        path = inst.install()
        print(f"  written: {path}")
        print("  Restart OpenCode to load the memory tools.\n")
    elif not (args.verify or args.agents):
        inst.print_config()
        print("  Run with --install to write this automatically.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
