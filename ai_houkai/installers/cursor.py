"""
Cursor installer for the AI-Houkai MCP server.

Registers `ai-houkai-mcp` in Cursor's MCP config so the editor's agent gains a
persistent memory. Cursor uses the same `mcpServers` schema as Claude Desktop /
Claude Code, but reads it from a different file:

    ~/.cursor/mcp.json            global (all projects)
    <project>/.cursor/mcp.json    project-scoped

Library use:

    from ai_houkai.installers import CursorInstaller

    inst = CursorInstaller(memory_path="~/.ai_houkai/.chroma")
    inst.install()                # patch ~/.cursor/mcp.json
    inst.print_config()           # preview the JSON block
    inst.verify()                 # smoke-test the server
    print(inst.rule_snippet())    # .cursor/rules/*.mdc content

CLI use (also exposed as the `ai-houkai-install-cursor` console script):

    python -m ai_houkai.installers.cursor --install
    python -m ai_houkai.installers.cursor --project        # ./.cursor/mcp.json
    python -m ai_houkai.installers.cursor --verify
    python -m ai_houkai.installers.cursor --rule

Everything that is not Cursor-specific — the merge-and-write install, the
preview, verify, and the argparse front end — lives in
:class:`ai_houkai.installers.common.JSONConfigInstaller`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import ClassVar, Optional

from ai_houkai.installers.common import (
    DEFAULT_MEMORY_PATH,
    MEMORY_GUIDE,
    SERVER_NAME,
    JSONConfigInstaller,
)

GLOBAL_CONFIG_PATH  = os.path.expanduser("~/.cursor/mcp.json")
PROJECT_CONFIG_PATH = os.path.join(".cursor", "mcp.json")
DEFAULT_COLLECTION  = "cursor"

# Re-exported from common so `from ai_houkai.installers.cursor import
# DEFAULT_MEMORY_PATH` keeps working; the values are shared across clients.
__all__ = [
    "CursorInstaller",
    "DEFAULT_COLLECTION",
    "DEFAULT_MEMORY_PATH",
    "GLOBAL_CONFIG_PATH",
    "PROJECT_CONFIG_PATH",
    "RULE_SNIPPET",
    "SERVER_NAME",
]


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
class CursorInstaller(JSONConfigInstaller):
    """Register the AI-Houkai MCP server with Cursor."""

    client_name:         ClassVar[str] = "Cursor"
    slug:                ClassVar[str] = "cursor"
    config_key:          ClassVar[str] = "mcpServers"
    default_collection:  ClassVar[str] = DEFAULT_COLLECTION
    global_config_path:  ClassVar[str] = GLOBAL_CONFIG_PATH
    project_config_path: ClassVar[str] = PROJECT_CONFIG_PATH
    preview_hint:        ClassVar[str] = (
        "Then reload Cursor and open Settings → MCP to confirm "
        "'{server_name}' is listed.")
    installed_hint:      ClassVar[str] = "Reload Cursor, then check Settings → MCP."
    snippet_flag:        ClassVar[str] = "rule"
    snippet_help:        ClassVar[str] = (
        "Print a .cursor/rules/*.mdc memory-usage snippet")
    snippet_heading:     ClassVar[str] = ".cursor/rules/ai-houkai-memory.mdc"
    snippet:             ClassVar[str] = RULE_SNIPPET
    snippet_trailer:     ClassVar[str] = "\n\n"

    def build_mcp_block(self) -> dict:
        return {"command": self.mcp_command, "env": self.build_env()}

    @staticmethod
    def rule_snippet() -> str:
        return RULE_SNIPPET


def _main(argv: Optional[list] = None) -> int:
    return CursorInstaller.main(argv)


if __name__ == "__main__":
    raise SystemExit(_main())
