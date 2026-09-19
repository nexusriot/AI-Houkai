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

    inst = OpenCodeInstaller(memory_path="~/.ai_houkai/.chroma")
    inst.install()                # patch ~/.config/opencode/opencode.json
    inst.print_config()           # preview the JSON block
    inst.verify()                 # smoke-test the server
    print(inst.agents_snippet())  # AGENTS.md content

CLI use (also exposed as the `ai-houkai-install-opencode` console script):

    python -m ai_houkai.installers.opencode --install
    python -m ai_houkai.installers.opencode --project       # ./opencode.json
    python -m ai_houkai.installers.opencode --verify
    python -m ai_houkai.installers.opencode --agents

Everything that is not OpenCode-specific — the merge-and-write install, the
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

GLOBAL_CONFIG_PATH  = os.path.expanduser("~/.config/opencode/opencode.json")
PROJECT_CONFIG_PATH = "opencode.json"
DEFAULT_COLLECTION  = "opencode"
CONFIG_SCHEMA_URL   = "https://opencode.ai/config.json"

# Re-exported from common so `from ai_houkai.installers.opencode import
# DEFAULT_MEMORY_PATH` keeps working; the values are shared across clients.
__all__ = [
    "CONFIG_SCHEMA_URL",
    "AGENTS_SNIPPET",
    "DEFAULT_COLLECTION",
    "DEFAULT_MEMORY_PATH",
    "GLOBAL_CONFIG_PATH",
    "OpenCodeInstaller",
    "PROJECT_CONFIG_PATH",
    "SERVER_NAME",
]


# OpenCode reads project/global instructions from AGENTS.md.
AGENTS_SNIPPET = f"## Memory (AI-Houkai MCP)\n\n{MEMORY_GUIDE}"


@dataclass
class OpenCodeInstaller(JSONConfigInstaller):
    """Register the AI-Houkai MCP server with OpenCode."""

    client_name:         ClassVar[str] = "OpenCode"
    slug:                ClassVar[str] = "opencode"
    config_key:          ClassVar[str] = "mcp"
    default_collection:  ClassVar[str] = DEFAULT_COLLECTION
    global_config_path:  ClassVar[str] = GLOBAL_CONFIG_PATH
    project_config_path: ClassVar[str] = PROJECT_CONFIG_PATH
    preview_hint:        ClassVar[str] = (
        "Then restart OpenCode — '{server_name}' tools become available "
        "to the agent.")
    installed_hint:      ClassVar[str] = "Restart OpenCode to load the memory tools."
    snippet_flag:        ClassVar[str] = "agents"
    snippet_help:        ClassVar[str] = "Print an AGENTS.md memory-usage snippet"
    snippet_heading:     ClassVar[str] = "AGENTS.md snippet"
    snippet:             ClassVar[str] = AGENTS_SNIPPET

    def config_defaults(self) -> dict:
        return {"$schema": CONFIG_SCHEMA_URL}

    def build_mcp_block(self) -> dict:
        return {
            "type":        "local",
            "command":     [self.mcp_command],
            "enabled":     True,
            "environment": self.build_env(),
        }

    @staticmethod
    def agents_snippet() -> str:
        return AGENTS_SNIPPET


def _main(argv: Optional[list] = None) -> int:
    return OpenCodeInstaller.main(argv)


if __name__ == "__main__":
    raise SystemExit(_main())
