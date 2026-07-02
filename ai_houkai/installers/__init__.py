"""
AI-Houkai installers — register the MCP server with various clients.

Submodules:
    claude_code  — Claude Code CLI (`claude mcp add` / `~/.claude.json` / `.mcp.json`)
    cursor       — Cursor editor (`~/.cursor/mcp.json`)
    opencode     — OpenCode agent (`~/.config/opencode/opencode.json`)

Future:
    claude_desktop — Claude Desktop GUI app (per-OS config path)
"""

from ai_houkai.installers.claude_code import ClaudeCodeInstaller
from ai_houkai.installers.cursor import CursorInstaller
from ai_houkai.installers.opencode import OpenCodeInstaller

__all__ = ["ClaudeCodeInstaller", "CursorInstaller", "OpenCodeInstaller"]
