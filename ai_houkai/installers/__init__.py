"""
AI-Houkai installers — register the MCP server with various clients.

Submodules:
    claude_code  — Claude Code CLI (`~/.claude/settings.json`)

Future:
    claude_desktop — Claude Desktop GUI app (per-OS config path)
"""

from ai_houkai.installers.claude_code import ClaudeCodeInstaller

__all__ = ["ClaudeCodeInstaller"]
