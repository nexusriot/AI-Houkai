"""
Claude Code installer for the AI-Houkai MCP server.

Registers `ai-houkai-mcp` with the Claude Code CLI so it launches the memory
MCP server automatically. Claude Code reads MCP servers from `~/.claude.json`
(user scope) or a project-level `.mcp.json` — NOT from settings.json, which
has no `mcpServers` key. The installer therefore prefers the supported
interface (`claude mcp add --scope user`), falling back to editing the
config file directly when the `claude` CLI is not on PATH.

Library use:

    from ai_houkai.installers import ClaudeCodeInstaller

    inst = ClaudeCodeInstaller(memory_path="~/.ai_houkai/.chroma")
    inst.install()                # register (CLI or ~/.claude.json)
    inst.install(scope="project") # register in ./.mcp.json
    inst.print_config()           # preview the JSON block
    inst.verify()                 # smoke-test the server + CLI
    print(inst.claudemd_snippet())

CLI use (also exposed as the `ai-houkai-install-claude-code` console script):

    python -m ai_houkai.installers.claude_code --install
    python -m ai_houkai.installers.claude_code --install --project
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

from ai_houkai.installers.common import (
    load_json,
    resolve_mcp_command,
    verify_server,
    write_json,
)


DEFAULT_CONFIG_PATH  = os.path.expanduser("~/.claude.json")
PROJECT_CONFIG_PATH  = ".mcp.json"
# `.chroma` leaf matches the CLI default (~/.ai_houkai/.chroma) so `houkai
# list` sees installed-client memories, and the store's journal.log lands in
# ~/.ai_houkai/ instead of $HOME (it is written to the store path's parent).
DEFAULT_MEMORY_PATH  = os.path.expanduser("~/.ai_houkai/.chroma")
DEFAULT_COLLECTION   = "claude_code"
SERVER_NAME          = "ai-houkai"


CLAUDEMD_SNIPPET = textwrap.dedent("""
    ## Memory (AI-Houkai MCP)

    You have access to a persistent memory store via MCP tools:

    - **remember(text, type, tags, importance)** — store a fact, decision, or preference
    - **recall(query, k)** — semantic search across stored memories
    - **edit(memory_id, …)** — update a memory in place (keeps id, links, history)
    - **forget(memory_id)** — remove a specific memory
    - **list_recent()** — see the most recently created memories

    ### When to use memory

    | Situation | Action |
    |---|---|
    | User states a preference or coding convention | `remember` with `type="feedback"` or `"procedural"` |
    | You learn something about the codebase | `remember` with `type="semantic"` |
    | Starting a new task | `recall` relevant context first |
    | A stored fact is outdated or has a typo | `edit` it in place — don't forget+remember |
    | User corrects you | `remember` the correction, `forget` the wrong fact |

    ### Memory types
    - `episodic` — time-stamped events ("Fixed auth bug in PR #441")
    - `semantic` — distilled facts ("API versioned at /api/v1/")
    - `procedural` — how-to rules ("Always use tmp_path in tests")
    - `feedback` — user preferences ("Prefers concise answers")
""").strip()


@dataclass
class ClaudeCodeInstaller:
    """Register the AI-Houkai MCP server with Claude Code."""

    memory_path: str = DEFAULT_MEMORY_PATH
    collection:  str = DEFAULT_COLLECTION
    # Direct-write fallback target for scope="user" (used when the `claude`
    # CLI is not on PATH). Claude Code's user-scope MCP servers live in
    # ~/.claude.json — settings.json has no mcpServers key.
    config_path: str = DEFAULT_CONFIG_PATH
    server_name: str = SERVER_NAME
    extra_env:   dict = field(default_factory=dict)

    @property
    def mcp_command(self) -> str:
        return resolve_mcp_command()

    def build_env(self) -> dict:
        return {
            "AI_HOUKAI_PATH":       self.memory_path,
            "AI_HOUKAI_COLLECTION": self.collection,
            **self.extra_env,
        }

    def build_mcp_block(self) -> dict:
        return {
            "type":    "stdio",
            "command": self.mcp_command,
            "args":    [],
            "env":     self.build_env(),
        }

    def build_settings_block(self) -> dict:
        return {"mcpServers": {self.server_name: self.build_mcp_block()}}

    def install(
        self,
        *,
        scope: str = "user",
        overwrite_unparseable: bool = True,
    ) -> str:
        """Register the MCP server with Claude Code. Returns a description of
        what was written (a command line or a config file path).

        Prefers `claude mcp add --scope …` (the supported interface, robust
        to config-layout changes); falls back to editing the config file
        directly when the `claude` CLI is not on PATH: ~/.claude.json for
        scope="user", ./.mcp.json for scope="project".
        """
        if scope not in ("user", "project"):
            raise ValueError(f"scope must be 'user' or 'project', got {scope!r}")
        if shutil.which("claude"):
            return self._install_via_cli(scope)
        return self._install_direct(
            scope, overwrite_unparseable=overwrite_unparseable)

    def _install_via_cli(self, scope: str) -> str:
        """Register through `claude mcp add`. Re-adds if already present."""
        # `claude mcp add` refuses to overwrite an existing entry — remove
        # first so install() is idempotent. Failure (not registered) is fine.
        subprocess.run(
            ["claude", "mcp", "remove", "--scope", scope, self.server_name],
            capture_output=True, text=True, timeout=15,
        )
        # The server name MUST precede the --env options: `-e/--env <env...>`
        # is variadic and would swallow a following bare name, leaving
        # `commandOrUrl` missing and the add failing with rc=1.
        cmd = ["claude", "mcp", "add", "--scope", scope, self.server_name]
        for key, value in self.build_env().items():
            cmd += ["--env", f"{key}={value}"]
        cmd += ["--", self.mcp_command]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode != 0:
            raise RuntimeError(
                f"`claude mcp add` failed (rc={result.returncode}): "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )
        return f"claude mcp add --scope {scope} {self.server_name}"

    def _install_direct(self, scope: str, *, overwrite_unparseable: bool) -> str:
        """Merge the server block into the config file Claude Code reads."""
        path = self.config_path if scope == "user" else PROJECT_CONFIG_PATH
        config = load_json(path, overwrite_unparseable=overwrite_unparseable)
        config.setdefault("mcpServers", {})
        config["mcpServers"][self.server_name] = self.build_mcp_block()
        return write_json(path, config)

    def print_config(self, *, stream=sys.stdout) -> None:
        print(f"\nRegister with the Claude Code CLI (preferred):\n", file=stream)
        print(f"    claude mcp add --scope user {self.server_name} "
              f"--env AI_HOUKAI_PATH={self.memory_path} "
              f"--env AI_HOUKAI_COLLECTION={self.collection} "
              f"-- {self.mcp_command}\n", file=stream)
        print(f"Or paste this into the `mcpServers` block of {self.config_path} "
              f"(user scope)\nor a project's .mcp.json:\n", file=stream)
        print(json.dumps(self.build_settings_block(), indent=2), file=stream)

    def verify(self, *, stream=sys.stdout) -> bool:
        """Smoke-test: console script present, server module importable, and
        the *target* store (memory_path/collection) reachable. Returns True
        on success."""
        ok = verify_server(
            self.server_name,
            memory_path=self.memory_path,
            collection=self.collection,
            stream=stream,
        )

        if shutil.which("claude"):
            try:
                result = subprocess.run(
                    ["claude", "mcp", "list"],
                    capture_output=True, text=True, timeout=15,
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
                    help="Register the MCP server (claude mcp add, or the "
                         "config file directly if the CLI is missing)")
    ap.add_argument("--project", action="store_true",
                    help="Register at project scope (./.mcp.json) instead of user")
    ap.add_argument("--memory-path", default=DEFAULT_MEMORY_PATH, metavar="PATH",
                    help=f"ChromaDB directory (default: {DEFAULT_MEMORY_PATH})")
    ap.add_argument("--collection", default=DEFAULT_COLLECTION,
                    help=f"Collection name (default: {DEFAULT_COLLECTION})")
    ap.add_argument("--config", default=DEFAULT_CONFIG_PATH,
                    help="Path to the user-scope config for the direct-write "
                         f"fallback (default: {DEFAULT_CONFIG_PATH})")
    ap.add_argument("--verify", action="store_true",
                    help="Smoke-test the MCP server")
    ap.add_argument("--claudemd", action="store_true",
                    help="Print a CLAUDE.md memory-usage snippet")
    args = ap.parse_args(argv)

    scope = "project" if args.project else "user"
    inst = ClaudeCodeInstaller(
        memory_path=args.memory_path,
        collection=args.collection,
        config_path=args.config,
    )

    print(f"\nAI-Houkai · Claude Code installer")
    print(f"  Scope       : {scope}")
    print(f"  Memory path : {inst.memory_path}")
    print(f"  MCP command : {inst.mcp_command}\n")

    if args.verify:
        ok = inst.verify()
        if not ok:
            return 1

    if args.claudemd:
        print("\n── CLAUDE.md snippet ─────────────────────────────────────\n")
        print(inst.claudemd_snippet())
        print("\n──────────────────────────────────────────────────────────\n")

    if args.install:
        try:
            written = inst.install(scope=scope)
        except RuntimeError as exc:
            print(f"  err  {exc}")
            return 1
        print(f"  registered via: {written}")
        print(f"  verify:  claude mcp list\n")
    elif not (args.verify or args.claudemd):
        inst.print_config()
        print("  Run with --install to register this automatically.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
