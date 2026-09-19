"""Shared helpers for the AI-Houkai client installers.

Each installer (Claude Code, Cursor, OpenCode, …) registers the same stdio
MCP server — `ai-houkai-mcp` — into a client-specific config file. The only
differences are the file location and the JSON schema the client expects.
This module collects the bits every installer needs.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from typing import ClassVar, Optional

from ai_houkai.memory_system import MemoryStore
from ai_houkai.mcp_server import server as srv

CONSOLE_SCRIPT = "ai-houkai-mcp"
SERVER_NAME = "ai-houkai"
# `.chroma` leaf matches the CLI default (~/.ai_houkai/.chroma) so `houkai
# list` sees installed-client memories, and the store's journal.log lands in
# ~/.ai_houkai/ instead of $HOME (it is written to the store path's parent).
DEFAULT_MEMORY_PATH = os.path.expanduser("~/.ai_houkai/.chroma")


def resolve_mcp_command() -> str:
    """Return the absolute path to the `ai-houkai-mcp` console script if found,
    otherwise the bare name (resolved via PATH at runtime)."""
    candidate = os.path.join(os.path.dirname(sys.executable), CONSOLE_SCRIPT)
    if os.path.isfile(candidate):
        return candidate
    return CONSOLE_SCRIPT


def load_json(path: str, *, overwrite_unparseable: bool = False) -> dict:
    """Load a JSON config file, returning {} when it is missing.

    An existing-but-invalid (or non-object) file raises ValueError by
    default: these are the user's own client configs, and install() writes
    the merged result back — treating garbage as {} would silently replace
    the whole file with just our server block. overwrite_unparseable=True
    opts into exactly that, after parking a ``.bak`` copy of the original.
    """
    if not os.path.isfile(path):
        return {}
    try:
        with open(path) as f:
            loaded = json.load(f)
    except json.JSONDecodeError as e:
        if overwrite_unparseable:
            shutil.copy2(path, path + ".bak")
            return {}
        raise ValueError(
            f"{path} is not valid JSON ({e}) — fix or remove it, then re-run"
        ) from e
    if isinstance(loaded, dict):
        return loaded
    if overwrite_unparseable:
        shutil.copy2(path, path + ".bak")
        return {}
    raise ValueError(
        f"{path}: expected a JSON object at the top level, "
        f"got {type(loaded).__name__} — fix or remove it, then re-run"
    )


def write_json(path: str, config: dict) -> str:
    """Atomically write `config` to `path` (creating parent dirs). Returns the path.

    Write-to-temp + os.replace so a crash mid-write can never leave the
    user's client config truncated."""
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=parent, prefix=".ahk-", suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(config, f, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def merge_server_block(
    path: str,
    config_key: str,
    server_name: str,
    block: dict,
    *,
    defaults: dict | None = None,
    overwrite_unparseable: bool = False,
) -> str:
    """Merge one MCP server block into a client config file. Returns the path.

    Read-modify-write rather than overwrite: every client config also holds
    settings that are none of our business, and other MCP servers besides
    ours. *defaults* seeds top-level keys only when absent (OpenCode's
    ``$schema``).
    """
    config = load_json(path, overwrite_unparseable=overwrite_unparseable)
    for key, value in (defaults or {}).items():
        config.setdefault(key, value)
    config.setdefault(config_key, {})
    config[config_key][server_name] = block
    return write_json(path, config)


def verify_server(
    *,
    memory_path: str | None = None,
    collection: str | None = None,
    stream=sys.stdout,
) -> bool:
    """Smoke-test shared across installers: the `ai-houkai-mcp` console script is
    reachable and the MCP server module imports with its core tools. Returns True
    on success.

    When *memory_path*/*collection* are given (the values the installer wrote
    into the client config), the reported store count comes from THAT store —
    not from whatever env-default store the server would open."""
    ok = True
    cmd = resolve_mcp_command()
    found = shutil.which(cmd) or (os.path.isfile(cmd) and cmd)
    if found:
        print(f"  ok   console script: {found}", file=stream)
    else:
        print(f"  err  '{cmd}' not on PATH — run: pip install ai-houkai",
              file=stream)
        ok = False

    tools = ["remember", "recall", "forget", "list_recent", "stats"]
    missing = [t for t in tools if not hasattr(srv, t)]
    if missing:
        print(f"  err  missing tools: {missing}", file=stream)
        ok = False
    else:
        try:
            count = _target_store_count(memory_path, collection)
            print(f"  ok   tools: {', '.join(tools)} | "
                  f"store count = {count}", file=stream)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"  err  store check failed: {exc}", file=stream)
            ok = False

    return ok


def _target_store_count(memory_path: str | None, collection: str | None) -> int:
    """Count memories in the store the installer is configuring.

    Opens the target store explicitly — verify is a smoke test, and creating
    the directory the MCP server will use anyway is fine. Without a
    memory_path, falls back to the server's own (env-configured) store."""
    if memory_path is None:
        return srv.get_store().count()
    store = MemoryStore(path=os.path.expanduser(memory_path),
                        collection=collection or "ai_houkai")
    try:
        return store.count()
    finally:
        store.client.close()


# A client-agnostic description of when/how to use the memory tools. Each
# installer wraps this in its own instruction-file format (CLAUDE.md, AGENTS.md,
# .cursor/rules/*.mdc, …).
MEMORY_GUIDE = textwrap.dedent("""
    You have access to a persistent memory store via AI-Houkai MCP tools:

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
class JSONConfigInstaller:
    """Register the MCP server in a client's own JSON config file.

    Cursor and OpenCode differ in exactly four places: where the config file
    lives, which top-level key holds the server map, what the per-server block
    inside it looks like, and the wording of the "now reload the client"
    hints. Everything else — env building, the merge-and-write install, the
    paste-this-block preview, the verify report, and the argparse front end —
    was byte-for-byte identical in both, so it lives here and each client
    declares only its differences as class attributes.

    Claude Code deliberately does *not* subclass this: it prefers the
    `claude mcp add` CLI over writing a config file, and only falls back to a
    direct write. It shares the file-level helpers above instead.
    """

    memory_path: str = DEFAULT_MEMORY_PATH
    #: Empty means "the client's default" — resolved in __post_init__ so a
    #: subclass declares its default once, as a class attribute.
    collection: str = ""
    settings_path: str = ""
    server_name: str = SERVER_NAME
    extra_env: dict = field(default_factory=dict)

    #: Display name used in CLI banners and messages ("Cursor").
    client_name: ClassVar[str] = ""
    #: Console-script suffix: ai-houkai-install-<slug>.
    slug: ClassVar[str] = ""
    #: Top-level config key whose object maps server name -> server block.
    config_key: ClassVar[str] = "mcpServers"
    default_collection: ClassVar[str] = ""
    global_config_path: ClassVar[str] = ""
    project_config_path: ClassVar[str] = ""
    #: Templates, formatted with ``server_name``. Printed after the
    #: paste-this-block preview and after a successful --install.
    preview_hint: ClassVar[str] = ""
    installed_hint: ClassVar[str] = ""
    #: The instruction-file snippet: the --flag that prints it, the help
    #: text for that flag, the heading above it, and its content.
    snippet_flag: ClassVar[str] = ""
    snippet_help: ClassVar[str] = ""
    snippet_heading: ClassVar[str] = ""
    snippet: ClassVar[str] = ""
    #: Blank lines printed after the snippet, so it stays visually separate
    #: from whatever the shell prints next.
    snippet_trailer: ClassVar[str] = "\n"

    def __post_init__(self) -> None:
        if not self.collection:
            self.collection = self.default_collection
        if not self.settings_path:
            self.settings_path = self.global_config_path

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
        """The per-server block in this client's schema."""
        raise NotImplementedError

    def config_defaults(self) -> dict:
        """Top-level keys to seed when absent (OpenCode's ``$schema``)."""
        return {}

    def build_settings_block(self) -> dict:
        return {
            **self.config_defaults(),
            self.config_key: {self.server_name: self.build_mcp_block()},
        }

    def install(self, *, overwrite_unparseable: bool = False) -> str:
        """Merge the MCP server block into the client config. Returns the path."""
        return merge_server_block(
            self.settings_path, self.config_key, self.server_name,
            self.build_mcp_block(),
            defaults=self.config_defaults(),
            overwrite_unparseable=overwrite_unparseable,
        )

    def print_config(self, *, stream=sys.stdout) -> None:
        print(f"\nPaste this into {self.settings_path}:\n", file=stream)
        print(json.dumps(self.build_settings_block(), indent=2), file=stream)
        print(f"\n{self.preview_hint.format(server_name=self.server_name)}\n",
              file=stream)

    def verify(self, *, stream=sys.stdout) -> bool:
        ok = verify_server(memory_path=self.memory_path,
                           collection=self.collection, stream=stream)
        if os.path.isfile(self.settings_path):
            try:
                cfg = load_json(self.settings_path)
            except ValueError as exc:
                print(f"  warn {exc}", file=stream)
                cfg = {}
            if self.server_name in cfg.get(self.config_key, {}):
                print(f"  ok   registered in {self.settings_path}", file=stream)
            else:
                print(f"  warn not yet in {self.settings_path} — run --install",
                      file=stream)
        else:
            print(f"  warn {self.settings_path} not found — run --install",
                  file=stream)
        return ok

    @classmethod
    def main(cls, argv: Optional[list] = None) -> int:
        """The `ai-houkai-install-<client>` console script."""
        ap = argparse.ArgumentParser(
            prog=f"ai-houkai-install-{cls.slug}",
            description=f"Register the AI-Houkai MCP server with {cls.client_name}.",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        config_file = os.path.basename(cls.global_config_path)
        ap.add_argument("--install", action="store_true",
                        help=f"Write the MCP block to {cls.client_name}'s "
                             f"{config_file}")
        ap.add_argument("--project", action="store_true",
                        help=f"Target ./{cls.project_config_path} instead of "
                             "the global config")
        ap.add_argument("--memory-path", default=DEFAULT_MEMORY_PATH,
                        metavar="PATH",
                        help=f"ChromaDB directory (default: {DEFAULT_MEMORY_PATH})")
        ap.add_argument("--collection", default=cls.default_collection,
                        help=f"Collection name (default: {cls.default_collection})")
        ap.add_argument("--settings", default=None,
                        help=f"Explicit path to {config_file} "
                             "(overrides --project)")
        ap.add_argument("--verify", action="store_true",
                        help="Smoke-test the MCP server + check registration")
        ap.add_argument(f"--{cls.snippet_flag}", action="store_true",
                        help=cls.snippet_help)
        args = ap.parse_args(argv)
        want_snippet = getattr(args, cls.snippet_flag.replace("-", "_"))

        inst = cls(
            memory_path=args.memory_path,
            collection=args.collection,
            settings_path=(args.settings
                           or (cls.project_config_path if args.project
                               else cls.global_config_path)),
        )

        print(f"\nAI-Houkai · {cls.client_name} installer")
        print(f"  Config file : {inst.settings_path}")
        print(f"  Memory path : {inst.memory_path}")
        print(f"  MCP command : {inst.mcp_command}\n")

        if args.verify and not inst.verify():
            return 1

        if want_snippet:
            print(f"\n{cls.snippet_heading}\n")
            print(cls.snippet)
            print(cls.snippet_trailer)

        if args.install:
            try:
                path = inst.install()
            except ValueError as exc:
                print(f"  err  {exc}")
                return 1
            print(f"  written: {path}")
            print(f"  {inst.installed_hint.format(server_name=inst.server_name)}\n")
        elif not (args.verify or want_snippet):
            inst.print_config()
            print("  Run with --install to write this automatically.\n")
        return 0
