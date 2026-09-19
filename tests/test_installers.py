"""Tests for the client installers (Claude Code registration paths, shared
helpers, and the no-side-effects-on-import guarantee)."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ai_houkai.cli import config as cfg_mod
from ai_houkai.installers import claude_code as cc_mod
from ai_houkai.installers import common
from ai_houkai.installers import cursor as cur_mod
from ai_houkai.installers import opencode as oc_mod
from ai_houkai.installers.claude_code import ClaudeCodeInstaller
from ai_houkai.installers.common import load_json, write_json


@pytest.fixture()
def no_claude_cli(monkeypatch):
    """Pretend the `claude` CLI is not on PATH."""
    monkeypatch.setattr(cc_mod.shutil, "which", lambda name: None)


def test_direct_install_writes_claude_json(tmp_path: Path, no_claude_cli) -> None:
    cfg = tmp_path / ".claude.json"
    inst = ClaudeCodeInstaller(
        memory_path="/mem", collection="col", config_path=str(cfg))
    written = inst.install()

    assert written == str(cfg)
    data = json.loads(cfg.read_text())
    block = data["mcpServers"]["ai-houkai"]
    assert block["type"] == "stdio"
    assert block["env"]["AI_HOUKAI_PATH"] == "/mem"
    assert block["env"]["AI_HOUKAI_COLLECTION"] == "col"


def test_direct_install_preserves_existing_config(
    tmp_path: Path, no_claude_cli
) -> None:
    cfg = tmp_path / ".claude.json"
    cfg.write_text(json.dumps({
        "numStartups": 42,
        "mcpServers": {"other": {"type": "stdio", "command": "x"}},
    }))
    inst = ClaudeCodeInstaller(config_path=str(cfg))
    inst.install()

    data = json.loads(cfg.read_text())
    assert data["numStartups"] == 42                # unrelated keys survive
    assert "other" in data["mcpServers"]            # other servers survive
    assert "ai-houkai" in data["mcpServers"]


def test_direct_install_project_scope_writes_mcp_json(
    tmp_path: Path, no_claude_cli, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    inst = ClaudeCodeInstaller(memory_path="/mem")
    written = inst.install(scope="project")

    assert written == ".mcp.json"
    data = json.loads((tmp_path / ".mcp.json").read_text())
    assert data["mcpServers"]["ai-houkai"]["env"]["AI_HOUKAI_PATH"] == "/mem"


def test_install_rejects_bad_scope(no_claude_cli) -> None:
    with pytest.raises(ValueError, match="scope"):
        ClaudeCodeInstaller().install(scope="global")


def test_cli_install_invokes_claude_mcp_add(monkeypatch) -> None:
    monkeypatch.setattr(cc_mod.shutil, "which", lambda name: "/usr/bin/claude")
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(cc_mod.subprocess, "run", fake_run)
    inst = ClaudeCodeInstaller(memory_path="/mem", collection="col")
    result = inst.install()

    assert result == "claude mcp add --scope user ai-houkai"
    # idempotency: stale entry removed first, then re-added
    assert calls[0][:4] == ["claude", "mcp", "remove", "--scope"]
    add = calls[1]
    assert add[:6] == ["claude", "mcp", "add", "--scope", "user", "ai-houkai"]
    # the name MUST precede --env: the CLI's variadic -e/--env would
    # otherwise swallow it and the add fails with rc=1
    assert add.index("ai-houkai") < add.index("--env")
    assert "AI_HOUKAI_PATH=/mem" in add
    assert "AI_HOUKAI_COLLECTION=col" in add
    assert add[-2:-1] == ["--"]                     # command after separator
    # never touches settings.json
    assert not any("settings.json" in " ".join(c) for c in calls)


def test_cli_install_failure_raises(monkeypatch) -> None:
    monkeypatch.setattr(cc_mod.shutil, "which", lambda name: "/usr/bin/claude")

    def fake_run(cmd, **kwargs):
        rc = 1 if cmd[:3] == ["claude", "mcp", "add"] else 0
        return subprocess.CompletedProcess(cmd, rc, stdout="", stderr="boom")

    monkeypatch.setattr(cc_mod.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="boom"):
        ClaudeCodeInstaller().install()


def test_write_json_atomic_and_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "cfg.json"
    write_json(str(path), {"a": 1})
    assert load_json(str(path)) == {"a": 1}
    # no temp files left behind
    assert [p.name for p in path.parent.iterdir()] == ["cfg.json"]

    # atomicity: a crash mid-serialisation must leave the existing file
    # untouched and clean up its temp file
    with pytest.raises(TypeError):
        write_json(str(path), {"bad": {1, 2, 3}})   # sets are unserialisable
    assert load_json(str(path)) == {"a": 1}          # original intact
    assert [p.name for p in path.parent.iterdir()] == ["cfg.json"]


def test_load_json_refuses_garbage_by_default(tmp_path: Path) -> None:
    """An unparseable config must never be silently treated as empty —
    install() writes the merged result back, so {} here would replace the
    user's whole client config with just our server block."""
    path = tmp_path / "bad.json"
    path.write_text("{not json")
    with pytest.raises(ValueError, match="bad.json"):
        load_json(str(path))
    assert path.read_text() == "{not json"   # original untouched

    # Opting into the overwrite parks a .bak copy of the original first.
    assert load_json(str(path), overwrite_unparseable=True) == {}
    assert (tmp_path / "bad.json.bak").read_text() == "{not json"


def test_load_json_refuses_non_object(tmp_path: Path) -> None:
    """Valid JSON that is not an object would crash the setdefault-based
    merge — reject it with the same clear error as unparseable JSON."""
    path = tmp_path / "list.json"
    path.write_text('["not", "an", "object"]')
    with pytest.raises(ValueError, match="expected a JSON object"):
        load_json(str(path))
    assert load_json(str(path), overwrite_unparseable=True) == {}
    assert (tmp_path / "list.json.bak").exists()


def test_install_direct_refuses_corrupt_config(tmp_path: Path,
                                               monkeypatch) -> None:
    """--install against a corrupt user config must fail loudly, not
    replace the file (the pre-fix behavior lost the user's entire
    ~/.claude.json to one trailing comma)."""
    cfg = tmp_path / "claude.json"
    cfg.write_text('{"mcpServers": {},}')   # trailing comma: invalid JSON
    monkeypatch.setattr(cc_mod.shutil, "which", lambda _: None)
    inst = ClaudeCodeInstaller(config_path=str(cfg))
    with pytest.raises(ValueError, match="claude.json"):
        inst.install()
    assert cfg.read_text() == '{"mcpServers": {},}'   # file untouched


def test_importing_installers_creates_no_store(tmp_path: Path) -> None:
    """Importing the installers package (which imports the MCP server module)
    must not materialise a ./.chroma directory — the server's store is
    created lazily, on first tool use."""
    proc = subprocess.run(
        [sys.executable, "-c", "import ai_houkai.installers"],
        cwd=tmp_path, capture_output=True, text=True, timeout=120,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parent.parent)},
    )
    assert proc.returncode == 0, proc.stderr
    assert not (tmp_path / ".chroma").exists()


def test_importing_mcp_server_creates_no_store(tmp_path: Path) -> None:
    """The MCP server module itself must be import-side-effect-free."""
    proc = subprocess.run(
        [sys.executable, "-c", "import ai_houkai.mcp_server.server"],
        cwd=tmp_path, capture_output=True, text=True, timeout=120,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parent.parent)},
    )
    assert proc.returncode == 0, proc.stderr
    assert not (tmp_path / ".chroma").exists()


class TestDefaultMemoryPath:
    """Installer defaults must point at ~/.ai_houkai/.chroma — the CLI default
    — so `houkai list` sees installed-client memories and the store journal
    lands in ~/.ai_houkai/journal.log instead of $HOME/journal.log (the
    journal is written to the store path's PARENT directory)."""

    def test_all_installers_default_to_chroma_leaf(self):
        expected = os.path.expanduser("~/.ai_houkai/.chroma")
        assert cc_mod.DEFAULT_MEMORY_PATH == expected
        assert cur_mod.DEFAULT_MEMORY_PATH == expected
        assert oc_mod.DEFAULT_MEMORY_PATH == expected

    def test_default_journal_parent_is_houkai_dir(self):
        parent = Path(ClaudeCodeInstaller().memory_path).parent
        assert parent == Path(os.path.expanduser("~/.ai_houkai"))
        assert parent != Path(os.path.expanduser("~"))

    def test_matches_cli_default(self):
        assert ClaudeCodeInstaller().memory_path == cfg_mod._DEFAULT_PATH


def test_claude_code_uses_shared_mcp_command_resolver(monkeypatch):
    monkeypatch.setattr(common, "resolve_mcp_command", lambda: "/stub/mcp")
    monkeypatch.setattr(cc_mod, "resolve_mcp_command", lambda: "/stub/mcp")
    assert ClaudeCodeInstaller().mcp_command == "/stub/mcp"
    assert not hasattr(cc_mod, "_resolve_mcp_command")   # duplicate removed


class TestJSONConfigInstallers:
    """Cursor and OpenCode share JSONConfigInstaller.

    Before the shared base they were two 190-line near-copies with no
    coverage of their own beyond the default memory path, so these pin the
    behaviour the base now owns for both: the merge-and-write install, the
    client-specific block schema, and the verify report.
    """

    @pytest.fixture(params=["cursor", "opencode"])
    def client(self, request, tmp_path: Path):
        """(installer, config_path, config_key) for each JSON-config client."""
        if request.param == "cursor":
            cfg = tmp_path / "mcp.json"
            return (cur_mod.CursorInstaller(
                memory_path="/mem", collection="col",
                settings_path=str(cfg)), cfg, "mcpServers")
        cfg = tmp_path / "opencode.json"
        return (oc_mod.OpenCodeInstaller(
            memory_path="/mem", collection="col",
            settings_path=str(cfg)), cfg, "mcp")

    def test_install_writes_the_server_block(self, client) -> None:
        inst, cfg, key = client
        assert inst.install() == str(cfg)
        block = json.loads(cfg.read_text())[key]["ai-houkai"]
        env = block.get("env") or block["environment"]
        assert env["AI_HOUKAI_PATH"] == "/mem"
        assert env["AI_HOUKAI_COLLECTION"] == "col"

    def test_install_preserves_unrelated_config(self, client) -> None:
        """A client config also holds settings that are none of our business,
        and other MCP servers besides ours."""
        inst, cfg, key = client
        cfg.write_text(json.dumps({"theme": "dark",
                                   key: {"other": {"command": "x"}}}))
        inst.install()

        data = json.loads(cfg.read_text())
        assert data["theme"] == "dark"
        assert set(data[key]) == {"other", "ai-houkai"}

    def test_install_refuses_corrupt_config(self, client) -> None:
        inst, cfg, _ = client
        cfg.write_text('{"mcp": {},}')          # trailing comma: invalid JSON
        with pytest.raises(ValueError):
            inst.install()
        assert cfg.read_text() == '{"mcp": {},}'   # file untouched

    def test_verify_reports_registration(self, client, monkeypatch) -> None:
        inst, cfg, _ = client
        monkeypatch.setattr(common, "verify_server",
                            lambda **kw: True)

        out = io.StringIO()
        inst.verify(stream=out)
        assert "not found" in out.getvalue()        # nothing written yet

        inst.install()
        out = io.StringIO()
        assert inst.verify(stream=out) is True
        assert f"ok   registered in {cfg}" in out.getvalue()

    def test_install_is_idempotent(self, client) -> None:
        inst, cfg, key = client
        inst.install()
        first = cfg.read_text()
        inst.install()
        assert cfg.read_text() == first

    def test_defaults_come_from_the_class(self) -> None:
        """__post_init__ resolves the blank field defaults, so constructing
        an installer with no arguments still targets the right client."""
        assert cur_mod.CursorInstaller().collection == "cursor"
        assert cur_mod.CursorInstaller().settings_path == cur_mod.GLOBAL_CONFIG_PATH
        assert oc_mod.OpenCodeInstaller().collection == "opencode"
        assert oc_mod.OpenCodeInstaller().settings_path == oc_mod.GLOBAL_CONFIG_PATH


def test_cursor_block_uses_the_mcpservers_schema() -> None:
    """Cursor reads the same `mcpServers` schema as Claude Desktop."""
    block = cur_mod.CursorInstaller(memory_path="/mem").build_settings_block()
    assert set(block) == {"mcpServers"}
    assert set(block["mcpServers"]["ai-houkai"]) == {"command", "env"}


def test_opencode_block_uses_its_own_schema() -> None:
    """OpenCode's schema is not `mcpServers`: a `command` ARRAY under `mcp`,
    an `environment` object, an `enabled` flag, and a top-level `$schema`."""
    block = oc_mod.OpenCodeInstaller(memory_path="/mem").build_settings_block()
    assert block["$schema"] == oc_mod.CONFIG_SCHEMA_URL
    entry = block["mcp"]["ai-houkai"]
    assert entry["type"] == "local"
    assert isinstance(entry["command"], list)
    assert entry["enabled"] is True
    assert entry["environment"]["AI_HOUKAI_PATH"] == "/mem"


def test_opencode_install_seeds_schema_without_clobbering_it(tmp_path: Path) -> None:
    """`$schema` is a default, not an override — a user pinning their own
    must keep it."""
    cfg = tmp_path / "opencode.json"
    inst = oc_mod.OpenCodeInstaller(settings_path=str(cfg))
    inst.install()
    assert json.loads(cfg.read_text())["$schema"] == oc_mod.CONFIG_SCHEMA_URL

    cfg.write_text(json.dumps({"$schema": "https://example.invalid/mine.json"}))
    inst.install()
    data = json.loads(cfg.read_text())
    assert data["$schema"] == "https://example.invalid/mine.json"
    assert "ai-houkai" in data["mcp"]


@pytest.mark.parametrize("module,flag,marker", [
    (cur_mod, "--rule", "alwaysApply: true"),
    (oc_mod, "--agents", "## Memory (AI-Houkai MCP)"),
])
def test_installer_cli_prints_its_instruction_snippet(
    module, flag, marker, capsys
) -> None:
    assert module._main([flag]) == 0
    assert marker in capsys.readouterr().out


@pytest.mark.parametrize("module", [cur_mod, oc_mod])
def test_installer_cli_writes_the_config(module, tmp_path: Path, capsys) -> None:
    cfg = tmp_path / "client.json"
    assert module._main(["--install", "--settings", str(cfg)]) == 0
    assert f"written: {cfg}" in capsys.readouterr().out
    assert "ai-houkai" in json.dumps(json.loads(cfg.read_text()))


def test_memory_guide_is_shared_not_copied() -> None:
    """The three instruction snippets wrap ONE guide. They were three
    hand-maintained copies, and the Claude Code one had already lost the
    `edit` rows the others carry."""
    for snippet in (cc_mod.CLAUDEMD_SNIPPET,
                    cur_mod.RULE_SNIPPET,
                    oc_mod.AGENTS_SNIPPET):
        assert common.MEMORY_GUIDE in snippet
        assert "edit(memory_id" in snippet
