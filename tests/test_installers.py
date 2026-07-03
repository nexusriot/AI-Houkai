"""Tests for the client installers (Claude Code registration paths, shared
helpers, and the no-side-effects-on-import guarantee)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ai_houkai.installers import claude_code as cc_mod
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


def test_load_json_tolerates_garbage(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not json")
    assert load_json(str(path)) == {}
    with pytest.raises(json.JSONDecodeError):
        load_json(str(path), overwrite_unparseable=False)


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
