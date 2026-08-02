"""CLI smoke tests using Typer's CliRunner."""

from __future__ import annotations

import re
import json
import stat
import sys

import pytest
from typer.testing import CliRunner

import ai_houkai.cli.commands.decay as decay_mod
import ai_houkai.cli.commands.maintenance as maint_mod
import ai_houkai.cli.main as main_mod
from ai_houkai.cli import config as cfg_mod
from ai_houkai.cli.config import MaintenanceConfig
from ai_houkai.cli.main import app
from ai_houkai.memory_system import MemoryStore

runner = CliRunner()

_UUID_LEN = 36


@pytest.fixture()
def cli_store(tmp_path):
    s = MemoryStore(path=str(tmp_path / "chroma"), collection="cli_test")
    yield s, str(tmp_path / "chroma")
    s.client.close()


def _invoke(args: list[str], store_path: str) -> "Result":
    return runner.invoke(app, ["--store", store_path, "--collection", "cli_test"] + args)


def _last_line(output: str) -> str:
    """Return the last non-empty line (strips HF Hub noise from earlier lines)."""
    return next((l for l in reversed(output.splitlines()) if l.strip()), "")


def _first_uuid(output: str) -> str:
    """Extract the first UUID-shaped token from output."""
    m = re.search(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", output)
    return m.group(0) if m else ""


def test_remember_and_list(tmp_path):
    store_path = str(tmp_path / "chroma")
    result = _invoke(["remember", "The sky is blue", "--type", "semantic", "--tag", "nature"], store_path)
    assert result.exit_code == 0, result.output
    mem_id = _first_uuid(result.output)
    assert len(mem_id) == _UUID_LEN, f"UUID not found in output: {result.output!r}"

    result2 = _invoke(["list", "--format", "json"], store_path)
    assert result2.exit_code == 0, result2.output
    data = json.loads(result2.output)
    assert len(data) == 1
    assert data[0]["text"] == "The sky is blue"
    assert "nature" in data[0]["tags"]


def test_recall(tmp_path):
    store_path = str(tmp_path / "chroma")
    _invoke(["remember", "Roses are red", "--type", "episodic"], store_path)
    _invoke(["remember", "Violets are blue", "--type", "episodic"], store_path)

    result = _invoke(["recall", "flowers colors", "--format", "json", "-k", "2"], store_path)
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert len(data) >= 1
    assert "score" in data[0]


def test_pack_text(tmp_path):
    store_path = str(tmp_path / "chroma")
    _invoke(["remember", "Roses are red", "--type", "episodic"], store_path)
    _invoke(["remember", "Violets are blue", "--type", "episodic"], store_path)

    result = _invoke(["pack", "flowers colors", "--budget", "1000"], store_path)
    assert result.exit_code == 0, result.output
    assert "## Relevant memory" in result.output
    assert "Roses are red" in result.output or "Violets are blue" in result.output


def test_pack_json(tmp_path):
    store_path = str(tmp_path / "chroma")
    _invoke(["remember", "Python uses indentation", "--type", "semantic"], store_path)

    result = _invoke(["pack", "python syntax", "--format", "json", "--budget", "1000"], store_path)
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert "text" in data and "items" in data
    assert data["budget"] == 1000
    assert data["used_tokens"] == sum(i["tokens"] for i in data["items"])


def test_show(tmp_path):
    store_path = str(tmp_path / "chroma")
    r = _invoke(["remember", "Test memory for show"], store_path)
    mem_id = _first_uuid(r.output)

    result = _invoke(["show", mem_id[:8]], store_path)
    assert result.exit_code == 0
    assert "Test memory for show" in result.output


def test_forget(tmp_path):
    store_path = str(tmp_path / "chroma")
    r = _invoke(["remember", "To be forgotten"], store_path)
    mem_id = _first_uuid(r.output)

    result = _invoke(["forget", mem_id[:8], "--yes"], store_path)
    assert result.exit_code == 0
    assert "Deleted" in result.output

    list_result = _invoke(["list", "--format", "json"], store_path)
    data = json.loads(list_result.output) if list_result.output.strip().startswith("[") else []
    assert all(d["id"] != mem_id for d in data)


def test_tag_and_bump(tmp_path):
    store_path = str(tmp_path / "chroma")
    r = _invoke(["remember", "Hardware note", "--tag", "old"], store_path)
    mem_id = _first_uuid(r.output)

    result = _invoke(["tag", mem_id[:8], "--add", "hardware", "--remove", "old"], store_path)
    assert result.exit_code == 0, result.output
    assert "hardware" in result.output

    result2 = _invoke(["bump", mem_id[:8], "=0.9"], store_path)
    assert result2.exit_code == 0
    assert "0.90" in result2.output

    show_r = _invoke(["show", mem_id[:8]], store_path)
    assert "hardware" in show_r.output


def test_link_and_neighbors(tmp_path):
    store_path = str(tmp_path / "chroma")
    r1 = _invoke(["remember", "Parent memory"], store_path)
    r2 = _invoke(["remember", "Child memory"], store_path)
    id1 = _first_uuid(r1.output)
    id2 = _first_uuid(r2.output)

    link_r = _invoke(["link", id1[:8], id2[:8], "--rel", "refines"], store_path)
    assert link_r.exit_code == 0
    assert "refines" in link_r.output

    nb_r = _invoke(["neighbors", id1[:8], "--direction", "out"], store_path)
    assert nb_r.exit_code == 0
    assert id2[:8] in nb_r.output or "Child memory" in nb_r.output

    unlink_r = _invoke(["unlink", id1[:8], id2[:8]], store_path)
    assert unlink_r.exit_code == 0
    assert "Removed 1" in unlink_r.output


def test_supersede_and_restore(tmp_path):
    store_path = str(tmp_path / "chroma")
    r_old = _invoke(["remember", "Old fact about X"], store_path)
    r_new = _invoke(["remember", "New fact about X"], store_path)
    old_id = _first_uuid(r_old.output)
    new_id = _first_uuid(r_new.output)

    sup_r = _invoke(["supersede", old_id[:8], new_id[:8]], store_path)
    assert sup_r.exit_code == 0

    # List without superseded should hide old
    list_r = _invoke(["list", "--format", "json"], store_path)
    data = json.loads(list_r.output)
    active_ids = [d["id"] for d in data]
    assert old_id not in active_ids

    restore_r = _invoke(["restore", old_id[:8]], store_path)
    assert restore_r.exit_code == 0
    assert "restored" in restore_r.output


def test_export_import_roundtrip(tmp_path):
    store_path = str(tmp_path / "chroma")
    store2_path = str(tmp_path / "chroma2")
    export_file = str(tmp_path / "dump.jsonl")

    _invoke(["remember", "Export me 1", "--tag", "exp"], store_path)
    _invoke(["remember", "Export me 2", "--tag", "exp"], store_path)

    exp_r = _invoke(["export", export_file], store_path)
    assert exp_r.exit_code == 0
    assert "Exported 2 memories" in exp_r.output

    imp_r = _invoke(["import", export_file, "--yes"], store2_path)
    assert imp_r.exit_code == 0
    assert "imported=2" in imp_r.output

    list_r = _invoke(["list", "--format", "json"], store2_path)
    data = json.loads(list_r.output)
    assert len(data) == 2


def test_import_conflict_error_exits_cleanly(tmp_path):
    # `--on-conflict error` into a store that already has the ids used to
    # raise ImportConflictError past the CLI's except clause → raw traceback.
    store_path = str(tmp_path / "chroma")
    export_file = str(tmp_path / "dump.jsonl")
    _invoke(["remember", "collide me", "--tag", "x"], store_path)
    assert _invoke(["export", export_file], store_path).exit_code == 0

    # Import back into the SAME store so every id collides.
    result = _invoke(
        ["import", export_file, "--on-conflict", "error", "--yes"], store_path
    )
    assert result.exit_code == 1
    assert "Error" in result.output
    assert "Traceback" not in result.output


def test_import_bad_on_conflict_value_exits_cleanly(tmp_path):
    store_path = str(tmp_path / "chroma")
    export_file = str(tmp_path / "dump.jsonl")
    _invoke(["remember", "something", "--tag", "x"], store_path)
    _invoke(["export", export_file], store_path)
    result = _invoke(
        ["import", export_file, "--on-conflict", "bogus", "--yes"],
        str(tmp_path / "chroma2"),
    )
    assert result.exit_code == 1
    assert "Traceback" not in result.output


def test_stats(tmp_path):
    store_path = str(tmp_path / "chroma")
    _invoke(["remember", "Stat test", "--type", "procedural"], store_path)

    result = _invoke(["stats", "--format", "json"], store_path)
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["active"] == 1
    assert data["by_type"].get("procedural") == 1


def test_prune_dryrun(tmp_path):
    store_path = str(tmp_path / "chroma")
    _invoke(["remember", "Low importance memory", "--importance", "0.01"], store_path)

    result = _invoke(["prune", "--min-score", "0.5"], store_path)
    assert result.exit_code == 0
    # dry-run by default — memory should still be there
    list_r = _invoke(["list", "--format", "json"], store_path)
    data = json.loads(list_r.output)
    assert len(data) == 1


def test_remember_from_stdin(tmp_path):
    store_path = str(tmp_path / "chroma")
    result = runner.invoke(
        app,
        ["--store", store_path, "--collection", "cli_test", "remember", "--stdin"],
        input="Memory from stdin\n",
    )
    assert result.exit_code == 0
    mem_id = _first_uuid(result.output)
    assert len(mem_id) == _UUID_LEN, f"UUID not found in output: {result.output!r}"


def _make_fake_editor(tmp_path, new_text: str) -> str:
    """Write an executable 'editor' that rewrites the body after '---'."""
    script = tmp_path / "fake_editor.py"
    script.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "path = sys.argv[1]\n"
        "with open(path) as f:\n"
        "    content = f.read()\n"
        "head, sep, _ = content.partition('---')\n"
        f"new_text = {new_text!r}\n"
        "with open(path, 'w') as f:\n"
        "    f.write(head + sep + '\\n' + new_text + '\\n')\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IRWXU)
    return str(script)


def test_edit_text_change_preserves_id_and_links(tmp_path, monkeypatch):
    """Editing a memory's text re-embeds it in place: the id is unchanged and
    both its own links and incoming links survive (regression — the old
    forget()+remember() path minted a new id and orphaned the links)."""
    store_path = str(tmp_path / "chroma")

    a_id = _first_uuid(_invoke(["remember", "Original text about cats"], store_path).output)
    b_id = _first_uuid(_invoke(["remember", "An unrelated note about dogs"], store_path).output)

    # A -> B (A's own outgoing link) and B -> A (an incoming reference to A).
    assert _invoke(["link", a_id, b_id, "--rel", "related"], store_path).exit_code == 0
    assert _invoke(["link", b_id, a_id, "--rel", "refines"], store_path).exit_code == 0

    editor = _make_fake_editor(tmp_path, "Updated text about cats and kittens")
    monkeypatch.setenv("EDITOR", editor)

    res = _invoke(["edit", a_id], store_path)
    assert res.exit_code == 0, res.output

    store = MemoryStore(path=store_path, collection="cli_test")
    try:
        a = store._get_by_id(a_id)
        b = store._get_by_id(b_id)

        assert a is not None, "edit must not change the memory id"
        assert a.text == "Updated text about cats and kittens"
        # A's own outgoing link is intact.
        assert any(l.to == b_id and l.rel == "related" for l in a.links)
        # B's incoming link still points at a memory that exists (not orphaned).
        assert any(l.to == a_id and l.rel == "refines" for l in b.links)
        assert store._get_by_id(b.links[0].to) is not None
    finally:
        store.client.close()


def test_edit_preserves_markdown_headings_in_body(tmp_path, monkeypatch):
    """A '#'-prefixed line in the memory body is content, not an editor comment:
    editing must not strip markdown headings from the text (regression — the
    old code filtered '#' lines across the whole file, including the body)."""
    store_path = str(tmp_path / "chroma")
    a_id = _first_uuid(_invoke(["remember", "Plain note to be replaced"], store_path).output)

    new_text = "# Deploy guide\n\nRun make release.\n## Notes\nNever on a Friday."
    editor = _make_fake_editor(tmp_path, new_text)
    monkeypatch.setenv("EDITOR", editor)

    res = _invoke(["edit", a_id], store_path)
    assert res.exit_code == 0, res.output

    store = MemoryStore(path=store_path, collection="cli_test")
    try:
        a = store._get_by_id(a_id)
        assert a is not None
        assert a.text == new_text          # nothing stripped
        assert "# Deploy guide" in a.text
        assert "## Notes" in a.text
    finally:
        store.client.close()


def test_bump_accepts_negative_delta(tmp_path):
    """`bump <id> -0.1` must parse as the delta argument, not be rejected as an
    unknown option (regression — Click treated the leading '-' as a flag)."""
    store_path = str(tmp_path / "chroma")
    mem_id = _first_uuid(
        _invoke(["remember", "Tweak my importance", "--importance", "0.5"], store_path).output
    )
    res = _invoke(["bump", mem_id[:8], "-0.1"], store_path)
    assert res.exit_code == 0, res.output
    assert "0.50 → 0.40" in res.output


def test_journal_tail_exposes_raw_ts(tmp_path):
    """`journal tail` must print the raw epoch ts so `journal show`/`undo` —
    which take a float ts — are usable from its output (regression — tail only
    rendered a formatted datetime, leaving no way to obtain the ts)."""
    store_path = str(tmp_path / "chroma")
    _invoke(["remember", "An auditable memory"], store_path)

    store = MemoryStore(path=store_path, collection="cli_test")
    try:
        ts = next(e.ts for e in store.journal.read())
    finally:
        store.client.close()

    # Wide terminal so rich doesn't truncate the ts column.
    tail = runner.invoke(
        app,
        ["--store", store_path, "--collection", "cli_test", "journal", "tail"],
        env={"COLUMNS": "200"},
    )
    assert tail.exit_code == 0, tail.output
    assert f"{ts:.3f}" in tail.output      # raw epoch printed

    show = _invoke(["journal", "show", f"{ts:.3f}"], store_path)
    assert show.exit_code == 0, show.output
    assert "remember" in show.output


def test_conflicts_interactive_skips_resolved_memories(tmp_path):
    """Interactive conflict resolution must not crash when a memory resolved in
    one pair reappears in a later pair (regression — the old code called
    supersede/forget on the missing id and raised KeyError mid-scan)."""
    store_path = str(tmp_path / "chroma")

    # Three mutually-similar memories => find_conflicts yields three pairs,
    # each sharing members, so a resolved memory recurs in a later pair.
    for text in (
        "The database uses Postgres for storage.",
        "Our primary database is Postgres.",
        "We run Postgres as the database backend.",
    ):
        _invoke(["remember", text], store_path)

    # Pair 1: delete A. The remaining pairs are answered with supersede — one of
    # them references the just-deleted memory and is skipped instead of crashing.
    res = runner.invoke(
        app,
        ["--store", store_path, "--collection", "cli_test",
         "conflicts", "--threshold", "0.3", "--resolve", "interactive"],
        input="d\ns\ns\n",
    )

    assert res.exception is None, f"interactive resolve crashed: {res.exception!r}"
    assert res.exit_code == 0, res.output


def test_nuke_deletes_all_memories(tmp_path):
    store_path = str(tmp_path / "chroma")
    for text in ("alpha", "beta", "gamma"):
        _invoke(["remember", text], store_path)

    result = _invoke(["nuke", "--yes"], store_path)
    assert result.exit_code == 0, result.output
    assert "3" in _last_line(result.output)

    result2 = _invoke(["list"], store_path)
    assert "No memories found" in result2.output


def test_nuke_empty_collection(tmp_path):
    store_path = str(tmp_path / "chroma")
    result = _invoke(["nuke", "--yes"], store_path)
    assert result.exit_code == 0, result.output
    assert "empty" in result.output.lower()


def test_nuke_requires_confirmation(tmp_path):
    store_path = str(tmp_path / "chroma")
    _invoke(["remember", "keep me"], store_path)

    result = runner.invoke(
        app,
        ["--store", store_path, "--collection", "cli_test", "nuke"],
        input="n\n",
    )
    assert result.exit_code == 0
    assert "Aborted" in result.output

    result2 = _invoke(["list", "--format", "json"], store_path)
    assert len(json.loads(result2.output)) == 1


def test_graph_json_edges_use_full_ids(tmp_path):
    """Regression: edges carried 8-char id prefixes while nodes carried full
    ids, making the JSON unjoinable for machine consumers."""
    store_path = str(tmp_path / "chroma")
    a = _first_uuid(_invoke(["remember", "graph json node a"], store_path).output)
    b = _first_uuid(_invoke(["remember", "graph json node b"], store_path).output)
    _invoke(["link", a, b, "--rel", "refines"], store_path)

    result = _invoke(["graph", a, "--format", "json"], store_path)
    assert result.exit_code == 0, result.output
    data = json.loads(result.output[result.output.index("{"):])
    node_ids = {n["id"] for n in data["nodes"]}
    for edge in data["edges"]:
        assert len(edge["src"]) == _UUID_LEN
        assert len(edge["dst"]) == _UUID_LEN
        assert edge["src"] in node_ids and edge["dst"] in node_ids


def test_config_expands_tilde_from_env(monkeypatch, tmp_path):

    monkeypatch.setattr(cfg_mod, "_CONFIG_FILE", tmp_path / "missing.toml")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("AI_HOUKAI_PATH", "~/env-store/.chroma")
    cfg = cfg_mod.load()
    assert cfg.store_path == str(tmp_path / "env-store" / ".chroma")
    assert "~" not in cfg.store_path


def test_config_expands_tilde_from_config_file(monkeypatch, tmp_path):

    toml = tmp_path / "config.toml"
    toml.write_text('store_path = "~/file-store/.chroma"\n')
    monkeypatch.setattr(cfg_mod, "_CONFIG_FILE", toml)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("AI_HOUKAI_PATH", raising=False)
    cfg = cfg_mod.load()
    assert cfg.store_path == str(tmp_path / "file-store" / ".chroma")


def test_cli_store_option_expands_tilde(monkeypatch, tmp_path):
    """`houkai --store '~/x'` (quoted, so the shell didn't expand it) must not
    create a literal ./~ directory."""

    captured = {}

    def fake_store(*, path, collection, actor, **kwargs):
        # **kwargs so the double survives new constructor options (index=…)
        # without pretending to care about them.
        captured["path"] = path
        raise RuntimeError("stop before touching chroma")

    monkeypatch.setattr(main_mod, "MemoryStore", fake_store)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("AI_HOUKAI_PATH", raising=False)
    runner.invoke(app, ["--store", "~/tilde/chroma", "list"])
    assert captured["path"] == str(tmp_path / "tilde" / "chroma")


def test_maintenance_config_consolidate_parsing(monkeypatch, tmp_path):

    toml = tmp_path / "config.toml"

    def load_with(consolidate_line: str):
        toml.write_text("[maintenance.reflect]\n" + consolidate_line)
        monkeypatch.setattr(cfg_mod, "_CONFIG_FILE", toml)
        return cfg_mod.load_maintenance()

    assert load_with("").reflect_consolidate is True            # default: soft
    assert load_with('consolidate = "none"').reflect_consolidate is False
    assert load_with('consolidate = "soft"').reflect_consolidate is True
    assert load_with('consolidate = "hard"').reflect_consolidate == "hard"
    with pytest.raises(ValueError, match="consolidate"):
        load_with('consolidate = "sideways"')


def test_maintenance_config_enabled_defaults_true(monkeypatch, tmp_path):

    monkeypatch.setattr(cfg_mod, "_CONFIG_FILE", tmp_path / "missing.toml")
    assert cfg_mod.load_maintenance().enabled is True

    toml = tmp_path / "config.toml"
    toml.write_text("[maintenance]\nenabled = false\n")
    monkeypatch.setattr(cfg_mod, "_CONFIG_FILE", toml)
    assert cfg_mod.load_maintenance().enabled is False


def test_maintenance_tick_noops_when_disabled(tmp_path, monkeypatch):

    def fake_mcfg():
        return MaintenanceConfig(
            enabled=False, decay_every=1, reflect_every=1, purge_every=1,
            tick_interval=300,
            log_path=str(tmp_path / "m.log"),
            state_path=str(tmp_path / "m.state.json"),
            pid_path=str(tmp_path / "m.pid"), decay_rate=0.1, min_score=0.05,
            protect_types=("procedural",), frequency_weight=0.0,
            min_cluster_size=2, reflect_apply=True, summarizer=None,
        )

    monkeypatch.setattr(maint_mod, "load_maintenance", fake_mcfg)
    store_path = str(tmp_path / "chroma")
    result = _invoke(["maintenance", "tick"], store_path)
    assert result.exit_code == 0, result.output
    assert "disabled" in result.output
    assert "Running maintenance tick" not in result.output


def test_prune_defaults_come_from_config(tmp_path, monkeypatch):

    def fake_mcfg():
        return MaintenanceConfig(
            enabled=True, decay_every=None, reflect_every=None, purge_every=None,
            tick_interval=300,
            log_path="", state_path="", pid_path="",
            decay_rate=0.1, min_score=0.99,          # everything is at risk
            protect_types=(), frequency_weight=0.0,
            min_cluster_size=3, reflect_apply=False, summarizer=None,
        )

    monkeypatch.setattr(decay_mod, "load_maintenance", fake_mcfg)
    store_path = str(tmp_path / "chroma")
    _invoke(["remember", "fresh but doomed", "-i", "0.5"], store_path)

    # Config min_score=0.99 → the fresh memory is a prune candidate.
    result = _invoke(["prune"], store_path)
    assert result.exit_code == 0, result.output
    assert "Prune candidates (1)" in result.output

    # An explicit flag still wins over the config value.
    result2 = _invoke(["prune", "--min-score", "0.0000001"], store_path)
    assert result2.exit_code == 0, result2.output
    assert "Nothing to prune." in result2.output


def test_ingest_journals_as_ingest_actor(tmp_path):
    store_path = str(tmp_path / "chroma")
    doc = tmp_path / "notes.md"
    doc.write_text("A paragraph long enough to survive the min-chars filter.\n")
    result = _invoke(["ingest", str(doc), "--yes"], store_path)
    assert result.exit_code == 0, result.output

    store = MemoryStore(path=store_path, collection="cli_test")
    try:
        entries = [e for e in store.journal.read() if e.op == "remember"]
        assert entries
        assert all(e.actor == "ingest" for e in entries)
    finally:
        store.client.close()


def test_ingest_bad_type_is_friendly_error(tmp_path):
    store_path = str(tmp_path / "chroma")
    doc = tmp_path / "notes.md"
    doc.write_text("A paragraph long enough to survive the min-chars filter.\n")
    result = _invoke(["ingest", str(doc), "--type", "bogus", "--yes"], store_path)
    assert result.exit_code == 1
    assert "Error:" in result.output
    assert "Traceback" not in result.output


def test_auto_context_text(tmp_path):
    store_path = str(tmp_path / "chroma")
    _invoke(["remember", "The deploy pipeline runs through GitHub Actions",
             "--type", "procedural"], store_path)
    result = _invoke(["auto-context", "deploy the api to production",
                      "--budget", "500"], store_path)
    assert result.exit_code == 0, result.output
    assert "## Relevant memory" in result.output


def test_auto_context_json(tmp_path):
    store_path = str(tmp_path / "chroma")
    _invoke(["remember", "The deploy pipeline runs through GitHub Actions",
             "--type", "procedural"], store_path)
    result = _invoke(["auto-context", "deploy the api", "-f", "json"], store_path)
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert set(data) >= {"text", "used_tokens", "budget", "truncated", "items"}
    assert data["items"]


def test_remember_ttl_hidden_from_recall(tmp_path):
    store_path = str(tmp_path / "chroma")
    # expires_at in the past → immediately expired.
    _invoke(["remember", "ephemeral cli note", "--expires-at", "1"], store_path)
    r = _invoke(["recall", "ephemeral cli note", "-f", "json"], store_path)
    # Expired: recall prints "No memories found." (stderr); the text is absent.
    assert "ephemeral cli note" not in r.output
    # With --include-expired it shows up.
    r2 = _invoke(["recall", "ephemeral cli note", "--include-expired", "-f", "json"],
                 store_path)
    assert "ephemeral cli note" in r2.output


def test_purge_command_dry_run_then_apply(tmp_path):
    store_path = str(tmp_path / "chroma")
    _invoke(["remember", "expired cli doc", "--expires-at", "1"], store_path)
    _invoke(["remember", "live cli doc"], store_path)

    dry = _invoke(["purge"], store_path)
    assert dry.exit_code == 0
    assert "Dry-run" in dry.output
    # Nothing deleted yet.
    listed = _invoke(["list", "--format", "json", "--include-expired"], store_path)
    assert len(json.loads(listed.output)) == 2

    applied = _invoke(["purge", "--apply", "--yes"], store_path)
    assert applied.exit_code == 0
    assert "Purged 1" in applied.output
    listed2 = _invoke(["list", "--format", "json", "--include-expired"], store_path)
    assert len(json.loads(listed2.output)) == 1


def test_purge_nothing_to_do(tmp_path):
    store_path = str(tmp_path / "chroma")
    _invoke(["remember", "permanent doc"], store_path)
    r = _invoke(["purge"], store_path)
    assert r.exit_code == 0
    assert "Nothing to purge" in r.output
