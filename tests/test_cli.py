"""CLI smoke tests using Typer's CliRunner."""

from __future__ import annotations

import re
import json
import stat
import sys

import pytest
from typer.testing import CliRunner

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
    assert "already resolved" in res.output
