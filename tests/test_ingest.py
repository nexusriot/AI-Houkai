"""Tests for chunk_text and the houkai ingest / collections commands."""

from __future__ import annotations

import io
import json

import pytest
from typer.testing import CliRunner

from ai_houkai.cli import output as out
from ai_houkai.cli.main import app
from ai_houkai.memory_system.ingest import chunk_text

runner = CliRunner()


def _piped_stdin(monkeypatch) -> None:
    """Simulate stdin being a consumed pipe (not a terminal)."""
    monkeypatch.setattr(out.sys, "stdin",
                        type("S", (), {"isatty": lambda self: False})())


def _input_must_not_run(*a):
    raise AssertionError("input() must not be called when stdin is a pipe")


def _invoke(args, store_path, collection="cli_test", input=None):
    return runner.invoke(
        app, ["--store", store_path, "--collection", collection] + args,
        input=input,
    )


class TestChunkText:
    def test_splits_on_blank_lines(self):
        chunks = chunk_text(
            "First paragraph about deployment process.\n\n"
            "Second paragraph about testing conventions."
        )
        assert chunks == [
            "First paragraph about deployment process.",
            "Second paragraph about testing conventions.",
        ]

    def test_heading_glued_to_next_paragraph(self):
        chunks = chunk_text(
            "## Deploy\n\nUse make release to push a new version out."
        )
        assert len(chunks) == 1
        assert chunks[0].startswith("## Deploy\n")
        assert "make release" in chunks[0]

    def test_short_chunks_dropped(self):
        chunks = chunk_text("ok\n\nThis paragraph is long enough to survive the cut.")
        assert chunks == ["This paragraph is long enough to survive the cut."]

    def test_long_paragraph_split_on_sentences(self):
        para = " ".join(
            f"Sentence number {i} talks about something moderately interesting."
            for i in range(10)
        )
        chunks = chunk_text(para, max_chars=150)
        assert len(chunks) > 1
        assert all(len(c) <= 150 for c in chunks)
        # nothing lost
        assert "Sentence number 9" in chunks[-1]

    def test_single_oversized_sentence_kept_whole(self):
        sent = "word " * 60  # one "sentence", no terminator boundaries
        chunks = chunk_text(sent.strip() + ".", max_chars=100)
        assert len(chunks) == 1

    def test_crlf_normalised(self):
        chunks = chunk_text("Paragraph one is right here today.\r\n\r\nParagraph two follows after it.")
        assert len(chunks) == 2

    def test_empty_input(self):
        assert chunk_text("") == []
        assert chunk_text("\n\n\n") == []

    def test_trailing_heading_kept(self):
        chunks = chunk_text(
            "A real paragraph with enough words in it.\n\n## Dangling heading at the end"
        )
        assert chunks[-1] == "## Dangling heading at the end"


class TestIngestCommand:
    def test_ingest_file(self, tmp_path):
        store_path = str(tmp_path / "chroma")
        notes = tmp_path / "notes.md"
        notes.write_text(
            "## Deploy\n\nAlways deploy with make release, never by hand.\n\n"
            "The staging box is at 10.0.0.7 and runs Debian.\n"
        )
        result = _invoke(
            ["ingest", str(notes), "--type", "semantic", "--tag", "ops", "--yes"],
            store_path,
        )
        assert result.exit_code == 0, result.output
        assert "Stored 2 memories." in result.output

        listed = _invoke(["list", "--format", "json"], store_path)
        data = json.loads(listed.output)
        assert len(data) == 2
        assert all(d["source"] == "ingest:notes.md" for d in data)
        assert all("ops" in d["tags"] for d in data)

    def test_ingest_stdin(self, tmp_path):
        store_path = str(tmp_path / "chroma")
        result = _invoke(
            ["ingest", "--yes"], store_path,
            input="A paragraph that comes from standard input today.\n",
        )
        assert result.exit_code == 0, result.output
        assert "Stored 1 memories." in result.output
        data = json.loads(_invoke(["list", "--format", "json"], store_path).output)
        assert data[0]["source"] == "ingest:stdin"
        assert data[0]["type"] == "episodic"  # ingest default

    def test_dry_run_writes_nothing(self, tmp_path):
        store_path = str(tmp_path / "chroma")
        notes = tmp_path / "n.txt"
        notes.write_text("Some paragraph with a perfectly fine length.\n")
        result = _invoke(["ingest", str(notes), "--dry-run"], store_path)
        assert result.exit_code == 0, result.output
        assert "Dry-run" in result.output
        listed = _invoke(["list", "--format", "json"], store_path)
        assert "No memories found." in listed.output

    def test_auto_importance(self, tmp_path):
        store_path = str(tmp_path / "chroma")
        notes = tmp_path / "n.txt"
        notes.write_text(
            "Never deploy on a Friday afternoon, period.\n\n"
            "It seems the cache might be a little slow sometimes.\n"
        )
        _invoke(["ingest", str(notes), "--auto-importance", "--yes"], store_path)
        data = json.loads(_invoke(["list", "--format", "json"], store_path).output)
        by_text = {d["text"]: d["importance"] for d in data}
        assert by_text["Never deploy on a Friday afternoon, period."] >= 0.9
        assert by_text["It seems the cache might be a little slow sometimes."] < 0.5

    def test_missing_file_errors(self, tmp_path):
        result = _invoke(
            ["ingest", str(tmp_path / "nope.txt")], str(tmp_path / "chroma")
        )
        assert result.exit_code == 1


class TestConfirmViaTty:
    """out.confirm(use_tty=True): when input arrived on stdin (e.g. a piped
    `ingest`), stdin is at EOF, so input() would raise EOFError and silently
    abort. confirm must prompt on the controlling terminal instead, and fail
    loudly (not silently) when there is no terminal. (Regression.)"""

    def test_reads_yes_from_terminal_when_stdin_piped(self, monkeypatch):
        _piped_stdin(monkeypatch)
        monkeypatch.setattr(out, "open",
                            lambda *a, **k: io.StringIO("y\n"), raising=False)
        monkeypatch.setattr("builtins.input", _input_must_not_run)
        assert out.confirm("Store?", use_tty=True) is True

    def test_reads_no_from_terminal(self, monkeypatch):
        _piped_stdin(monkeypatch)
        monkeypatch.setattr(out, "open",
                            lambda *a, **k: io.StringIO("n\n"), raising=False)
        monkeypatch.setattr("builtins.input", _input_must_not_run)
        assert out.confirm("Store?", use_tty=True) is False

    def test_no_terminal_returns_false_without_crash(self, monkeypatch):
        _piped_stdin(monkeypatch)

        def no_tty(*a, **k):
            raise OSError("no controlling terminal")

        monkeypatch.setattr(out, "open", no_tty, raising=False)
        # Must not raise; declines rather than silently EOF-aborting.
        assert out.confirm("Store?", use_tty=True) is False

    def test_yes_flag_short_circuits(self, monkeypatch):
        def boom(*a, **k):
            raise AssertionError("--yes must not touch the terminal")

        monkeypatch.setattr(out, "open", boom, raising=False)
        assert out.confirm("Store?", yes=True, use_tty=True) is True


class TestCollectionsCommands:
    def test_list_create_delete(self, tmp_path):
        store_path = str(tmp_path / "chroma")
        _invoke(["remember", "Something to hold the collection open"], store_path)

        result = _invoke(["collections", "create", "scratch"], store_path)
        assert result.exit_code == 0, result.output

        listed = _invoke(["collections", "list", "--format", "json"], store_path)
        rows = {r["name"]: r for r in json.loads(listed.output)}
        assert rows["cli_test"]["count"] == 1
        assert rows["cli_test"]["active"] is True
        assert rows["scratch"]["count"] == 0

        result = _invoke(["collections", "delete", "scratch", "--yes"], store_path)
        assert result.exit_code == 0, result.output
        listed = _invoke(["collections", "list", "--format", "json"], store_path)
        assert "scratch" not in {r["name"] for r in json.loads(listed.output)}

    def test_create_duplicate_errors(self, tmp_path):
        store_path = str(tmp_path / "chroma")
        _invoke(["collections", "create", "dup"], store_path)
        result = _invoke(["collections", "create", "dup"], store_path)
        assert result.exit_code == 1

    def test_delete_active_collection_refused(self, tmp_path):
        store_path = str(tmp_path / "chroma")
        _invoke(["remember", "Keep the active collection populated"], store_path)
        result = _invoke(["collections", "delete", "cli_test", "--yes"], store_path)
        assert result.exit_code == 1
        assert "active collection" in result.output

    @pytest.mark.needs_model
    def test_copy_preserves_memories_and_search(self, tmp_path):
        store_path = str(tmp_path / "chroma")
        _invoke(["remember", "The deploy target is the staging box", "--tag", "ops"],
                store_path)
        _invoke(["remember", "Cats are excellent debugging companions"], store_path)

        result = _invoke(["collections", "copy", "cli_test", "backup", "--yes"],
                         store_path)
        assert result.exit_code == 0, result.output
        assert "Copied 2 memories" in result.output

        # The copy is a fully working collection: list + semantic recall.
        data = json.loads(
            _invoke(["list", "--format", "json"], store_path,
                    collection="backup").output
        )
        assert len(data) == 2
        recalled = _invoke(["recall", "deployment", "-k", "1", "--format", "json"],
                           store_path, collection="backup")
        assert "staging box" in recalled.output

    def test_copy_missing_src_errors(self, tmp_path):
        result = _invoke(["collections", "copy", "ghost", "dst", "--yes"],
                         str(tmp_path / "chroma"))
        assert result.exit_code == 1


class TestSplittingNeverLosesText:
    """`min_chars` drops noise, not the tail of a real paragraph.

    Greedy sentence packing leaves a short fragment whenever the next sentence
    will not fit, and that fragment can land anywhere — including last. Applying
    the `min_chars` noise filter to it deletes ingested text with no warning,
    which is not what the filter is for: separators and stray list bullets are
    noise, the last sentence of a paragraph is content.
    """

    def test_a_short_split_tail_survives(self):
        tail = "Tiny tail."
        para = " ".join(["A" * 38 + ".", "B" * 38 + ".", "C" * 96 + ".", tail])
        chunks = chunk_text(para, max_chars=100, min_chars=30)
        assert any(tail in c for c in chunks), f"tail was dropped: {chunks}"

    def test_every_sentence_of_a_split_paragraph_is_kept(self):
        sentences = ["A" * 38 + ".", "B" * 38 + ".", "C" * 96 + ".", "Tiny tail."]
        chunks = chunk_text(" ".join(sentences), max_chars=100, min_chars=30)
        joined = " ".join(chunks)
        for s in sentences:
            assert s in joined, f"{s!r} vanished from {chunks}"

    def test_a_short_leading_fragment_survives(self):
        """A runt can also come first, when one huge sentence follows a tiny one."""
        chunks = chunk_text("Hi. " + "Z" * 300 + ".", max_chars=100, min_chars=30)
        assert any("Hi." in c for c in chunks), f"leading fragment lost: {chunks}"

    def test_noise_blocks_are_still_dropped(self):
        """The filter must keep doing its job on standalone short blocks."""
        text = "---\n\n" + "R" * 80 + "\n\n*\n"
        chunks = chunk_text(text, max_chars=500, min_chars=30)
        assert chunks == ["R" * 80]
