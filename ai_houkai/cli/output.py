"""Output formatting: rich tables, json, tsv, id helpers."""

from __future__ import annotations

import json
import os
import sys
import time
from contextlib import contextmanager
from typing import Any, Iterator

import typer

from ai_houkai.memory_system.store import Memory

from rich.console import Console
from rich.table import Table
from rich.panel import Panel


@contextmanager
def friendly_errors() -> Iterator[None]:
    """Turn store validation errors (bad enum value, unknown id, self-link,
    supersede cycle, …) into a clean one-line CLI error + exit 1 instead of
    a traceback."""
    try:
        yield
    except (ValueError, KeyError) as e:
        msg = e.args[0] if e.args else str(e)
        typer.echo(f"Error: {msg}", err=True)
        raise typer.Exit(1)

# Re-check at call time (can be overridden after import)
def _is_tty() -> bool:
    return sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def resolve_format(fmt: str) -> str:
    """Resolve "auto" against the terminal, so callers can branch on the real
    format. Empty results still have to print something machine-readable, and
    only the caller knows whether it has rows."""
    if fmt != "auto":
        return fmt
    return "rich" if _is_tty() else "tsv"


def is_machine_format(fmt: str) -> bool:
    """True when the resolved format is meant to be parsed, not read.

    An empty result must still emit a valid document in those formats — `[]` or
    a bare header — or `houkai list -f json | jq .` fails on a fresh store.
    """
    return resolve_format(fmt) in ("json", "tsv")


def short_id(mid: str) -> str:
    return mid[:8]


def resolve_id_prefix(store, prefix: str) -> str:
    """Return the full UUID for an id prefix. Raises if ambiguous or not found."""
    if len(prefix) == 36:  # already full UUID
        return prefix
    all_mems = store._get_all_memories()
    matches = [m.id for m in all_mems if m.id.startswith(prefix)]
    if not matches:
        raise ValueError(f"No memory with id prefix {prefix!r}")
    if len(matches) > 1:
        raise ValueError(f"Ambiguous prefix {prefix!r} matches {len(matches)} memories")
    return matches[0]


def fmt_age(ts: float) -> str:
    delta = time.time() - ts
    if delta < 60:
        return "just now"
    if delta < 3600:
        return f"{int(delta/60)}m ago"
    if delta < 86400:
        return f"{int(delta/3600)}h ago"
    if delta < 86400 * 7:
        return f"{int(delta/86400)}d ago"
    if delta < 86400 * 60:
        return f"{int(delta/86400/7)}w ago"
    return f"{int(delta/86400/30)}mo ago"


def fmt_importance(v: float) -> str:
    filled = round(v * 5)
    return "█" * filled + "░" * (5 - filled)


def _truncate(text: str, width: int = 60) -> str:
    return text if len(text) <= width else text[:width - 1] + "…"


def print_memories_table(
    rows: list[tuple[Memory, float] | Memory],
    *,
    show_score: bool = False,
    fmt: str = "auto",
) -> None:
    """Print a list of memories as a table. rows can be (Memory,score) or Memory."""
    normalized: list[tuple[Memory, float | None]] = []
    for r in rows:
        if isinstance(r, tuple):
            normalized.append(r)
        else:
            normalized.append((r, None))

    output_fmt = resolve_format(fmt)

    if output_fmt == "json":
        print_memories_json(normalized)
        return

    if output_fmt == "tsv":
        _print_tsv(normalized, show_score=show_score)
        return

    console = Console()
    table = Table(show_header=True, header_style="bold cyan", expand=False)
    table.add_column("ID", style="dim", width=10)
    table.add_column("TYPE", width=10)
    table.add_column("IMP", width=7)
    if show_score:
        table.add_column("SCORE", width=6)
    table.add_column("TAGS", width=22)
    table.add_column("CREATED", width=10)
    table.add_column("TEXT")

    for mem, score in normalized:
        tags_str = ",".join(mem.tags) if mem.tags else ""
        imp_str = fmt_importance(mem.importance)
        row = [
            short_id(mem.id),
            mem.type,
            imp_str,
        ]
        if show_score:
            row.append(f"{score:.3f}" if score is not None else "—")
        row += [
            _truncate(tags_str, 20),
            fmt_age(mem.created_at),
            _truncate(mem.text, 65),
        ]
        style = "dim" if mem.superseded_by else None
        table.add_row(*row, style=style)

    console.print(table)


def _print_tsv(rows: list[tuple[Memory, float | None]], *, show_score: bool) -> None:
    header = ["id", "type", "importance", "tags", "created_at", "text"]
    if show_score:
        header.append("score")
    print("\t".join(header))
    for mem, score in rows:
        cols = [
            mem.id,
            mem.type,
            str(round(mem.importance, 3)),
            ",".join(mem.tags),
            str(int(mem.created_at)),
            mem.text.replace("\t", " ").replace("\n", " "),
        ]
        if show_score:
            cols.append(str(round(score, 4)) if score is not None else "")
        print("\t".join(cols))


def print_memories_json(rows: list[tuple[Memory, float | None]]) -> None:
    out = []
    for mem, score in rows:
        d = _mem_to_dict(mem)
        if score is not None:
            d["score"] = round(score, 4)
        out.append(d)
    print(json.dumps(out, indent=2))


def print_memory_detail(mem: Memory) -> None:
    console = Console()

    meta = Table.grid(padding=(0, 2))
    meta.add_row("[bold]id[/]",           mem.id)
    meta.add_row("[bold]type[/]",         mem.type)
    meta.add_row("[bold]importance[/]",   f"{mem.importance:.2f}  {fmt_importance(mem.importance)}")
    meta.add_row("[bold]tags[/]",         ", ".join(mem.tags) or "—")
    meta.add_row("[bold]created[/]",      fmt_age(mem.created_at))
    meta.add_row("[bold]last_accessed[/]",fmt_age(mem.last_accessed))
    meta.add_row("[bold]access_count[/]", str(mem.access_count))
    meta.add_row("[bold]source[/]",       mem.source or "—")
    meta.add_row("[bold]polarity[/]",     str(mem.polarity))
    if mem.superseded_by:
        meta.add_row("[bold red]superseded_by[/]", mem.superseded_by[:8])
    if mem.links:
        links_str = "\n".join(f"  {l.rel} → {l.to[:8]}" for l in mem.links)
        meta.add_row("[bold]links[/]", links_str)

    console.print(Panel(mem.text, title=f"[cyan]{mem.id[:8]}[/]", border_style="cyan"))
    console.print(meta)


def _mem_to_dict(mem: Memory) -> dict[str, Any]:
    return {
        "id": mem.id,
        "text": mem.text,
        "type": mem.type,
        "tags": mem.tags,
        "importance": mem.importance,
        "source": mem.source,
        "created_at": mem.created_at,
        "last_accessed": mem.last_accessed,
        "access_count": mem.access_count,
        "polarity": mem.polarity,
        "superseded_by": mem.superseded_by or None,
        "expires_at": mem.expires_at or None,
        "links": [{"to": l.to, "rel": l.rel} for l in mem.links],
    }


def confirm(prompt: str, *, yes: bool = False, use_tty: bool = False) -> bool:
    if yes:
        return True
    # When the data came in on stdin (e.g. a piped `ingest`), stdin is at EOF,
    # so input() would raise EOFError and we'd silently abort. Prompt on the
    # controlling terminal instead; if there is none, tell the user to pass
    # --yes rather than failing without explanation.
    if use_tty and not sys.stdin.isatty():
        try:
            tty = open("/dev/tty")
        except OSError:
            print(
                f"{prompt} — input was piped and no terminal is available "
                "to confirm; re-run with --yes.",
                file=sys.stderr,
            )
            return False
        try:
            print(f"{prompt} [y/N] ", end="", file=sys.stderr, flush=True)
            answer = tty.readline().strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        finally:
            tty.close()
        return answer in ("y", "yes")
    try:
        answer = input(f"{prompt} [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return answer in ("y", "yes")
