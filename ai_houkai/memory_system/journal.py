"""Append-only audit journal for the memory store.

One JSON object per line, gzipped on rotation. Writes are best-effort:
a failure to journal never breaks the underlying memory operation. Use
``Journal.read(...)`` to stream historical entries with filtering and
``MemoryStore.undo(entry)`` to reverse a single operation.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

log = logging.getLogger(__name__)


JournalOp = str  # "remember"|"forget"|"edit"|"supersede"|"restore"|"link"
                 # |"unlink"|"reflect"|"decay"|"import"|"export"|"undo"


@dataclass(frozen=True)
class JournalEntry:
    ts:     float
    op:     JournalOp
    actor:  str
    id:     str
    before: dict[str, Any] | None
    after:  dict[str, Any] | None
    meta:   dict[str, Any]

    def to_line(self) -> str:
        rec = {
            "ts": self.ts, "op": self.op, "actor": self.actor, "id": self.id,
            "before": self.before, "after": self.after, "meta": self.meta,
        }
        return json.dumps(rec, separators=(",", ":")) + "\n"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "JournalEntry":
        return cls(
            ts=float(d["ts"]),
            op=str(d["op"]),
            actor=str(d.get("actor", "lib")),
            id=str(d.get("id", "")),
            before=d.get("before"),
            after=d.get("after"),
            meta=d.get("meta") or {},
        )

    def summary(self) -> str:
        """Short, human-readable one-line description."""
        if self.op == "remember" and self.after:
            txt = (self.after.get("text") or "")[:60]
            return f"remember {self.id[:8]} «{txt}»"
        if self.op == "forget" and self.before:
            txt = (self.before.get("text") or "")[:60]
            return f"forget {self.id[:8]} «{txt}»"
        if self.op == "edit" and self.after:
            txt = (self.after.get("text") or "")[:60]
            return f"edit {self.id[:8]} «{txt}»"
        if self.op == "supersede":
            new_id = self.meta.get("new_id", "")
            return f"supersede {self.id[:8]} → {new_id[:8]}"
        if self.op in ("link", "unlink"):
            dst = self.meta.get("dst_id", "")
            rel = self.meta.get("rel", "")
            return f"{self.op} {self.id[:8]} → {dst[:8]} ({rel})"
        return f"{self.op} {self.id[:8]}"


class Journal:
    """Append-only JSONL journal with size-based gzip rotation.

    Writes are atomic at the line level on POSIX (O_APPEND linearises
    across processes for writes ≤ PIPE_BUF). Reads tolerate truncated
    last lines (a crash mid-write) by skipping un-parseable JSON.
    """

    _ROTATE_CHECK_EVERY = 256   # how often to size-check on append

    def __init__(
        self,
        path: str | Path,
        *,
        rotate_mb: int = 64,
        keep_days: int = 90,
    ) -> None:
        self.path = Path(path)
        self.rotate_mb = rotate_mb
        self.keep_days = keep_days
        self._writes_since_check = 0

    def append(self, entry: JournalEntry) -> None:
        """Append one entry. Failures are logged, not raised."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(entry.to_line())
        except OSError as e:
            log.warning("journal append failed: %s", e)
            return

        self._writes_since_check += 1
        if self._writes_since_check >= self._ROTATE_CHECK_EVERY:
            self._writes_since_check = 0
            try:
                self._maybe_rotate()
            except OSError as e:
                log.warning("journal rotate failed: %s", e)

    def _maybe_rotate(self) -> None:
        if self.rotate_mb > 0 and self.path.exists():
            size_mb = self.path.stat().st_size / (1024 * 1024)
            if size_mb >= self.rotate_mb:
                self._rotate()
        if self.keep_days > 0:
            self._prune_archives()

    def _rotate(self) -> None:
        stamp = time.strftime("%Y%m%dT%H%M%S")
        base = f"{self.path.stem}-{stamp}"
        rotated = self.path.with_name(f"{base}.log")
        archive = self.path.with_name(f"{base}.log.gz")
        if rotated.exists() or archive.exists():
            base = f"{base}-{os.getpid()}"
            rotated = self.path.with_name(f"{base}.log")
            archive = self.path.with_name(f"{base}.log.gz")
        # Rename first (atomic), then compress the renamed file. Appends that
        # land during/after the rename open the path fresh and so go to a new
        # active file — the old compress-then-truncate order silently dropped
        # any entry appended between the copy and the truncate. A crash
        # mid-compression leaves the plain rotated file behind; read() still
        # includes it, so no entries are lost either way.
        os.rename(self.path, rotated)
        with open(rotated, "rb") as src, gzip.open(archive, "wb") as dst:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)
        rotated.unlink()

    def _prune_archives(self) -> None:
        cutoff = time.time() - self.keep_days * 86_400
        # `-*.log` covers plain rotated files orphaned by a crash mid-compression.
        for pattern in (f"{self.path.stem}-*.log.gz", f"{self.path.stem}-*.log"):
            for p in self.path.parent.glob(pattern):
                try:
                    if p.stat().st_mtime < cutoff:
                        p.unlink()
                except OSError:
                    continue

    def read(
        self,
        *,
        since:     float | None = None,
        until:     float | None = None,
        op:        str | None = None,
        actor:     str | None = None,
        memory_id: str | None = None,
        limit:     int | None = None,
        include_archives: bool = False,
    ) -> Iterator[JournalEntry]:
        """Stream entries matching the filters, oldest first.

        Truncated / corrupted lines are silently skipped — the journal's
        purpose is best-effort forensics, not strict consistency.
        """
        files: list[Path] = []
        if include_archives:
            # Plain `-*.log` files are rotations orphaned by a crash before
            # compression finished — their entries are still valid history.
            files.extend(sorted(
                list(self.path.parent.glob(f"{self.path.stem}-*.log.gz"))
                + list(self.path.parent.glob(f"{self.path.stem}-*.log"))
            ))
        if self.path.exists():
            files.append(self.path)

        n_yielded = 0
        for fp in files:
            opener = gzip.open if fp.suffix == ".gz" else open
            try:
                handle = opener(fp, "rt", encoding="utf-8")
            except OSError:
                continue
            with handle as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = JournalEntry.from_dict(json.loads(line))
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue
                    if since is not None and entry.ts < since:
                        continue
                    if until is not None and entry.ts > until:
                        continue
                    if op is not None and entry.op != op:
                        continue
                    if actor is not None and entry.actor != actor:
                        continue
                    if memory_id is not None and entry.id != memory_id:
                        continue
                    yield entry
                    n_yielded += 1
                    if limit is not None and n_yielded >= limit:
                        return

    def find_by_ts(self, ts: float, *, tol: float = 1e-3) -> JournalEntry | None:
        """Locate the entry with timestamp matching *ts* (within *tol* seconds)."""
        for entry in self.read(since=ts - tol, until=ts + tol, include_archives=True):
            if abs(entry.ts - ts) <= tol:
                return entry
        return None
