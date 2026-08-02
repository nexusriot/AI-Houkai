"""Derived SQLite index beside the Chroma store.


Chroma is authoritative for text, metadata and vectors. It is not, however, a
metadata query engine: it has no reverse-link lookup, no cursor pagination, no
tag aggregation, and its lexical matching is whatever the caller does to the
rows it already fetched. So the store ends up scanning the whole collection for
work that is a one-line query in SQL:

    list_recent          load every row, sort in Python, slice
    _get_all_memories    load every row
    export / stats       load every row
    purge_expired        load every row
    neighbors(in|both)   load every row, once per frontier node per hop
    DecayEngine          list_recent(limit=100_000)
    ReflectionEngine     every episodic row, with embeddings


This module keeps a SQLite file next to `.chroma` mirroring the metadata, plus
an FTS5 table over the text. That single index delivers:

  1. full-corpus BM25 — `_bm25_score_pool` only ever scored the vector
     over-fetch pool, so a strong exact-token match with a weak embedding was
     unreachable *at any corpus size*
  2. cursor pagination on list_recent
  3. O(1) reverse links
  4. cheap tag counts and per-type aggregates
  5. an indexed expiry sweep


**It is a cache, never a source of truth.** Every read has a scan fallback, and
a sidecar that disagrees with Chroma's row count is declared unhealthy and
ignored until `houkai reindex` rebuilds it — a stale index must degrade to
"slower", never to "wrong".


Opt in with ``MemoryStore(index="sqlite")`` or ``AI_HOUKAI_INDEX=sqlite``;
the default keeps the previous scan behaviour byte for byte.
"""


from __future__ import annotations


import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


log = logging.getLogger("ai_houkai.sidecar")


SCHEMA_VERSION = 1


_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);


CREATE TABLE IF NOT EXISTS memories (
    id            TEXT PRIMARY KEY,
    text          TEXT NOT NULL,
    type          TEXT NOT NULL,
    tags          TEXT NOT NULL DEFAULT '',
    importance    REAL NOT NULL DEFAULT 0.5,
    created_at    REAL NOT NULL DEFAULT 0,
    last_accessed REAL NOT NULL DEFAULT 0,
    access_count  INTEGER NOT NULL DEFAULT 0,
    source        TEXT,
    superseded_by TEXT NOT NULL DEFAULT '',
    expires_at    REAL NOT NULL DEFAULT 0
);


CREATE INDEX IF NOT EXISTS idx_mem_created  ON memories(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_mem_type     ON memories(type);
CREATE INDEX IF NOT EXISTS idx_mem_expires  ON memories(expires_at)
    WHERE expires_at > 0;
CREATE INDEX IF NOT EXISTS idx_mem_active   ON memories(superseded_by);


CREATE TABLE IF NOT EXISTS links (
    src TEXT NOT NULL,
    dst TEXT NOT NULL,
    rel TEXT NOT NULL,
    PRIMARY KEY (src, dst, rel)
);


-- The whole point of the links table: an index on dst turns "who points at
-- me?" from a full-collection scan into a lookup.
CREATE INDEX IF NOT EXISTS idx_links_dst ON links(dst);


CREATE TABLE IF NOT EXISTS tags (
    memory_id TEXT NOT NULL,
    tag       TEXT NOT NULL,
    PRIMARY KEY (memory_id, tag)
);


CREATE INDEX IF NOT EXISTS idx_tags_tag ON tags(tag);
"""


# FTS5 is a compile-time option; a Python built without it still gets every
# other benefit of the sidecar, so its absence is a downgrade, not an error.
#
# Deliberately a STANDALONE table, not an external-content one. External
# content halves the storage but makes correctness depend on hand-maintaining
# rowid deltas — issue the 'delete' command for a row that was never indexed
# and SQLite reports "database disk image is malformed". Duplicating the text
# into a rebuildable cache is the cheaper mistake.
_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    id UNINDEXED,
    text,
    tokenize='unicode61 remove_diacritics 2'
);
"""


class SidecarUnavailable(RuntimeError):
    """The sidecar could not be opened or is not usable."""


def fts5_available() -> bool:
    """Whether this interpreter's SQLite was built with FTS5."""
    probe = sqlite3.connect(":memory:")
    try:
        probe.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
        return True
    except sqlite3.OperationalError:
        return False
    finally:
        probe.close()


class SidecarIndex:
    """SQLite metadata + full-text index mirroring a Chroma collection.

    One instance per store. Writes are small and synchronous; the connection is
    guarded by a lock because the HTTP server serves requests on threads.
    """

    def __init__(self, path: str | Path, *, collection: str = "ai_houkai") -> None:
        self.path = Path(path)
        self.collection = collection
        self._lock = threading.RLock()
        self.healthy = True
        # Reason the index was disabled, surfaced by `doctor` / stats.
        self.disabled_reason: str | None = None

        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        except sqlite3.Error as exc:
            raise SidecarUnavailable(f"cannot open {self.path}: {exc}") from exc

        self._conn.row_factory = sqlite3.Row
        # WAL keeps a reader from blocking the writer; NORMAL is the right
        # durability tradeoff for a rebuildable cache.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        self.fts = fts5_available()
        if self.fts:
            self._conn.executescript(_FTS_SCHEMA)
        self._conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass

    def disable(self, reason: str) -> None:
        """Mark the index unusable so every read falls back to scanning.

        Called when the sidecar disagrees with Chroma or a write fails: a stale
        index must make things slower, never wrong.
        """
        if self.healthy:
            log.warning("ai-houkai: sidecar index disabled (%s) — "
                        "run `houkai reindex` to rebuild", reason)
        self.healthy = False
        self.disabled_reason = reason

    def count(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM memories").fetchone()
        return int(row["n"])

    def verify(self, authoritative_count: int) -> bool:
        """Compare row counts with Chroma; disable on a mismatch.

        A cheap, conservative health check: it cannot detect a same-count
        divergence, but it does catch the common cases (index built for a
        different collection, writes lost while the sidecar was missing,
        somebody edited Chroma directly).
        """
        if not self.healthy:
            return False
        mine = self.count()
        if mine != authoritative_count:
            self.disable(
                f"row count {mine} != collection count {authoritative_count}")
            return False
        return True

    def upsert(self, rows: Iterable[dict[str, Any]]) -> None:
        """Insert or replace memory rows, with their tags and outgoing links."""
        rows = list(rows)
        if not rows or not self.healthy:
            return
        try:
            with self._lock, self._conn:
                for row in rows:
                    self._upsert_one(row)
        except sqlite3.Error as exc:
            self.disable(f"write failed: {exc}")

    def _upsert_one(self, row: dict[str, Any]) -> None:
        mid = row["id"]
        tags = row.get("tags") or []
        self._conn.execute(
            """INSERT INTO memories
               (id, text, type, tags, importance, created_at, last_accessed,
                access_count, source, superseded_by, expires_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET
                 text=excluded.text, type=excluded.type, tags=excluded.tags,
                 importance=excluded.importance, created_at=excluded.created_at,
                 last_accessed=excluded.last_accessed,
                 access_count=excluded.access_count, source=excluded.source,
                 superseded_by=excluded.superseded_by,
                 expires_at=excluded.expires_at""",
            (mid, row.get("text", ""), row.get("type", "semantic"),
             ",".join(tags), float(row.get("importance", 0.5)),
             float(row.get("created_at", 0.0)),
             float(row.get("last_accessed", 0.0)),
             int(row.get("access_count", 0)), row.get("source"),
             row.get("superseded_by") or "", float(row.get("expires_at", 0.0))),
        )
        self._conn.execute("DELETE FROM tags WHERE memory_id = ?", (mid,))
        if tags:
            self._conn.executemany(
                "INSERT OR IGNORE INTO tags (memory_id, tag) VALUES (?, ?)",
                [(mid, t) for t in tags])
        self._conn.execute("DELETE FROM links WHERE src = ?", (mid,))
        links = row.get("links") or []
        if links:
            self._conn.executemany(
                "INSERT OR IGNORE INTO links (src, dst, rel) VALUES (?, ?, ?)",
                [(mid, lnk["to"], lnk["rel"]) for lnk in links])
        if self.fts:
            # Delete-then-insert: an edit must replace the indexed text, not
            # add a second copy that would return the same id twice.
            self._conn.execute("DELETE FROM memories_fts WHERE id = ?", (mid,))
            self._conn.execute(
                "INSERT INTO memories_fts (id, text) VALUES (?, ?)",
                (mid, row.get("text", "")))

    def delete(self, ids: Sequence[str]) -> None:
        if not ids or not self.healthy:
            return
        try:
            with self._lock, self._conn:
                for mid in ids:
                    if self.fts:
                        self._conn.execute(
                            "DELETE FROM memories_fts WHERE id = ?", (mid,))
                    self._conn.execute("DELETE FROM memories WHERE id = ?", (mid,))
                    self._conn.execute("DELETE FROM tags WHERE memory_id = ?", (mid,))
                    # Drop edges in both directions: a dangling dst would make
                    # reverse lookups report neighbours that no longer exist.
                    self._conn.execute(
                        "DELETE FROM links WHERE src = ? OR dst = ?", (mid, mid))
        except sqlite3.Error as exc:
            self.disable(f"delete failed: {exc}")

    def clear(self) -> None:
        """Empty the index (mirrors a nuke)."""
        try:
            with self._lock, self._conn:
                self._conn.execute("DELETE FROM memories")
                self._conn.execute("DELETE FROM tags")
                self._conn.execute("DELETE FROM links")
                if self.fts:
                    self._conn.execute("DELETE FROM memories_fts")
            self.healthy = True
            self.disabled_reason = None
        except sqlite3.Error as exc:
            self.disable(f"clear failed: {exc}")

    def rebuild(self, rows: Iterable[dict[str, Any]]) -> int:
        """Drop everything and re-index *rows*. Restores health on success."""
        rows = list(rows)
        try:
            with self._lock, self._conn:
                self._conn.execute("DELETE FROM memories")
                self._conn.execute("DELETE FROM tags")
                self._conn.execute("DELETE FROM links")
                if self.fts:
                    self._conn.execute("DELETE FROM memories_fts")
                # Re-enable before writing: _upsert_one is a no-op guard away
                # from silently skipping the rebuild it was asked to perform.
                self.healthy = True
                self.disabled_reason = None
                for row in rows:
                    self._upsert_one(row)
        except sqlite3.Error as exc:
            self.disable(f"rebuild failed: {exc}")
            return 0
        return len(rows)

    def search_lexical(self, query: str, limit: int = 100) -> list[str]:
        """Return ids of the best full-corpus BM25 matches, best first.

        This is the piece the vector over-fetch pool cannot provide: an exact
        token match whose embedding is weak is invisible to `_bm25_score_pool`
        at any corpus size, because it never enters the pool to be scored.
        """
        if not (self.healthy and self.fts):
            return []
        match = _fts_query(query)
        if not match:
            return []
        try:
            with self._lock:
                rows = self._conn.execute(
                    """SELECT id
                       FROM memories_fts
                       WHERE memories_fts MATCH ?
                       ORDER BY bm25(memories_fts)
                       LIMIT ?""",
                    (match, limit),
                ).fetchall()
        except sqlite3.Error as exc:
            # A malformed MATCH expression is a query problem, not corruption —
            # return nothing rather than disabling the whole index.
            log.debug("ai-houkai: fts query failed (%s)", exc)
            return []
        return [r["id"] for r in rows]

    def incoming(self, dst: str, rel: str | None = None) -> list[tuple[str, str]]:
        """(src_id, rel) pairs pointing at *dst* — the reverse-link lookup."""
        if not self.healthy:
            return []
        sql = "SELECT src, rel FROM links WHERE dst = ?"
        params: list[Any] = [dst]
        if rel is not None:
            sql += " AND rel = ?"
            params.append(rel)
        with self._lock:
            rows = self._conn.execute(sql + " ORDER BY src, rel", params).fetchall()
        return [(r["src"], r["rel"]) for r in rows]

    def recent_ids(
        self,
        limit: int,
        *,
        before: float | None = None,
        include_superseded: bool = False,
        include_expired: bool = False,
        now: float = 0.0,
        type: str | None = None,
    ) -> list[str]:
        """Ids ordered newest-first, with an optional ``created_at`` cursor."""
        if not self.healthy:
            return []
        sql = ["SELECT id FROM memories WHERE 1=1"]
        params: list[Any] = []
        if not include_superseded:
            sql.append("AND superseded_by = ''")
        if not include_expired:
            sql.append("AND (expires_at = 0 OR expires_at > ?)")
            params.append(now)
        if type is not None:
            sql.append("AND type = ?")
            params.append(type)
        if before is not None:
            sql.append("AND created_at < ?")
            params.append(before)
        sql.append("ORDER BY created_at DESC, id DESC")
        if limit > 0:
            sql.append("LIMIT ?")
            params.append(limit)
        with self._lock:
            rows = self._conn.execute(" ".join(sql), params).fetchall()
        return [r["id"] for r in rows]

    def expired_ids(self, now: float) -> list[str]:
        """Ids whose TTL has passed — an index range scan, not a full load."""
        if not self.healthy:
            return []
        with self._lock:
            rows = self._conn.execute(
                "SELECT id FROM memories WHERE expires_at > 0 AND expires_at <= ?",
                (now,)).fetchall()
        return [r["id"] for r in rows]

    def tag_counts(self, *, include_superseded: bool = False) -> dict[str, int]:
        """How many memories carry each tag."""
        if not self.healthy:
            return {}
        sql = ("SELECT t.tag AS tag, COUNT(*) AS n FROM tags t "
               "JOIN memories m ON m.id = t.memory_id")
        if not include_superseded:
            sql += " WHERE m.superseded_by = ''"
        sql += " GROUP BY t.tag ORDER BY n DESC, t.tag"
        with self._lock:
            rows = self._conn.execute(sql).fetchall()
        return {r["tag"]: int(r["n"]) for r in rows}

    def type_counts(self, *, include_superseded: bool = False) -> dict[str, int]:
        if not self.healthy:
            return {}
        sql = "SELECT type, COUNT(*) AS n FROM memories"
        if not include_superseded:
            sql += " WHERE superseded_by = ''"
        sql += " GROUP BY type ORDER BY type"
        with self._lock:
            rows = self._conn.execute(sql).fetchall()
        return {r["type"]: int(r["n"]) for r in rows}

    def ids_with_tag(self, tag: str) -> list[str]:
        if not self.healthy:
            return []
        with self._lock:
            rows = self._conn.execute(
                "SELECT memory_id FROM tags WHERE tag = ? ORDER BY memory_id",
                (tag,)).fetchall()
        return [r["memory_id"] for r in rows]


class IndexedCollection:
    """Write-through proxy over a Chroma collection that mirrors into a sidecar.

    Wrapping the collection rather than patching each call site is deliberate:
    the store writes through ``collection.add`` / ``update`` / ``delete`` from
    sixteen places, and an index that misses one is worse than no index at all.
    Everything else is delegated untouched, so the store keeps using the
    collection exactly as before.

    ``row_builder`` turns ``(id, document, metadata)`` into the flat dict the
    index stores; the store supplies it so this module never imports Memory.
    """

    def __init__(self, collection, index: SidecarIndex, row_builder) -> None:
        self._collection = collection
        self._index = index
        self._row = row_builder

    def __getattr__(self, name):
        return getattr(self._collection, name)

    def _mirror(self, ids: Sequence[str], documents=None, metadatas=None) -> None:
        if not self._index.healthy or not ids:
            return
        # Fast path: an add supplies both halves, so no read-back is needed.
        if (documents is not None and metadatas is not None
                and len(documents) == len(ids) == len(metadatas)):
            rows = [self._row(i, d, m)
                    for i, d, m in zip(ids, documents, metadatas)]
        else:
            # A metadata-only update: re-read so the index records the current
            # text, not a stale copy.
            try:
                res = self._collection.get(
                    ids=list(ids), include=["documents", "metadatas"])
            except Exception as exc:  # noqa: BLE001 — never break the write
                self._index.disable(f"read-back failed: {exc}")
                return
            rows = [self._row(i, d, m) for i, d, m in
                    zip(res["ids"], res["documents"], res["metadatas"])]
        self._index.upsert(rows)

    def add(self, *args, **kwargs):
        result = self._collection.add(*args, **kwargs)
        self._mirror(kwargs.get("ids") or [], kwargs.get("documents"),
                     kwargs.get("metadatas"))
        return result

    def upsert(self, *args, **kwargs):
        result = self._collection.upsert(*args, **kwargs)
        self._mirror(kwargs.get("ids") or [], kwargs.get("documents"),
                     kwargs.get("metadatas"))
        return result

    def update(self, *args, **kwargs):
        result = self._collection.update(*args, **kwargs)
        self._mirror(kwargs.get("ids") or [], kwargs.get("documents"),
                     kwargs.get("metadatas"))
        return result

    def delete(self, *args, **kwargs):
        ids = kwargs.get("ids")
        # A delete-by-filter has no id list to mirror; the index cannot know
        # what went, so fall back to scanning until the next reindex.
        if ids is None and (args or kwargs):
            self._index.disable("delete without an explicit id list")
        result = self._collection.delete(*args, **kwargs)
        if ids:
            self._index.delete(list(ids))
        return result


def _fts_query(query: str) -> str:
    """Turn free text into a safe FTS5 MATCH expression.

    Every token is quoted and OR-ed: users type prose, not FTS5 syntax, and an
    unquoted `-` or `"` would otherwise be read as an operator (or a syntax
    error). OR rather than AND because this feeds a *candidate pool* — recall
    matters more than precision at this stage; BM25 does the ranking.
    """
    tokens = [t for t in _tokenize_for_fts(query) if t]
    if not tokens:
        return ""
    return " OR ".join('"' + t.replace('"', '""') + '"' for t in tokens)


def _tokenize_for_fts(text: str) -> Iterator[str]:
    current: list[str] = []
    for ch in text:
        if ch.isalnum() or ch == "_":
            current.append(ch)
        elif current:
            yield "".join(current)
            current = []
    if current:
        yield "".join(current)
