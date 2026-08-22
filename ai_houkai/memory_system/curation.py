"""Curation operations: merge, versions, tag management, path-finding, trash.

These grew up in ai-houkai-service, which implemented them by reaching through
the library's private API — ``store._get_by_id``, ``store._get_all_memories``,
``store.collection.update``, ``store._journal``. Every one is a store-level
primitive: re-pointing an incoming link, rewriting a tag across the collection
and walking the link graph all need the store's own write path and journal, and
a downstream consumer cannot do them correctly from outside.

They live in a separate module rather than growing ``store.py`` further, and
attach to :class:`MemoryStore` as mixin methods, so ``store.merge(...)`` works
without a second object to thread around.

**Trash** fills the gap between ``supersede`` (soft, but semantically "replaced
by X") and ``forget`` (irreversible): a recoverable delete. Decay pruning
hard-deletes today, so a mis-tuned ``min_score`` is unrecoverable.
"""

from __future__ import annotations

import gzip
import json
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover — non-POSIX platforms degrade to no lock
    fcntl = None  # type: ignore[assignment]

from .trust import worst_trust

__all__ = [
    "CurationMixin",
    "MergeError",
    "TagRename",
    "TrashEntry",
    "Version",
]

# Where trashed memories are parked, relative to the store's parent directory.
TRASH_FILENAME = "trash.jsonl.gz"

# Process-wide floor of the trash lock (see _trash_lock): covers every store
# in this process on every platform; POSIX adds a cross-process flock on top.
_TRASH_MUTEX = threading.Lock()


class MergeError(ValueError):
    """merge() could not proceed (missing memory, or a self-merge)."""

    def __init__(self, message: str, *, not_found: bool = False) -> None:
        super().__init__(message)
        self.not_found = not_found


@dataclass(frozen=True)
class Version:
    """One past text state of a memory, recovered from the journal."""
    ts: float
    text: str
    tags: list[str]
    importance: float
    source: str | None
    type: str


@dataclass(frozen=True)
class TagRename:
    """Result of a tag-curation operation."""
    changed: int
    tag: str


@dataclass(frozen=True)
class TrashEntry:
    """A soft-deleted memory parked in the trash file.

    ``collection`` scopes the entry: the trash file lives beside the store
    path and is shared by every collection opened on it, so without the tag
    a restore from collection B would materialize collection A's memory into
    B. Entries written before the field existed have ``""`` and stay visible
    from every collection — hiding them retroactively would strand them.
    """
    memory_id: str
    deleted_at: float
    actor: str
    memory: dict[str, Any]
    collection: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id, "deleted_at": self.deleted_at,
            "actor": self.actor, "memory": self.memory,
            "collection": self.collection,
        }


class CurationMixin:
    """Curation methods mixed into :class:`MemoryStore`."""

    def merge(self, target_id: str, other_id: str, *,
              separator: str = "\n\n") -> "Any":
        """Fold ``other`` into ``target`` and return the updated target.

        Combines the text, transfers ``other``'s outgoing links, and — the part
        no caller can do from outside — **re-points every incoming link**
        ``x -> other`` at ``x -> target``. ``forget`` does not clean up incoming
        edges, so without this step merging would silently strand every
        relationship that pointed at the absorbed memory.

        Both sides' changes are journaled as ``edit`` entries, so a merge is
        auditable and each rewritten source is individually undoable.

        Raises :class:`MergeError` on a missing memory or a self-merge.
        """
        target = self.get(target_id)
        if target is None:
            raise MergeError(f"memory not found: {target_id}", not_found=True)
        other = self.get(other_id)
        if other is None:
            raise MergeError(f"memory not found: {other_id}", not_found=True)
        if target.id == other.id:
            raise MergeError("cannot merge a memory with itself")

        before = target.to_dict()
        target.text = target.text + separator + other.text

        # The merged text now contains `other`'s content, so the result can be
        # no more trustworthy than its least trustworthy half. Without this,
        # merging is a laundering path: absorb an untrusted memory into a
        # trusted one and the provenance label silently survives.
        target.trust = worst_trust((target.trust, other.trust))
        # The pin is the union of the two sides: `other` is deleted by the
        # merge, so a pin that does not travel is destroyed outright and the
        # working set silently loses a standing instruction.
        target.pinned = target.pinned or other.pinned

        existing = {(lnk.to, lnk.rel) for lnk in target.links}
        for lnk in other.links:
            # Skip self-loops and edges whose destination is already gone.
            if lnk.to == target.id or self.get(lnk.to) is None:
                continue
            if (lnk.to, lnk.rel) in existing:
                continue
            existing.add((lnk.to, lnk.rel))
            target.links.append(type(lnk)(to=lnk.to, rel=lnk.rel))

        # The dedup hash has to move with the text, exactly as edit() moves it.
        # Left stale, the merged row still answers to its *pre-merge* text: the
        # next idempotent write of that text is absorbed as a duplicate and
        # silently lost, while the text the row now actually holds never
        # matches at all.
        self._rehash(target)

        # Text changed, so the vector must be recomputed — a merged memory that
        # kept the pre-merge embedding would not be findable by its new half.
        self.collection.update(
            ids=[target.id], documents=[target.text],
            metadatas=[target.to_metadata()])
        after = self.get(target.id)
        self._journal("edit", target.id, before=before,
                      after=(after or target).to_dict(),
                      meta={"merged_from": other.id})

        self._repoint_incoming(other.id, target.id)
        self.forget(other.id)
        return self.get(target.id)

    def _repoint_incoming(self, old_dst: str, new_dst: str) -> int:
        """Rewrite every ``x -> old_dst`` edge to ``x -> new_dst``.

        Writes each source's link list directly rather than going through
        unlink+link: that path re-validates the rel vocabulary (rejecting a
        legacy custom rel outright) and costs two journal entries per edge.
        A pre-existing ``new_dst -> old_dst`` edge is dropped rather than
        turned into a self-loop.
        """
        rewritten = 0
        for src in self._link_sources(old_dst):
            if src.id == old_dst or not any(l.to == old_dst for l in src.links):
                continue
            src_before = src.to_dict()
            seen = {(l.to, l.rel) for l in src.links if l.to != old_dst}
            new_links = [l for l in src.links if l.to != old_dst]
            if src.id != new_dst:
                for l in src.links:
                    if l.to != old_dst:
                        continue
                    if (new_dst, l.rel) not in seen:
                        seen.add((new_dst, l.rel))
                        new_links.append(type(l)(to=new_dst, rel=l.rel))
            src.links = new_links
            self.collection.update(ids=[src.id], metadatas=[src.to_metadata()])
            self._journal("edit", src.id, before=src_before, after=src.to_dict())
            rewritten += 1
        return rewritten

    def _link_sources(self, dst: str) -> list[Any]:
        """Memories with an edge pointing at *dst* (index-backed when enabled)."""
        return self._incoming_candidates(dst, None)

    def versions(self, memory_id: str, *,
                 include_archives: bool = True) -> list[Version]:
        """Past text states of a memory, oldest first.

        Each entry is the state *before* an edit; the current live state is
        excluded (fetch it with :meth:`get`). Reads rotated journal segments
        too, so version history survives a rollover.
        """
        out: list[Version] = []
        for e in self.journal.read(include_archives=include_archives):
            if e.op != "edit" or e.id != memory_id or not e.before:
                continue
            out.append(Version(
                ts=e.ts,
                text=e.before.get("text", ""),
                tags=list(e.before.get("tags") or []),
                importance=float(e.before.get("importance", 0.5)),
                source=e.before.get("source"),
                type=e.before.get("type", "semantic"),
            ))
        return out

    def list_tags(self, *, include_superseded: bool = False
                  ) -> list[tuple[str, int]]:
        """Every tag with its usage count, most-used first then alphabetical.

        Requires a full read: tags are stored comma-joined in a single metadata
        field, so Chroma cannot group by them. Tag cardinality is low and this
        is a curation command rather than a hot path, so one scan is acceptable.
        """
        counts: dict[str, int] = {}
        for m in self.list_recent(limit=10**9,
                                  include_superseded=include_superseded,
                                  include_expired=True):
            for t in m.tags:
                counts[t] = counts.get(t, 0) + 1
        return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))

    def _rewrite_tags(self, fn: Callable[[list[str]], list[str] | None]) -> int:
        """Apply *fn* to every memory's tag list, persisting + journaling changes.

        Includes superseded and expired memories: tag curation that skipped them
        would leave the old spelling alive in rows a later `restore` brings back.
        Returns the number of memories changed.
        """
        changed = 0
        for m in self.list_recent(limit=10**9, include_superseded=True,
                                  include_expired=True):
            new_tags = fn(list(m.tags))
            if new_tags is None or new_tags == m.tags:
                continue
            before = m.to_dict()
            m.tags = new_tags
            self.collection.update(ids=[m.id], metadatas=[m.to_metadata()])
            self._journal("edit", m.id, before=before, after=m.to_dict())
            changed += 1
        return changed

    def rename_tag(self, old: str, new: str) -> TagRename:
        """Rename a tag across the collection, de-duplicating on collision."""
        _validate_tag(new)

        def fn(tags: list[str]) -> list[str] | None:
            if old not in tags:
                return None
            out: list[str] = []
            for t in tags:
                t2 = new if t == old else t
                if t2 not in out:
                    out.append(t2)
            return out

        with self.as_actor("curation"):
            return TagRename(changed=self._rewrite_tags(fn), tag=new)

    def merge_tags(self, sources: Iterable[str], into: str) -> TagRename:
        """Fold several tags into one across the collection."""
        _validate_tag(into)
        src = set(sources)

        def fn(tags: list[str]) -> list[str] | None:
            if not src.intersection(tags):
                return None
            out: list[str] = []
            for t in tags:
                t2 = into if t in src else t
                if t2 not in out:
                    out.append(t2)
            return out

        with self.as_actor("curation"):
            return TagRename(changed=self._rewrite_tags(fn), tag=into)

    def delete_tag(self, tag: str) -> TagRename:
        """Strip a tag from every memory that carries it."""

        def fn(tags: list[str]) -> list[str] | None:
            if tag not in tags:
                return None
            return [t for t in tags if t != tag]

        with self.as_actor("curation"):
            return TagRename(changed=self._rewrite_tags(fn), tag=tag)

    def find_path(self, from_id: str, to_id: str, *,
                  max_depth: int = 6) -> list[tuple[str, str]]:
        """Shortest *undirected* link path between two memories.

        Returns ``[(memory_id, rel_used_to_reach_it), ...]`` starting at
        ``from_id`` (whose rel is ``""``), or ``[]`` when no path exists within
        ``max_depth``. Undirected because "how are these two related?" does not
        care which way the author happened to draw the arrow.

        Adjacency requires reading every memory, because Chroma cannot query
        the link metadata; it is
        built from a single full scan — one scan, not one per hop.
        """
        if self.get(from_id) is None or self.get(to_id) is None:
            return []
        if from_id == to_id:
            return [(from_id, "")]

        adjacency = self._undirected_adjacency()
        queue: deque[list[tuple[str, str]]] = deque([[(from_id, "")]])
        visited = {from_id}
        while queue:
            path = queue.popleft()
            if len(path) > max_depth:
                break
            for nxt, rel in adjacency.get(path[-1][0], []):
                if nxt in visited:
                    continue
                extended = path + [(nxt, rel)]
                if nxt == to_id:
                    return extended
                visited.add(nxt)
                queue.append(extended)
        return []

    def _undirected_adjacency(self) -> dict[str, list[tuple[str, str]]]:
        """Both-directions adjacency built from one pass over the link graph."""
        adjacency: dict[str, list[tuple[str, str]]] = {}
        for m in self.list_recent(limit=10**9, include_superseded=True,
                                  include_expired=True):
            for lnk in m.links:
                adjacency.setdefault(m.id, []).append((lnk.to, lnk.rel))
                adjacency.setdefault(lnk.to, []).append((m.id, lnk.rel))
        for edges in adjacency.values():
            edges.sort()
        return adjacency

    @property
    def trash_path(self) -> Path:
        """Where soft-deleted memories are parked."""
        return Path(self.path).parent / TRASH_FILENAME

    @contextmanager
    def _trash_lock(self) -> Iterator[None]:
        """Serialize trash mutations.

        The trash is a shared read-modify-write file: two stores (one per
        collection on the same path — a supported layout, or two processes)
        mutating it concurrently would lose whichever rewrite lands first.
        A process-wide mutex is the floor on every platform; on POSIX an
        exclusive flock on a lock file beside the trash extends the exclusion
        across processes (its errors propagate — proceeding unlocked risks
        exactly the corruption the lock exists to prevent). Without fcntl,
        cross-PROCESS sharing of one trash file is unsynchronized — a
        documented limitation; in-process multi-store use is covered.
        """
        with _TRASH_MUTEX:
            if fcntl is None:
                yield
                return
            lock_path = self.trash_path.with_name(
                self.trash_path.name + ".lock")
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            with open(lock_path, "a") as lf:
                fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(lf.fileno(), fcntl.LOCK_UN)

    def trash(self, memory_id: str) -> bool:
        """Soft-delete: park the memory in the trash, then remove it.

        The missing middle between ``supersede`` (which asserts "replaced by
        X") and ``forget`` (irreversible). ``trash_restore`` brings it back with
        its id, tags, links and timestamps intact — but not its vector, which is
        recomputed from the text on restore.
        """
        mem = self.get(memory_id)
        if mem is None:
            return False
        entry = TrashEntry(memory_id=mem.id, deleted_at=time.time(),
                           actor=self._actor, memory=mem.to_dict(),
                           collection=self.collection_name)
        self.trash_path.parent.mkdir(parents=True, exist_ok=True)
        with self._trash_lock(), \
                gzip.open(self.trash_path, "at", encoding="utf-8") as f:
            f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
        with self.as_actor("trash"):
            self.forget(mem.id)
        return True

    def _trash_visible(self, entry: TrashEntry) -> bool:
        """Whether a trash entry belongs to this store's collection.

        Legacy entries carry no collection tag and stay visible everywhere —
        see :class:`TrashEntry`.
        """
        return entry.collection in ("", self.collection_name)

    def trash_list(self) -> list[TrashEntry]:
        """This collection's trash, oldest first."""
        return [e for e in self._read_trash() if self._trash_visible(e)]

    def trash_restore(self, memory_id: str) -> "Any":
        """Bring a trashed memory back, or None when it cannot be restored.

        The row is re-added with its original id and metadata and re-embedded
        from its text; the restored trash entry is then dropped. Returns None
        when the id is not in this collection's trash — or when it is live
        again (an export/import can resurrect a trashed id): restoring over a
        live row would be silently ignored by the collection while the trash
        entry was destroyed, so the entry is kept instead. With several
        entries for one id (trash → import → trash), the newest snapshot is
        restored and older ones stay recoverable.
        """
        # The read-filter-rewrite below is atomic against other stores sharing
        # the trash file (one per collection, or another process) — unlocked,
        # two concurrent mutations lose whichever rewrite lands first.
        with self._trash_lock():
            return self._trash_restore_locked(memory_id)

    def _trash_restore_locked(self, memory_id: str) -> "Any":
        entries = self._read_trash()
        mine = [e for e in entries
                if self._trash_visible(e) and e.memory_id == memory_id]
        if not mine or self.get(memory_id) is not None:
            return None
        restored = max(mine, key=lambda e: e.deleted_at)
        entries.remove(restored)

        # Rehydrated through the store rather than by importing Memory here:
        # store.py imports this module for the mixin, so a module-level import
        # back the other way would be a cycle.
        mem = self._memory_from_dict(restored.memory)
        self.collection.add(ids=[mem.id], documents=[mem.text],
                            metadatas=[mem.to_metadata()])
        with self.as_actor("trash"):
            self._journal("restore", mem.id, after=mem.to_dict(),
                          meta={"from": "trash"})
        self._write_trash(entries)
        return mem

    def trash_purge(self, memory_id: str | None = None) -> int:
        """Permanently drop one trashed memory, or empty this collection's trash.

        Irreversible — this is the only operation in the trash path that loses
        data, which is why it is separate from ``trash``. Other collections'
        entries in the shared trash file are untouched.
        """
        with self._trash_lock():
            entries = self._read_trash()
            if memory_id is None:
                keep = [e for e in entries if not self._trash_visible(e)]
            else:
                keep = [e for e in entries
                        if not (self._trash_visible(e)
                                and e.memory_id == memory_id)]
            purged = len(entries) - len(keep)
            if purged:
                self._write_trash(keep)
        return purged

    def trash_purge_expired(self, ttl_days: float, *,
                            now: float | None = None) -> int:
        """Drop trashed memories deleted more than *ttl_days* ago.

        Retention, not reclamation: without it a recoverable delete is really a
        permanent archive, and the trash file grows without bound. Meant to be
        driven from a scheduler so the policy holds whether or not anyone runs a
        maintenance pass by hand.

        ``ttl_days <= 0`` is a no-op rather than "purge everything" — a
        misconfigured or unset retention must never be read as "delete it all".
        """
        if ttl_days <= 0:
            return 0
        cutoff = (now if now is not None else time.time()) - ttl_days * 86_400
        with self._trash_lock():
            entries = self._read_trash()
            keep = [e for e in entries
                    if not (self._trash_visible(e) and e.deleted_at < cutoff)]
            purged = len(entries) - len(keep)
            if purged:
                self._write_trash(keep)
        return purged

    def _read_trash(self) -> list[TrashEntry]:
        if not self.trash_path.exists():
            return []
        out: list[TrashEntry] = []
        with gzip.open(self.trash_path, "rt", encoding="utf-8") as f:
            while True:
                try:
                    line = f.readline()
                except (EOFError, OSError):
                    # A gzip member truncated by a crash mid-append raises
                    # here, before any JSON is seen — it must not make the
                    # whole trash unreadable. Keep what parsed so far.
                    break
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    # A truncated tail (crash mid-write) must not make the
                    # whole trash unreadable.
                    continue
                out.append(TrashEntry(
                    memory_id=d.get("memory_id", ""),
                    deleted_at=float(d.get("deleted_at", 0.0)),
                    actor=d.get("actor", ""),
                    memory=d.get("memory") or {},
                    collection=str(d.get("collection") or ""),
                ))
        return out

    def _write_trash(self, entries: list[TrashEntry]) -> None:
        tmp = self.trash_path.with_suffix(".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e.to_dict(), ensure_ascii=False) + "\n")
        tmp.replace(self.trash_path)


def _validate_tag(tag: str) -> None:
    """Tags are stored comma-joined, so a comma would split one tag into two."""
    if not tag:
        raise ValueError("tag must not be empty")
    if "," in tag:
        raise ValueError(f"tags must not contain commas — got {tag!r}")
