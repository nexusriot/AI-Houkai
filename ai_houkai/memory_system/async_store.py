"""AsyncMemoryStore — async wrapper around MemoryStore.

All blocking ChromaDB operations run in a single-threaded
``ThreadPoolExecutor`` so concurrent async callers are serialised and the
store never sees races.  The executor is owned by the instance and shut down
in ``close()`` / ``aclose()`` / ``async with``.

Usage::

    from ai_houkai.memory_system import AsyncMemoryStore

    async with AsyncMemoryStore(path=".chroma") as store:
        mem = await store.remember("Python favours duck typing.", type="semantic")
        hits = await store.recall("typing", k=3)

Or without async-context-manager::

    store = AsyncMemoryStore(path=".chroma")
    mem = await store.remember("hello")
    await store.aclose()

The underlying ``MemoryStore`` is accessible via ``store.sync`` for
operations that have not yet been wrapped (pass-through via ``run()``)::

    result = await store.run(store.sync.export, "/tmp/backup.ahkai")
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterable, Literal, TypeVar

from .store import (
    Conflict,
    ConflictFn,
    ExpandSpec,
    ExportSummary,
    Graph,
    HybridWeights,
    ImportSummary,
    Memory,
    MemoryStore,
    MemoryType,
    PackResult,
    RebuildSummary,
    RememberItem,
    Reranker,
    TrustLevel,
)
from .curation import TagRename, TrashEntry, Version
from .journal import JournalEntry

_T = TypeVar("_T")


class AsyncMemoryStore:
    """Async wrapper around :class:`MemoryStore`.

    Every public method is a coroutine that offloads the underlying
    synchronous ChromaDB call to a dedicated single-threaded executor,
    keeping the event loop unblocked.
    """

    def __init__(
        self,
        path: str = "./.chroma",
        collection: str = "ai_houkai",
        embedding_model: str = "all-MiniLM-L6-v2",
        *,
        conflict_policy: Literal["ignore", "warn", "supersede", "raise"] = "ignore",
        conflict_threshold: float = 0.80,
        contradiction_fn: ConflictFn | None = None,
        hybrid_weights: HybridWeights | None = None,
        importance_fn: "Callable[[str, str, list[str]], float] | None" = None,
        actor: str = "lib",
        journal_enabled: bool = True,
        journal_path: str | None = None,
        journal_rotate_mb: int = 64,
        journal_keep_days: int = 90,
    ) -> None:
        self.sync = MemoryStore(
            path=path,
            collection=collection,
            embedding_model=embedding_model,
            conflict_policy=conflict_policy,
            conflict_threshold=conflict_threshold,
            contradiction_fn=contradiction_fn,
            hybrid_weights=hybrid_weights,
            importance_fn=importance_fn,
            actor=actor,
            journal_enabled=journal_enabled,
            journal_path=journal_path,
            journal_rotate_mb=journal_rotate_mb,
            journal_keep_days=journal_keep_days,
        )
        # Single thread: ChromaDB (SQLite) is not thread-safe under writes.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="houkai")


    async def aclose(self) -> None:
        """Flush pending work and release resources."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self.sync.client.close)
        self._executor.shutdown(wait=True)

    def close(self) -> None:
        """Synchronous close — use in non-async teardown.

        Drain the executor BEFORE closing the client: any queued job would
        otherwise run against a closed ChromaDB connection. (aclose() gets
        the same ordering for free — its close runs FIFO in the executor.)"""
        self._executor.shutdown(wait=True)
        self.sync.client.close()

    async def __aenter__(self) -> "AsyncMemoryStore":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()


    async def run(self, fn: Callable[..., _T], *args: Any, **kwargs: Any) -> _T:
        """Run any synchronous callable in the store's executor and await it.

        Use this to call sync-only methods not yet wrapped below::

            await store.run(store.sync.export, "/tmp/out.ahkai")
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: fn(*args, **kwargs),
        )


    async def remember(
        self,
        text: str,
        type: MemoryType = "semantic",
        tags: Iterable[str] = (),
        importance: float | None = None,
        source: str | None = None,
        *,
        polarity: int = 0,
        expires_at: float | None = None,
        ttl_seconds: float | None = None,
        on_conflict: Literal["ignore", "warn", "supersede", "raise"] | None = None,
        contradiction_fn: ConflictFn | None = None,
        pinned: bool = False,
        trust: TrustLevel = "trusted",
        idempotent: bool = False,
        valid_from: float | None = None,
        valid_until: float | None = None,
    ) -> Memory:
        return await self.run(
            self.sync.remember,
            text,
            type,
            tags,
            importance,
            source,
            polarity=polarity,
            expires_at=expires_at,
            ttl_seconds=ttl_seconds,
            pinned=pinned,
            trust=trust,
            idempotent=idempotent,
            valid_from=valid_from,
            valid_until=valid_until,
            on_conflict=on_conflict,
            contradiction_fn=contradiction_fn,
        )

    async def remember_many(
        self,
        items: "Iterable[str | RememberItem | Mapping[str, Any]]",
        *,
        batch_size: int = 128,
        on_conflict: Literal["ignore", "warn", "supersede"] | None = None,
        contradiction_fn: ConflictFn | None = None,
        idempotent: bool = False,
    ) -> list[Memory]:
        return await self.run(
            self.sync.remember_many,
            items,
            batch_size=batch_size,
            on_conflict=on_conflict,
            contradiction_fn=contradiction_fn,
            idempotent=idempotent,
        )

    async def forget(self, memory_id: str) -> bool:
        return await self.run(self.sync.forget, memory_id)

    async def edit(
        self,
        memory_id: str,
        *,
        text: str | None = None,
        type: MemoryType | None = None,
        tags: Iterable[str] | None = None,
        importance: float | None = None,
        polarity: int | None = None,
        expires_at: float | None = None,
        source: str | None = MemoryStore._UNSET,
        pinned: bool | None = None,
        trust: TrustLevel | None = None,
        valid_from: float | None = None,
        valid_until: float | None = None,
    ) -> Memory:
        """Update fields of an existing memory in place — see MemoryStore.edit."""
        return await self.run(
            self.sync.edit, memory_id,
            text=text, type=type, tags=tags, importance=importance,
            polarity=polarity, expires_at=expires_at, source=source,
            pinned=pinned, trust=trust,
            valid_from=valid_from, valid_until=valid_until,
        )

    async def nuke(self) -> int:
        """Delete every memory in the current collection. Returns the count deleted."""
        return await self.run(self.sync.nuke)

    async def recall(
        self,
        query: str,
        k: int = 5,
        type: MemoryType | None = None,
        tag: str | None = None,
        min_importance: float | None = None,
        *,
        source: str | None = None,
        since: float | None = None,
        until: float | None = None,
        mode: Literal["semantic", "hybrid"] = "semantic",
        weights: HybridWeights | None = None,
        fusion: Literal["weighted", "rrf"] = "weighted",
        diversity: float | None = None,
        dedup_threshold: float | None = None,
        min_cosine: float | None = None,
        overfetch: int = 4,
        expand: ExpandSpec | None = None,
        include_superseded: bool = False,
        include_expired: bool = False,
        reranker: Reranker | None = None,
        touch: bool = True,
        explain: bool = False,
        min_trust: TrustLevel | None = None,
        lexical_index: Literal["pool", "corpus"] = "pool",
        as_of: float | None = None,
    ) -> list[tuple[Memory, float]] | list[tuple[Memory, float, dict[str, Any]]]:
        return await self.run(
            self.sync.recall,
            query,
            k,
            type,
            tag,
            min_importance,
            source=source,
            since=since,
            until=until,
            mode=mode,
            weights=weights,
            fusion=fusion,
            diversity=diversity,
            dedup_threshold=dedup_threshold,
            min_cosine=min_cosine,
            overfetch=overfetch,
            expand=expand,
            include_superseded=include_superseded,
            include_expired=include_expired,
            reranker=reranker,
            touch=touch,
            min_trust=min_trust,
            lexical_index=lexical_index,
            as_of=as_of,
            explain=explain,
        )

    async def recall_pack(
        self,
        query: str,
        *,
        token_budget: int = 800,
        type: MemoryType | None = None,
        tag: str | None = None,
        min_importance: float | None = None,
        source: str | None = None,
        since: float | None = None,
        until: float | None = None,
        mode: Literal["semantic", "hybrid"] = "hybrid",
        weights: HybridWeights | None = None,
        fusion: Literal["weighted", "rrf"] = "weighted",
        diversity: float | None = None,
        dedup_threshold: float | None = None,
        min_cosine: float | None = None,
        expand: ExpandSpec | None = None,
        include_superseded: bool = False,
        max_items: int = 50,
        touch: bool = True,
        token_counter: Callable[[str], int] | None = None,
        header: str = "## Relevant memory",
        compress: bool = False,
        compress_threshold: float = 0.30,
        compress_min_group: int = 2,
        min_trust: TrustLevel | None = None,
        lexical_index: Literal["pool", "corpus"] = "pool",
        include_pinned: bool = False,
        as_of: float | None = None,
    ) -> PackResult:
        return await self.run(
            self.sync.recall_pack,
            query,
            token_budget=token_budget,
            type=type,
            tag=tag,
            min_importance=min_importance,
            source=source,
            since=since,
            until=until,
            mode=mode,
            weights=weights,
            fusion=fusion,
            diversity=diversity,
            dedup_threshold=dedup_threshold,
            min_cosine=min_cosine,
            expand=expand,
            include_superseded=include_superseded,
            max_items=max_items,
            touch=touch,
            token_counter=token_counter,
            header=header,
            compress=compress,
            compress_threshold=compress_threshold,
            compress_min_group=compress_min_group,
            min_trust=min_trust,
            lexical_index=lexical_index,
            include_pinned=include_pinned,
            as_of=as_of,
        )

    async def auto_context_pack(
        self,
        task: str,
        *,
        token_budget: int = 800,
        max_phrases: int = 3,
        mode: Literal["semantic", "hybrid"] = "hybrid",
        min_cosine: float | None = None,
        header: str = "## Relevant memory",
        token_counter: Callable[[str], int] | None = None,
        touch: bool = True,
        compress: bool = False,
        compress_threshold: float = 0.30,
        compress_min_group: int = 2,
        lexical_index: Literal["pool", "corpus"] = "pool",
        min_trust: TrustLevel | None = None,
    ) -> PackResult:
        return await self.run(
            self.sync.auto_context_pack,
            task,
            token_budget=token_budget,
            max_phrases=max_phrases,
            mode=mode,
            min_cosine=min_cosine,
            header=header,
            token_counter=token_counter,
            touch=touch,
            compress=compress,
            compress_threshold=compress_threshold,
            compress_min_group=compress_min_group,
            lexical_index=lexical_index,
            min_trust=min_trust,
        )

    async def get(self, memory_id: str) -> Memory | None:
        return await self.run(self.sync.get, memory_id)

    async def find_by_content_hash(self, text: str) -> Memory | None:
        return await self.run(self.sync.find_by_content_hash, text)

    async def list_recent(
        self,
        limit: int = 20,
        *,
        include_superseded: bool = False,
        include_expired: bool = False,
        before: float | None = None,
    ) -> list[Memory]:
        """Newest-first memories — see MemoryStore.list_recent.

        ``before`` is the keyset cursor that makes paging a large store viable.
        """
        return await self.run(
            self.sync.list_recent,
            limit,
            include_superseded=include_superseded,
            include_expired=include_expired,
            before=before,
        )

    async def purge_expired(
        self, *, now: float | None = None, dry_run: bool = False
    ) -> list[Memory]:
        return await self.run(self.sync.purge_expired, now=now, dry_run=dry_run)

    async def metrics(self) -> dict[str, Any]:
        return await self.run(self.sync.metrics)

    async def history(
        self, memory_id: str, *, include_archives: bool = True
    ) -> list[JournalEntry]:
        return await self.run(
            self.sync.history, memory_id, include_archives=include_archives)

    async def state_at(
        self, timestamp: float, *, include_archives: bool = True
    ) -> list[Memory]:
        return await self.run(
            self.sync.state_at, timestamp, include_archives=include_archives)

    async def get_at(
        self, memory_id: str, timestamp: float, *, include_archives: bool = True
    ) -> Memory | None:
        return await self.run(
            self.sync.get_at, memory_id, timestamp,
            include_archives=include_archives)

    async def count(self) -> int:
        return await self.run(self.sync.count)

    async def probe_embedding(
        self, text: str = "ai-houkai health probe"
    ) -> dict[str, Any]:
        return await self.run(self.sync.probe_embedding, text)

    async def readiness(self, *, cache_ttl: float = 0.0) -> dict[str, Any]:
        return await self.run(self.sync.readiness, cache_ttl=cache_ttl)


    async def link(self, src_id: str, dst_id: str, rel: str = "related") -> None:
        await self.run(self.sync.link, src_id, dst_id, rel)

    async def unlink(self, src_id: str, dst_id: str, rel: str | None = None) -> int:
        return await self.run(self.sync.unlink, src_id, dst_id, rel)

    async def neighbors(
        self,
        memory_id: str,
        *,
        rel: str | None = None,
        direction: Literal["out", "in", "both"] = "both",
        depth: int = 1,
    ) -> list[tuple[Memory, str]]:
        return await self.run(
            self.sync.neighbors,
            memory_id,
            rel=rel,
            direction=direction,
            depth=depth,
        )

    async def subgraph(
        self,
        memory_ids: Iterable[str],
        *,
        depth: int = 1,
    ) -> Graph:
        return await self.run(self.sync.subgraph, memory_ids, depth=depth)


    async def find_conflicts(
        self,
        memory_id: str | None = None,
        *,
        threshold: float | None = None,
    ) -> list[Conflict]:
        return await self.run(
            self.sync.find_conflicts,
            memory_id,
            threshold=threshold,
        )

    async def supersede(self, old_id: str, new_id: str) -> None:
        await self.run(self.sync.supersede, old_id, new_id)

    async def restore(self, memory_id: str) -> bool:
        return await self.run(self.sync.restore, memory_id)


    async def export(
        self,
        path: str,
        *,
        include_vectors: bool = True,
        include_superseded: bool = False,
        types: Iterable[MemoryType] | None = None,
        tags: Iterable[str] | None = None,
        since: float | None = None,
    ) -> ExportSummary:
        return await self.run(
            self.sync.export,
            path,
            include_vectors=include_vectors,
            include_superseded=include_superseded,
            types=types,
            tags=tags,
            since=since,
        )

    async def vector_index_ok(self) -> bool:
        return await self.run(self.sync.vector_index_ok)

    async def rebuild_vectors(
        self,
        *,
        batch_size: int = 128,
        backup_path: str | None = None,
    ) -> RebuildSummary:
        return await self.run(
            self.sync.rebuild_vectors,
            batch_size=batch_size,
            backup_path=backup_path,
        )

    async def import_(
        self,
        path: str,
        *,
        on_conflict: Literal["skip", "overwrite", "rename", "error"] = "skip",
        regenerate_vectors: bool = False,
        dry_run: bool = False,
    ) -> ImportSummary:
        return await self.run(
            self.sync.import_,
            path,
            on_conflict=on_conflict,
            regenerate_vectors=regenerate_vectors,
            dry_run=dry_run,
        )


    async def undo(self, entry: JournalEntry) -> bool:
        return await self.run(self.sync.undo, entry)

    async def merge(self, target_id: str, other_id: str, *,
                    separator: str = "\n\n") -> Memory:
        """Fold one memory into another — see MemoryStore.merge."""
        return await self.run(self.sync.merge, target_id, other_id,
                              separator=separator)

    async def versions(self, memory_id: str, *,
                       include_archives: bool = True) -> list[Version]:
        return await self.run(self.sync.versions, memory_id,
                              include_archives=include_archives)

    async def list_tags(self, *, include_superseded: bool = False
                        ) -> list[tuple[str, int]]:
        return await self.run(self.sync.list_tags,
                              include_superseded=include_superseded)

    async def rename_tag(self, old: str, new: str) -> TagRename:
        return await self.run(self.sync.rename_tag, old, new)

    async def merge_tags(self, sources: Iterable[str], into: str) -> TagRename:
        return await self.run(self.sync.merge_tags, sources, into)

    async def delete_tag(self, tag: str) -> TagRename:
        return await self.run(self.sync.delete_tag, tag)

    async def find_path(self, from_id: str, to_id: str, *,
                        max_depth: int = 6) -> list[Memory]:
        return await self.run(self.sync.find_path, from_id, to_id,
                              max_depth=max_depth)

    async def trash(self, memory_id: str) -> bool:
        """Soft-delete into the trash — see MemoryStore.trash."""
        return await self.run(self.sync.trash, memory_id)

    async def trash_list(self) -> list[TrashEntry]:
        return await self.run(self.sync.trash_list)

    async def trash_restore(self, memory_id: str) -> Memory | None:
        return await self.run(self.sync.trash_restore, memory_id)

    async def trash_purge(self, memory_id: str | None = None) -> int:
        return await self.run(self.sync.trash_purge, memory_id)

    async def trash_purge_expired(self, ttl_days: float, *,
                                  now: float | None = None) -> int:
        return await self.run(self.sync.trash_purge_expired, ttl_days, now=now)
