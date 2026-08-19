"""MCP server exposing the AI-Houkai memory store.

Run with:
    ai-houkai-mcp
    # or: python -m ai_houkai.mcp_server.server

Tools exposed (41):
    remember(text, type?, tags?, importance?, source?, on_conflict?, polarity?, expires_at?, ttl_seconds?, pinned?, trust?, idempotent?)
    remember_many(items, batch_size?, on_conflict?, idempotent?)
    get(memory_id)
    edit(memory_id, text?, type?, tags?, importance?, polarity?, source?, clear_source?, expires_at?, pinned?, trust?)
    recall(query, k?, type?, tag?, min_importance?, source?, since?, until?, mode?, overfetch?, include_superseded?, include_expired?, explain?, fusion?, diversity?, dedup_threshold?, min_cosine?, graph?, touch?, expand_*?)
    recall_pack(query, token_budget?, type?, tag?, min_importance?, source?, since?, until?, mode?, max_items?, compress?, compress_threshold?, compress_min_group?, fusion?, diversity?, dedup_threshold?, min_cosine?, graph?, touch?, header?, expand_*?)
    auto_context(task, token_budget?, max_phrases?, mode?, min_cosine?, touch?, header?, compress?, compress_threshold?, compress_min_group?)
    forget(memory_id)
    purge_expired(dry_run?)
    list_recent(limit?, include_superseded?, include_expired?)
    stats()
    metrics()
    history(memory_id)
    state_at(ts)
    get_at(memory_id, ts)
    link(src_id, dst_id, rel?)
    unlink(src_id, dst_id, rel?)
    neighbors(memory_id, rel?, direction?, depth?)
    subgraph(memory_ids, depth?)
    find_conflicts(memory_id?, threshold?)
    supersede(old_id, new_id)
    restore(memory_id)
    undo(ts?, memory_id?)
    nuke(confirm)
    eval_recall(cases, k?, mode?, fusion?, graph?, diversity?, dedup_threshold?, min_cosine?)
    merge(target_id, other_id, separator?)
    versions(memory_id)
    list_tags(include_superseded?)
    rename_tag(old, new)
    merge_tags(sources, into)
    delete_tag(tag)
    find_path(from_id, to_id, max_depth?)
    trash(memory_id)
    trash_list()
    trash_restore(memory_id)
    trash_purge(memory_id?, older_than_days?)
    ready()
    maintenance_tick(reflect_apply?)
    journal_tail(n?, op?, since_seconds?)
    export(path, include_vectors?, include_superseded?, type?, tag?, since?)
    import(path, on_conflict?, regenerate_vectors?, dry_run?)
"""

from __future__ import annotations

import os
import time
from typing import Any

from mcp.server.fastmcp import FastMCP

from ai_houkai.eval import EvalCase, evaluate
from ai_houkai.maintenance.scheduler import MaintenanceScheduler
from ai_houkai.cli.config import load_maintenance
from ai_houkai.memory_system import MemoryStore
from ai_houkai.memory_system.importance import score_importance
from ai_houkai.memory_system.curation import MergeError
from ai_houkai.memory_system.store import (
    ConflictError,
    ExpandSpec,
    HybridWeights,
    ImportConflictError,
    extract_key_phrases,
)
from ai_houkai.memory_system.summarizers import build_summarizer
from ai_houkai.timeparse import parse_timestamp

mcp = FastMCP("AI-Houkai")

_store: MemoryStore | None = None


def get_store() -> MemoryStore:
    """Return the process-wide store, creating it on first use.

    Importing this module is side-effect-free: no ChromaDB directory is
    materialised and no embedding model is loaded until a tool actually
    runs. The installers rely on this to inspect the tool surface, and env
    configuration (AI_HOUKAI_PATH / AI_HOUKAI_COLLECTION /
    AI_HOUKAI_AUTO_IMPORTANCE) is read here, at first use, not at import.
    """
    global _store
    if _store is None:
        auto_importance = os.environ.get(
            "AI_HOUKAI_AUTO_IMPORTANCE", "").lower() in ("1", "true", "yes", "on")
        _store = MemoryStore(
            path=os.path.expanduser(os.environ.get("AI_HOUKAI_PATH", "./.chroma")),
            collection=os.environ.get("AI_HOUKAI_COLLECTION", "ai_houkai"),
            actor="mcp",
            importance_fn=score_importance if auto_importance else None,
        )
    return _store


@mcp.tool()
def remember(
    text: str,
    type: str = "semantic",
    tags: list[str] | None = None,
    importance: float | None = None,
    source: str | None = None,
    on_conflict: str | None = None,
    polarity: int = 0,
    expires_at: float | None = None,
    ttl_seconds: float | None = None,
    pinned: bool = False,
    trust: str = "trusted",
    idempotent: bool = False,
    valid_from: float | None = None,
    valid_until: float | None = None,
) -> dict[str, Any]:
    """Store a new memory.
    type: episodic | semantic | procedural | feedback.
    importance: 0..1; omit for the default (0.5, or a heuristic score when
    the server runs with AI_HOUKAI_AUTO_IMPORTANCE=1).
    on_conflict: ignore | warn | supersede | raise (default: store policy).
    polarity: -1 (negative) | 0 (neutral) | +1 (positive).
    expires_at / ttl_seconds: optional TTL — an absolute epoch (expires_at) or
    a relative lifetime in seconds (ttl_seconds). Expired memories are hidden
    from recall and reclaimable by purge_expired. Pass at most one.
    pinned: mark a standing instruction — always offered to the packer
    (recall_pack include_pinned), never pruned by decay.
    trust: trusted (default) | reported | untrusted — how much the memory's
    ORIGIN is trusted. Use "untrusted" for anything read from content the agent
    did not author (a web page, a document, another agent's output): it stays
    recallable but is filterable and is marked in packed context.
    idempotent: true makes a repeated assertion a no-op — if a live memory has
    the same normalised text, its access count is bumped and it is returned
    unchanged instead of writing a duplicate.
    """

    policy = on_conflict  # type: ignore[assignment]
    # Asked before the write, because remember() returns the existing row on a
    # dedupe hit and there is otherwise no way to tell it from a fresh one — an
    # agent re-asserting known facts was told each one was newly stored.
    deduped = (idempotent
               and get_store().find_by_content_hash(text) is not None)
    try:
        mem = get_store().remember(
            text=text,
            type=type,           # type: ignore[arg-type]
            tags=tags or [],
            importance=importance,
            source=source,
            polarity=polarity,
            expires_at=expires_at,
            ttl_seconds=ttl_seconds,
            on_conflict=policy,  # type: ignore[arg-type]
            pinned=pinned,
            trust=trust,          # type: ignore[arg-type]
            idempotent=idempotent,
            valid_from=valid_from,
            valid_until=valid_until,
        )
    except ValueError as e:
        return {"stored": False, "error": str(e)}
    except ConflictError as e:
        return {
            "stored": False,
            "conflicts": [
                {"kind": c.kind, "similarity": c.similarity,
                 "other_id": c.b.id, "other_text": c.b.text[:100]}
                for c in e.conflicts
            ],
        }
    return {"id": mem.id, "stored": not deduped, "importance": mem.importance,
            "expires_at": mem.expires_at or None, "pinned": mem.pinned,
            "trust": mem.trust}


@mcp.tool()
def remember_many(
    items: list[dict[str, Any]],
    batch_size: int = 128,
    on_conflict: str | None = None,
    idempotent: bool = False,
) -> dict[str, Any]:
    """Store many memories in one batched, embedding-efficient call.
    items: a list of objects, each with a required "text" plus optional "type",
    "tags", "importance", "source", "polarity", "expires_at", "ttl_seconds"
    (the same fields as remember). Embedding is batched, so N items cost
    ceil(N / batch_size) encode passes instead of N.
    on_conflict: ignore | warn | supersede (default: store policy). "raise" is
    not supported in bulk — use remember per item.
    idempotent: true collapses re-assertions by normalised text, both against
    rows already stored and within this call, so a batch replayed every session
    does not accumulate near-duplicates. Every input still maps to an entry in
    `ids`, with duplicates sharing one id.
    Returns {stored, ids}, where `stored` counts the rows actually created — 0
    for a fully replayed batch.
    """
    started = time.time()
    try:
        mems = get_store().remember_many(
            items,
            batch_size=batch_size,
            on_conflict=on_conflict,  # type: ignore[arg-type]
            idempotent=idempotent,
        )
    except (ValueError, TypeError) as e:
        return {"stored": 0, "error": str(e)}
    # Rows created, not items submitted: an idempotent replay returns the
    # pre-existing rows, and reporting len(mems) told the agent it had written N
    # facts when it had written none. Distinct ids also collapse intra-batch
    # duplicates, which map to one row.
    created = {m.id for m in mems if m.created_at >= started}
    return {"stored": len(created), "ids": [m.id for m in mems]}


def _weights(graph: float | None) -> "HybridWeights | None":
    """Build HybridWeights from a ``graph`` param, or None for the store default.

    Only the graph-proximity weight is tunable per call: the core weights are a
    server-configuration concern, and the dataclass keeps its defaults so
    ``graph`` is a pure add-on (matching the HTTP surface).
    """
    return None if graph is None else HybridWeights(graph=graph)


def _expand(
    expand_rels: list[str] | None,
    expand_depth: int | None,
    expand_cap: int | None,
    expand_score: float | None,
    expand_decay: float | None,
    expand_rerank: bool | None,
) -> "ExpandSpec | None":
    """Build an ExpandSpec from flat ``expand_*`` params, or None for no
    expansion. MCP tool schemas are flat, so the HTTP body's nested ``expand``
    object is spelled here as one param per field; unset fields fall back to
    ExpandSpec defaults. Expansion is off unless at least one is supplied.
    """
    parts = (expand_rels, expand_depth, expand_cap, expand_score,
             expand_decay, expand_rerank)
    if all(p is None for p in parts):
        return None
    kwargs: dict[str, Any] = {}
    if expand_rels:
        kwargs["rels"] = tuple(expand_rels)
    if expand_depth is not None:
        kwargs["depth"] = expand_depth
    if expand_cap is not None:
        kwargs["cap"] = expand_cap
    if expand_score is not None:
        kwargs["score"] = expand_score
    if expand_decay is not None:
        kwargs["decay"] = expand_decay
    if expand_rerank is not None:
        kwargs["rerank"] = expand_rerank
    return ExpandSpec(**kwargs)


@mcp.tool()
def recall(
    query: str,
    k: int = 5,
    type: str | None = None,
    tag: str | None = None,
    min_importance: float | None = None,
    source: str | None = None,
    since: str | None = None,
    until: str | None = None,
    mode: str = "semantic",
    overfetch: int = 4,
    include_superseded: bool = False,
    include_expired: bool = False,
    explain: bool = False,
    fusion: str = "weighted",
    diversity: float | None = None,
    dedup_threshold: float | None = None,
    min_cosine: float | None = None,
    graph: float | None = None,
    touch: bool = True,
    expand_rels: list[str] | None = None,
    expand_depth: int | None = None,
    expand_cap: int | None = None,
    expand_score: float | None = None,
    expand_decay: float | None = None,
    expand_rerank: bool | None = None,
    lexical_index: str = "pool",
    min_trust: str | None = None,
    as_of: float | None = None,
) -> list[dict[str, Any]]:
    """Semantic (or hybrid) search across stored memories.
    mode: "semantic" (default) | "hybrid" (cosine + BM25 + recency + importance).
    source: keep only memories with this exact provenance string.
    since/until: bound created_at — epoch seconds, an ISO-8601 date/datetime,
    or a relative span like "7d" / "24h" (since="7d" → last 7 days).
    include_expired: also return memories whose TTL has passed (hidden by default).
    explain: attach a per-signal score breakdown to each hit under "explain".

    Ranking controls:
    fusion: "weighted" (default) | "rrf" — Reciprocal Rank Fusion of the hybrid
    signals (scale-free; hybrid mode only).
    diversity: MMR λ in [0,1] — higher favours relevance, lower novelty.
    dedup_threshold: drop a candidate whose cosine to an already-selected result
    exceeds this [0,1].
    min_cosine: absolute cosine floor in [-1,1] — return nothing rather than
    weak hits.
    graph: graph-proximity weight (hybrid mode only) — lifts candidates linked
    to other strong hits. 0.0/omitted disables the channel.
    touch: false = read-only recall (no access-count / last_accessed bump).

    Graph-walk expansion (all optional; supply any to enable):
    expand_rels / expand_depth / expand_cap / expand_score / expand_decay, and
    expand_rerank=true to merge expanded neighbours into the pool BEFORE
    dedup/MMR/top-k instead of appending them after.

    lexical_index: "pool" (default) scores BM25 only over the vector over-fetch
    pool; "corpus" also pulls in memories whose text contains the query's
    index into the pool, so an exact-token match with a weak embedding can be
    found at all (hybrid mode; requires an enabled index).

    min_trust: keep only memories whose provenance is at least this trusted —
    "trusted" admits only trusted, "reported" also admits reported,
    "untrusted" admits everything (the default when omitted).
    """
    hits = get_store().recall(
        query=query,
        k=k,
        type=type,            # type: ignore[arg-type]
        tag=tag,
        min_importance=min_importance,
        source=source,
        since=parse_timestamp(since),
        until=parse_timestamp(until),
        mode=mode,            # type: ignore[arg-type]
        overfetch=overfetch,
        include_superseded=include_superseded,
        include_expired=include_expired,
        explain=explain,
        fusion=fusion,        # type: ignore[arg-type]
        diversity=diversity,
        dedup_threshold=dedup_threshold,
        min_cosine=min_cosine,
        weights=_weights(graph),
        touch=touch,
        expand=_expand(expand_rels, expand_depth, expand_cap,
                       expand_score, expand_decay, expand_rerank),
        lexical_index=lexical_index,  # type: ignore[arg-type]
        min_trust=min_trust,          # type: ignore[arg-type]
        as_of=as_of,
    )

    def _hit(m: Any, score: float) -> dict[str, Any]:
        return {
            "id": m.id,
            "text": m.text,
            "type": m.type,
            "tags": m.tags,
            "importance": m.importance,
            "score": round(score, 4),
            "created_at": m.created_at,
            "superseded_by": m.superseded_by or None,
            "expires_at": m.expires_at or None,
        }

    if explain:
        return [{**_hit(m, score), "explain": expl} for m, score, expl in hits]
    return [_hit(m, score) for m, score in hits]


@mcp.tool()
def recall_pack(
    query: str,
    token_budget: int = 800,
    type: str | None = None,
    tag: str | None = None,
    min_importance: float | None = None,
    source: str | None = None,
    since: str | None = None,
    until: str | None = None,
    mode: str = "hybrid",
    max_items: int = 50,
    include_superseded: bool = False,
    compress: bool = False,
    compress_threshold: float = 0.30,
    compress_min_group: int = 2,
    fusion: str = "weighted",
    diversity: float | None = None,
    dedup_threshold: float | None = None,
    min_cosine: float | None = None,
    graph: float | None = None,
    touch: bool = True,
    header: str = "## Relevant memory",
    expand_rels: list[str] | None = None,
    expand_depth: int | None = None,
    expand_cap: int | None = None,
    expand_score: float | None = None,
    expand_decay: float | None = None,
    expand_rerank: bool | None = None,
    lexical_index: str = "pool",
    min_trust: str | None = None,
    include_pinned: bool = False,
    as_of: float | None = None,
) -> dict[str, Any]:
    """Assemble the most relevant memories into a token-budgeted context block.

    Ranks with hybrid scoring (cosine + BM25 + recency + importance + polarity)
    by default, then greedily packs results until token_budget is reached.
    Returns a ready-to-inject `text` block plus the packed items.
    token_budget is a soft ceiling (~4 chars/token) covering memory lines only.

    compress=True: when candidates are dropped for exceeding the budget, similar
    ones are clustered by token-Jaccard and folded into a single summary line
    (marked "compressed") that may fit in the remaining space.
    compress_threshold: Jaccard similarity threshold for grouping (default 0.30).
    compress_min_group: minimum cluster size to produce a compressed line (default 2).

    header: heading prepended to the block (not counted against token_budget);
    pass "" for a bare list.

    include_pinned: prepend every pinned memory ahead of the ranked hits, so a
    standing instruction is present whether or not it matches the query. They
    compete for the same budget. min_trust filters by provenance.

    Ranking controls (fusion / diversity / dedup_threshold / min_cosine / graph /
    touch / expand_*) behave exactly as on the `recall` tool — diversity in
    particular stops the budget being spent on near-duplicates, and min_cosine
    keeps an off-topic query from padding context with weak hits.
    """
    pack = get_store().recall_pack(
        query=query,
        token_budget=token_budget,
        type=type,             # type: ignore[arg-type]
        tag=tag,
        min_importance=min_importance,
        source=source,
        since=parse_timestamp(since),
        until=parse_timestamp(until),
        mode=mode,             # type: ignore[arg-type]
        max_items=max_items,
        include_superseded=include_superseded,
        compress=compress,
        compress_threshold=compress_threshold,
        compress_min_group=compress_min_group,
        fusion=fusion,         # type: ignore[arg-type]
        diversity=diversity,
        dedup_threshold=dedup_threshold,
        min_cosine=min_cosine,
        weights=_weights(graph),
        touch=touch,
        header=header,
        expand=_expand(expand_rels, expand_depth, expand_cap,
                       expand_score, expand_decay, expand_rerank),
        lexical_index=lexical_index,  # type: ignore[arg-type]
        min_trust=min_trust,          # type: ignore[arg-type]
        as_of=as_of,
        include_pinned=include_pinned,
    )
    return {
        "text": pack.text,
        "used_tokens": pack.used_tokens,
        "budget": pack.budget,
        "truncated": pack.truncated,
        "items": [
            {
                "id": p.memory.id,
                "text": p.memory.text,
                "type": p.memory.type,
                "tags": p.memory.tags,
                "importance": p.memory.importance,
                "score": round(p.score, 4),
                "tokens": p.tokens,
            }
            for p in pack.items
        ],
        "compressed_groups": [
            {
                "ids": [m.id for m in cg.memories],
                "text": cg.text,
                "tokens": cg.tokens,
                "count": len(cg.memories),
            }
            for cg in pack.compressed_groups
        ],
    }


@mcp.tool()
def auto_context(
    task: str,
    token_budget: int = 800,
    max_phrases: int = 3,
    mode: str = "hybrid",
    min_cosine: float | None = None,
    touch: bool = True,
    header: str = "## Relevant memory",
    compress: bool = False,
    compress_threshold: float = 0.30,
    compress_min_group: int = 2,
    lexical_index: str = "pool",
    min_trust: str | None = None,
) -> dict[str, Any]:
    """Build a ready-to-inject context block by fanning out over multiple recall angles.

    Extracts up to max_phrases key bigram/keyword phrases from the task description,
    recalls memories for each angle independently, deduplicates (keeping the highest
    score per memory), then packs greedily within token_budget.

    More thorough than a single recall_pack call for tasks with compound concepts
    (e.g. "deploy the API to production" → also searches "deploy api", "api production").
    Returns the same structure as recall_pack plus the queries that were used.

    min_cosine applies an absolute relevance floor to every fan-out query, so an
    off-topic task injects nothing rather than padding context with weak hits.
    touch=false makes the whole fan-out read-only. compress* behave as on
    recall_pack. fusion is not offered here: RRF scores are rank-relative to each
    query's own pool, so they cannot be compared across the fan-out.

    min_trust and lexical_index apply to every fan-out query, as on recall_pack.
    The trust floor is worth setting here in particular: this is the tool you call
    without choosing a query, so it is the one most likely to pull scraped
    material into a context block unattended.
    """
    queries = [task] + extract_key_phrases(task, max_phrases)
    pack = get_store().auto_context_pack(
        task=task,
        token_budget=token_budget,
        max_phrases=max_phrases,
        mode=mode,             # type: ignore[arg-type]
        min_cosine=min_cosine,
        touch=touch,
        header=header,
        compress=compress,
        compress_threshold=compress_threshold,
        compress_min_group=compress_min_group,
        lexical_index=lexical_index,   # type: ignore[arg-type]
        min_trust=min_trust,           # type: ignore[arg-type]
    )
    return {
        "text": pack.text,
        "queries": queries,
        "used_tokens": pack.used_tokens,
        "budget": pack.budget,
        "truncated": pack.truncated,
        "items": [
            {
                "id": p.memory.id,
                "text": p.memory.text,
                "type": p.memory.type,
                "tags": p.memory.tags,
                "importance": p.memory.importance,
                "score": round(p.score, 4),
                "tokens": p.tokens,
            }
            for p in pack.items
        ],
        "compressed_groups": [
            {
                "ids": [m.id for m in cg.memories],
                "text": cg.text,
                "tokens": cg.tokens,
                "count": len(cg.memories),
            }
            for cg in pack.compressed_groups
        ],
    }


def _mem_dict(mem: Any) -> dict[str, Any]:
    """Full memory record shared by the get / restore / subgraph tools."""
    return {
        "id": mem.id,
        "text": mem.text,
        "type": mem.type,
        "tags": mem.tags,
        "importance": mem.importance,
        "source": mem.source,
        "polarity": mem.polarity,
        "created_at": mem.created_at,
        "last_accessed": mem.last_accessed,
        "access_count": mem.access_count,
        "superseded_by": mem.superseded_by or None,
        "superseded_at": mem.superseded_at or None,
        "expires_at": mem.expires_at or None,
        "links": [{"to": lnk.to, "rel": lnk.rel} for lnk in mem.links],
    }


@mcp.tool()
def get(memory_id: str) -> dict[str, Any]:
    """Fetch one memory by its exact id.

    A plain read: no access-count bump and no filtering — a superseded or
    expired memory is still returned, with its state visible in the response.
    Use `recall` for ranked search and `get_at` for a past point in time.
    """
    mem = get_store().get(memory_id)
    if mem is None:
        return {"found": False, "id": memory_id}
    return {"found": True, **_mem_dict(mem)}


@mcp.tool()
def forget(memory_id: str) -> dict[str, Any]:
    """Permanently delete a memory by id."""
    return {"deleted": get_store().forget(memory_id)}


@mcp.tool()
def edit(
    memory_id: str,
    text: str | None = None,
    type: str | None = None,
    tags: list[str] | None = None,
    importance: float | None = None,
    polarity: int | None = None,
    source: str | None = None,
    clear_source: bool = False,
    expires_at: float | None = None,
    pinned: bool | None = None,
    trust: str | None = None,
    valid_from: float | None = None,
    valid_until: float | None = None,
) -> dict[str, Any]:
    """Update fields of an existing memory in place, keeping its id.

    Omitted fields stay unchanged. Text changes are re-embedded; links,
    supersede state, and access tracking are preserved (do NOT forget+remember
    to fix a typo — that loses them). The change is journaled and undoable.
    type: episodic | semantic | procedural | feedback.
    clear_source: true removes the provenance string (an omitted `source`
    otherwise means "leave unchanged"); do not combine with `source`.
    expires_at: set the TTL to this absolute epoch; pass 0 to clear it (omit to
    leave unchanged).
    """
    if clear_source and source is not None:
        return {"ok": False,
                "error": "pass either source or clear_source, not both"}
    kwargs: dict[str, Any] = {}
    if text is not None:
        kwargs["text"] = text
    if type is not None:
        kwargs["type"] = type
    if tags is not None:
        kwargs["tags"] = tags
    if importance is not None:
        kwargs["importance"] = importance
    if polarity is not None:
        kwargs["polarity"] = polarity
    if pinned is not None:
        kwargs["pinned"] = pinned
    if trust is not None:
        kwargs["trust"] = trust
    if expires_at is not None:
        kwargs["expires_at"] = expires_at
    if valid_from is not None:
        kwargs["valid_from"] = valid_from
    if valid_until is not None:
        kwargs["valid_until"] = valid_until
    if source is not None:
        kwargs["source"] = source
    if clear_source:
        kwargs["source"] = None
    try:
        mem = get_store().edit(memory_id, **kwargs)
    except (KeyError, ValueError) as e:
        return {"ok": False, "error": e.args[0] if e.args else str(e)}
    return {
        "ok": True,
        "id": mem.id,
        "text": mem.text,
        "type": mem.type,
        "tags": mem.tags,
        "importance": mem.importance,
        "polarity": mem.polarity,
        "source": mem.source,
        "expires_at": mem.expires_at or None,
    }


@mcp.tool()
def list_recent(
    limit: int = 20,
    include_superseded: bool = False,
    include_expired: bool = False,
) -> list[dict[str, Any]]:
    """List the most recently created memories."""
    return [
        {
            "id": m.id,
            "text": m.text,
            "type": m.type,
            "tags": m.tags,
            "created_at": m.created_at,
            "superseded_by": m.superseded_by or None,
            "expires_at": m.expires_at or None,
        }
        for m in get_store().list_recent(
            limit=limit, include_superseded=include_superseded,
            include_expired=include_expired)
    ]


@mcp.tool()
def purge_expired(dry_run: bool = False) -> dict[str, Any]:
    """Hard-delete memories whose TTL has passed (reclaims storage).

    Expired memories are already hidden from recall; this removes them for
    good. dry_run=True reports what would be purged without deleting.
    """
    purged = get_store().purge_expired(dry_run=dry_run)
    return {"purged": len(purged), "dry_run": dry_run,
            "ids": [p.id for p in purged]}


@mcp.tool()
def metrics() -> dict[str, Any]:
    """Runtime metrics: op counters + recall latency since server start.

    Process-local and in-memory (reset on restart). Complements `stats`
    (content aggregates) with operational counts.
    """
    return get_store().metrics()


@mcp.tool()
def history(memory_id: str) -> list[dict[str, Any]]:
    """Full journaled timeline of one memory, oldest first.

    Every event touching the memory — creation, edits, supersede/restore,
    links pointing at it, forget — with before/after snapshots. Bounded by
    journal retention.
    """
    return [
        {"ts": e.ts, "op": e.op, "actor": e.actor, "id": e.id,
         "before": e.before, "after": e.after, "meta": e.meta,
         "summary": e.summary()}
        for e in get_store().history(memory_id)
    ]


@mcp.tool()
def state_at(ts: str) -> dict[str, Any]:
    """Reconstruct the store's live memories as of a past time.

    ts: epoch seconds, an ISO-8601 date/datetime, or a relative span like "7d".
    Best-effort replay of the journal (see the store's state_at docs for its
    limits: retention window, nuke resets, journaling-disabled gaps).
    """
    t = parse_timestamp(ts)
    if t is None:
        return {"error": "ts is required"}
    mems = get_store().state_at(t)
    return {
        "ts": t,
        "count": len(mems),
        "memories": [
            {"id": m.id, "text": m.text, "type": m.type, "tags": m.tags,
             "importance": m.importance, "created_at": m.created_at}
            for m in mems
        ],
    }


@mcp.tool()
def get_at(memory_id: str, ts: str) -> dict[str, Any]:
    """Reconstruct a single memory as it was at a past time (see state_at)."""
    t = parse_timestamp(ts)
    if t is None:
        return {"error": "ts is required"}
    mem = get_store().get_at(memory_id, t)
    if mem is None:
        return {"ok": False, "error": "memory did not exist at that time"}
    return {"ok": True, "ts": t, **{
        "id": mem.id, "text": mem.text, "type": mem.type, "tags": mem.tags,
        "importance": mem.importance, "created_at": mem.created_at,
        "superseded_by": mem.superseded_by or None,
        "expires_at": mem.expires_at or None,
    }}


@mcp.tool()
def stats() -> dict[str, Any]:
    """Return basic store statistics."""
    store = get_store()
    return {"count": store.count(), "path": store.path,
            "collection": store.collection_name}


@mcp.tool()
def link(
    src_id: str,
    dst_id: str,
    rel: str = "related",
) -> dict[str, Any]:
    """Add a directed link src → dst.
    rel: supersedes | refines | derived_from | example_of | contradicts | related.
    Idempotent — calling twice with the same arguments is safe.
    """
    try:
        get_store().link(src_id=src_id, dst_id=dst_id, rel=rel)
        return {"ok": True, "src_id": src_id, "dst_id": dst_id, "rel": rel}
    except (KeyError, ValueError) as e:
        return {"ok": False, "error": str(e)}


@mcp.tool()
def unlink(
    src_id: str,
    dst_id: str,
    rel: str | None = None,
) -> dict[str, Any]:
    """Remove link(s) from src to dst. If rel is omitted, all rels are removed."""
    removed = get_store().unlink(src_id=src_id, dst_id=dst_id, rel=rel)
    return {"removed": removed}


@mcp.tool()
def neighbors(
    memory_id: str,
    rel: str | None = None,
    direction: str = "both",
    depth: int = 1,
) -> list[dict[str, Any]]:
    """Return memories reachable from memory_id via links.
    direction: out | in | both (default).
    """
    hits = get_store().neighbors(
        memory_id,
        rel=rel,
        direction=direction,  # type: ignore[arg-type]
        depth=depth,
    )
    return [
        {
            "id": m.id,
            "text": m.text,
            "type": m.type,
            "tags": m.tags,
            "importance": m.importance,
            "rel": r,
        }
        for m, r in hits
    ]


@mcp.tool()
def find_conflicts(
    memory_id: str | None = None,
    threshold: float | None = None,
) -> list[dict[str, Any]]:
    """Detect duplicate or contradicting memories.
    memory_id: check one specific memory; omit for a full-store scan.
    threshold: cosine similarity cutoff (default: 0.80).
    """
    conflicts = get_store().find_conflicts(memory_id=memory_id, threshold=threshold)
    return [
        {
            "kind":       c.kind,
            "reason":     c.reason,
            "similarity": c.similarity,
            "a": {"id": c.a.id, "text": c.a.text[:120], "type": c.a.type},
            "b": {"id": c.b.id, "text": c.b.text[:120], "type": c.b.type},
        }
        for c in conflicts
    ]


@mcp.tool()
def supersede(old_id: str, new_id: str) -> dict[str, Any]:
    """Mark old_id as superseded by new_id (soft-delete old, add 'supersedes' link).
    The old memory stays in the store but is hidden from default queries.
    Use restore() / forget() to undo or hard-delete.
    """
    try:
        get_store().supersede(old_id=old_id, new_id=new_id)
        return {"ok": True, "old_id": old_id, "new_id": new_id}
    except (KeyError, ValueError) as e:
        return {"ok": False, "error": str(e)}


@mcp.tool()
def restore(memory_id: str) -> dict[str, Any]:
    """Undo a supersede: clear the soft-delete so the memory is visible again.

    Also removes the 'supersedes' link the superseder gained. Returns
    restored:false when the memory does not exist or was not superseded.
    """
    mem = get_store().get(memory_id)
    ok = get_store().restore(memory_id)
    return {"restored": ok, "id": memory_id,
            "was_superseded_by": (mem.superseded_by or None) if mem else None}


@mcp.tool()
def subgraph(memory_ids: list[str], depth: int = 1) -> dict[str, Any]:
    """Return the link graph reachable from the given memory ids within depth hops.

    Follows OUTGOING links only (use `neighbors` with direction="in" for the
    reverse). Returns {nodes: [...], edges: [{src, dst, rel}]}.
    """
    graph = get_store().subgraph(memory_ids, depth=depth)
    return {
        "nodes": [_mem_dict(m) for m in graph.nodes.values()],
        "edges": [{"src": s, "dst": d, "rel": r} for s, d, r in graph.edges],
    }


@mcp.tool()
def undo(ts: float | None = None, memory_id: str | None = None) -> dict[str, Any]:
    """Reverse a journaled mutation — the newest one by default.

    Pass `ts` to undo the entry with that exact journal timestamp (as reported
    by `journal_tail`), or `memory_id` to undo the newest entry touching that
    memory. Undo refuses when the current state has diverged from the entry's
    "after" snapshot, so it cannot silently clobber a later change. The undo
    itself is journaled.
    """
    store = get_store()
    entry = None
    if ts is not None:
        entry = store.journal.find_by_ts(ts)
        if entry is None:
            return {"ok": False, "error": f"no journal entry at ts={ts}"}
    else:
        candidates = [
            e for e in store.journal.read()
            if memory_id is None or store._entry_touches(e, memory_id)
        ]
        if not candidates:
            return {"ok": False, "error": "no journal entry to undo"}
        entry = candidates[-1]
    ok = store.undo(entry)
    return {"ok": ok, "op": entry.op, "id": entry.id, "ts": entry.ts,
            "actor": entry.actor,
            "error": None if ok else "state diverged or nothing to undo"}


@mcp.tool()
def nuke(confirm: str = "") -> dict[str, Any]:
    """Delete EVERY memory in the collection. Irreversible.

    Guarded: pass confirm="DELETE ALL" to proceed. The journal keeps a single
    'nuke' entry with the count, but the memories themselves are gone — undo
    cannot bring them back.
    """
    if confirm != "DELETE ALL":
        return {"ok": False, "deleted": 0,
                "error": 'refusing to nuke: pass confirm="DELETE ALL"'}
    return {"ok": True, "deleted": get_store().nuke()}


@mcp.tool()
def eval_recall(
    cases: list[dict[str, Any]],
    k: int = 5,
    mode: str = "hybrid",
    fusion: str = "weighted",
    graph: float | None = None,
    diversity: float | None = None,
    dedup_threshold: float | None = None,
    min_cosine: float | None = None,
) -> dict[str, Any]:
    """Score retrieval quality against a gold set. Read-only.

    Each case is {"query": str, "relevant_ids": [id, ...], "k"?: int,
    "mode"?: str}. Returns recall@k, precision@k, MRR, MAP and nDCG@k averaged
    over the cases, plus a per-case breakdown.

    Recall runs with touch=false, so evaluating never perturbs access counts or
    recency. Pass the ranking knobs to A/B a configuration — this is the only
    way to tell whether a weight change actually helped.
    """
    parsed: list[EvalCase] = []
    for i, raw in enumerate(cases):
        if not isinstance(raw, dict):
            return {"error": f"case {i}: expected an object"}
        query = raw.get("query")
        ids = raw.get("relevant_ids")
        if not query:
            return {"error": f"case {i}: missing 'query'"}
        if not isinstance(ids, list) or not ids:
            return {"error": f"case {i}: 'relevant_ids' must be a non-empty list"}
        parsed.append(EvalCase(
            query=str(query),
            relevant_ids=[str(x) for x in ids],
            k=raw.get("k"),
            mode=raw.get("mode"),
        ))
    if not parsed:
        return {"error": "no cases supplied"}

    kwargs: dict[str, Any] = {"fusion": fusion}
    if graph is not None:
        kwargs["weights"] = HybridWeights(graph=graph)
    if diversity is not None:
        kwargs["diversity"] = diversity
    if dedup_threshold is not None:
        kwargs["dedup_threshold"] = dedup_threshold
    if min_cosine is not None:
        kwargs["min_cosine"] = min_cosine

    try:
        result = evaluate(get_store(), parsed, default_k=k, default_mode=mode,
                          **kwargs)
    except ValueError as e:
        return {"error": str(e)}
    return {
        "n": result.n,
        "k": result.k,
        "recall_at_k": round(result.recall_at_k, 4),
        "precision_at_k": round(result.precision_at_k, 4),
        "mrr": round(result.mrr, 4),
        "map": round(result.map, 4),
        "ndcg_at_k": round(result.ndcg_at_k, 4),
        "per_case": result.per_case,
    }


@mcp.tool()
def merge(target_id: str, other_id: str, separator: str = "\n\n") -> dict[str, Any]:
    """Fold one memory into another and delete the absorbed one.

    Combines the text, transfers the absorbed memory's outgoing links, and
    re-points every INCOMING link at the target — `forget` does not clean up
    incoming edges, so a plain delete would strand every relationship pointing
    at the absorbed memory. Journaled on both sides.
    """
    try:
        mem = get_store().merge(target_id, other_id, separator=separator)
    except MergeError as e:
        return {"ok": False, "error": str(e), "not_found": e.not_found}
    return {"ok": True, **_mem_dict(mem)}


@mcp.tool()
def versions(memory_id: str) -> list[dict[str, Any]]:
    """Past text states of a memory, oldest first.

    Each entry is the state BEFORE an edit; the current live state is excluded
    (use `get`). Reads rotated journal segments, so history survives a rollover.
    """
    return [
        {"ts": v.ts, "text": v.text, "tags": v.tags,
         "importance": v.importance, "source": v.source, "type": v.type}
        for v in get_store().versions(memory_id)
    ]


@mcp.tool()
def list_tags(include_superseded: bool = False) -> list[dict[str, Any]]:
    """Every tag with its usage count, most-used first."""
    return [{"tag": t, "count": n} for t, n in
            get_store().list_tags(include_superseded=include_superseded)]


@mcp.tool()
def rename_tag(old: str, new: str) -> dict[str, Any]:
    """Rename a tag across the collection, de-duplicating on collision."""
    try:
        res = get_store().rename_tag(old, new)
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "changed": res.changed, "tag": res.tag}


@mcp.tool()
def merge_tags(sources: list[str], into: str) -> dict[str, Any]:
    """Fold several tags into one across the collection."""
    try:
        res = get_store().merge_tags(sources, into)
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "changed": res.changed, "tag": res.tag}


@mcp.tool()
def delete_tag(tag: str) -> dict[str, Any]:
    """Strip a tag from every memory that carries it."""
    res = get_store().delete_tag(tag)
    return {"ok": True, "changed": res.changed, "tag": res.tag}


@mcp.tool()
def find_path(from_id: str, to_id: str, max_depth: int = 6) -> dict[str, Any]:
    """Shortest undirected link path between two memories.

    Undirected because "how are these related?" does not care which way the
    author drew the arrow. Returns {found, path:[{id, rel, text}]}; an empty
    path means no route within max_depth.
    """
    store = get_store()
    hops = store.find_path(from_id, to_id, max_depth=max_depth)
    out = []
    for mid, rel in hops:
        mem = store.get(mid)
        out.append({"id": mid, "rel": rel,
                    "text": (mem.text[:120] if mem else None)})
    return {"found": bool(hops), "length": max(0, len(hops) - 1), "path": out}


@mcp.tool()
def trash(memory_id: str) -> dict[str, Any]:
    """Soft-delete a memory: recoverable, unlike `forget`.

    The missing middle between `supersede` (which asserts "replaced by X") and
    `forget` (irreversible). Use `trash_restore` to bring it back.
    """
    return {"trashed": get_store().trash(memory_id), "id": memory_id}


@mcp.tool()
def trash_list() -> list[dict[str, Any]]:
    """Everything currently in the trash, oldest first."""
    return [
        {"memory_id": e.memory_id, "deleted_at": e.deleted_at,
         "actor": e.actor, "text": (e.memory.get("text") or "")[:200]}
        for e in get_store().trash_list()
    ]


@mcp.tool()
def trash_restore(memory_id: str) -> dict[str, Any]:
    """Bring a trashed memory back with its id, tags and links intact."""
    mem = get_store().trash_restore(memory_id)
    if mem is None:
        return {"restored": False, "id": memory_id,
                "error": "not in the trash"}
    return {"restored": True, **_mem_dict(mem)}


@mcp.tool()
def trash_purge(memory_id: str | None = None,
                older_than_days: float | None = None) -> dict[str, Any]:
    """Permanently drop trashed memories. Irreversible.

    Pass memory_id for one entry, older_than_days to apply a retention cutoff,
    or neither to empty the whole trash. The two are mutually exclusive.
    """
    if memory_id is not None and older_than_days is not None:
        return {"purged": 0,
                "error": "pass either memory_id or older_than_days, not both"}
    if older_than_days is not None:
        return {"purged": get_store().trash_purge_expired(older_than_days)}
    return {"purged": get_store().trash_purge(memory_id)}


@mcp.tool()
def ready() -> dict[str, Any]:
    """Readiness probe: is the store reachable and the embedder working?

    Returns {ready, checks:{store, embedder}} with the embedder check carrying
    its measured dimension and latency. Unlike the HTTP /ready endpoint this is
    not sanitized — an MCP client is already authenticated.
    """
    return get_store().readiness()


@mcp.tool()
def maintenance_tick(
    reflect_apply: bool | None = None,
) -> dict[str, Any]:
    """Run one maintenance tick: prune stale memories via decay and optionally
    consolidate episodic clusters via reflection.

    Uses the schedule configured in ~/.config/ai_houkai/config.toml
    [maintenance] section (or built-in defaults).  Jobs only run when their
    interval has elapsed since the last run — safe to call frequently.

    reflect_apply
        If True, reflection summaries are written to the store.
        If False, reflection runs in dry-run mode (reports what it would
        create, writes nothing). Omit to use the config file's
        [maintenance.reflect] apply setting (default False).

    When [maintenance].enabled = false in the config, nothing runs and the
    result reports enabled=false.

    Returns a summary dict with counts and any errors.
    """
    mcfg = load_maintenance()
    if not mcfg.enabled:
        return {
            "enabled": False,
            "summary": "maintenance disabled ([maintenance].enabled = false)",
            "ran_decay": False,
            "ran_reflect": False,
            "ran_purge": False,
            "decayed": 0,
            "reflected": 0,
            "purged": 0,
            "trash_purged": 0,
            "reflect_applied": False,
            "decay_error": None,
            "reflect_error": None,
            "purge_error": None,
        }
    if reflect_apply is None:
        reflect_apply = mcfg.reflect_apply
    sched = MaintenanceScheduler(
        store=get_store(),
        decay_every=mcfg.decay_every,
        reflect_every=mcfg.reflect_every,
        purge_every=mcfg.purge_every,
        trash_ttl_days=mcfg.trash_ttl_days,
        tick_interval=mcfg.tick_interval,
        state_path=mcfg.state_path,
        decay_rate=mcfg.decay_rate,
        min_score=mcfg.min_score,
        protect_types=mcfg.protect_types,
        frequency_weight=mcfg.frequency_weight,
        min_cluster_size=mcfg.min_cluster_size,
        reflect_apply=reflect_apply,
        reflect_consolidate=mcfg.reflect_consolidate,
        summarizer=build_summarizer(mcfg.summarizer),
    )
    result = sched.tick()
    return {
        "enabled": True,
        "summary": result.summary(),
        "ran_decay": result.ran_decay,
        "ran_reflect": result.ran_reflect,
        "ran_purge": result.ran_purge,
        "decayed": result.decayed,
        "reflected": result.reflected,
        "purged": result.purged,
        "trash_purged": result.trash_purged,
        "reflect_applied": result.reflect_applied,
        "decay_error": result.decay_error,
        "reflect_error": result.reflect_error,
        "purge_error": result.purge_error,
    }


@mcp.tool()
def journal_tail(
    n: int = 20,
    op: str | None = None,
    since_seconds: float | None = None,
) -> list[dict[str, Any]]:
    """Return the most recent audit-journal entries (newest first).

    op: filter by remember|forget|supersede|restore|link|unlink|reflect|decay|...
    since_seconds: limit to entries within the last N seconds.
    """
    if n <= 0:  # entries[-0:] would be the WHOLE journal, not none of it
        return []
    since = (time.time() - since_seconds) if since_seconds is not None else None
    entries = list(get_store().journal.read(since=since, op=op))
    entries = entries[-n:][::-1]
    return [
        {"ts": e.ts, "op": e.op, "actor": e.actor,
         "id": e.id, "summary": e.summary(), "meta": e.meta}
        for e in entries
    ]


@mcp.tool(name="export")
def export(
    path: str,
    include_vectors: bool = True,
    include_superseded: bool = False,
    type: str | None = None,
    tag: str | None = None,
    since: str | float | None = None,
) -> dict[str, Any]:
    """Export memories to a portable .ahkai file (gzipped JSONL).

    The path is server-local. Returns summary counts.
    since: epoch float, ISO date, or relative like "7d" — same as recall.
    """
    summary = get_store().export(
        path,
        include_vectors=include_vectors,
        include_superseded=include_superseded,
        types=[type] if type else None,
        tags=[tag] if tag else None,
        since=parse_timestamp(since),
    )
    return {
        "path":    str(summary.path),
        "count":   summary.count,
        "bytes":   summary.bytes,
        "elapsed": summary.elapsed,
    }


@mcp.tool(name="import")
def import_(
    path: str,
    on_conflict: str = "skip",
    regenerate_vectors: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Import memories from a portable .ahkai file.

    on_conflict: skip | overwrite | rename | error.
    """
    try:
        summary = get_store().import_(
            path,
            on_conflict=on_conflict,            # type: ignore[arg-type]
            regenerate_vectors=regenerate_vectors,
            dry_run=dry_run,
        )
    except (ImportError, ImportConflictError, FileNotFoundError) as e:
        return {"ok": False, "error": str(e)}
    return {
        "ok": True,
        "imported":    summary.imported,
        "skipped":     summary.skipped,
        "overwritten": summary.overwritten,
        "renamed":     summary.renamed,
        "errors":      summary.errors,
        "vectors_regenerated": summary.vectors_regenerated,
    }


def run() -> None:
    """Console-script entry point: ``ai-houkai-mcp``."""
    mcp.run()


if __name__ == "__main__":
    run()
