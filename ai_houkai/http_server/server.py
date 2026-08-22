"""Standard-library JSON HTTP server over a :class:`MemoryStore`.

Routes (all JSON in / JSON out):

    GET    /health                         liveness + memory count
    GET    /ready                           readiness — backend + embedder probe (200/503)
    GET    /stats                          store statistics
    GET    /metrics                        runtime counters + recall latency
    GET    /memories?limit=&include_superseded=&include_expired=
                                           recent memories (list_recent)
    POST   /memories                       store a memory (remember; ttl_seconds/expires_at)
    POST   /memories/batch                 bulk store with batched embedding (remember_many; idempotent?)
    GET    /memories/{id}                  fetch one memory
    PATCH  /memories/{id}                  edit fields in place (journaled; expires_at)
    DELETE /memories/{id}                  forget one memory
    GET    /memories/{id}/neighbors?rel=&direction=&depth=
                                           linked memories
    GET    /memories/{id}/history          full journaled timeline of one memory
    GET    /memories/{id}/at?ts=           reconstruct one memory as of a past time
    GET    /state_at?ts=                   reconstruct all live memories as of a past time
    POST   /purge_expired  {dry_run?}      hard-delete memories whose TTL passed
    GET    /recall?query=&k=&type=&tag=&min_importance=&source=&since=&until=&mode=&overfetch=&include_expired=&explain=
    POST   /recall                         same, via JSON body
    POST   /recall_pack                    token-budgeted context block
    POST   /auto_context                   multi-angle context block (auto_context_pack)
    POST   /links        {src_id,dst_id,rel?}      add a directed link
    POST   /unlink       {src_id,dst_id,rel?}      remove link(s)
    POST   /supersede    {old_id,new_id}           soft-delete + supersede link
    POST   /restore      {memory_id}               clear a supersede (un-soft-delete)
    POST   /conflicts    {memory_id?,threshold?}   duplicate / contradiction scan
    POST   /subgraph     {memory_ids,depth?}       link graph reachable from ids
    POST   /undo         {ts?,memory_id?}          reverse a journaled mutation
    POST   /nuke         {confirm}                 delete every memory (guarded)
    GET    /journal?n=&op=&since=                  audit-journal tail
    POST   /export       {path,…}                  write a .ahkai archive (server-side path)
    POST   /import       {path,on_conflict?,…}     read a .ahkai archive (server-side path)
    POST   /merge        {target_id,other_id,…}    fold one memory into another
    GET    /memories/{id}/versions                 past text states from the journal
    GET    /tags?include_superseded=               tag usage counts
    POST   /tags/rename  {old,new}                 rename a tag collection-wide
    POST   /tags/merge   {sources,into}            fold several tags into one
    DELETE /tags/{tag}                             strip a tag from every memory
    POST   /find_path    {from_id,to_id,max_depth?} shortest undirected link path
    POST   /trash        {memory_id}               soft-delete (recoverable)
    GET    /trash                                  list soft-deleted memories
    POST   /trash/restore {memory_id}              bring one back
    POST   /trash/purge  {memory_id?, older_than_days?}  permanently drop (irreversible)

Optional bearer-token auth: pass ``auth_token`` (or set ``AI_HOUKAI_HTTP_TOKEN``)
and every request must carry ``Authorization: Bearer <token>``.  ``/health`` and
``/ready`` stay reachable so liveness/readiness probes work without the secret.
``/export`` and ``/import`` (which read/write server-side paths) additionally
require a token to be configured at all; a tokenless server refuses them.

The handler is intentionally framework-free: a single regex routing table maps
``(method, path)`` to small functions taking ``(store, match, query, body)``.
"""

from __future__ import annotations

import hmac
import json
import os
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Optional
from urllib.parse import parse_qs, urlsplit

from ai_houkai.memory_system import ExpandSpec, HybridWeights, MemoryStore, RememberItem
from ai_houkai.memory_system.curation import MergeError
from ai_houkai.memory_system.store import (
    ConflictError,
    ImportConflictError,
    extract_key_phrases,
)
from ai_houkai.timeparse import parse_timestamp


class HttpError(Exception):
    """Raised by a route to short-circuit with a specific status + message."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def _mem_dict(mem: Any) -> dict[str, Any]:
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
        "links": [{"to": l.to, "rel": l.rel} for l in mem.links],
        "superseded_by": mem.superseded_by or None,
        "superseded_at": mem.superseded_at or None,
        "expires_at": mem.expires_at or None,
        # Always present, never elided to null: a REST client can set these on
        # a write, so it has to be able to read them back — and "absent" would
        # be indistinguishable from "not pinned" / "unlabelled".
        "pinned": mem.pinned,
        "trust": mem.trust,
        # Valid time — when the memory was true, as opposed to when we learned
        # it. 0 on either end means unbounded.
        "valid_from": mem.valid_from,
        "valid_until": mem.valid_until,
    }


def _hit_dict(mem: Any, score: float,
              explanation: dict[str, Any] | None = None) -> dict[str, Any]:
    d = _mem_dict(mem)
    d["score"] = round(score, 4)
    if explanation is not None:
        d["explain"] = explanation
    return d


def _qs_one(query: dict[str, list[str]], key: str) -> Optional[str]:
    vals = query.get(key)
    return vals[0] if vals else None


def _as_int(value: Optional[str], default: int) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise HttpError(400, f"'{value}' is not a valid integer") from exc


def _as_float(value: Optional[str]) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise HttpError(400, f"'{value}' is not a valid number") from exc


def _as_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


# JSON-body coercers — the POST twins of _as_int/_as_float/_as_bool. A JSON
# body can carry any type (null, string, object, …), and passing those raw
# into the store used to surface as 500s where the GET path returned 400.

def _body_int(body: dict[str, Any], key: str, default: int) -> int:
    v = body.get(key)
    if v is None:
        return default
    if isinstance(v, bool):  # bool subclasses int — reject explicitly
        raise HttpError(400, f"{key}: {v!r} is not a valid integer")
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, str):
        try:
            return int(v)
        except ValueError:
            pass
    raise HttpError(400, f"{key}: {v!r} is not a valid integer")


def _body_float(body: dict[str, Any], key: str) -> Optional[float]:
    v = body.get(key)
    if v is None:
        return None
    if isinstance(v, bool):
        raise HttpError(400, f"{key}: {v!r} is not a valid number")
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            pass
    raise HttpError(400, f"{key}: {v!r} is not a valid number")


def _body_bool(body: dict[str, Any], key: str, default: bool = False) -> bool:
    v = body.get(key)
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, str):  # JSON string "false" must not be truthy
        return v.lower() in ("1", "true", "yes", "on")
    if isinstance(v, (int, float)):
        return bool(v)
    raise HttpError(400, f"{key}: {v!r} is not a valid boolean")


def _body_tags(body: dict[str, Any], key: str = "tags") -> list[str]:
    """Coerce a JSON tags value to a list of strings.

    A lone string becomes a one-element list — passing it through raw would
    iterate the string and store every character as its own tag. Anything
    other than a string / list of strings is a 400.
    """
    v = body.get(key)
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    if isinstance(v, list) and all(isinstance(t, str) for t in v):
        return v
    raise HttpError(400, f"{key}: must be a string or a list of strings")


def _time(value: Any) -> Optional[float]:
    try:
        return parse_timestamp(value)
    except ValueError as exc:
        raise HttpError(400, str(exc)) from exc


def _require(body: dict[str, Any], key: str) -> str:
    if key not in body or body[key] in (None, ""):
        raise HttpError(400, f"missing required field: {key}")
    v = body[key]
    # Every required field is a string; a JSON number/list/object here used
    # to sail into the store and surface as a 500.
    if not isinstance(v, str):
        raise HttpError(400, f"{key}: must be a string")
    return v


# Each takes (store, match, query, body) and returns (status, payload).

def _health(store: MemoryStore, m, q, b):
    # Liveness only — deliberately does not leak the collection name / topology.
    return 200, {"status": "ok", "count": store.count()}


def _ready(store: MemoryStore, m, q, b):
    # Readiness (distinct from the always-open liveness /health): exercises the
    # backend and an actual embed so orchestrators learn whether the store can
    # serve requests. 503 when any dependency check fails.
    #
    # Auth-exempt like /health, so the body is deliberately minimal: only the
    # overall flag and a per-check ok bool. Raw error strings, provider
    # latency/dim, and file paths are withheld so an unauthenticated probe can't
    # enumerate backend internals — use `houkai doctor` (local/authenticated)
    # for the detailed report. cache_ttl bounds embed calls under rapid polling.
    r = store.readiness(cache_ttl=5.0)
    safe = {
        "ready": r["ready"],
        "checks": {name: {"ok": bool(c.get("ok"))}
                   for name, c in r["checks"].items()},
    }
    return (200 if r["ready"] else 503), safe


def _stats(store: MemoryStore, m, q, b):
    return 200, {"count": store.count(), "path": store.path,
                 "collection": store.collection_name}


def _metrics(store: MemoryStore, m, q, b):
    # Runtime counters + recall latency (process-local, reset on restart).
    return 200, store.metrics()


def _list(store: MemoryStore, m, q, b):
    limit = _as_int(_qs_one(q, "limit"), 20)
    inc = _as_bool(_qs_one(q, "include_superseded"))
    inc_exp = _as_bool(_qs_one(q, "include_expired"))
    mems = store.list_recent(limit=limit, include_superseded=inc,
                             include_expired=inc_exp)
    return 200, {"memories": [_mem_dict(x) for x in mems]}


def _purge_expired(store: MemoryStore, m, q, b):
    dry = _body_bool(b, "dry_run")
    purged = store.purge_expired(dry_run=dry)
    return 200, {"purged": len(purged), "dry_run": dry,
                 "ids": [p.id for p in purged]}


def _journal_entry_dict(e: Any) -> dict[str, Any]:
    return {"ts": e.ts, "op": e.op, "actor": e.actor, "id": e.id,
            "before": e.before, "after": e.after, "meta": e.meta,
            "summary": e.summary()}


def _history(store: MemoryStore, m, q, b):
    mid = m.group("id")
    entries = store.history(mid)
    # Distinguish an unknown id from a live memory with no journal history.
    if not entries and store.get(mid) is None:
        raise HttpError(404, "memory not found")
    return 200, {"id": mid,
                 "history": [_journal_entry_dict(e) for e in entries]}


def _state_at(store: MemoryStore, m, q, b):
    ts = _time(_qs_one(q, "ts"))
    if ts is None:
        raise HttpError(400, "missing required field: ts")
    mems = store.state_at(ts)
    return 200, {"ts": ts, "count": len(mems),
                 "memories": [_mem_dict(x) for x in mems]}


def _get_at(store: MemoryStore, m, q, b):
    ts = _time(_qs_one(q, "ts"))
    if ts is None:
        raise HttpError(400, "missing required field: ts")
    mem = store.get_at(m.group("id"), ts)
    if mem is None:
        raise HttpError(404, "memory did not exist at that time")
    return 200, _mem_dict(mem)


def _remember(store: MemoryStore, m, q, b):
    text = _require(b, "text")
    # Asked before the write, because remember() returns the existing row on a
    # dedupe hit and there is otherwise no way to tell it from a fresh one.
    # 201/stored:true has to mean "I created something": a client replaying a
    # batch every session was told it wrote N new rows when it wrote none.
    deduped = (_body_bool(b, "idempotent")
               and store.find_by_content_hash(text) is not None)
    try:
        mem = store.remember(
            text=text,
            # `or` (not a .get default) so an explicit JSON null also means
            # "use the default", matching the null convention in _edit.
            type=b.get("type") or "semantic",
            tags=_body_tags(b),
            importance=_body_float(b, "importance"),
            source=b.get("source"),
            polarity=_body_int(b, "polarity", 0),
            expires_at=_body_float(b, "expires_at"),
            ttl_seconds=_body_float(b, "ttl_seconds"),
            pinned=_body_bool(b, "pinned"),
            trust=b.get("trust") or "trusted",
            idempotent=_body_bool(b, "idempotent"),
            valid_from=_body_float(b, "valid_from"),
            valid_until=_body_float(b, "valid_until"),
            on_conflict=b.get("on_conflict"),
        )
    except ConflictError as e:
        return 409, {
            "stored": False,
            "conflicts": [
                {"kind": c.kind, "similarity": c.similarity,
                 "other_id": c.b.id, "other_text": c.b.text[:100]}
                for c in e.conflicts
            ],
        }
    if deduped:
        return 200, {"stored": False, **_mem_dict(mem)}
    return 201, {"stored": True, **_mem_dict(mem)}


def _remember_many(store: MemoryStore, m, q, b):
    raw = b.get("items")
    if not isinstance(raw, list):
        raise HttpError(400, "body must include an 'items' array")
    if not raw:
        return 201, {"stored": 0, "memories": []}
    items: list[RememberItem] = []
    for i, it in enumerate(raw):
        if not isinstance(it, dict):
            raise HttpError(400, f"items[{i}]: must be an object")
        text = it.get("text")
        if not isinstance(text, str) or not text.strip():
            raise HttpError(400, f"items[{i}]: missing or empty 'text'")
        # Reuse the per-field body coercers on each item so a string tag,
        # numeric-string importance, etc. behave exactly as on POST /memories.
        items.append(RememberItem(
            text=text,
            type=it.get("type") or "semantic",
            tags=tuple(_body_tags(it)),
            importance=_body_float(it, "importance"),
            source=it.get("source"),
            polarity=_body_int(it, "polarity", 0),
            expires_at=_body_float(it, "expires_at"),
            ttl_seconds=_body_float(it, "ttl_seconds"),
        ))
    started = time.time()
    try:
        mems = store.remember_many(
            items,
            batch_size=_body_int(b, "batch_size", 128),
            on_conflict=b.get("on_conflict"),
            idempotent=_body_bool(b, "idempotent"),
        )
    except ValueError as e:
        # bad type/tag/policy (incl. on_conflict='raise') → 400, not 500
        raise HttpError(400, str(e))
    # `stored` is how many rows the batch CREATED. An idempotent replay returns
    # the pre-existing rows, and reporting len(mems) told the client it had
    # written N rows when it had written none. Counting distinct new ids also
    # collapses intra-batch duplicates, which map to one row.
    created = {x.id for x in mems if x.created_at >= started}
    status = 201 if created else 200
    return status, {"stored": len(created),
                    "memories": [_mem_dict(x) for x in mems]}


def _get_one(store: MemoryStore, m, q, b):
    mem = store.get(m.group("id"))
    if mem is None:
        raise HttpError(404, "memory not found")
    return 200, _mem_dict(mem)


def _forget(store: MemoryStore, m, q, b):
    ok = store.forget(m.group("id"))
    if not ok:
        raise HttpError(404, "memory not found")
    return 200, {"forgotten": True, "id": m.group("id")}


def _edit(store: MemoryStore, m, q, b):
    if not b:
        raise HttpError(400, "empty edit: provide at least one field")
    # Uniform null semantics: null means "leave unchanged" for every field
    # except `source`, where null explicitly clears (matching the store's
    # sentinel-based edit()). An explicit [] clears tags.
    kwargs: dict[str, Any] = {}
    if b.get("text") is not None:
        kwargs["text"] = b["text"]
    if b.get("type") is not None:
        kwargs["type"] = b["type"]
    if b.get("tags") is not None:
        kwargs["tags"] = _body_tags(b)
    if b.get("importance") is not None:
        kwargs["importance"] = _body_float(b, "importance")
    if b.get("polarity") is not None:
        kwargs["polarity"] = _body_int(b, "polarity", 0)
    if b.get("pinned") is not None:
        kwargs["pinned"] = _body_bool(b, "pinned")
    if b.get("trust") is not None:
        kwargs["trust"] = b["trust"]
    if b.get("valid_from") is not None:
        kwargs["valid_from"] = _body_float(b, "valid_from")
    if b.get("valid_until") is not None:
        kwargs["valid_until"] = _body_float(b, "valid_until")
    if b.get("expires_at") is not None:
        # null = unchanged; an explicit 0 clears the TTL.
        kwargs["expires_at"] = _body_float(b, "expires_at")
    if "source" in b:
        kwargs["source"] = b["source"]
    if not kwargs:
        raise HttpError(400, "no editable fields in body")
    try:
        mem = store.edit(m.group("id"), **kwargs)
    except KeyError:
        raise HttpError(404, "memory not found")
    return 200, _mem_dict(mem)


def _neighbors(store: MemoryStore, m, q, b):
    # neighbors() can't distinguish "no links" from "no such memory" — check
    # existence first so a typo'd id is a 404, consistent with GET /memories/{id}.
    if store.get(m.group("id")) is None:
        raise HttpError(404, "memory not found")
    hits = store.neighbors(
        m.group("id"),
        rel=_qs_one(q, "rel"),
        direction=_qs_one(q, "direction") or "both",
        depth=_as_int(_qs_one(q, "depth"), 1),
    )
    return 200, {"neighbors": [{**_mem_dict(mem), "rel": rel} for mem, rel in hits]}


def _weights_from_body(b: dict[str, Any]) -> "HybridWeights | None":
    """Build HybridWeights from a body ``graph`` field, or None to use the
    store default. Only the graph-proximity weight is exposed over HTTP; the
    dataclass keeps its default core weights, so ``graph`` is a pure add-on."""
    graph = _body_float(b, "graph")
    if graph is None:
        return None
    return HybridWeights(graph=graph)


def _expand_from_body(b: dict[str, Any]) -> "ExpandSpec | None":
    """Build ExpandSpec from a body ``expand`` object, or None for no
    expansion. Unspecified fields fall back to ExpandSpec defaults."""
    exp = b.get("expand")
    if not isinstance(exp, dict):
        return None
    kwargs: dict[str, Any] = {}
    rels = exp.get("rels")
    if isinstance(rels, list) and rels:
        kwargs["rels"] = tuple(str(r) for r in rels)
    if exp.get("depth") is not None:
        kwargs["depth"] = int(exp["depth"])
    if exp.get("cap") is not None:
        kwargs["cap"] = int(exp["cap"])
    if exp.get("score") is not None:
        kwargs["score"] = float(exp["score"])
    if exp.get("decay") is not None:
        kwargs["decay"] = float(exp["decay"])
    if exp.get("rerank") is not None:
        kwargs["rerank"] = bool(exp["rerank"])
    return ExpandSpec(**kwargs)


def _recall_params(q, b):
    """Pull recall arguments from a query string (GET) or JSON body (POST)."""
    if b:
        get = b.get
        return {
            "query": _require(b, "query"),
            "k": _body_int(b, "k", 5),
            "type": get("type"),
            "tag": get("tag"),
            "min_importance": _body_float(b, "min_importance"),
            "source": get("source"),
            "since": _time(get("since")),
            "until": _time(get("until")),
            "mode": get("mode") or "semantic",
            "overfetch": _body_int(b, "overfetch", 4),
            "include_superseded": _body_bool(b, "include_superseded"),
            "include_expired": _body_bool(b, "include_expired"),
            "explain": _body_bool(b, "explain"),
            # Advanced retrieval tuning is POST-body-only (nested/typed values
            # don't map cleanly onto a query string), matching the Go port's
            # POST surface: fusion, MMR diversity, dedup, the min_cosine gate,
            # the graph-proximity weight, and graph-walk expansion (incl.
            # rerank gating).
            "fusion": get("fusion") or "weighted",
            "diversity": _body_float(b, "diversity"),
            "dedup_threshold": _body_float(b, "dedup_threshold"),
            "min_cosine": _body_float(b, "min_cosine"),
            "weights": _weights_from_body(b),
            "expand": _expand_from_body(b),
            "lexical_index": get("lexical_index") or "pool",
            "min_trust": get("min_trust"),
            "as_of": _body_float(b, "as_of"),
            # touch=False lets eval/monitoring traffic recall without
            # inflating access counters, which feed decay reinforcement.
            "touch": _body_bool(b, "touch", True),
        }
    query = _qs_one(q, "query")
    if not query:
        raise HttpError(400, "missing required field: query")
    return {
        "query": query,
        "k": _as_int(_qs_one(q, "k"), 5),
        "type": _qs_one(q, "type"),
        "tag": _qs_one(q, "tag"),
        "min_importance": _as_float(_qs_one(q, "min_importance")),
        "source": _qs_one(q, "source"),
        "since": _time(_qs_one(q, "since")),
        "until": _time(_qs_one(q, "until")),
        "mode": _qs_one(q, "mode") or "semantic",
        "overfetch": _as_int(_qs_one(q, "overfetch"), 4),
        "include_superseded": _as_bool(_qs_one(q, "include_superseded")),
        "include_expired": _as_bool(_qs_one(q, "include_expired")),
        "explain": _as_bool(_qs_one(q, "explain")),
        # Plain scalars, so unlike the nested tuning knobs they map fine onto
        # a query string and are offered on GET too.
        "as_of": _as_float(_qs_one(q, "as_of")),
        "min_trust": _qs_one(q, "min_trust"),
        "lexical_index": _qs_one(q, "lexical_index") or "pool",
        "touch": _as_bool(_qs_one(q, "touch"), True),
    }


def _recall(store: MemoryStore, m, q, b):
    p = _recall_params(q, b)
    # The store's configured reranker (if any) applies automatically; a
    # callable can't cross the JSON boundary, so it's a library/server-config
    # concern, not a per-request param.
    if p.get("explain"):
        hits = store.recall(**p)
        return 200, {"results": [_hit_dict(mem, s, expl) for mem, s, expl in hits]}
    hits = store.recall(**p)
    return 200, {"results": [_hit_dict(mem, s) for mem, s in hits]}


def _compress_threshold(body: dict[str, Any]) -> float:
    # `or 0.30` would swallow an explicit 0.0 ("cluster everything") and
    # silently substitute the default — only None may mean "default".
    v = _body_float(body, "compress_threshold")
    return 0.30 if v is None else v


def _pack_payload(res: Any) -> dict[str, Any]:
    """Shared response shape for /recall_pack and /auto_context."""
    return {
        "text": res.text,
        "used_tokens": res.used_tokens,
        "budget": res.budget,
        "truncated": res.truncated,
        "items": [
            {**_mem_dict(p.memory), "score": round(p.score, 4), "tokens": p.tokens}
            for p in res.items
        ],
        "compressed_groups": [
            {"ids": [mem.id for mem in cg.memories], "text": cg.text,
             "tokens": cg.tokens, "count": len(cg.memories)}
            for cg in res.compressed_groups
        ],
    }


def _recall_pack(store: MemoryStore, m, q, b):
    res = store.recall_pack(
        query=_require(b, "query"),
        token_budget=_body_int(b, "token_budget", 800),
        type=b.get("type"),
        tag=b.get("tag"),
        min_importance=_body_float(b, "min_importance"),
        source=b.get("source"),
        since=_time(b.get("since")),
        until=_time(b.get("until")),
        mode=b.get("mode") or "hybrid",
        fusion=b.get("fusion") or "weighted",
        weights=_weights_from_body(b),
        diversity=_body_float(b, "diversity"),
        dedup_threshold=_body_float(b, "dedup_threshold"),
        min_cosine=_body_float(b, "min_cosine"),
        expand=_expand_from_body(b),
        lexical_index=b.get("lexical_index") or "pool",
        min_trust=b.get("min_trust"),
        as_of=_body_float(b, "as_of"),
        include_pinned=_body_bool(b, "include_pinned"),
        max_items=_body_int(b, "max_items", 50),
        include_superseded=_body_bool(b, "include_superseded"),
        header=b.get("header", "## Relevant memory"),
        compress=_body_bool(b, "compress"),
        compress_threshold=_compress_threshold(b),
        compress_min_group=_body_int(b, "compress_min_group", 2),
    )
    return 200, _pack_payload(res)


def _auto_context(store: MemoryStore, m, q, b):
    task = _require(b, "task")
    res = store.auto_context_pack(
        task=task,
        token_budget=_body_int(b, "token_budget", 800),
        max_phrases=_body_int(b, "max_phrases", 3),
        mode=b.get("mode") or "hybrid",
        min_cosine=_body_float(b, "min_cosine"),
        header=b.get("header", "## Relevant memory"),
        compress=_body_bool(b, "compress"),
        compress_threshold=_compress_threshold(b),
        compress_min_group=_body_int(b, "compress_min_group", 2),
        lexical_index=b.get("lexical_index") or "pool",
        min_trust=b.get("min_trust") or None,
    )
    payload = _pack_payload(res)
    payload["queries"] = [task] + extract_key_phrases(
        task, _body_int(b, "max_phrases", 3))
    return 200, payload


def _link(store: MemoryStore, m, q, b):
    try:
        store.link(src_id=_require(b, "src_id"), dst_id=_require(b, "dst_id"),
                   rel=b.get("rel", "related"))
    except KeyError as e:
        # unknown id → 404; a bad rel/self-link (ValueError) becomes 400
        # via the dispatcher.
        raise HttpError(404, e.args[0] if e.args else str(e))
    return 200, {"ok": True}


def _unlink(store: MemoryStore, m, q, b):
    removed = store.unlink(src_id=_require(b, "src_id"),
                           dst_id=_require(b, "dst_id"), rel=b.get("rel"))
    return 200, {"removed": removed}


def _supersede(store: MemoryStore, m, q, b):
    try:
        store.supersede(old_id=_require(b, "old_id"), new_id=_require(b, "new_id"))
    except KeyError as e:
        # unknown id → 404; self-supersede/cycle (ValueError) becomes 400
        # via the dispatcher.
        raise HttpError(404, e.args[0] if e.args else str(e))
    return 200, {"ok": True}


def _conflicts(store: MemoryStore, m, q, b):
    found = store.find_conflicts(memory_id=b.get("memory_id"),
                                 threshold=_body_float(b, "threshold"))
    return 200, {"conflicts": [
        {"kind": c.kind, "reason": c.reason, "similarity": c.similarity,
         "a": {"id": c.a.id, "text": c.a.text[:120], "type": c.a.type},
         "b": {"id": c.b.id, "text": c.b.text[:120], "type": c.b.type}}
        for c in found
    ]}


def _restore(store: MemoryStore, m, q, b):
    mid = _require(b, "memory_id")
    if store.get(mid) is None:
        raise HttpError(404, "memory not found")
    return 200, {"restored": store.restore(mid), "id": mid}


def _subgraph(store: MemoryStore, m, q, b):
    ids = b.get("memory_ids")
    if isinstance(ids, str):
        ids = [ids]
    if not isinstance(ids, list) or not ids:
        raise HttpError(400, "missing required field: memory_ids")
    graph = store.subgraph([str(i) for i in ids], depth=_body_int(b, "depth", 1))
    return 200, {
        "nodes": [_mem_dict(x) for x in graph.nodes.values()],
        "edges": [{"src": s, "dst": d, "rel": r} for s, d, r in graph.edges],
    }


def _undo(store: MemoryStore, m, q, b):
    """Reverse a journaled mutation: the newest, one by exact ts, or the newest
    touching a given memory."""
    ts = _body_float(b, "ts")
    mid = b.get("memory_id")
    if ts is not None:
        entry = store.journal.find_by_ts(ts)
        if entry is None:
            raise HttpError(404, f"no journal entry at ts={ts}")
    else:
        candidates = [e for e in store.journal.read()
                      if mid is None or store._entry_touches(e, str(mid))]
        if not candidates:
            raise HttpError(404, "no journal entry to undo")
        entry = candidates[-1]
    ok = store.undo(entry)
    return 200, {"ok": ok, "op": entry.op, "id": entry.id, "ts": entry.ts,
                 "actor": entry.actor}


def _nuke(store: MemoryStore, m, q, b):
    """Delete every memory. Guarded by an explicit confirm string so a stray
    DELETE can't empty the store."""
    if b.get("confirm") != "DELETE ALL":
        raise HttpError(400, 'refusing to nuke: pass {"confirm": "DELETE ALL"}')
    return 200, {"ok": True, "deleted": store.nuke()}


def _journal(store: MemoryStore, m, q, b):
    entries = list(store.journal.read(
        op=_qs_one(q, "op"),
        since=_time(_qs_one(q, "since")),
    ))
    n = _as_int(_qs_one(q, "n"), 20)
    if n > 0:
        entries = entries[-n:]
    return 200, {"count": len(entries),
                 "entries": [_journal_entry_dict(e) for e in entries]}


def _str_list(b: dict[str, Any], key: str) -> "list[str] | None":
    """Coerce a body field that may be a single string or a list of strings."""
    v = b.get(key)
    if v is None:
        return None
    if isinstance(v, str):
        return [v]
    if isinstance(v, list):
        return [str(x) for x in v]
    raise HttpError(400, f"{key}: expected a string or list of strings")


_ARCHIVE_DENIED = (
    "archive routes read/write server-side paths and require a configured "
    "auth token; use the `houkai export` / `houkai import` CLI commands for "
    "tokenless local use"
)


def _archive_route_allowed(path: str, auth_token: str | None) -> bool:
    """Whether /export & /import may run for this caller.

    These two routes reach past the store onto the server's filesystem
    (arbitrary read into the store / arbitrary .gz write), so unlike the
    store-scoped routes they are never offered without a configured token.
    A peer-address check is not a substitute: the stdlib handler accepts any
    Content-Type, so even a loopback bind is reachable by a drive-by
    cross-origin POST from a browser, and localhost proxies forward remote
    callers with a loopback peer address. Local tokenless workflows use the
    CLI, which talks to the store directly. An EMPTY token counts as
    unconfigured — its "Bearer " header carries no secret.
    """
    return path not in ("/export", "/import") or bool(auth_token)


def _export(store: MemoryStore, m, q, b):
    """Write a .ahkai archive to a server-side path. The path is resolved on
    the server; these routes require a configured auth token (see
    _archive_route_allowed)."""
    summary = store.export(
        _require(b, "path"),
        include_vectors=_body_bool(b, "include_vectors", True),
        include_superseded=_body_bool(b, "include_superseded"),
        types=_str_list(b, "types"),
        tags=_str_list(b, "tags"),
        since=_time(b.get("since")),
    )
    return 200, {"path": str(summary.path), "count": summary.count,
                 "bytes": summary.bytes, "elapsed": round(summary.elapsed, 4)}


def _import(store: MemoryStore, m, q, b):
    try:
        summary = store.import_(
            _require(b, "path"),
            on_conflict=b.get("on_conflict", "skip"),
            regenerate_vectors=_body_bool(b, "regenerate_vectors"),
            dry_run=_body_bool(b, "dry_run"),
        )
    except FileNotFoundError as e:
        raise HttpError(404, f"archive not found: {e}")
    except ImportConflictError as e:
        raise HttpError(409, str(e))
    except ImportError as e:
        raise HttpError(400, str(e))
    return 200, {
        "imported": summary.imported, "skipped": summary.skipped,
        "overwritten": summary.overwritten, "renamed": summary.renamed,
        "vectors_regenerated": summary.vectors_regenerated,
        "errors": [{"id": i, "error": msg} for i, msg in summary.errors],
    }


def _merge(store: MemoryStore, m, q, b):
    try:
        mem = store.merge(_require(b, "target_id"), _require(b, "other_id"),
                          separator=b.get("separator", "\n\n"))
    except MergeError as e:
        raise HttpError(404 if e.not_found else 400, str(e))
    return 200, _mem_dict(mem)


def _versions(store: MemoryStore, m, q, b):
    mid = m.group("id")
    if store.get(mid) is None:
        raise HttpError(404, "memory not found")
    return 200, {"id": mid, "versions": [
        {"ts": v.ts, "text": v.text, "tags": v.tags,
         "importance": v.importance, "source": v.source, "type": v.type}
        for v in store.versions(mid)
    ]}


def _tags(store: MemoryStore, m, q, b):
    include = _as_bool(_qs_one(q, "include_superseded"))
    return 200, {"tags": [{"tag": t, "count": n} for t, n
                          in store.list_tags(include_superseded=include)]}


def _rename_tag(store: MemoryStore, m, q, b):
    res = store.rename_tag(_require(b, "old"), _require(b, "new"))
    return 200, {"changed": res.changed, "tag": res.tag}


def _merge_tags(store: MemoryStore, m, q, b):
    sources = b.get("sources")
    if not isinstance(sources, list) or not sources:
        raise HttpError(400, "missing required field: sources")
    res = store.merge_tags([str(s) for s in sources], _require(b, "into"))
    return 200, {"changed": res.changed, "tag": res.tag}


def _delete_tag(store: MemoryStore, m, q, b):
    res = store.delete_tag(m.group("tag"))
    return 200, {"changed": res.changed, "tag": res.tag}


def _find_path(store: MemoryStore, m, q, b):
    hops = store.find_path(_require(b, "from_id"), _require(b, "to_id"),
                           max_depth=_body_int(b, "max_depth", 6))
    path = []
    for mid, rel in hops:
        mem = store.get(mid)
        path.append({"id": mid, "rel": rel,
                     "text": (mem.text[:120] if mem else None)})
    return 200, {"found": bool(hops), "length": max(0, len(hops) - 1),
                 "path": path}


def _trash(store: MemoryStore, m, q, b):
    mid = _require(b, "memory_id")
    trashed = store.trash(mid)
    if not trashed:
        raise HttpError(404, "memory not found")
    return 200, {"trashed": True, "id": mid}


def _trash_list(store: MemoryStore, m, q, b):
    return 200, {"entries": [
        {"memory_id": e.memory_id, "deleted_at": e.deleted_at,
         "actor": e.actor, "text": (e.memory.get("text") or "")[:200]}
        for e in store.trash_list()
    ]}


def _trash_restore(store: MemoryStore, m, q, b):
    mem = store.trash_restore(_require(b, "memory_id"))
    if mem is None:
        raise HttpError(404, "not in the trash")
    return 200, _mem_dict(mem)


def _trash_purge(store: MemoryStore, m, q, b):
    """Drop one entry, apply a retention cutoff, or empty the trash.

    ``older_than_days`` is honoured here rather than ignored: read as a plain
    "no memory_id" request it would fall through to emptying the *whole* trash,
    so a client asking to reclaim month-old entries would irreversibly destroy
    every recoverable memory instead. The Go port and the MCP tool both take
    the two arguments, and both refuse them together.
    """
    memory_id = b.get("memory_id") or None
    older_than = _body_float(b, "older_than_days")
    if memory_id is not None and older_than is not None:
        raise HttpError(400,
                        "pass either memory_id or older_than_days, not both")
    if older_than is not None:
        return 200, {"purged": store.trash_purge_expired(older_than)}
    return 200, {"purged": store.trash_purge(memory_id)}


Route = tuple[str, "re.Pattern[str]", Callable, bool]  # method, pat, fn, needs_body

_ROUTES: list[Route] = [
    ("GET",    re.compile(r"^/health$"),                         _health,      False),
    ("GET",    re.compile(r"^/ready$"),                          _ready,       False),
    ("GET",    re.compile(r"^/stats$"),                          _stats,       False),
    ("GET",    re.compile(r"^/metrics$"),                        _metrics,     False),
    ("GET",    re.compile(r"^/memories$"),                       _list,        False),
    ("POST",   re.compile(r"^/memories$"),                       _remember,    True),
    ("POST",   re.compile(r"^/memories/batch$"),                 _remember_many, True),
    ("GET",    re.compile(r"^/memories/(?P<id>[^/]+)$"),         _get_one,     False),
    ("PATCH",  re.compile(r"^/memories/(?P<id>[^/]+)$"),         _edit,        True),
    ("DELETE", re.compile(r"^/memories/(?P<id>[^/]+)$"),         _forget,      False),
    ("GET",    re.compile(r"^/memories/(?P<id>[^/]+)/neighbors$"), _neighbors, False),
    ("GET",    re.compile(r"^/memories/(?P<id>[^/]+)/history$"), _history,     False),
    ("GET",    re.compile(r"^/memories/(?P<id>[^/]+)/at$"),      _get_at,      False),
    ("GET",    re.compile(r"^/state_at$"),                       _state_at,    False),
    ("POST",   re.compile(r"^/purge_expired$"),                  _purge_expired, True),
    ("GET",    re.compile(r"^/recall$"),                         _recall,      False),
    ("POST",   re.compile(r"^/recall$"),                         _recall,      True),
    ("POST",   re.compile(r"^/recall_pack$"),                    _recall_pack, True),
    ("POST",   re.compile(r"^/auto_context$"),                   _auto_context, True),
    ("POST",   re.compile(r"^/links$"),                          _link,        True),
    ("POST",   re.compile(r"^/unlink$"),                         _unlink,      True),
    ("POST",   re.compile(r"^/supersede$"),                      _supersede,   True),
    ("POST",   re.compile(r"^/restore$"),                        _restore,     True),
    ("POST",   re.compile(r"^/conflicts$"),                      _conflicts,   True),
    ("POST",   re.compile(r"^/subgraph$"),                       _subgraph,    True),
    ("POST",   re.compile(r"^/undo$"),                           _undo,        True),
    ("POST",   re.compile(r"^/nuke$"),                           _nuke,        True),
    ("GET",    re.compile(r"^/journal$"),                        _journal,     False),
    ("POST",   re.compile(r"^/export$"),                         _export,      True),
    ("POST",   re.compile(r"^/import$"),                         _import,      True),
    ("POST",   re.compile(r"^/merge$"),                          _merge,       True),
    ("GET",    re.compile(r"^/memories/(?P<id>[^/]+)/versions$"), _versions,   False),
    ("GET",    re.compile(r"^/tags$"),                           _tags,        False),
    ("POST",   re.compile(r"^/tags/rename$"),                    _rename_tag,  True),
    ("POST",   re.compile(r"^/tags/merge$"),                     _merge_tags,  True),
    ("DELETE", re.compile(r"^/tags/(?P<tag>[^/]+)$"),            _delete_tag,  False),
    ("POST",   re.compile(r"^/find_path$"),                      _find_path,   True),
    ("POST",   re.compile(r"^/trash$"),                          _trash,       True),
    ("GET",    re.compile(r"^/trash$"),                          _trash_list,  False),
    ("POST",   re.compile(r"^/trash/restore$"),                  _trash_restore, True),
    ("POST",   re.compile(r"^/trash/purge$"),                    _trash_purge, True),
]

_MAX_BODY = 4 * 1024 * 1024  # 4 MiB cap on request bodies


def build_handler(
    store: MemoryStore,
    *,
    auth_token: str | None = None,
) -> type[BaseHTTPRequestHandler]:
    """Return a request-handler class bound to *store* and an optional token."""
    # An empty token is a misconfiguration, not a credential: it would make
    # every route "protected" by a forgeable bare "Bearer " header (and count
    # as configured for the archive gate). Normalize it to no-auth.
    auth_token = auth_token or None

    # ThreadingHTTPServer dispatches each request on its own thread, but
    # MemoryStore mutations (link/unlink/supersede and the access-count bump in
    # _touch) are read-modify-write against ChromaDB and so race under
    # concurrency — concurrent writers clobber each other's updates. Serialise
    # all store access through one lock; ChromaDB already serialises internally,
    # so the throughput cost is negligible.
    store_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        server_version = "AIHoukai-HTTP"
        protocol_version = "HTTP/1.1"

        # quieter logging — one tidy line per request
        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def _send(self, status: int, payload: Any) -> None:
            self._drain_unread_body()
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            if self.close_connection:
                self.send_header("Connection", "close")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _drain_unread_body(self) -> None:
            # HTTP/1.1 keeps connections alive, so a request body that was
            # never read (404/405 path, 401 rejection, bodied GET/DELETE …)
            # stays in the socket buffer and gets parsed as the *start of the
            # next request*, desyncing every client that pools connections.
            # Read and discard it; if it's too large to drain cheaply, close
            # the connection instead.
            if self._body_consumed:
                return
            self._body_consumed = True
            try:
                length = int(self.headers.get("Content-Length", 0) or 0)
            except (TypeError, ValueError):
                length = 0
            if length <= 0:
                return
            if length > _MAX_BODY:
                self.close_connection = True
                return
            remaining = length
            while remaining > 0:
                chunk = self.rfile.read(min(remaining, 65536))
                if not chunk:
                    break
                remaining -= len(chunk)

        def _authorized(self, path: str) -> bool:
            if auth_token is None or path in ("/health", "/ready"):
                return True
            header = self.headers.get("Authorization", "")
            # Constant-time compare so a wrong token can't be recovered by
            # timing. Compare BYTES: the str overload raises TypeError on
            # non-ASCII input, which would crash the request thread with no
            # response instead of returning 401.
            return hmac.compare_digest(
                header.encode("utf-8"), f"Bearer {auth_token}".encode("utf-8")
            )

        def _read_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", 0) or 0)
            if length <= 0:
                return {}
            if length > _MAX_BODY:
                raise HttpError(413, "request body too large")
            raw = self.rfile.read(length)
            self._body_consumed = True
            try:
                data = json.loads(raw or b"{}")
            except json.JSONDecodeError as exc:
                raise HttpError(400, f"invalid JSON body: {exc}") from exc
            if not isinstance(data, dict):
                raise HttpError(400, "JSON body must be an object")
            return data

        def _dispatch(self) -> None:
            self._body_consumed = False
            parts = urlsplit(self.path)
            path = parts.path.rstrip("/") or "/"
            query = parse_qs(parts.query)

            if not self._authorized(path):
                self._send(401, {"error": "unauthorized"})
                return

            if not _archive_route_allowed(path, auth_token):
                self._send(403, {"error": _ARCHIVE_DENIED})
                return

            # HEAD is GET without a body (_send suppresses it) — without this
            # mapping every HEAD, including HEAD /health probes, was a 405.
            command = "GET" if self.command == "HEAD" else self.command

            matched_path = False
            for method, pattern, fn, needs_body in _ROUTES:
                match = pattern.match(path)
                if not match:
                    continue
                matched_path = True
                if method != command:
                    continue
                try:
                    body = self._read_body() if needs_body else {}
                    with store_lock:
                        status, payload = fn(store, match, query, body)
                    self._send(status, payload)
                except HttpError as e:
                    self._send(e.status, {"error": e.message})
                except ValueError as e:
                    # Store-level validation (bad mode/type/rel/polarity …)
                    # is caller error, not an internal fault.
                    self._send(400, {"error": str(e)})
                except Exception:  # noqa: BLE001 — surface as 500, keep serving
                    # Don't leak internals (exception type/message/trace) to the
                    # client; tag with a request id so it can be correlated to a
                    # server-side log if one is wired up.
                    rid = uuid.uuid4().hex[:12]
                    self._send(500, {"error": "internal server error",
                                     "request_id": rid})
                return

            if matched_path:
                self._send(405, {"error": "method not allowed"})
            else:
                self._send(404, {"error": "not found"})

        # all verbs funnel through one dispatcher
        do_GET = _dispatch
        do_POST = _dispatch
        do_PATCH = _dispatch
        do_DELETE = _dispatch
        do_HEAD = _dispatch

    return Handler


class _Server(ThreadingHTTPServer):
    # The stdlib default listen backlog is 5, so a burst of simultaneous
    # connections gets reset by the kernel (ECONNRESET) before accept(). Raise
    # it so concurrent clients are queued rather than dropped. daemon_threads
    # lets the process exit promptly without joining in-flight workers.
    request_queue_size = 128
    daemon_threads = True


def make_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8077,
    store: MemoryStore | None = None,
    path: str = "./.chroma",
    collection: str = "ai_houkai",
    auth_token: str | None = None,
) -> ThreadingHTTPServer:
    """Construct (but do not start) a threaded HTTP server.

    Pass an existing *store* to reuse it, or let one be created from *path* /
    *collection*.  The store's actor is set to ``"http"`` so journal entries are
    attributable to this front-end.
    """
    if store is None:
        store = MemoryStore(path=path, collection=collection, actor="http")
    handler = build_handler(store, auth_token=auth_token)
    return _Server((host, port), handler)


def serve(
    *,
    host: str = "127.0.0.1",
    port: int = 8077,
    store: MemoryStore | None = None,
    path: str = "./.chroma",
    collection: str = "ai_houkai",
    auth_token: str | None = None,
) -> None:
    """Create and run the HTTP server until interrupted (blocking)."""
    httpd = make_server(
        host=host, port=port, store=store, path=path,
        collection=collection, auth_token=auth_token,
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def run() -> None:
    """Console-script entrypoint (``ai-houkai-serve``).

    Configured entirely through the environment so it needs no CLI deps:
      AI_HOUKAI_PATH         Chroma store path        (default ./.chroma)
      AI_HOUKAI_COLLECTION   collection name          (default ai_houkai)
      AI_HOUKAI_HTTP_HOST    bind address             (default 127.0.0.1)
      AI_HOUKAI_HTTP_PORT    bind port                (default 8077)
      AI_HOUKAI_HTTP_TOKEN   optional bearer token
    """

    serve(
        host=os.environ.get("AI_HOUKAI_HTTP_HOST", "127.0.0.1"),
        port=int(os.environ.get("AI_HOUKAI_HTTP_PORT", "8077")),
        # expanduser: AI_HOUKAI_PATH=~/mem must not create a literal ./~ dir
        path=os.path.expanduser(os.environ.get("AI_HOUKAI_PATH", "./.chroma")),
        collection=os.environ.get("AI_HOUKAI_COLLECTION", "ai_houkai"),
        auth_token=os.environ.get("AI_HOUKAI_HTTP_TOKEN") or None,
    )


if __name__ == "__main__":
    run()
