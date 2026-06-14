"""Standard-library JSON HTTP server over a :class:`MemoryStore`.

Routes (all JSON in / JSON out):

    GET    /health                         liveness + memory count
    GET    /stats                          store statistics
    GET    /memories?limit=&include_superseded=
                                           recent memories (list_recent)
    POST   /memories                       store a memory (remember)
    GET    /memories/{id}                  fetch one memory
    DELETE /memories/{id}                  forget one memory
    GET    /memories/{id}/neighbors?rel=&direction=&depth=
                                           linked memories
    GET    /recall?query=&k=&type=&tag=&min_importance=&source=&since=&until=&mode=
    POST   /recall                         same, via JSON body
    POST   /recall_pack                    token-budgeted context block
    POST   /links        {src_id,dst_id,rel?}      add a directed link
    POST   /unlink       {src_id,dst_id,rel?}      remove link(s)
    POST   /supersede    {old_id,new_id}           soft-delete + supersede link
    POST   /conflicts    {memory_id?,threshold?}   duplicate / contradiction scan

Optional bearer-token auth: pass ``auth_token`` (or set ``AI_HOUKAI_HTTP_TOKEN``)
and every request must carry ``Authorization: Bearer <token>``.  ``/health`` is
always reachable so liveness probes work without the secret.

The handler is intentionally framework-free: a single regex routing table maps
``(method, path)`` to small functions taking ``(store, match, query, body)``.
"""

from __future__ import annotations

import json
import re
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Optional
from urllib.parse import parse_qs, urlsplit

from ai_houkai.memory_system import MemoryStore
from ai_houkai.memory_system.store import ConflictError
from ai_houkai.timeparse import parse_timestamp


class HttpError(Exception):
    """Raised by a route to short-circuit with a specific status + message."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


# ---- serialisation helpers ------------------------------------------------

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
        "links": [{"to": l.to, "rel": l.rel} for l in mem.links],
        "superseded_by": mem.superseded_by or None,
    }


def _hit_dict(mem: Any, score: float) -> dict[str, Any]:
    d = _mem_dict(mem)
    d["score"] = round(score, 4)
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


def _as_bool(value: Optional[str]) -> bool:
    return (value or "").lower() in ("1", "true", "yes", "on")


def _time(value: Any) -> Optional[float]:
    try:
        return parse_timestamp(value)
    except ValueError as exc:
        raise HttpError(400, str(exc)) from exc


def _require(body: dict[str, Any], key: str) -> Any:
    if key not in body or body[key] in (None, ""):
        raise HttpError(400, f"missing required field: {key}")
    return body[key]


# ---- route handlers -------------------------------------------------------
# Each takes (store, match, query, body) and returns (status, payload).

def _health(store: MemoryStore, m, q, b):
    return 200, {"status": "ok", "count": store.count(),
                 "collection": store.collection_name}


def _stats(store: MemoryStore, m, q, b):
    return 200, {"count": store.count(), "path": store.path,
                 "collection": store.collection_name}


def _list(store: MemoryStore, m, q, b):
    limit = _as_int(_qs_one(q, "limit"), 20)
    inc = _as_bool(_qs_one(q, "include_superseded"))
    mems = store.list_recent(limit=limit, include_superseded=inc)
    return 200, {"memories": [_mem_dict(x) for x in mems]}


def _remember(store: MemoryStore, m, q, b):
    text = _require(b, "text")
    try:
        mem = store.remember(
            text=text,
            type=b.get("type", "semantic"),
            tags=b.get("tags") or [],
            importance=b.get("importance"),
            source=b.get("source"),
            polarity=int(b.get("polarity", 0)),
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
    return 201, {"stored": True, **_mem_dict(mem)}


def _get_one(store: MemoryStore, m, q, b):
    mem = store._get_by_id(m.group("id"))
    if mem is None:
        raise HttpError(404, "memory not found")
    return 200, _mem_dict(mem)


def _forget(store: MemoryStore, m, q, b):
    ok = store.forget(m.group("id"))
    if not ok:
        raise HttpError(404, "memory not found")
    return 200, {"forgotten": True, "id": m.group("id")}


def _neighbors(store: MemoryStore, m, q, b):
    hits = store.neighbors(
        m.group("id"),
        rel=_qs_one(q, "rel"),
        direction=_qs_one(q, "direction") or "both",
        depth=_as_int(_qs_one(q, "depth"), 1),
    )
    return 200, {"neighbors": [{**_mem_dict(mem), "rel": rel} for mem, rel in hits]}


def _recall_params(q, b):
    """Pull recall arguments from a query string (GET) or JSON body (POST)."""
    if b:
        get = b.get
        return {
            "query": _require(b, "query"),
            "k": int(get("k", 5)),
            "type": get("type"),
            "tag": get("tag"),
            "min_importance": get("min_importance"),
            "source": get("source"),
            "since": _time(get("since")),
            "until": _time(get("until")),
            "mode": get("mode", "semantic"),
            "include_superseded": bool(get("include_superseded", False)),
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
        "include_superseded": _as_bool(_qs_one(q, "include_superseded")),
    }


def _recall(store: MemoryStore, m, q, b):
    p = _recall_params(q, b)
    hits = store.recall(**p)
    return 200, {"results": [_hit_dict(mem, s) for mem, s in hits]}


def _recall_pack(store: MemoryStore, m, q, b):
    res = store.recall_pack(
        query=_require(b, "query"),
        token_budget=int(b.get("token_budget", 800)),
        type=b.get("type"),
        tag=b.get("tag"),
        min_importance=b.get("min_importance"),
        source=b.get("source"),
        since=_time(b.get("since")),
        until=_time(b.get("until")),
        mode=b.get("mode", "hybrid"),
        max_items=int(b.get("max_items", 50)),
        include_superseded=bool(b.get("include_superseded", False)),
        header=b.get("header", "## Relevant memory"),
    )
    return 200, {
        "text": res.text,
        "used_tokens": res.used_tokens,
        "budget": res.budget,
        "truncated": res.truncated,
        "items": [
            {**_mem_dict(p.memory), "score": round(p.score, 4), "tokens": p.tokens}
            for p in res.items
        ],
    }


def _link(store: MemoryStore, m, q, b):
    try:
        store.link(src_id=_require(b, "src_id"), dst_id=_require(b, "dst_id"),
                   rel=b.get("rel", "related"))
    except (KeyError, ValueError) as e:
        raise HttpError(404, str(e))
    return 200, {"ok": True}


def _unlink(store: MemoryStore, m, q, b):
    removed = store.unlink(src_id=_require(b, "src_id"),
                           dst_id=_require(b, "dst_id"), rel=b.get("rel"))
    return 200, {"removed": removed}


def _supersede(store: MemoryStore, m, q, b):
    try:
        store.supersede(old_id=_require(b, "old_id"), new_id=_require(b, "new_id"))
    except (KeyError, ValueError) as e:
        raise HttpError(404, str(e))
    return 200, {"ok": True}


def _conflicts(store: MemoryStore, m, q, b):
    found = store.find_conflicts(memory_id=b.get("memory_id"),
                                 threshold=b.get("threshold"))
    return 200, {"conflicts": [
        {"kind": c.kind, "reason": c.reason, "similarity": c.similarity,
         "a": {"id": c.a.id, "text": c.a.text[:120], "type": c.a.type},
         "b": {"id": c.b.id, "text": c.b.text[:120], "type": c.b.type}}
        for c in found
    ]}


Route = tuple[str, "re.Pattern[str]", Callable, bool]  # method, pat, fn, needs_body

_ROUTES: list[Route] = [
    ("GET",    re.compile(r"^/health$"),                         _health,      False),
    ("GET",    re.compile(r"^/stats$"),                          _stats,       False),
    ("GET",    re.compile(r"^/memories$"),                       _list,        False),
    ("POST",   re.compile(r"^/memories$"),                       _remember,    True),
    ("GET",    re.compile(r"^/memories/(?P<id>[^/]+)$"),         _get_one,     False),
    ("DELETE", re.compile(r"^/memories/(?P<id>[^/]+)$"),         _forget,      False),
    ("GET",    re.compile(r"^/memories/(?P<id>[^/]+)/neighbors$"), _neighbors, False),
    ("GET",    re.compile(r"^/recall$"),                         _recall,      False),
    ("POST",   re.compile(r"^/recall$"),                         _recall,      True),
    ("POST",   re.compile(r"^/recall_pack$"),                    _recall_pack, True),
    ("POST",   re.compile(r"^/links$"),                          _link,        True),
    ("POST",   re.compile(r"^/unlink$"),                         _unlink,      True),
    ("POST",   re.compile(r"^/supersede$"),                      _supersede,   True),
    ("POST",   re.compile(r"^/conflicts$"),                      _conflicts,   True),
]

_MAX_BODY = 4 * 1024 * 1024  # 4 MiB cap on request bodies


def build_handler(
    store: MemoryStore,
    *,
    auth_token: str | None = None,
) -> type[BaseHTTPRequestHandler]:
    """Return a request-handler class bound to *store* and an optional token."""

    class Handler(BaseHTTPRequestHandler):
        server_version = "AIHoukai-HTTP"
        protocol_version = "HTTP/1.1"

        # quieter logging — one tidy line per request
        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def _send(self, status: int, payload: Any) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _authorized(self, path: str) -> bool:
            if auth_token is None or path == "/health":
                return True
            header = self.headers.get("Authorization", "")
            return header == f"Bearer {auth_token}"

        def _read_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", 0) or 0)
            if length <= 0:
                return {}
            if length > _MAX_BODY:
                raise HttpError(413, "request body too large")
            raw = self.rfile.read(length)
            try:
                data = json.loads(raw or b"{}")
            except json.JSONDecodeError as exc:
                raise HttpError(400, f"invalid JSON body: {exc}") from exc
            if not isinstance(data, dict):
                raise HttpError(400, "JSON body must be an object")
            return data

        def _dispatch(self) -> None:
            parts = urlsplit(self.path)
            path = parts.path.rstrip("/") or "/"
            query = parse_qs(parts.query)

            if not self._authorized(path):
                self._send(401, {"error": "unauthorized"})
                return

            matched_path = False
            for method, pattern, fn, needs_body in _ROUTES:
                match = pattern.match(path)
                if not match:
                    continue
                matched_path = True
                if method != self.command:
                    continue
                try:
                    body = self._read_body() if needs_body else {}
                    status, payload = fn(store, match, query, body)
                    self._send(status, payload)
                except HttpError as e:
                    self._send(e.status, {"error": e.message})
                except Exception as e:  # noqa: BLE001 — surface as 500, keep serving
                    self._send(500, {"error": f"{type(e).__name__}: {e}"})
                return

            if matched_path:
                self._send(405, {"error": "method not allowed"})
            else:
                self._send(404, {"error": "not found"})

        # all verbs funnel through one dispatcher
        do_GET = _dispatch
        do_POST = _dispatch
        do_DELETE = _dispatch
        do_HEAD = _dispatch

    return Handler


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
    return ThreadingHTTPServer((host, port), handler)


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
        path=os.environ.get("AI_HOUKAI_PATH", "./.chroma"),
        collection=os.environ.get("AI_HOUKAI_COLLECTION", "ai_houkai"),
        auth_token=os.environ.get("AI_HOUKAI_HTTP_TOKEN") or None,
    )


if __name__ == "__main__":
    run()
