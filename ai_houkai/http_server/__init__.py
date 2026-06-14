"""Optional HTTP/REST front-end for an AI-Houkai memory store.

Exposes the same `MemoryStore` the MCP server and CLI use over a small JSON
API, so web apps, shell scripts and non-MCP agents can read and write memories
with a plain HTTP request.  Built on the standard library only — no extra
dependency beyond the core memory layer.

    from ai_houkai.http_server import serve
    serve(host="127.0.0.1", port=8077)

or from the CLI:  ``houkai serve --port 8077``
"""

from __future__ import annotations

from ai_houkai.http_server.server import build_handler, make_server, serve

__all__ = ["serve", "make_server", "build_handler"]
