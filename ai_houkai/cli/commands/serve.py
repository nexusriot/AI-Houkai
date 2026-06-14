from __future__ import annotations

import os
from typing import Optional

import typer

from ai_houkai.http_server import serve as run_server


def serve(
    ctx: typer.Context,
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Bind address"),
    port: int = typer.Option(8077, "--port", "-p", help="Bind port"),
    token: Optional[str] = typer.Option(
        None, "--token",
        envvar="AI_HOUKAI_HTTP_TOKEN",
        help="Require 'Authorization: Bearer <token>' on every request (except /health)",
    ),
) -> None:
    """Serve the memory store over a JSON HTTP/REST API.

    Reuses the store selected by --store/--collection. Examples:

      houkai serve --port 8077
      curl -s localhost:8077/health
      curl -s 'localhost:8077/recall?query=auth&k=3&since=7d'
      curl -s localhost:8077/memories -d '{"text":"remember this","type":"semantic"}'
    """
    store = ctx.obj["store"]
    # Attribute journal writes to the HTTP front-end while it owns the store.
    store._actor = "http"

    auth = "(token required)" if token else "(open — no auth)"
    typer.echo(
        f"AI-Houkai HTTP API → http://{host}:{port}  {auth}\n"
        f"store: {store.path}  collection: {store.collection_name}\n"
        f"Press Ctrl-C to stop.",
        err=True,
    )
    run_server(host=host, port=port, store=store, auth_token=token)
