from __future__ import annotations

import json

import typer

from ai_houkai.cli import output as out


def doctor(
    ctx: typer.Context,
    as_json: bool = typer.Option(False, "--json", help="Emit the report as JSON"),
) -> None:
    """Diagnose store, embedding backend, and configuration health.

    Actively probes the embedding backend (which is otherwise only contacted
    lazily on the first write/recall), checks the store is reachable, guards
    against an embedding-dimension mismatch between the model and the stored
    vectors, and reports the resolved configuration. Exits non-zero if any
    check fails, so it doubles as a scriptable readiness gate.
    """
    store = ctx.obj["store"]
    checks: list[dict] = []

    def add(name: str, ok: bool, **detail: object) -> None:
        checks.append({"name": name, "ok": ok, **detail})

    # 1. Resolved configuration.
    add(
        "config", True,
        store_path=store.path,
        collection=store.collection_name,
        embedding_model=store.embedding_model,
        embedder=store.embedder_name,
    )

    # 2. Store reachable.
    count: int | None = None
    try:
        count = store.count()
        add("store", True, count=count)
    except Exception as exc:  # noqa: BLE001 — report, don't crash the diagnosis
        add("store", False, error=f"{type(exc).__name__}: {exc}")

    # 3. Embedding backend reachability + dimension + latency.
    probe = store.probe_embedding()
    add(
        "embedder", bool(probe.get("ok")),
        model=store.embedder_name,
        dim=probe.get("dim"),
        latency_ms=probe.get("latency_ms"),
        error=probe.get("error"),
    )

    # 4. Vector index readable, and its dimension matching the probe's — a
    #    mismatch makes cosine similarity silently meaningless, and an index
    #    that cannot be read at all makes every query answer with nothing
    #    while the collection still counts and lists its memories. Reading one
    #    row's embedding settles both. This used to swallow the read error and
    #    skip the check, so a collection that could not answer a single query
    #    was diagnosed "OK — ready".
    stored_dim: int | None = None
    if count:
        hint = "run `houkai rebuild-vectors` to rebuild it from the stored text"
        try:
            res = store.collection.get(limit=1, include=["embeddings"])
            embs = res.get("embeddings")
            if embs is None or len(embs) == 0:
                add("vector_index", False, memories=count,
                    error=f"the collection holds text but no vectors — {hint}")
            else:
                stored_dim = len(embs[0])
                add("vector_index", True, memories=count)
        except Exception as exc:  # noqa: BLE001 — report, don't crash the diagnosis
            add("vector_index", False, memories=count,
                error=f"{type(exc).__name__}: {exc} — {hint}")
    if stored_dim is not None and probe.get("ok"):
        add(
            "embed_dim", stored_dim == probe.get("dim"),
            probe_dim=probe.get("dim"), stored_dim=stored_dim,
        )

    # 5. Audit journal readable + freshness.
    jp = store.journal.path
    exists = jp.exists()
    detail: dict[str, object] = {
        "enabled": store._journal_enabled, "path": str(jp), "exists": exists,
    }
    if exists:
        st = jp.stat()
        detail["size_bytes"] = st.st_size
        detail["last_write_age"] = out.fmt_age(st.st_mtime)
    add("journal", True, **detail)

    ok = all(c["ok"] for c in checks)
    report = {"ok": ok, "checks": checks}

    if as_json:
        typer.echo(json.dumps(report, indent=2))
    else:
        for c in checks:
            mark = "[ok]" if c["ok"] else "[!!]"
            extras = " ".join(
                f"{k}={v}" for k, v in c.items()
                if k not in ("name", "ok") and v is not None
            )
            typer.echo(f"{mark} {c['name']:<11} {extras}")
        typer.echo("")
        typer.echo("OK — ready" if ok else "NOT READY — see failures above")

    if not ok:
        raise typer.Exit(1)
