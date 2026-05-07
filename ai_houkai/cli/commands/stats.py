"""Stats command."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

def stats(
    ctx: typer.Context,
    fmt: str = typer.Option("auto", "--format", "-f", help="auto|json"),
) -> None:
    """Show memory store statistics."""
    store = ctx.obj["store"]
    cfg = ctx.obj["config"]

    memories = store.list_recent(limit=999_999, include_superseded=True)
    active = [m for m in memories if not m.superseded_by]
    superseded = [m for m in memories if m.superseded_by]

    type_counts = Counter(m.type for m in active)
    tag_counts: Counter = Counter()
    for m in active:
        for t in m.tags:
            tag_counts[t] += 1

    store_path = Path(cfg.store_path)
    store_size = (
        sum(f.stat().st_size for f in store_path.rglob("*") if f.is_file())
        if store_path.exists() else 0
    )

    data = {
        "store_path": str(store_path),
        "collection": cfg.collection,
        "total": len(memories),
        "active": len(active),
        "superseded": len(superseded),
        "by_type": dict(type_counts),
        "top_tags": dict(tag_counts.most_common(15)),
        "store_size_bytes": store_size,
    }

    if fmt == "json":
        print(json.dumps(data, indent=2))
        return

    try:

        console = Console()

        console.print(f"[bold]Store[/]       {store_path}")
        console.print(f"[bold]Collection[/]  {cfg.collection}")
        console.print(
            f"[bold]Total[/]       {len(memories)}  "
            f"([green]{len(active)} active[/], [dim]{len(superseded)} superseded[/])"
        )
        console.print(f"[bold]Size[/]        {store_size / 1024:.1f} KB")

        if type_counts:
            t = Table(title="By type", show_header=True, header_style="bold cyan")
            t.add_column("TYPE")
            t.add_column("COUNT", justify="right")
            for tp, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
                t.add_row(tp, str(cnt))
            console.print(t)

        if tag_counts:
            t2 = Table(title="Top tags", show_header=True, header_style="bold cyan")
            t2.add_column("TAG")
            t2.add_column("COUNT", justify="right")
            for tg, cnt in tag_counts.most_common(15):
                t2.add_row(tg, str(cnt))
            console.print(t2)

    except ImportError:
        print(json.dumps(data, indent=2))
