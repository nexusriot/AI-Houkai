"""Stats command."""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from ai_houkai.cli import config as _config

# Decay defaults mirror DecayEngine.__init__ so the health view is consistent
# with the engine's behaviour when no custom engine has been constructed. The
# effective values are loaded from the [maintenance.decay] config so the report
# matches what `houkai prune` / the maintenance daemon would actually remove.
_DEFAULT_DECAY_RATE     = 0.1
_DEFAULT_MIN_SCORE      = 0.05
# DecayEngine.prune() never removes these types regardless of score, so they
# must not be reported as "at risk" either.
_DEFAULT_PROTECT_TYPES  = ("procedural",)

_BUCKET_LABELS = ["0.0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"]


def _decay_score(
    importance: float,
    last_accessed: float,
    decay_rate: float,
    access_count: int = 0,
    frequency_weight: float = 0.0,
) -> float:
    """Mirror DecayEngine.score exactly, including frequency reinforcement."""
    days = max(0.0, (time.time() - last_accessed) / 86_400.0)
    base = importance * math.exp(-decay_rate * days)
    if frequency_weight:
        base *= 1.0 + frequency_weight * math.log1p(max(0, access_count))
    return base


def _bucket(score: float) -> int:
    return min(4, int(score * 5))


def stats(
    ctx: typer.Context,
    fmt: str  = typer.Option("auto", "--format", "-f", help="auto|json"),
    health: bool = typer.Option(
        False, "--health", "-H",
        help="Show a detailed health report: decay histogram, stale memories, "
             "cluster readiness, link density, and top recalled.",
    ),
    stale_days: int = typer.Option(
        30, "--stale-days",
        help="Days without access before a memory is considered stale (--health only).",
    ),
    decay_rate: Optional[float] = typer.Option(
        None, "--decay-rate",
        help="Decay λ for the health histogram (--health only). "
             "Defaults to [maintenance.decay].decay_rate from config.",
    ),
    frequency_weight: Optional[float] = typer.Option(
        None, "--frequency-weight",
        help="Recall-reinforcement weight for the health scores (--health only). "
             "Defaults to [maintenance.decay].frequency_weight from config.",
    ),
) -> None:
    """Show memory store statistics."""
    store = ctx.obj["store"]
    cfg   = ctx.obj["config"]

    # Pull the real decay parameters so --health reflects what prune() removes.
    mcfg = _config.load_maintenance()
    eff_decay_rate  = decay_rate if decay_rate is not None else mcfg.decay_rate
    eff_freq_weight = (frequency_weight if frequency_weight is not None
                       else mcfg.frequency_weight)

    memories    = store.list_recent(limit=999_999, include_superseded=True)
    active      = [m for m in memories if not m.superseded_by]
    superseded  = [m for m in memories if m.superseded_by]

    type_counts: Counter = Counter(m.type for m in active)
    tag_counts:  Counter = Counter()
    for m in active:
        for t in m.tags:
            tag_counts[t] += 1

    store_path = Path(cfg.store_path)
    store_size = (
        sum(f.stat().st_size for f in store_path.rglob("*") if f.is_file())
        if store_path.exists() else 0
    )

    data: dict = {
        "store_path":       str(store_path),
        "collection":       cfg.collection,
        "total":            len(memories),
        "active":           len(active),
        "superseded":       len(superseded),
        "by_type":          dict(type_counts),
        "top_tags":         dict(tag_counts.most_common(15)),
        "store_size_bytes": store_size,
    }

    if health:
        data["health"] = _compute_health(
            active, stale_days=stale_days, decay_rate=eff_decay_rate,
            min_score=mcfg.min_score, protect_types=tuple(mcfg.protect_types),
            frequency_weight=eff_freq_weight,
        )

    if fmt == "json":
        print(json.dumps(data, indent=2))
        return

    console = Console()
    _render_basic(console, data, store_path, active, superseded, type_counts, tag_counts, store_size)
    if health:
        _render_health(console, data["health"])


def _compute_health(
    active: list,
    *,
    stale_days: int,
    decay_rate: float,
    min_score: float = _DEFAULT_MIN_SCORE,
    protect_types: tuple[str, ...] = _DEFAULT_PROTECT_TYPES,
    frequency_weight: float = 0.0,
) -> dict:
    now       = time.time()
    stale_ts  = now - stale_days * 86_400.0
    at_risk_threshold = min_score

    # Decay score per memory — same formula as DecayEngine.score (incl. the
    # frequency-reinforcement term) so the histogram and at-risk count match
    # what the engine would actually prune.
    scored = [
        (m, _decay_score(m.importance, m.last_accessed, decay_rate,
                         m.access_count, frequency_weight))
        for m in active
    ]

    # Histogram buckets 0–1 in 5 bands
    hist = [0, 0, 0, 0, 0]
    for _, s in scored:
        hist[_bucket(s)] += 1

    # At-risk: below prune threshold AND not a protected type — this matches
    # DecayEngine.prune(), which skips protect_types regardless of score.
    at_risk = [
        m for m, s in scored
        if s < at_risk_threshold and m.type not in protect_types
    ]

    # Never recalled
    never_recalled = [m for m in active if m.access_count == 0]

    # Stale: last accessed before the cutoff
    stale = [m for m in active if m.last_accessed < stale_ts]

    # Reflection cluster readiness: unsuperseded episodics
    episodic_active = [m for m in active if m.type == "episodic"]

    # Top recalled (by access_count descending)
    top_recalled = sorted(active, key=lambda m: m.access_count, reverse=True)[:5]

    # Link density: average outgoing links per memory
    total_links = sum(len(m.links) for m in active)
    link_density = total_links / len(active) if active else 0.0

    # Avg importance
    avg_importance = (
        sum(m.importance for m in active) / len(active) if active else 0.0
    )

    return {
        "decay_histogram": {label: cnt for label, cnt in zip(_BUCKET_LABELS, hist)},
        "at_risk_count":          len(at_risk),
        "never_recalled_count":   len(never_recalled),
        "stale_count":            len(stale),
        "stale_days":             stale_days,
        "episodic_active_count":  len(episodic_active),
        "link_density":           round(link_density, 3),
        "total_links":            total_links,
        "avg_importance":         round(avg_importance, 3),
        "top_recalled": [
            {
                "id":           m.id[:8],
                "access_count": m.access_count,
                "text_snippet": m.text[:60] + ("…" if len(m.text) > 60 else ""),
            }
            for m in top_recalled
            if m.access_count > 0
        ],
    }


def _render_basic(
    console,
    data: dict,
    store_path,
    active: list,
    superseded: list,
    type_counts: Counter,
    tag_counts: Counter,
    store_size: int,
) -> None:
    console.print(f"[bold]Store[/]       {store_path}")
    console.print(f"[bold]Collection[/]  {data['collection']}")
    console.print(
        f"[bold]Total[/]       {data['total']}  "
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


def _render_health(console, h: dict) -> None:
    console.rule("[bold yellow]Health Report[/]")

    n_total = sum(h["decay_histogram"].values())
    at_risk_pct = (h["at_risk_count"] / n_total * 100) if n_total else 0.0
    stale_pct   = (h["stale_count"]   / n_total * 100) if n_total else 0.0

    at_risk_style = "red"    if at_risk_pct > 20 else ("yellow" if at_risk_pct > 5 else "green")
    stale_style   = "yellow" if stale_pct   > 30 else "green"

    console.print(
        f"[bold]Avg importance[/]  {h['avg_importance']:.2f}   "
        f"[bold]Link density[/]  {h['link_density']:.2f} links/memory"
    )
    console.print(
        f"[bold]At-risk[/]         [{at_risk_style}]{h['at_risk_count']}[/] "
        f"({at_risk_pct:.1f}% below prune threshold)   "
        f"[bold]Stale[/]  [{stale_style}]{h['stale_count']}[/] "
        f"({stale_pct:.1f}% idle >{h['stale_days']}d)"
    )
    console.print(
        f"[bold]Never recalled[/]  {h['never_recalled_count']}   "
        f"[bold]Episodic (ripe for reflection)[/]  {h['episodic_active_count']}"
    )

    t = Table(title="Decay score distribution", show_header=True, header_style="bold cyan")
    t.add_column("SCORE BAND")
    t.add_column("COUNT", justify="right")
    t.add_column("BAR")
    max_cnt = max(h["decay_histogram"].values()) if h["decay_histogram"] else 1
    for label, cnt in h["decay_histogram"].items():
        bar_len = round(cnt / max_cnt * 20) if max_cnt else 0
        bar = "█" * bar_len + "░" * (20 - bar_len)
        # 0.0–0.2 band is "at risk"
        style = "red" if label == "0.0–0.2" else "default"
        t.add_row(f"[{style}]{label}[/]", f"[{style}]{cnt}[/]", f"[{style}]{bar}[/]")
    console.print(t)

    if h["top_recalled"]:
        t2 = Table(title="Top recalled", show_header=True, header_style="bold cyan")
        t2.add_column("ID")
        t2.add_column("RECALLS", justify="right")
        t2.add_column("TEXT")
        for row in h["top_recalled"]:
            t2.add_row(row["id"], str(row["access_count"]), row["text_snippet"])
        console.print(t2)
