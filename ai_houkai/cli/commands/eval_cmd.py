"""Retrieval-quality evaluation against a gold set.

`ai_houkai.eval` has always been able to score a ranking config; until now it
was reachable from no surface at all, so the graph damping, the β=0.20 lexical
weight, the MMR defaults and the `graph` weight were all set by intuition with
no way to measure a change. This command is the ruler.

Gold sets are JSONL — one case per line — to keep the stdlib-only ethos::

    {"query": "how do we deploy", "relevant_ids": ["<uuid>"]}
    {"query": "test isolation", "relevant_ids": ["<uuid>", "<uuid>"], "k": 3}

`relevant_ids` accepts 8-char prefixes as well as full UUIDs, so a gold set can
be written by eye from `houkai list`.
"""

from __future__ import annotations

import json as jsonlib
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from ai_houkai.cli import output as out
from ai_houkai.eval import EvalCase, evaluate
from ai_houkai.memory_system import ExpandSpec, HybridWeights


def load_goldset(path: Path, store=None) -> list[EvalCase]:
    """Parse a JSONL gold set into EvalCases.

    Blank lines and ``#`` comments are skipped. When *store* is given, id
    prefixes are resolved to full UUIDs — an unresolvable id is an error, not a
    silent zero score, because a typo would otherwise look like a ranking
    regression.
    """
    cases: list[EvalCase] = []
    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            obj = jsonlib.loads(line)
        except jsonlib.JSONDecodeError as e:
            raise ValueError(f"{path}:{lineno}: invalid JSON — {e}") from e
        if not isinstance(obj, dict):
            raise ValueError(f"{path}:{lineno}: expected a JSON object")
        query = obj.get("query")
        if not query:
            raise ValueError(f"{path}:{lineno}: missing 'query'")
        ids = obj.get("relevant_ids") or []
        if not isinstance(ids, list) or not ids:
            raise ValueError(f"{path}:{lineno}: 'relevant_ids' must be a non-empty list")
        if store is not None:
            resolved = []
            for mid in ids:
                try:
                    resolved.append(out.resolve_id_prefix(store, str(mid)))
                except ValueError as e:
                    raise ValueError(f"{path}:{lineno}: {e}") from e
            ids = resolved
        cases.append(EvalCase(
            query=str(query),
            relevant_ids=[str(i) for i in ids],
            k=obj.get("k"),
            mode=obj.get("mode"),
        ))
    if not cases:
        raise ValueError(f"{path}: no evaluation cases found")
    return cases


def eval_cmd(
    ctx: typer.Context,
    goldset: Path = typer.Argument(..., help="JSONL gold set (see --help)"),
    k: int = typer.Option(5, "-k", help="Default top-k when a case omits it"),
    mode: str = typer.Option("hybrid", "--mode", help="semantic | hybrid"),
    fusion: str = typer.Option("weighted", "--fusion", help="weighted | rrf"),
    graph: Optional[float] = typer.Option(
        None, "--graph", help="Graph-proximity weight (hybrid only)"),
    diversity: Optional[float] = typer.Option(
        None, "--diversity", help="MMR λ in [0,1]"),
    dedup_threshold: Optional[float] = typer.Option(
        None, "--dedup", help="Drop near-duplicates above this cosine"),
    min_cosine: Optional[float] = typer.Option(
        None, "--min-cosine", help="Absolute relevance floor"),
    expand_rerank: bool = typer.Option(
        False, "--expand-rerank",
        help="Merge graph-expanded nodes into the pool before top-k"),
    per_case: bool = typer.Option(
        False, "--per-case", help="Show a row per query"),
    json: bool = typer.Option(False, "--json", help="Emit raw JSON"),
) -> None:
    """Score retrieval quality against a gold set (recall/precision/MRR/MAP/nDCG).

    Recall runs read-only (touch=False), so evaluating never perturbs
    access-count or recency. Pass ranking flags to A/B a config:

        houkai eval gold.jsonl --mode hybrid --fusion rrf
        houkai eval gold.jsonl --graph 0.15 --expand-rerank
    """
    store = ctx.obj["store"]
    try:
        cases = load_goldset(goldset, store)
    except (OSError, ValueError) as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    kwargs: dict = {"fusion": fusion}
    if graph is not None:
        # Start from the defaults so a lone --graph does not zero the core
        # weights (HybridWeights rejects that outright).
        kwargs["weights"] = HybridWeights(graph=graph)
    if diversity is not None:
        kwargs["diversity"] = diversity
    if dedup_threshold is not None:
        kwargs["dedup_threshold"] = dedup_threshold
    if min_cosine is not None:
        kwargs["min_cosine"] = min_cosine
    if expand_rerank:
        kwargs["expand"] = ExpandSpec(rerank=True)

    try:
        result = evaluate(store, cases, default_k=k, default_mode=mode, **kwargs)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    if json:
        typer.echo(jsonlib.dumps({
            "n": result.n, "k": result.k,
            "recall_at_k": result.recall_at_k,
            "precision_at_k": result.precision_at_k,
            "mrr": result.mrr, "map": result.map,
            "ndcg_at_k": result.ndcg_at_k,
            "config": {"mode": mode, "fusion": fusion, "graph": graph,
                       "diversity": diversity, "dedup_threshold": dedup_threshold,
                       "min_cosine": min_cosine, "expand_rerank": expand_rerank},
            "per_case": result.per_case,
        }, indent=2))
        return

    table = Table(title=f"Retrieval eval — {goldset.name} ({result.n} cases)")
    table.add_column("metric")
    table.add_column("value", justify="right")
    ks = "mixed" if result.k < 0 else str(result.k)
    for label, value in (
        (f"recall@{ks}", result.recall_at_k),
        (f"precision@{ks}", result.precision_at_k),
        ("MRR", result.mrr),
        ("MAP", result.map),
        (f"nDCG@{ks}", result.ndcg_at_k),
    ):
        table.add_row(label, f"{value:.3f}")
    Console().print(table)

    if per_case:
        detail = Table(title="Per case")
        detail.add_column("query")
        detail.add_column("k", justify="right")
        detail.add_column("recall", justify="right")
        detail.add_column("RR", justify="right")
        detail.add_column("nDCG", justify="right")
        for case in result.per_case:
            detail.add_row(
                case["query"][:50], str(case["k"]),
                f"{case['recall_at_k']:.3f}", f"{case['rr']:.3f}",
                f"{case['ndcg_at_k']:.3f}",
            )
        Console().print(detail)
