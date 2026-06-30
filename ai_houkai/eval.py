"""Retrieval-quality metrics and a tiny evaluation harness.

Dependency-free (stdlib only), matching the project's local-first philosophy.
Lets you measure ranking changes against a small gold set so retrieval tweaks
(weights, fusion, diversity, …) are no longer flying blind.

    from ai_houkai.eval import EvalCase, evaluate

    cases = [
        EvalCase(query="how do we deploy", relevant_ids=[deploy_mem.id]),
        EvalCase(query="test isolation",   relevant_ids=[a.id, b.id], k=3),
    ]
    result = evaluate(store, cases, default_mode="hybrid")
    print(result.recall_at_k, result.mrr, result.ndcg_at_k)

The metric functions also work standalone on any ranked list of ids:

    recall_at_k(["a", "b", "c"], relevant={"b"}, k=2)   # -> 1.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

__all__ = [
    "recall_at_k", "precision_at_k", "reciprocal_rank", "average_precision",
    "dcg_at_k", "ndcg_at_k", "EvalCase", "EvalResult", "evaluate",
]


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of relevant ids that appear in the top-k retrieved.

    Counts *distinct* matches so a duplicated retrieved id can't push the value
    above 1.0.
    """
    rel = set(relevant)
    if not rel:
        return 0.0
    top = set(retrieved[:k])
    return len(top & rel) / len(rel)


def precision_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of the top-k retrieved that are relevant."""
    rel = set(relevant)
    top = list(retrieved[:k])
    if not top:
        return 0.0
    return sum(1 for r in top if r in rel) / len(top)


def reciprocal_rank(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    """1 / rank of the first relevant id (0 if none retrieved)."""
    rel = set(relevant)
    for i, r in enumerate(retrieved, start=1):
        if r in rel:
            return 1.0 / i
    return 0.0


def average_precision(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    """Average precision over the retrieved list (basis of MAP).

    Each relevant id is credited once, at its first occurrence.
    """
    rel = set(relevant)
    if not rel:
        return 0.0
    seen: set[str] = set()
    hits = 0
    total = 0.0
    for i, r in enumerate(retrieved, start=1):
        if r in rel and r not in seen:
            seen.add(r)
            hits += 1
            total += hits / i
    return total / len(rel)


def dcg_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Discounted cumulative gain (binary relevance) over the top-k.

    A relevant id is credited once (at its first occurrence) so duplicate
    retrieved ids cannot inflate the gain.
    """
    rel = set(relevant)
    seen: set[str] = set()
    total = 0.0
    for i, r in enumerate(retrieved[:k], start=1):
        if r in rel and r not in seen:
            seen.add(r)
            total += 1.0 / math.log2(i + 1)
    return total


def ndcg_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Normalised DCG@k (binary relevance), in [0, 1]."""
    rel = set(relevant)
    if not rel:
        return 0.0
    ideal_hits = min(len(rel), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg == 0:
        return 0.0
    return dcg_at_k(retrieved, relevant, k) / idcg


@dataclass
class EvalCase:
    """One query with its known-relevant memory ids.

    ``k``/``mode`` are ``None`` by default so they fall back to the
    ``default_k``/``default_mode`` passed to :func:`evaluate`; set them per-case
    only to override.
    """
    query: str
    relevant_ids: list[str]
    k: int | None = None
    mode: str | None = None


@dataclass
class EvalResult:
    n: int
    k: int
    recall_at_k: float
    precision_at_k: float
    mrr: float
    map: float
    ndcg_at_k: float
    per_case: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        ks = "mixed" if self.k < 0 else str(self.k)
        return (f"n={self.n} | recall@{ks}={self.recall_at_k:.3f} "
                f"P@{ks}={self.precision_at_k:.3f} MRR={self.mrr:.3f} "
                f"MAP={self.map:.3f} nDCG@{ks}={self.ndcg_at_k:.3f}")


def evaluate(
    store,
    cases: Sequence[EvalCase],
    *,
    default_k: int = 5,
    default_mode: str = "hybrid",
    **recall_kwargs,
) -> EvalResult:
    """Run each case through ``store.recall`` and aggregate ranking metrics.

    Recall is invoked read-only (``touch=False``) so evaluating does not perturb
    access-count / recency. Extra keyword args (e.g. ``weights=``, ``fusion=``,
    ``diversity=``) are forwarded to ``recall`` so you can A/B ranking configs.
    """
    rs = ps = rr = ap = ng = 0.0
    per: list[dict] = []
    ks_used: set[int] = set()
    for c in cases:
        k = c.k if getattr(c, "k", None) is not None else default_k
        mode = c.mode if getattr(c, "mode", None) is not None else default_mode
        ks_used.add(k)
        hits = store.recall(c.query, k=k, mode=mode, touch=False, **recall_kwargs)
        ids = [m.id for m, *_ in hits]
        m = {
            "query": c.query,
            "k": k,
            "recall_at_k": recall_at_k(ids, c.relevant_ids, k),
            "precision_at_k": precision_at_k(ids, c.relevant_ids, k),
            "rr": reciprocal_rank(ids, c.relevant_ids),
            "ap": average_precision(ids, c.relevant_ids),
            "ndcg_at_k": ndcg_at_k(ids, c.relevant_ids, k),
            "retrieved": ids,
        }
        per.append(m)
        rs += m["recall_at_k"]; ps += m["precision_at_k"]
        rr += m["rr"]; ap += m["ap"]; ng += m["ndcg_at_k"]
    n = len(cases)
    denom = n or 1
    # Label the result with the k actually used when uniform, else -1 ("mixed").
    result_k = next(iter(ks_used)) if len(ks_used) == 1 else -1
    return EvalResult(
        n=n, k=result_k,
        recall_at_k=rs / denom, precision_at_k=ps / denom,
        mrr=rr / denom, map=ap / denom, ndcg_at_k=ng / denom,
        per_case=per,
    )
