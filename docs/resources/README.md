# docs/resources — design charts

Data charts for [`DESIGN.md`](../DESIGN.md) and [`PROPOSALS.md`](../../PROPOSALS.md),
provided as **PNG** (embedded in the docs) and **SVG** (vector, selectable text).

Every value is computed from the **shipped source**, not hand-copied:

- the importance charts call the real `score_importance()` from
  [`ai_houkai/memory_system/importance.py`](../../ai_houkai/memory_system/importance.py); and
- the decay / hybrid / BM25 / RRF / recall_pack constants are re-parsed out of
  `decay.py`, `store.py`, and `reflection.py` at generation time and asserted
  against the numbers the charts draw (`_validate_against_source`), so a
  parameter change in the code makes regeneration **fail loudly** instead of
  emitting a stale chart.

## Charts

| File | Section | Shows |
|---|---|---|
| `decay_curves` | §6 Decay | Decay score vs. days for importance 0.90/0.50/0.30; where each hits the 0.05 prune line |
| `decay_heatmap` | §6 Decay | 2-D survival region (importance × idle days) with the prune contour |
| `reinforcement` | §6 Decay | `frequency_weight` multiplier and the extra survival days recalls buy |
| `halflife` | §6 Decay | Same curve for λ = 0.05/0.10/0.20 → 14/7/3.5-day half-lives |
| `halflife_vs_lambda` | §6 Decay | Half-life as a function of `decay_rate`, `t½ = ln2/λ` |
| `importance_tiers` | §15 Importance | The heuristic 0.90/0.75/0.60/0.50/0.35 scorer, values from real `score_importance()` |
| `importance_waterfall` | §15 Importance | Base tier → modifiers → final importance, clamped `[0.05, 0.98]` |
| `hybrid_weights` | §14 Hybrid / Proposal §1 | Blend weights (0.55/0.20/0.15/0.10 + 0.05 polarity) + a worked ranking |
| `bm25_saturation` | §14 Hybrid / Proposal §1 | BM25 term-frequency saturation (k1 = 1.5, b = 0.75) |
| `rrf_fusion` | §14 Hybrid | Reciprocal Rank Fusion — scale-free `Σ w/(60+rank)` blending |
| `mmr_tradeoff` | §14 Hybrid | MMR relevance-vs-novelty trade-off and the crossover λ |
| `recall_pack_budget` | §5 Lifecycle | Greedy token-budget packing at the default `token_budget = 800`, default vs. `compress=True` |

## Key defaults (as drawn)

| Quantity | Default | Source |
|---|---|---|
| `decay_rate` (λ) | 0.10 (≈ 7-day half-life) | `decay.py` |
| `min_score` (prune) | 0.05 | `decay.py` |
| `frequency_weight` | 0.0 (pure recency) | `decay.py` |
| Hybrid weights | cosine 0.55 · BM25 0.20 · recency 0.15 · importance 0.10 · polarity ±0.05 | `store.py` |
| BM25 | k1 = 1.5, b = 0.75 | `store.py` |
| RRF | `rrf_k` = 60 | `store.py` |
| `recall_pack` budget | 800 tokens (compress: Jaccard 0.30, min group 2) | `store.py` |
| Importance tiers | 0.90 / 0.75 / 0.60 / 0.50 / 0.35, clamp [0.05, 0.98] | `importance.py` |
| Reflection | similarity ≥ 0.75, min cluster size 2 | `reflection.py` |

## Regenerating

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install matplotlib numpy cairosvg
python docs/resources/make_charts.py       # → docs/resources/*.svg + *.png
```

Charts are pure matplotlib (SVG `fonttype=none`, so text stays editable).
