#!/usr/bin/env python3
"""Generate AI-Houkai design charts as SVG + PNG into ``docs/resources``.

Every value here is the *actual* shipped default, not a hand-copied number:

* the importance charts call the real ``score_importance`` from
  ``ai_houkai/memory_system/importance.py`` (dependency-free), so the tiers,
  modifiers and clamp are exactly what the store assigns; and
* the decay / hybrid / BM25 / RRF / recall_pack constants are re-parsed out of
  the source at start-up by :func:`_validate_against_source`, which raises if
  the code drifts from the numbers the charts draw. Regenerating after a
  parameter change therefore fails loudly instead of shipping a stale chart.

Formulas mirrored from source:
  decay:      importance * exp(-lambda*days) * (1 + fw*ln(1+access))   decay.py
  hybrid:     0.55 cos + 0.20 bm25 + 0.15 recency + 0.10 imp + 0.05 pol store.py
  recency:    exp(-0.10 * age_days)                                     store.py
  bm25:       k1=1.5, b=0.75 (normalised)                              store.py
  rrf:        sum_s  w_s / (60 + rank_s)                                store.py
  importance: tiers 0.90/0.75/0.60/0.50/0.35 (+/- modifiers, clip 0.05..0.98)

Usage:
    python docs/resources/make_charts.py         # -> docs/resources/*.svg + *.png
"""
from __future__ import annotations

import importlib.util
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent                       # docs/resources -> repo root
SRC = REPO / "ai_houkai" / "memory_system"
OUT = HERE
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.edgecolor": "#cbd5e0", "axes.labelcolor": "#2d3748", "text.color": "#1a202c",
    "xtick.color": "#4a5568", "ytick.color": "#4a5568",
    "axes.grid": True, "grid.color": "#edf2f7", "grid.linewidth": 1,
    "figure.dpi": 100, "svg.fonttype": "none",   # selectable/editable text in SVG
})

INK, PRUNE, REFLECT, BLUE, TEAL, AMBER = (
    "#1a202c", "#742a2a", "#553c9a", "#1a365d", "#2c7a7b", "#b7791f")


# Actual defaults — imported live where possible, else parsed from source and
# asserted, so the charts cannot silently drift from the shipped code.
def _load_score_importance():
    p = SRC / "importance.py"
    spec = importlib.util.spec_from_file_location("_ah_importance", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                 # only imports re + typing
    return mod


_IMP = _load_score_importance()
score_importance = _IMP.score_importance

# Constants the charts draw. Kept here for readability, then verified against
# the source by _validate_against_source() below.
LAMBDA = 0.10          # decay_rate / recency λ            decay.py, store.py
MIN_SCORE = 0.05       # prune threshold                   decay.py
FREQ_WEIGHT = 0.0      # reinforcement (off by default)    decay.py
W = dict(cos=0.55, bm=0.20, rec=0.15, imp=0.10, pol=0.05)  # HybridWeights, store.py
BM25_K1, BM25_B = 1.5, 0.75        # _BM25_K1, _BM25_B      store.py
RRF_K = 60                          # _rrf_score rrf_k       store.py
PACK_BUDGET = 800                   # recall_pack token_budget default  store.py
COMPRESS_MIN_GROUP = 2              # recall_pack compress_min_group     store.py
FLOOR, CEIL = _IMP._FLOOR, _IMP._CEIL             # importance clamp (live)


def _grab(text: str, pattern: str) -> float:
    m = re.search(pattern, text)
    if not m:
        raise AssertionError(f"could not find /{pattern}/ in source")
    return float(m.group(1))


def _validate_against_source() -> None:
    """Re-parse the load-bearing constants from source and assert they match
    what the charts draw. Fails loudly rather than emitting a stale chart."""
    decay = (SRC / "decay.py").read_text()
    store = (SRC / "store.py").read_text()
    refl = (SRC / "reflection.py").read_text()

    checks = {
        "decay_rate":        (_grab(decay, r"decay_rate:\s*float\s*=\s*([\d.]+)"), LAMBDA),
        "min_score":         (_grab(decay, r"min_score:\s*float\s*=\s*([\d.]+)"), MIN_SCORE),
        "frequency_weight":  (_grab(decay, r"frequency_weight:\s*float\s*=\s*([\d.]+)"), FREQ_WEIGHT),
        "cosine":            (_grab(store, r"cosine:\s*float\s*=\s*([\d.]+)"), W["cos"]),
        "lexical":           (_grab(store, r"lexical:\s*float\s*=\s*([\d.]+)"), W["bm"]),
        "recency":           (_grab(store, r"recency:\s*float\s*=\s*([\d.]+)"), W["rec"]),
        "importance":        (_grab(store, r"importance:\s*float\s*=\s*([\d.]+)"), W["imp"]),
        "polarity_weight":   (_grab(store, r"polarity_weight:\s*float\s*=\s*([\d.]+)"), W["pol"]),
        "bm25_k1":           (_grab(store, r"_BM25_K1\s*=\s*([\d.]+)"), BM25_K1),
        "bm25_b":            (_grab(store, r"_BM25_B\s*=\s*([\d.]+)"), BM25_B),
        "rrf_k":             (_grab(store, r"rrf_k:\s*int\s*=\s*(\d+)"), RRF_K),
        "pack_budget":       (_grab(store, r"token_budget:\s*int\s*=\s*(\d+)"), PACK_BUDGET),
        "reflect_sim":       (_grab(refl, r"similarity_threshold:\s*float\s*=\s*([\d.]+)"), 0.75),
        "reflect_min":       (_grab(refl, r"min_cluster_size:\s*int\s*=\s*(\d+)"), 2),
    }
    bad = {k: v for k, (v, exp) in checks.items() if abs(v - exp) > 1e-9}
    if bad:
        raise AssertionError(
            "charts are stale — source constants changed: "
            + ", ".join(f"{k}={checks[k][0]} (chart uses {checks[k][1]})" for k in bad))
    print("validated 14 source constants — charts match shipped defaults")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.svg", format="svg", bbox_inches="tight", transparent=True)
    fig.savefig(OUT / f"{name}.png", format="png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("chart:", name)


# Decay charts
def decay_curves():
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    days = np.linspace(0, 60, 400); lam, thr = LAMBDA, MIN_SCORE; cross = []
    for imp, c in [(0.90, BLUE), (0.50, TEAL), (0.30, AMBER)]:
        ax.plot(days, imp * np.exp(-lam * days), color=c, lw=2.4, label=f"importance = {imp:.2f}")
        d = math.log(imp / thr) / lam
        ax.scatter([d], [thr], color=c, zorder=6, s=42, edgecolor="white", linewidth=1.2)
        cross.append((imp, d, c))
    ax.axhline(thr, ls="--", color=PRUNE, lw=1.6)
    ax.text(0.3, thr + 0.015, f"prune threshold  min_score = {thr}", color=PRUNE, fontsize=9, fontweight="bold")
    ax.text(40, 0.42, "crosses 0.05 at:", fontsize=9.5, color="#2d3748", fontweight="bold")
    for j, (imp, d, c) in enumerate(cross):
        ax.text(40, 0.36 - j * 0.05, f"importance {imp:.2f}  →  {d:.0f} days", fontsize=9.5, color=c, fontweight="bold")
    hl = math.log(2) / lam
    ax.axvline(hl, ls=":", color="#718096", lw=1.4)
    ax.text(hl + 0.6, 0.83, f"half-life ≈ {hl:.1f} d", color="#4a5568", fontsize=9)
    ax.set(xlabel="days since last access", ylabel="decay score",
           title=f"Exponential decay  ·  score = importance × e^(−{lam} × days)", xlim=(0, 60), ylim=(0, 0.95))
    ax.legend(frameon=False, loc="upper right"); save(fig, "decay_curves")


def halflife():
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    days = np.linspace(0, 60, 400); imp, thr = 0.50, MIN_SCORE
    for lam, c in [(0.05, TEAL), (0.10, BLUE), (0.20, PRUNE)]:
        ax.plot(days, imp * np.exp(-lam * days), color=c, lw=2.4,
                label=f"λ = {lam:.2f}  (half-life {math.log(2)/lam:.0f} d)")
    ax.axhline(thr, ls="--", color="#718096", lw=1.4)
    ax.text(60, thr + 0.012, f"min_score = {thr}", ha="right", color="#4a5568", fontsize=9)
    ax.set(xlabel="days since last access", ylabel="decay score",
           title="Tuning the forgetting rate  ·  importance = 0.50", xlim=(0, 60), ylim=(0, 0.55))
    ax.legend(frameon=False, loc="upper right"); save(fig, "halflife")


def halflife_vs_lambda():
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    lam = np.linspace(0.02, 0.5, 300); ax.plot(lam, np.log(2) / lam, color=BLUE, lw=2.6)
    for L, c in [(0.05, TEAL), (0.10, BLUE), (0.20, PRUNE)]:
        hl = math.log(2) / L; ax.scatter([L], [hl], color=c, s=46, zorder=5, edgecolor="white", lw=1.2)
        ax.annotate(f"λ={L} → {hl:.1f} d", (L, hl), textcoords="offset points", xytext=(8, 6),
                    fontsize=9.5, color=c, fontweight="bold")
    ax.set(xlabel="decay_rate  λ", ylabel="half-life (days)",
           title="Half-life vs decay rate   t½ = ln(2) / λ", xlim=(0.02, 0.5), ylim=(0, 36))
    save(fig, "halflife_vs_lambda")


def reinforcement():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.9)); ac = np.arange(0, 61)
    for fw, c in [(0.1, AMBER), (0.3, REFLECT), (0.5, BLUE)]:
        ax1.plot(ac, 1 + fw * np.log1p(ac), color=c, lw=2.3, label=f"frequency_weight = {fw}")
    ax1.scatter([20], [1 + 0.3 * math.log1p(20)], color=REFLECT, zorder=5, s=34)
    ax1.annotate("fw 0.3, recalled 20× ≈ 1.9×", (20, 1 + 0.3 * math.log1p(20)),
                 textcoords="offset points", xytext=(6, -2), fontsize=8.5, color=REFLECT, fontweight="bold")
    ax1.set(xlabel="recall count (access_count)", ylabel="score multiplier", title="Reinforcement: 1 + fw·ln(1+recalls)")
    ax1.legend(frameon=False, fontsize=9, loc="upper left")
    imp, lam, thr = 0.5, LAMBDA, MIN_SCORE
    for fw, c in [(0.0, "#a0aec0"), (0.3, REFLECT), (0.5, BLUE)]:
        surv = np.log(imp * (1 + fw * np.log1p(ac)) / thr) / lam
        ax2.plot(ac, surv, color=c, lw=2.3, label="recency-only (fw 0)" if fw == 0 else f"fw = {fw}")
    ax2.set(xlabel="recall count (access_count)", ylabel="days until pruned", title="Recalls extend a memory's lifetime")
    ax2.legend(frameon=False, fontsize=9, loc="lower right"); save(fig, "reinforcement")


def decay_heatmap():
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    days = np.linspace(0, 45, 240); imp = np.linspace(0.05, 1.0, 200)
    D, I = np.meshgrid(days, imp); Z = I * np.exp(-LAMBDA * D)
    pcm = ax.pcolormesh(D, I, Z, cmap="viridis", shading="auto", vmin=0, vmax=1)
    cs = ax.contour(D, I, Z, levels=[MIN_SCORE], colors=["#e53e3e"], linewidths=2.4)
    ax.clabel(cs, fmt=f"prune line {MIN_SCORE}", fontsize=9)
    ax.text(33, 0.85, "RETAINED", color="white", fontsize=12, fontweight="bold")
    ax.text(2.5, 0.12, "PRUNED", color="#fed7d7", fontsize=12, fontweight="bold")
    fig.colorbar(pcm, ax=ax, label=f"decay score  (importance × e^(−{LAMBDA}·days))")
    ax.set(xlabel="days since last access", ylabel="importance",
           title=f"Survival region: which memories the decay engine keeps (λ={LAMBDA})")
    ax.grid(False); save(fig, "decay_heatmap")


# Importance charts — values computed by the real score_importance()
def importance_tiers():
    fig, ax = plt.subplots(figsize=(8.4, 4.1))
    # Each row: (label, representative phrase, example type). The bar value is
    # what the shipped score_importance() actually returns for that phrase.
    rows = [
        ("Observation / hedge", "It seems fine, maybe revisit later", "semantic"),
        ("Neutral default", "The build produced three artifacts", "semantic"),
        ("Completion / fact", "Fixed and deployed the auth service", "semantic"),
        ("Decision / convention", "We decided the policy is trunk-based", "semantic"),
        ("Instruction / preference", "Always run the linter before pushing", "semantic"),
    ]
    colors = ["#a0aec0", "#90cdf4", TEAL, AMBER, PRUNE]
    ex_color = ["#2d3748", "#2d3748", "white", "white", "white"]
    vals = [score_importance(p, t) for _, p, t in rows]
    y = np.arange(len(rows)); ax.barh(y, vals, color=colors, height=0.62, zorder=3)
    for i, ((lab, phrase, _), v) in enumerate(zip(rows, vals)):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=10, fontweight="bold")
        ax.text(0.012, i, f'“{phrase}”', va="center", ha="left", fontsize=8.4,
                color=ex_color[i], style="italic")
    ax.set_yticks(y, [r[0] for r in rows], fontsize=10); ax.set_xlim(0, 1.05)
    ax.set_xlabel("auto-assigned importance  (score_importance)")
    ax.set_title("Heuristic importance tiers  (highest matching tier wins)")
    ax.grid(axis="y", visible=False); save(fig, "importance_tiers")


def importance_waterfall():
    fig, ax = plt.subplots(figsize=(8.6, 4.1))
    # (display label, text, type, base tier). Final = real score_importance();
    # the modifier is (final - base) so the annotation is always consistent.
    ex = [
        ("'Always run tests\nbefore pushing'\n(procedural)", "Always run tests before pushing", "procedural", 0.90),
        ("'We decided to use\nREST not GraphQL'", "We decided to use REST not GraphQL", "semantic", 0.75),
        ("'Fixed the flaky\nauth test'", "Fixed the flaky auth test", "semantic", 0.60),
        ("'Slow?'\n(short question)", "Slow?", "semantic", 0.50),
    ]
    x = np.arange(len(ex))
    base = [e[3] for e in ex]
    final = [score_importance(e[1], e[2]) for e in ex]
    ax.bar(x - 0.2, base, width=0.38, color="#cbd5e0", label="base tier", zorder=3)
    ax.bar(x + 0.2, final, width=0.38, color=BLUE, label="final (after modifiers, clamped)", zorder=3)
    for i, (lab, txt, ty, b) in enumerate(ex):
        f = final[i]; mod = round(f - b, 2)
        ax.text(x[i] - 0.2, b + 0.015, f"{b:.2f}", ha="center", fontsize=9)
        ax.text(x[i] + 0.2, f + 0.015, f"{f:.2f}", ha="center", fontsize=9.5, fontweight="bold")
        if abs(mod) > 1e-9:
            ax.text(x[i] + 0.2, f / 2, f"{'+' if mod > 0 else ''}{mod:.2f}",
                    ha="center", color="white", fontsize=9, fontweight="bold")
    ax.set_xticks(x, [e[0] for e in ex], fontsize=8.4); ax.set_ylim(0, 1.05)
    ax.set_ylabel("importance")
    ax.set_title(f"Importance: base tier → modifiers → final  (clamped [{FLOOR}, {CEIL}])")
    ax.legend(frameon=False, loc="upper right"); ax.grid(axis="x", visible=False)
    save(fig, "importance_waterfall")


# Hybrid-retrieval charts
def hybrid_weights():
    fig, (axp, axb) = plt.subplots(1, 2, figsize=(8.8, 4.0), gridspec_kw={"width_ratios": [1, 1.7]})
    w = [W["cos"], W["bm"], W["rec"], W["imp"]]
    wl = [f"cosine\n{W['cos']}", f"BM25\n{W['bm']}", f"recency\n{W['rec']}", f"importance\n{W['imp']}"]
    cols = [BLUE, TEAL, AMBER, REFLECT]
    axp.pie(w, labels=wl, colors=cols, startangle=90, counterclock=False,
            wedgeprops=dict(width=0.42, edgecolor="white"), textprops=dict(fontsize=9))
    axp.set_title(f"Blend weights\n(+{W['pol']} polarity, additive)", fontsize=11)
    cand = [("A  exact match,\nfresh, important", dict(cos=0.92, bm=0.80, rec=0.98, imp=0.90, pol=1)),
            ("B  semantic only,\nstale", dict(cos=0.74, bm=0.10, rec=0.30, imp=0.50, pol=0)),
            ("C  keyword hit,\nlow value", dict(cos=0.55, bm=0.95, rec=0.60, imp=0.30, pol=0))]
    parts = [("cos", BLUE, "cosine"), ("bm", TEAL, "BM25"), ("rec", AMBER, "recency"),
             ("imp", REFLECT, "importance"), ("pol", "#dd6b20", "polarity")]
    x = np.arange(len(cand)); bottom = np.zeros(len(cand))
    for key, col, lab in parts:
        vals = np.array([W[key] * c[1][key] for c in cand])
        axb.bar(x, vals, bottom=bottom, color=col, label=lab, width=0.6, zorder=3); bottom += vals
    for i, total in enumerate(bottom):
        axb.text(i, total + 0.008, f"{total:.2f}", ha="center", fontsize=10, fontweight="bold")
    axb.set_xticks(x, [c[0] for c in cand], fontsize=8.6); axb.set_ylabel("final hybrid score")
    axb.set_title("Worked example: how candidates are ranked")
    axb.legend(frameon=False, ncol=5, fontsize=8.2, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    axb.grid(axis="x", visible=False); save(fig, "hybrid_weights")


def bm25_saturation():
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    tf = np.linspace(0, 20, 300); b = BM25_B
    for k1, c in [(1.0, AMBER), (BM25_K1, BLUE), (2.5, REFLECT)]:
        s = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * 1.0))   # dl == avgdl
        ax.plot(tf, s, color=c, lw=2.4, label=f"k1 = {k1}" + ("  (AI-Houkai default)" if k1 == BM25_K1 else ""))
    ax.plot(tf, tf, ls=":", color="#a0aec0", lw=1.6, label="linear (no saturation)")
    ax.set(xlabel="term frequency in document", ylabel="BM25 tf component",
           title=f"BM25 term-frequency saturation  (b = {BM25_B}, dl = avgdl)", xlim=(0, 20), ylim=(0, 12))
    ax.legend(frameon=False, loc="upper left")
    ax.text(10, 1.2, "diminishing returns: the 10th hit\nadds far less than the 1st",
            fontsize=9, color="#4a5568", style="italic"); save(fig, "bm25_saturation")


def rrf_fusion():
    fig, ax = plt.subplots(figsize=(8.6, 4.1))
    cands = ["A  exact + fresh", "B  semantic only", "C  keyword hit"]
    ranks = {"cosine": [0, 1, 2], "BM25": [1, 2, 0],
             "recency": [0, 2, 1], "importance": [0, 1, 2]}
    weights = {"cosine": W["cos"], "BM25": W["bm"], "recency": W["rec"], "importance": W["imp"]}
    cols = {"cosine": BLUE, "BM25": TEAL, "recency": AMBER, "importance": REFLECT}
    x = np.arange(len(cands)); bottom = np.zeros(len(cands))
    for sig in ["cosine", "BM25", "recency", "importance"]:
        contrib = np.array([weights[sig] / (RRF_K + ranks[sig][i]) for i in range(len(cands))])
        ax.bar(x, contrib, bottom=bottom, color=cols[sig], width=0.55, label=sig, zorder=3)
        bottom += contrib
    ax.set_ylim(0, float(bottom.max()) * 1.32)
    for i, t in enumerate(bottom):
        ax.text(i, t + float(bottom.max()) * 0.015, f"{t:.4f}", ha="center",
                fontsize=10, fontweight="bold")
    ax.set_xticks(x, cands, fontsize=9)
    ax.set_ylabel(f"RRF score  =  Σ  w / ({RRF_K} + rank)")
    ax.set_title("Reciprocal Rank Fusion — scale-free, rank-based blending")
    ax.legend(frameon=False, ncol=4, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis="x", visible=False)
    ax.text(0.5, 0.985, "scores are small by design — only relative order matters; "
            "immune to the BM25 pool-normalization artifact",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.2,
            color="#4a5568", style="italic")
    save(fig, "rrf_fusion")


def mmr_tradeoff():
    fig, ax = plt.subplots(figsize=(8.4, 4.1))
    lam = np.linspace(0, 1, 200)
    dup = lam * 0.95 - (1 - lam) * 0.95     # near-duplicate of an already-picked item
    nov = lam * 0.80 - (1 - lam) * 0.20     # novel item, slightly less relevant
    ax.plot(lam, dup, color=PRUNE, lw=2.4, label="near-duplicate  (rel 0.95, sim 0.95)")
    ax.plot(lam, nov, color=TEAL, lw=2.4, label="novel item  (rel 0.80, sim 0.20)")
    cx = 0.75 / 0.9
    cy = cx * 0.80 - (1 - cx) * 0.20
    ax.set_ylim(-1.05, 1.05)
    ax.axvspan(0, cx, color="#e6fffa"); ax.axvspan(cx, 1, color="#fffaf0")
    ax.axvline(cx, ls=":", color="#718096", lw=1.4)
    ax.scatter([cx], [cy], color="#2d3748", zorder=5, s=36)
    ax.annotate(f"crossover  λ ≈ {cx:.2f}", (cx, cy), textcoords="offset points",
                xytext=(8, 10), fontsize=9, fontweight="bold")
    ax.text(cx / 2, -0.9, "← diversity wins", ha="center", color=TEAL, fontsize=9.5, fontweight="bold")
    ax.text((cx + 1) / 2, -0.9, "relevance wins →", ha="center", color=PRUNE, fontsize=9.5, fontweight="bold")
    ax.set(xlabel="diversity  λ", ylabel="MMR selection value", xlim=(0, 1),
           title="MMR trade-off:  λ · relevance − (1 − λ) · redundancy")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    save(fig, "mmr_tradeoff")


def recall_pack_budget():
    fig, ax = plt.subplots(figsize=(8.6, 3.6)); budget = PACK_BUDGET
    packed = [("mem A", 240), ("mem B", 220), ("mem C", 180)]
    greens = ["#1a365d", "#2c5282", "#2c7a7b", "#38a169"]
    used0 = sum(c for _, c in packed)            # 640
    dropped = [("mem D", 190), ("mem E", 170)]   # each > remaining (160) -> dropped
    comp = 120                                    # compressed D+E summary line
    # row 1: greedy default (no compress)
    left = 0
    for (lab, c), col in zip(packed, greens):
        ax.barh(1, c, left=left, color=col, edgecolor="white", height=0.5, zorder=3)
        ax.text(left + c / 2, 1, f"{lab}\n{c}", ha="center", va="center", color="white", fontsize=8)
        left += c
    ax.barh(1, budget - left, left=left, color="#edf2f7", edgecolor="#cbd5e0", height=0.5, hatch="//", zorder=2)
    ax.text(left + (budget - left) / 2, 1, f"unused\n{budget - left}", ha="center", va="center", color="#718096", fontsize=8)
    # row 0: greedy + compress=True
    left = 0
    for (lab, c), col in zip(packed, greens):
        ax.barh(0, c, left=left, color=col, edgecolor="white", height=0.5, zorder=3); left += c
    ax.barh(0, comp, left=left, color=REFLECT, edgecolor="white", height=0.5, zorder=3)
    ax.annotate(f"(compressed) D+E → {comp}", xy=(left + comp / 2, -0.25), xytext=(left + comp / 2, -0.46),
                ha="center", va="top", color=REFLECT, fontsize=8, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=REFLECT, lw=1))
    ax.barh(0, budget - left - comp, left=left + comp, color="#edf2f7", edgecolor="#cbd5e0", height=0.5, hatch="//", zorder=2)
    ax.axvline(budget, color=PRUNE, lw=2, ls="--")
    ax.text(budget + 8, 0.5, f"token_budget = {budget}", color=PRUNE, fontsize=9.5, fontweight="bold", rotation=90, va="center")
    ax.text(300, 1.42, f"dropped (didn't fit remaining {budget - used0}): "
            f"{dropped[0][0]} ({dropped[0][1]}), {dropped[1][0]} ({dropped[1][1]})",
            fontsize=8.6, color="#742a2a", style="italic")
    ax.set_yticks([0, 1], ["compress=True", "default"], fontsize=10)
    ax.set_xlim(0, budget * 1.2); ax.set_xlabel("tokens"); ax.set_ylim(-0.85, 1.7)
    ax.set_title("recall_pack: greedy fit to a token budget"); ax.grid(axis="y", visible=False)
    save(fig, "recall_pack_budget")


CHARTS = (decay_curves, halflife, halflife_vs_lambda, reinforcement, decay_heatmap,
          importance_tiers, importance_waterfall, hybrid_weights, bm25_saturation,
          rrf_fusion, mmr_tradeoff, recall_pack_budget)

if __name__ == "__main__":
    _validate_against_source()
    for f in CHARTS:
        f()
    print(f"done — {len(CHARTS)} charts (svg + png) ->", OUT)
