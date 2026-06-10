"""Heuristic importance auto-assignment.

``score_importance(text, type, tags)`` rates a memory 0..1 without an LLM,
following the tiers sketched in DESIGN.md §15:

    0.90+  explicit standing instructions, corrections, user preferences
    0.75   decisions, conventions, policies
    0.60   task completions, durable project facts
    0.50   neutral default (nothing matched)
    0.35   passing observations, hedged statements

The score is the strongest matching tier plus small modifiers (memory
type, questions, very short fragments), clamped to [0.05, 0.98] so an
auto-scored memory never claims absolute certainty in either direction.

Deterministic by design: same text in, same score out — recall ranking
and decay both consume importance, so it must be stable across runs.

Wiring: pass ``importance_fn=score_importance`` to MemoryStore (or set
``default_importance = "auto"`` in config.toml for the CLI, or
``AI_HOUKAI_AUTO_IMPORTANCE=1`` for the MCP server) and leave
``importance`` unset when remembering.
"""

from __future__ import annotations

import re
from typing import Iterable

# Each tier: (score, compiled patterns). First (highest) matching tier wins;
# patterns are case-insensitive and word-bounded where it matters.
_TIERS: list[tuple[float, list[re.Pattern[str]]]] = [
    (0.90, [re.compile(p, re.IGNORECASE) for p in (
        r"\b(always|never)\b",
        r"\bfrom now on\b",
        r"\b(must|must not|mustn't)\b",
        r"\bdo not ever\b",
        r"\bimportant\b",
        r"\bcritical\b",
        r"^correction\b",
        r"\bactually,",
        r"\bthat('s| is| was) (wrong|incorrect)\b",
        r"\bnot \w+([ ,]+\w+){0,3}[ ,]+but\b",
        r"\binstead of\b",
        r"\bprefers?\b",
        r"\bfavou?rite\b",
        r"\b(i|user) (like|love|hate|dislike)s?\b",
        r"\bdon'?t (like|want|use)\b",
    )]),
    (0.75, [re.compile(p, re.IGNORECASE) for p in (
        r"\bdecided\b",
        r"\bdecision\b",
        r"\bwe (use|chose|agreed|settled on)\b",
        r"\bconvention\b",
        r"\bpolicy\b",
        r"\bstandard\b",
        r"\brule\b",
        r"\bworkflow\b",
        r"\brequired\b",
    )]),
    (0.60, [re.compile(p, re.IGNORECASE) for p in (
        r"\b(fixed|solved|resolved)\b",
        r"\b(implemented|added|built|created)\b",
        r"\b(deployed|released|shipped|merged|published)\b",
        r"\b(configured|installed|migrated)\b",
    )]),
    (0.35, [re.compile(p, re.IGNORECASE) for p in (
        r"\b(noticed|observed|saw|spotted)\b",
        r"\b(seems|appears|looks like)\b",
        r"\b(maybe|perhaps|might|possibly)\b",
        r"\bnot sure\b",
        r"\bfor now\b",
        r"\btemporar(y|ily)\b",
    )]),
]

_DEFAULT = 0.50
_FLOOR, _CEIL = 0.05, 0.98

# Memory types that are durable by nature get a nudge upward.
_TYPE_BONUS = {"procedural": 0.10, "feedback": 0.10}


def score_importance(
    text: str,
    type: str = "semantic",
    tags: Iterable[str] = (),
) -> float:
    """Rate a memory's importance 0..1 from its text, type, and tags.

    The strongest matching tier wins (an instruction beats a hedge), then
    modifiers apply: +0.10 for procedural/feedback types, −0.15 for
    questions, −0.10 for fragments under 20 characters.
    """
    body = (text or "").strip()

    score = _DEFAULT
    for tier_score, patterns in _TIERS:
        if any(p.search(body) for p in patterns):
            score = tier_score
            break

    score += _TYPE_BONUS.get(type, 0.0)
    if body.endswith("?"):
        score -= 0.15
    if len(body) < 20:
        score -= 0.10

    return round(max(_FLOOR, min(_CEIL, score)), 3)
