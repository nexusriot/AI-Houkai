"""Provenance trust levels, and the rule for combining them.

A leaf module on purpose. ``store.py`` imports ``CurationMixin`` from
``curation.py``, so ``curation.py`` cannot import back from ``store.py`` — and
``reflection.py`` needs the same vocabulary. Putting the levels here lets every
one of them import it at module top, with no cycle and no deferred import.

``store.py`` re-exports ``TrustLevel`` / ``TRUST_LEVELS`` so
``from ai_houkai.memory_system.store import TRUST_LEVELS`` keeps working.
"""

from __future__ import annotations

from typing import Iterable, Literal

__all__ = ["TRUST_LEVELS", "TrustLevel", "worst_trust"]

# How much a memory's ORIGIN is trusted — not how confident we are that it is
# true (that is `importance`) and not whether it is current (that is
# `superseded_by`). Ordered best-first, and the order is load-bearing:
# `min_trust` compares by index, and `worst_trust` takes the maximum.
TrustLevel = Literal["trusted", "reported", "untrusted"]

TRUST_LEVELS: tuple[str, ...] = ("trusted", "reported", "untrusted")


def worst_trust(levels: Iterable[str]) -> str:
    """The least-trusted level among *levels*.

    Combining trust always takes the worst case, because a derived memory
    carries the content of every source it was derived from. A summary of
    untrusted material is untrusted; merging untrusted text into a trusted
    memory makes the result untrusted. Anything else is a laundering path:
    content the agent did not author ends up recallable under
    ``min_trust="trusted"``.

    An unrecognised level (from a hand-edited store, or a future level this
    build does not know) is treated as the worst case rather than ignored —
    failing safe is the only defensible default for a provenance label.
    """
    worst = 0
    for level in levels:
        try:
            worst = max(worst, TRUST_LEVELS.index(level))
        except ValueError:
            return TRUST_LEVELS[-1]
    return TRUST_LEVELS[worst]
