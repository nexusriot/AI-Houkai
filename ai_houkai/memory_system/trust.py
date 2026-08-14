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

__all__ = ["TRUST_LEVELS", "TrustLevel", "trust_rank", "worst_trust"]

# How much a memory's ORIGIN is trusted — not how confident we are that it is
# true (that is `importance`) and not whether it is current (that is
# `superseded_by`). Ordered best-first, and the order is load-bearing:
# `min_trust` compares by index, and `worst_trust` takes the maximum.
TrustLevel = Literal["trusted", "reported", "untrusted"]

TRUST_LEVELS: tuple[str, ...] = ("trusted", "reported", "untrusted")


def trust_rank(level: str) -> int:
    """Position of *level* in :data:`TRUST_LEVELS` — higher is less trusted.

    Two cases that look alike but are not:

    An **absent** level (``""``) ranks as trusted. Rows written before the
    trust field existed deserialise that way, and opening an old store with a
    newer build must not change what recall returns.

    An **unrecognised** non-empty level ranks as the worst case. It can only
    reach here from a hand-edited store or from a build that knows a level this
    one does not, and failing safe is the only defensible default for a
    provenance label. Reading it as trusted would launder unknown content into
    an answer the caller asked to be trusted; raising would let one odd row
    break every provenance-filtered recall.
    """
    if not level:
        return 0
    try:
        return TRUST_LEVELS.index(level)
    except ValueError:
        return len(TRUST_LEVELS) - 1


def worst_trust(levels: Iterable[str]) -> str:
    """The least-trusted level among *levels*.

    Combining trust always takes the worst case, because a derived memory
    carries the content of every source it was derived from. A summary of
    untrusted material is untrusted; merging untrusted text into a trusted
    memory makes the result untrusted. Anything else is a laundering path:
    content the agent did not author ends up recallable under
    ``min_trust="trusted"``.

    No levels at all yields ``"trusted"``. Every caller passes at least one — a
    cluster's members, or the two sides of a merge — so this is an unreachable
    edge rather than a policy choice; it is spelled out because it used to be
    silent.
    """
    return TRUST_LEVELS[max((trust_rank(level) for level in levels), default=0)]
