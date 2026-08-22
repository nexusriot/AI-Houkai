"""Provenance trust: combining levels, and surviving a level we don't know.

``trust`` is validated on write, so an unrecognised level can only arrive from
a hand-edited store or from a build that knows a level this one does not — both
of which ``trust.py`` names explicitly as expected. Two rules follow, and
neither used to hold:

  * Combining trust takes the **worst** case, including when there is nothing
    to combine. Anything else is a laundering path.
  * Reading an unrecognised level must **fail safe**, not raise. A recall that
    filters by provenance crashing on one odd row is worse than excluding it.

An *absent* level is a different thing from an unrecognised one: it means the
row predates the feature, and it deserialises as ``trusted`` so that opening an
old store with a newer build does not change what recall returns.
"""

from __future__ import annotations

import pytest

from ai_houkai.memory_system import MemoryStore
from ai_houkai.memory_system.trust import (
    TRUST_LEVELS,
    trust_rank,
    worst_trust,
)


def _force_trust(store: MemoryStore, memory_id: str, value: str) -> None:
    """Write *value* straight into metadata, bypassing validation.

    Stands in for a hand-edited store or a row written by a future build.
    """
    meta = dict(store.collection.get(
        ids=[memory_id], include=["metadatas"])["metadatas"][0])
    meta["trust"] = value
    store.collection.update(ids=[memory_id], metadatas=[meta])


class TestWorstTrust:
    @pytest.mark.parametrize("levels,expected", [
        (("trusted", "trusted"), "trusted"),
        (("trusted", "reported"), "reported"),
        (("reported", "untrusted"), "untrusted"),
        (("untrusted", "trusted"), "untrusted"),
    ])
    def test_takes_the_worst_of_known_levels(self, levels, expected):
        assert worst_trust(levels) == expected

    def test_an_unknown_level_is_the_worst_case(self):
        """Failing safe is the only defensible default for a provenance label."""
        assert worst_trust(("trusted", "from-the-future")) == TRUST_LEVELS[-1]

    def test_nothing_to_combine_reads_as_trusted(self):
        """An unreachable edge, pinned so it stops being a silent one.

        Every caller passes at least one level — a cluster's members, or the two
        sides of a merge. The Go port asserts the same value, so the two stay
        interchangeable.
        """
        assert worst_trust(()) == "trusted"

    def test_an_absent_level_still_reads_as_trusted(self):
        """Old rows deserialise with no trust key; that must stay benign."""
        assert worst_trust(("", "")) == "trusted"


class TestTrustRank:
    def test_orders_known_levels_best_first(self):
        ranks = [trust_rank(level) for level in TRUST_LEVELS]
        assert ranks == sorted(ranks) and ranks[0] == 0

    def test_absent_is_trusted(self):
        assert trust_rank("") == 0

    def test_unknown_is_least_trusted(self):
        assert trust_rank("from-the-future") == len(TRUST_LEVELS) - 1


class TestRecallSurvivesAnUnknownLevel:
    def test_min_trust_excludes_it_without_raising(self, store: MemoryStore):
        good = store.remember("a trusted fact", trust="trusted")
        odd = store.remember("a fact with a level we do not know")
        _force_trust(store, odd.id, "from-the-future")

        hits = store.recall("fact", k=10, min_trust="trusted")
        ids = [m.id for m, _ in hits]
        assert good.id in ids
        assert odd.id not in ids

    def test_it_is_still_returned_when_no_floor_is_asked_for(
            self, store: MemoryStore):
        """Fail-safe filtering, not silent deletion — the row still exists."""
        odd = store.remember("a fact with a level we do not know")
        _force_trust(store, odd.id, "from-the-future")

        ids = [m.id for m, _ in store.recall("fact", k=10)]
        assert odd.id in ids

    def test_the_pinned_lane_is_filtered_too(self, store: MemoryStore):
        """include_pinned prepends outside the ranked pool, so it needs the
        same fail-safe read of the level."""
        odd = store.remember("a pinned instruction", pinned=True)
        _force_trust(store, odd.id, "from-the-future")
        store.remember("something to rank")

        packed = store.recall_pack("something", include_pinned=True,
                                   min_trust="trusted")
        assert odd.id not in packed.ids()
