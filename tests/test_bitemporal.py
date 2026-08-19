"""Bi-temporal validity: when a memory was true, as distinct from when we learned it.

The journal already records TRANSACTION time — ``state_at`` replays it to answer
"as of when we *knew*". These tests cover VALID time: the half-open interval
``[valid_from, valid_until)`` during which a memory was true in the world, and
``recall(as_of=T)`` asking what held at a chosen moment.

The subtle half is the interaction with supersede. A memory that has since been
replaced is exactly what "what was true then" should return, so ``as_of`` has to
admit superseded rows and let the validity interval decide — otherwise it would
answer with today's beliefs wearing a past timestamp.
"""

from __future__ import annotations

import pytest

from ai_houkai.memory_system import Memory, MemoryStore

# A fixed timeline, well clear of "now", so the tests never race the clock.
JAN = 1_700_000_000.0
FEB = JAN + 30 * 86_400
MAR = JAN + 60 * 86_400
APR = JAN + 90 * 86_400


def _ids(hits):
    return {mem.id for mem, _ in hits}


class TestDefaultsAreInvisible:
    """A store that never sets the fields must behave exactly as before."""

    def test_a_plain_memory_is_unbounded(self, store: MemoryStore):
        mem = store.remember("no validity recorded")
        assert mem.valid_from == 0.0 and mem.valid_until == 0.0

    def test_an_unbounded_memory_is_valid_at_any_instant(self, store: MemoryStore):
        mem = store.remember("timeless fact about deployment")
        for ts in (1.0, JAN, APR):
            assert mem.id in _ids(store.recall("deployment", k=10, as_of=ts))

    def test_a_row_written_before_the_fields_existed_reads_as_unbounded(
            self, store: MemoryStore):
        """An old store must not change behaviour just by being opened."""
        store.collection.add(
            ids=["legacy-validity"],
            documents=["written before valid_from existed"],
            metadatas=[{"type": "semantic", "tags": "", "importance": 0.5,
                        "created_at": 1.0, "last_accessed": 1.0,
                        "access_count": 0, "source": "", "links": "[]",
                        "superseded_by": "", "superseded_at": 0.0,
                        "polarity": 0}])
        mem = store.get("legacy-validity")
        assert mem.valid_from == 0.0 and mem.valid_until == 0.0
        assert "legacy-validity" in _ids(
            store.recall("written before valid_from existed", k=5))


class TestValidityRoundTrips:
    def test_write_and_read_back(self, store: MemoryStore):
        mem = store.remember("the office is in Berlin",
                             valid_from=JAN, valid_until=MAR)
        got = store.get(mem.id)
        assert (got.valid_from, got.valid_until) == (JAN, MAR)

    def test_survives_a_dict_round_trip(self):
        """to_dict/from_dict is what the journal and export/import use."""
        mem = Memory(id="m1", text="t", type="semantic",
                     valid_from=JAN, valid_until=MAR)
        assert Memory.from_dict(mem.to_dict()).valid_from == JAN
        assert Memory.from_dict(mem.to_dict()).valid_until == MAR

    def test_only_one_end_may_be_bounded(self, store: MemoryStore):
        opened = store.remember("still true today", valid_from=JAN)
        assert opened.valid_until == 0.0
        closed = store.remember("true until March", valid_until=MAR)
        assert closed.valid_from == 0.0


class TestValidation:
    def test_an_inverted_interval_is_rejected(self, store: MemoryStore):
        with pytest.raises(ValueError, match="valid_until must be > valid_from"):
            store.remember("backwards", valid_from=MAR, valid_until=JAN)

    def test_a_zero_length_interval_is_rejected(self, store: MemoryStore):
        """Half-open means [T, T) contains nothing — a memory true at no instant
        is a bug in the caller, not a memory worth storing."""
        with pytest.raises(ValueError):
            store.remember("instantaneous", valid_from=JAN, valid_until=JAN)

    def test_negative_bounds_are_rejected(self, store: MemoryStore):
        with pytest.raises(ValueError):
            store.remember("negative", valid_from=-1)

    def test_as_of_must_not_be_negative(self, store: MemoryStore):
        with pytest.raises(ValueError, match="as_of must be >= 0"):
            store.recall("anything", as_of=-1)


class TestTheIntervalIsHalfOpen:
    """[from, until) — so two facts that succeed one another can share a
    boundary instant without both being true at it."""

    def test_valid_from_is_inclusive(self, store: MemoryStore):
        mem = store.remember("berlin office fact", valid_from=FEB)
        assert mem.id in _ids(store.recall("berlin office", k=10, as_of=FEB))

    def test_valid_until_is_exclusive(self, store: MemoryStore):
        mem = store.remember("berlin office fact", valid_until=MAR)
        assert mem.id not in _ids(store.recall("berlin office", k=10, as_of=MAR))
        assert mem.id in _ids(store.recall("berlin office", k=10, as_of=MAR - 1))

    def test_adjacent_intervals_never_overlap(self, store: MemoryStore):
        first = store.remember("the office is in Berlin",
                               valid_from=JAN, valid_until=MAR)
        second = store.remember("the office is in Munich", valid_from=MAR)
        at_boundary = _ids(store.recall("where is the office", k=10, as_of=MAR))
        assert second.id in at_boundary
        assert first.id not in at_boundary


class TestAsOfSelectsWhatWasTrue:
    def test_returns_the_fact_that_held_then(self, store: MemoryStore):
        old = store.remember("the office is in Berlin",
                             valid_from=JAN, valid_until=MAR)
        new = store.remember("the office is in Munich", valid_from=MAR)

        february = _ids(store.recall("where is the office", k=10, as_of=FEB))
        assert old.id in february and new.id not in february

        april = _ids(store.recall("where is the office", k=10, as_of=APR))
        assert new.id in april and old.id not in april

    def test_a_retired_fact_drops_out_of_ordinary_recall(self, store: MemoryStore):
        """Closing valid_until retires a fact without deleting it — that is the
        difference from TTL, which reclaims the row."""
        retired = store.remember("the office is in Berlin",
                                 valid_from=JAN, valid_until=MAR)
        assert retired.id not in _ids(store.recall("where is the office", k=10))
        # …but it is still there, and still reachable by asking about the past.
        assert store.get(retired.id) is not None
        assert retired.id in _ids(
            store.recall("where is the office", k=10, as_of=FEB))

    def test_a_future_fact_is_not_yet_current(self, store: MemoryStore):
        future = store.remember("the office moves to Munich next year",
                                valid_from=APR * 10)
        assert future.id not in _ids(store.recall("office munich", k=10))


class TestAsOfOverridesSupersedeHiding:
    """The subtle half, and the reason as_of is not just a metadata filter."""

    def test_a_superseded_memory_valid_at_t_is_returned(self, store: MemoryStore):
        old = store.remember("the office is in Berlin",
                             valid_from=JAN, valid_until=MAR)
        new = store.remember("the office is in Munich", valid_from=MAR)
        store.supersede(old_id=old.id, new_id=new.id)

        # Ordinary recall hides it, as always.
        assert old.id not in _ids(store.recall("where is the office", k=10))
        # Asking about February must still find it: it WAS true then, and
        # hiding it would make as_of answer with today's beliefs.
        assert old.id in _ids(
            store.recall("where is the office", k=10, as_of=FEB))

    def test_a_superseded_memory_not_valid_at_t_stays_hidden(self, store: MemoryStore):
        """Admitting superseded rows must not become "return everything"."""
        old = store.remember("the office is in Berlin",
                             valid_from=JAN, valid_until=FEB)
        new = store.remember("the office is in Munich", valid_from=FEB)
        store.supersede(old_id=old.id, new_id=new.id)

        march = _ids(store.recall("where is the office", k=10, as_of=MAR))
        assert old.id not in march
        assert new.id in march

    def test_without_as_of_supersede_hiding_is_unchanged(self, store: MemoryStore):
        old = store.remember("superseded and unbounded")
        new = store.remember("the replacement")
        store.supersede(old_id=old.id, new_id=new.id)
        assert old.id not in _ids(store.recall("superseded", k=10))


class TestEditAdjustsValidity:
    def test_closing_an_interval_retires_the_fact(self, store: MemoryStore):
        mem = store.remember("the office is in Berlin", valid_from=JAN)
        assert mem.id in _ids(store.recall("where is the office", k=10))

        store.edit(mem.id, valid_until=MAR)
        assert mem.id not in _ids(store.recall("where is the office", k=10))
        assert mem.id in _ids(store.recall("where is the office", k=10, as_of=FEB))

    def test_reopening_an_end_makes_it_current_again(self, store: MemoryStore):
        mem = store.remember("temporarily retired", valid_from=JAN, valid_until=MAR)
        assert mem.id not in _ids(store.recall("temporarily retired", k=10))
        store.edit(mem.id, valid_until=0)
        assert mem.id in _ids(store.recall("temporarily retired", k=10))

    def test_editing_one_end_validates_against_the_other(self, store: MemoryStore):
        """Closing an end must not silently produce until <= from."""
        mem = store.remember("bounded below", valid_from=MAR)
        with pytest.raises(ValueError):
            store.edit(mem.id, valid_until=JAN)

    def test_the_change_is_journaled(self, store: MemoryStore):
        mem = store.remember("journaled validity", valid_from=JAN)
        store.edit(mem.id, valid_until=MAR)
        edits = [e for e in store.journal.read()
                 if e.op == "edit" and e.id == mem.id]
        assert edits and edits[-1].after["valid_until"] == MAR


class TestBatchWritesCarryValidity:
    def test_remember_many_accepts_per_item_intervals(self, store: MemoryStore):
        mems = store.remember_many([
            {"text": "berlin era", "valid_from": JAN, "valid_until": MAR},
            {"text": "munich era", "valid_from": MAR},
        ])
        assert [(m.valid_from, m.valid_until) for m in mems] == \
            [(JAN, MAR), (MAR, 0.0)]


class TestValidTimeIsNotTransactionTime:
    def test_state_at_and_as_of_answer_different_questions(self, store: MemoryStore):
        """state_at replays the journal ("as of when we knew"); as_of reads the
        validity interval ("what was true then"). A memory written today about
        last year is invisible to the first and visible to the second."""
        mem = store.remember("the office was in Berlin all of February",
                             valid_from=JAN, valid_until=MAR)

        # We only learned it now, so a journal replay of a past instant has it
        # nowhere — JAN predates the write.
        assert all(m.id != mem.id for m in store.state_at(JAN))
        # But it was true in February, and as_of says so.
        assert mem.id in _ids(store.recall("office february", k=10, as_of=FEB))
