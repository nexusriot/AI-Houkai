"""Pinned tier, idempotent writes, trust tier, and tiered reflection.

F3 — `importance` was doing three jobs at once (ranking, decay survival, the
min_importance filter), so protecting a standing instruction meant distorting
every search it appeared in. `pinned` separates "always consider this" out.

F4 — agents re-assert the same fact every session; the only defence was a
per-write vector conflict scan that still created a row.

G — anything reaching remember() becomes durable, well-ranked agent context
later, so a fact scraped from a page and one stated by the user have to be
distinguishable at recall time.

F1 — reflection only ever clustered `episodic`, so summaries were never
themselves consolidated and a long-lived store accumulated them without bound.
"""

from __future__ import annotations

import pytest

from ai_houkai.memory_system import DecayEngine, MemoryStore, ReflectionEngine
from ai_houkai.memory_system.store import TRUST_LEVELS, content_hash
from ai_houkai.testing import FakeEmbedder


class TestContentHash:
    def test_normalises_case_and_whitespace(self):
        assert content_hash("Use ruff for linting") == \
            content_hash("use  ruff\nfor linting  ")

    def test_distinguishes_different_text(self):
        assert content_hash("alpha") != content_hash("beta")

    def test_is_stable_across_calls(self):
        assert content_hash("stable") == content_hash("stable")


class TestIdempotentRemember:
    def test_repeat_returns_the_original(self, fake_store):
        a = fake_store.remember("Use ruff for linting", idempotent=True)
        b = fake_store.remember("use  ruff for linting", idempotent=True)
        assert a.id == b.id
        assert fake_store.count() == 1

    def test_repeat_bumps_access_instead_of_writing(self, fake_store):
        a = fake_store.remember("counted repeat", idempotent=True)
        assert fake_store.get(a.id).access_count == 0
        fake_store.remember("counted repeat", idempotent=True)
        assert fake_store.get(a.id).access_count == 1

    def test_off_by_default(self, fake_store):
        fake_store.remember("plain write")
        fake_store.remember("plain write")
        assert fake_store.count() == 2

    def test_matches_a_row_written_without_the_flag(self, fake_store):
        """`idempotent` controls whether THIS write dedupes — not whether the
        earlier one volunteered to be a dedup target.

        The hash used to be recorded only when the caller opted in, which made
        the contract asymmetric: two idempotent writes collapsed, but a plain
        write followed by an idempotent re-assertion did not. Whether some
        earlier call passed the flag is invisible to the caller asking "don't
        duplicate this fact", so an agent adopting the flag mid-life silently
        accumulated one duplicate of every pre-existing memory.
        """
        first = fake_store.remember("no flag on the first write")
        again = fake_store.remember("no flag on the first write", idempotent=True)
        assert again.id == first.id
        assert fake_store.count() == 1

    def test_an_empty_digest_never_matches(self, fake_store):
        """Rows written before `content_hash` existed carry no hash.

        Their stored value reads back as "", and an empty digest must return
        None rather than matching every hash-less row — otherwise the first
        idempotent write after an upgrade would absorb an arbitrary old memory.
        (The state itself cannot be constructed through the public API: Chroma's
        `update` merges metadata, so a key cannot be removed once written.)
        """
        fake_store.remember("some row")
        assert fake_store._find_by_content_hash("") is None

    def test_edit_keeps_the_hash_in_step(self, fake_store):
        """An edited memory must answer to its NEW text, not its old one."""
        mem = fake_store.remember("original wording", idempotent=True)
        fake_store.edit(mem.id, text="revised wording")

        same = fake_store.remember("revised wording", idempotent=True)
        assert same.id == mem.id, "the new text should match the edited row"

        stale = fake_store.remember("original wording", idempotent=True)
        assert stale.id != mem.id, "the pre-edit text must no longer match"

    def test_superseded_rows_do_not_absorb_a_repeat(self, fake_store):
        """Re-asserting a fact that was explicitly replaced should create a new
        memory, not resurrect the old one."""
        old = fake_store.remember("the policy", idempotent=True)
        new = fake_store.remember("the replacement")
        fake_store.supersede(old_id=old.id, new_id=new.id)

        again = fake_store.remember("the policy", idempotent=True)
        assert again.id != old.id
        assert fake_store.get(old.id).superseded_by == new.id

    def test_expired_rows_do_not_absorb_a_repeat(self, fake_store):
        gone = fake_store.remember("ephemeral fact", idempotent=True,
                                   expires_at=1.0)
        again = fake_store.remember("ephemeral fact", idempotent=True)
        assert again.id != gone.id

    def test_editing_the_text_moves_the_hash(self, fake_store):
        """Otherwise an edited memory would still answer to its original text."""
        m = fake_store.remember("original wording", idempotent=True)
        fake_store.edit(m.id, text="revised wording")
        assert fake_store.get(m.id).content_hash == content_hash("revised wording")

        again = fake_store.remember("original wording", idempotent=True)
        assert again.id != m.id

    def test_works_in_remember_many(self, fake_store):
        mems = fake_store.remember_many([
            {"text": "batch a"}, {"text": "batch b"}])
        assert len({m.id for m in mems}) == 2


class TestPinned:
    def test_defaults_to_false_and_roundtrips(self, fake_store):
        plain = fake_store.remember("ordinary")
        pinned = fake_store.remember("standing instruction", pinned=True)
        assert fake_store.get(plain.id).pinned is False
        assert fake_store.get(pinned.id).pinned is True

    def test_editable(self, fake_store):
        m = fake_store.remember("promote me")
        fake_store.edit(m.id, pinned=True)
        assert fake_store.get(m.id).pinned is True
        fake_store.edit(m.id, pinned=False)
        assert fake_store.get(m.id).pinned is False

    def test_decay_never_prunes_a_pinned_memory(self, fake_store):
        """The point of the flag: protecting a standing instruction should not
        require inflating its importance, which distorts every search."""
        stale = fake_store.remember("stale and unimportant", importance=0.01)
        kept = fake_store.remember("stale but pinned", importance=0.01,
                                   pinned=True)
        engine = DecayEngine(fake_store, decay_rate=1.0, min_score=0.9)

        pruned = {m.id for m in engine.prune(dry_run=True)}
        assert stale.id in pruned
        assert kept.id not in pruned

    def test_pack_prepends_pinned(self, fake_store):
        fake_store.remember("ALWAYS run make lint", pinned=True,
                            type="procedural")
        fake_store.remember("unrelated gardening note")
        packed = fake_store.recall_pack("gardening", include_pinned=True)
        assert packed.text.splitlines()[1].startswith("- (procedural) [pinned]")

    def test_pack_omits_pinned_by_default(self, fake_store):
        fake_store.remember("ALWAYS run make lint", pinned=True)
        fake_store.remember("gardening note")
        packed = fake_store.recall_pack("gardening", max_items=1)
        assert "make lint" not in packed.text

    def test_pinned_is_not_duplicated_when_it_also_matches(self, fake_store):
        m = fake_store.remember("the pinned and matching subject", pinned=True)
        packed = fake_store.recall_pack("the pinned and matching subject",
                                        include_pinned=True)
        assert [p.memory.id for p in packed.items].count(m.id) == 1

    def test_pinned_still_respects_the_budget(self, fake_store):
        """A standing instruction is prioritised, not exempt: an over-budget
        pack must not silently blow past its ceiling."""
        fake_store.remember("x" * 400, pinned=True)
        packed = fake_store.recall_pack("anything", token_budget=10,
                                        include_pinned=True)
        assert packed.used_tokens <= 10


class TestTrust:
    def test_defaults_to_trusted(self, fake_store):
        assert fake_store.remember("from the user").trust == "trusted"

    @pytest.mark.parametrize("level", TRUST_LEVELS)
    def test_roundtrips_every_level(self, fake_store, level):
        m = fake_store.remember(f"a {level} memory", trust=level)
        assert fake_store.get(m.id).trust == level

    def test_rejects_an_unknown_level(self, fake_store):
        with pytest.raises(ValueError, match="trust"):
            fake_store.remember("bad", trust="probably-fine")

    def test_editable(self, fake_store):
        m = fake_store.remember("reclassify me")
        fake_store.edit(m.id, trust="untrusted")
        assert fake_store.get(m.id).trust == "untrusted"

    def test_min_trust_filters_recall(self, fake_store):
        user = fake_store.remember("stated by the user", trust="trusted")
        tool = fake_store.remember("relayed by a tool", trust="reported")
        page = fake_store.remember("scraped from a page", trust="untrusted")

        everything = {m.id for m, _ in fake_store.recall("a memory", k=5)}
        assert {user.id, tool.id, page.id} <= everything

        strict = {m.id for m, _ in
                  fake_store.recall("a memory", k=5, min_trust="trusted")}
        assert strict == {user.id}

        moderate = {m.id for m, _ in
                    fake_store.recall("a memory", k=5, min_trust="reported")}
        assert moderate == {user.id, tool.id}

    def test_min_trust_rejects_an_unknown_level(self, fake_store):
        with pytest.raises(ValueError, match="min_trust"):
            fake_store.recall("q", min_trust="nonsense")

    def test_pack_marks_untrusted_lines(self, fake_store):
        """The packed block goes straight into a model's context; a scraped
        fact must not be indistinguishable there from a user-stated one."""
        fake_store.remember("scraped claim", trust="untrusted")
        packed = fake_store.recall_pack("scraped claim")
        assert "[untrusted]" in packed.text

    def test_pack_does_not_mark_trusted_lines(self, fake_store):
        fake_store.remember("user stated fact")
        assert "[" not in fake_store.recall_pack("user stated fact").text

    def test_old_rows_read_as_trusted(self, tmp_path):
        """An existing store must not change behaviour just by being opened."""
        store = MemoryStore(path=str(tmp_path / "c"), collection="legacy",
                            embedding_function=FakeEmbedder())
        try:
            store.collection.add(
                ids=["legacy-1"], documents=["written before trust existed"],
                metadatas=[{"type": "semantic", "tags": "", "importance": 0.5,
                            "created_at": 1.0, "last_accessed": 1.0,
                            "access_count": 0, "source": "", "links": "[]",
                            "superseded_by": "", "superseded_at": 0.0,
                            "polarity": 0}])
            mem = store.get("legacy-1")
            assert mem.trust == "trusted"
            assert mem.pinned is False
            assert mem.content_hash == ""
            # And it survives a min_trust filter, rather than vanishing.
            hits = store.recall("written before trust existed", k=1,
                                min_trust="trusted")
            assert [m.id for m, _ in hits] == ["legacy-1"]
        finally:
            store.client.close()


class TestReflectionTypes:
    def _seed(self, store, n, type="episodic", tags=()):
        return [store.remember(f"{type} note {i}", type=type, tags=list(tags))
                for i in range(n)]

    def test_defaults_to_episodic_only(self, fake_store):
        self._seed(fake_store, 3, type="episodic")
        self._seed(fake_store, 3, type="semantic")
        engine = ReflectionEngine(fake_store, similarity_threshold=-1.0,
                                  min_cluster_size=2)
        clusters = engine.clusters()
        assert clusters
        assert all(m.type == "episodic" for c in clusters for m in c)

    def test_can_cluster_other_types(self, fake_store):
        self._seed(fake_store, 3, type="feedback")
        engine = ReflectionEngine(fake_store, similarity_threshold=-1.0,
                                  min_cluster_size=2, types=("feedback",))
        clusters = engine.clusters()
        assert clusters and all(m.type == "feedback" for c in clusters for m in c)

    def test_multiple_types(self, fake_store):
        self._seed(fake_store, 2, type="episodic")
        self._seed(fake_store, 2, type="feedback")
        engine = ReflectionEngine(fake_store, similarity_threshold=-1.0,
                                  min_cluster_size=2,
                                  types=("episodic", "feedback"))
        seen = {m.type for c in engine.clusters() for m in c}
        assert seen == {"episodic", "feedback"}

    def test_summaries_are_tagged_with_their_tier(self, fake_store):
        self._seed(fake_store, 3, type="episodic")
        engine = ReflectionEngine(fake_store, similarity_threshold=-1.0,
                                  min_cluster_size=2)
        made = engine.reflect()
        assert made and "level:1" in made[0].tags

    def test_reflections_of_reflections(self, fake_store):
        self._seed(fake_store, 4, type="episodic")
        tier1 = ReflectionEngine(fake_store, similarity_threshold=-1.0,
                                 min_cluster_size=2, max_level=2)
        assert tier1.reflect()
        # A second level-1 summary so the tier-2 cluster has two members.
        fake_store.remember("another summary", type="semantic",
                            tags=["reflection", "level:1"])

        tier2 = ReflectionEngine(fake_store, similarity_threshold=-1.0,
                                 min_cluster_size=2, types=("semantic",),
                                 max_level=2)
        made = tier2.reflect()
        assert made and "level:2" in made[0].tags

    def test_max_level_caps_the_hierarchy(self, fake_store):
        """The guard against runaway re-summarisation eating the store."""
        fake_store.remember("s1", type="semantic", tags=["reflection", "level:1"])
        fake_store.remember("s2", type="semantic", tags=["reflection", "level:1"])
        engine = ReflectionEngine(fake_store, similarity_threshold=-1.0,
                                  min_cluster_size=2, types=("semantic",),
                                  max_level=1)
        assert engine.reflect() == []

    def test_member_level_tags_are_not_inherited(self, fake_store):
        fake_store.remember("s1", type="semantic", tags=["reflection", "level:1", "topic"])
        fake_store.remember("s2", type="semantic", tags=["reflection", "level:1"])
        engine = ReflectionEngine(fake_store, similarity_threshold=-1.0,
                                  min_cluster_size=2, types=("semantic",),
                                  max_level=3)
        made = engine.reflect()
        assert made
        levels = [t for t in made[0].tags if t.startswith("level:")]
        assert levels == ["level:2"]
        assert "topic" in made[0].tags


class TestTrustPropagation:
    """A derived memory must inherit the least-trusted of its sources.

    Otherwise the trust tier has a laundering path: ingest a poisoned page as
    `untrusted`, let maintenance reflection summarise it, and the summary is
    born `trusted` — recallable under `min_trust="trusted"` with the poisoned
    content intact. `_cluster` already refuses to blend opposite polarities for
    the same reason; trust needs the same treatment.
    """

    def test_reflection_summary_inherits_least_trusted_source(self, store):
        for _ in range(3):
            store.remember("scraped claim about widgets from a random blog",
                           type="episodic", trust="untrusted")
        engine = ReflectionEngine(store, similarity_threshold=0.5,
                                  min_cluster_size=2)
        created = engine.reflect()
        assert created, "expected the identical texts to cluster"
        assert all(m.trust == "untrusted" for m in created), (
            "a summary of untrusted sources must not be born trusted")

    def test_reflection_mixed_sources_take_the_worst(self, store):
        store.remember("mixed provenance widget fact", type="episodic",
                       trust="trusted")
        store.remember("mixed provenance widget fact", type="episodic",
                       trust="reported")
        engine = ReflectionEngine(store, similarity_threshold=0.5,
                                  min_cluster_size=2)
        created = engine.reflect()
        assert created and created[0].trust == "reported"

    def test_reflection_of_trusted_sources_stays_trusted(self, store):
        for _ in range(2):
            store.remember("first-party widget fact", type="episodic",
                           trust="trusted")
        engine = ReflectionEngine(store, similarity_threshold=0.5,
                                  min_cluster_size=2)
        created = engine.reflect()
        assert created and created[0].trust == "trusted"

    def test_merge_inherits_the_least_trusted_side(self, store):
        target = store.remember("trusted target fact", trust="trusted")
        other = store.remember("untrusted absorbed fact", trust="untrusted")
        merged = store.merge(target.id, other.id)
        assert merged.trust == "untrusted", (
            "absorbing untrusted text must downgrade the target's provenance")

    def test_merge_keeps_the_better_trust_when_other_is_cleaner(self, store):
        target = store.remember("reported target", trust="reported")
        other = store.remember("trusted addition", trust="trusted")
        merged = store.merge(target.id, other.id)
        assert merged.trust == "reported", "merge must not upgrade trust"

    def test_merge_preserves_pinned(self, store):
        target = store.remember("pinned target", pinned=True)
        other = store.remember("absorbed text")
        assert store.merge(target.id, other.id).pinned is True


class TestPinnedLookupIsNotAFullScan:
    def test_prepend_pinned_does_not_load_the_whole_store(self, store):
        """The pinned lookup sits on the recall_pack / auto_context hot path.

        Reading every memory to find the handful that are pinned makes the
        cheapest feature in the packer the most expensive part of it.
        """
        store.remember_many([f"filler {i}" for i in range(40)])
        store.remember("ALWAYS run lint", pinned=True, type="procedural")

        seen: list = []
        real = store.list_recent

        def spy(*args, **kwargs):
            seen.append(kwargs.get("limit", args[0] if args else None))
            return real(*args, **kwargs)

        store.list_recent = spy
        try:
            store.recall_pack("filler", include_pinned=True)
        finally:
            store.list_recent = real

        unbounded = [n for n in seen if n is None or n > 10_000]
        assert not unbounded, (
            f"recall_pack loaded the whole collection to find pinned "
            f"memories (list_recent limits: {seen})")

    def test_pinned_lookup_still_finds_them(self, store):
        store.remember_many([f"filler {i}" for i in range(5)])
        pin = store.remember("ALWAYS run lint", pinned=True, type="procedural")
        packed = store.recall_pack("filler", include_pinned=True)
        assert pin.id in [p.memory.id for p in packed.items]

    def test_pinned_lookup_honours_min_trust(self, store):
        store.remember("untrusted standing instruction", pinned=True,
                       trust="untrusted")
        clean = store.remember("trusted standing instruction", pinned=True)
        store.remember("subject matter")
        packed = store.recall_pack("subject matter", include_pinned=True,
                                   min_trust="trusted")
        ids = [p.memory.id for p in packed.items]
        assert clean.id in ids
        assert all(m.trust == "trusted" for m in
                   (p.memory for p in packed.items))


class TestFastPathRespectsMinTrust:
    def test_min_trust_does_not_truncate_the_result(self, store):
        """`no_post_filter` fetches exactly k and skips the over-fetch pool.

        min_trust is a post-query filter, so taking that path with a trust floor
        set silently returns fewer than k results even when more qualifying
        memories exist.
        """
        for i in range(20):
            store.remember(f"widget note untrusted {i}", trust="untrusted")
        for i in range(20):
            store.remember(f"widget note trusted {i}", trust="trusted")

        got = store.recall("widget note", k=5, mode="semantic",
                           include_superseded=True, include_expired=True,
                           min_trust="trusted")
        assert len(got) == 5, (
            f"fast path returned {len(got)} of k=5 with a trust floor set")
        assert all(m.trust == "trusted" for m, _ in got)


class TestBatchIdempotency:
    def test_remember_many_accepts_idempotent(self, store):
        """Bulk write is where dedupe matters most — the conflict scan is per
        item, so a re-asserted batch is the expensive case."""
        mems = store.remember_many(
            ["Use ruff for linting", "use  ruff for linting  ",
             "Use ruff for linting"],
            idempotent=True)
        assert store.count() == 1, "normalised duplicates must collapse"
        assert len({m.id for m in mems}) == 1
        assert len(mems) == 3, "every input still maps to a returned memory"

    def test_remember_many_idempotent_matches_existing_rows(self, store):
        first = store.remember("Use ruff for linting")
        mems = store.remember_many(["use ruff for linting"], idempotent=True)
        assert [m.id for m in mems] == [first.id]
        assert store.count() == 1

    def test_remember_many_default_still_writes_duplicates(self, store):
        store.remember_many(["same text", "same text"])
        assert store.count() == 2, "idempotency must stay opt-in"


class TestPinSurvivesSupersede:
    """A pinned memory is a standing-instruction slot (docs/DESIGN.md §27).

    Superseding one is how you *correct* a standing instruction, but the pin was
    dropped: the slot silently emptied and the agent stopped seeing the rule
    until somebody noticed and re-pinned by hand.
    """

    def test_the_replacement_inherits_the_pin(self, store):
        old = store.remember("indent with tabs", pinned=True)
        new = store.remember("indent with four spaces")
        store.supersede(old.id, new.id)

        assert store.get(new.id).pinned is True
        assert [m.id for m in store._pinned_memories()] == [new.id]

    def test_superseding_an_unpinned_memory_pins_nothing(self, store):
        old = store.remember("a plain fact")
        new = store.remember("a corrected plain fact")
        store.supersede(old.id, new.id)

        assert store.get(new.id).pinned is False
        assert store._pinned_memories() == []

    def test_an_already_pinned_replacement_is_left_alone(self, store):
        old = store.remember("old rule", pinned=True)
        new = store.remember("new rule", pinned=True)
        store.supersede(old.id, new.id)

        assert store.get(new.id).pinned is True

    def test_trust_is_not_inherited(self, store):
        """Supersede keeps both rows, each with its own provenance — unlike
        merge, which folds two into one and must take the worse label."""
        old = store.remember("reported claim", pinned=True, trust="reported")
        new = store.remember("verified replacement")
        store.supersede(old.id, new.id)

        assert store.get(new.id).trust == "trusted"

    def test_restore_hands_the_pin_back(self, store):
        """Undoing a supersede must not leave the slot filled twice."""
        old = store.remember("original rule", pinned=True)
        new = store.remember("replacement rule")
        store.supersede(old.id, new.id)
        store.restore(old.id)

        assert store.get(old.id).pinned is True
        assert store.get(new.id).pinned is False
        assert [m.id for m in store._pinned_memories()] == [old.id]


class TestPinSurvivesConsolidation:
    """Reflection folds sources into a summary; a pinned source must not lose
    its standing-instruction slot on the way.

    Soft consolidate supersedes each source, which now carries the pin over —
    but `reflect()` returned the pre-supersede snapshot, so callers saw
    `pinned=False` for a row that is pinned. Hard consolidate deleted the
    sources outright and the pin went with them.
    """

    def _pair(self, store, pinned_first=True):
        store.remember("standing deploy rule", type="episodic",
                       pinned=pinned_first)
        store.remember("standing deploy rule", type="episodic")
        return ReflectionEngine(store, similarity_threshold=0.5,
                                min_cluster_size=2)

    def test_soft_consolidate_moves_the_pin_to_the_summary(self, store):
        engine = self._pair(store)
        created = engine.reflect(consolidate=True)

        assert created, "identical texts should cluster"
        assert created[0].pinned is True, "returned summary misreports its pin"
        assert store.get(created[0].id).pinned is True
        assert [m.id for m in store._pinned_memories()] == [created[0].id]

    def test_hard_consolidate_keeps_the_pin_alive(self, store):
        engine = self._pair(store)
        created = engine.reflect(consolidate="hard")

        assert created and created[0].pinned is True
        assert [m.id for m in store._pinned_memories()] == [created[0].id]

    def test_without_consolidation_the_summary_is_not_pinned(self, store):
        """The sources stay live and pinned — pinning the summary too would put
        two rows in the working set for one standing instruction."""
        engine = self._pair(store)
        created = engine.reflect()

        assert created and created[0].pinned is False
        assert [m.text for m in store._pinned_memories()] == ["standing deploy rule"]

    def test_unpinned_sources_produce_an_unpinned_summary(self, store):
        engine = self._pair(store, pinned_first=False)
        created = engine.reflect(consolidate=True)

        assert created and created[0].pinned is False
        assert store._pinned_memories() == []

    def test_dry_run_reports_the_pin_it_would_set(self, store):
        engine = self._pair(store)
        candidates = engine.reflect(dry_run=True, consolidate=True)

        assert candidates and candidates[0].pinned is True
        assert store.get(candidates[0].id) is None, "dry run must not write"


class TestMergeKeepsThePin:
    """`merge` folds `other` into `target` and deletes `other`.

    Trust already takes the worse of the two sides; the pin did not travel at
    all, so absorbing a standing instruction into an unpinned row emptied the
    working set and deleted the only copy of the flag.
    """

    def test_absorbing_a_pinned_memory_pins_the_target(self, store):
        target = store.remember("plain target")
        other = store.remember("ALWAYS run make lint", pinned=True)
        merged = store.merge(target.id, other.id)

        assert merged.pinned is True
        assert [m.id for m in store._pinned_memories()] == [target.id]

    def test_a_pinned_target_stays_pinned(self, store):
        target = store.remember("the standing rule", pinned=True)
        other = store.remember("an ordinary detail")
        merged = store.merge(target.id, other.id)

        assert merged.pinned is True

    def test_merging_two_unpinned_memories_pins_nothing(self, store):
        target = store.remember("first half")
        other = store.remember("second half")
        merged = store.merge(target.id, other.id)

        assert merged.pinned is False
        assert store._pinned_memories() == []


class TestIdempotentWritesReportWhatHappened:
    """A caller has to be able to tell a fresh write from a dedupe hit.

    `remember(idempotent=True)` returns the existing memory either way, and the
    surfaces reported `stored: true` regardless — so a client replaying a batch
    every session was told it had written N new rows when it had written none.
    (The Go port went further and answered HTTP 409 with an empty conflict
    list.) `find_by_content_hash` is the public form of the lookup the store
    already does, so a caller can also ask before writing.
    """

    def test_find_by_content_hash_locates_a_live_row(self, store):
        mem = store.remember("a fact worth repeating")
        assert store.find_by_content_hash("a fact worth repeating").id == mem.id

    def test_lookup_normalises_the_text_the_same_way_a_write_does(self, store):
        mem = store.remember("a fact worth repeating")
        assert store.find_by_content_hash(
            "  a fact worth repeating  ").id == mem.id

    def test_lookup_misses_when_nothing_matches(self, store):
        store.remember("something else entirely")
        assert store.find_by_content_hash("never written") is None

    def test_a_superseded_row_does_not_match(self, store):
        """Same rule the write path uses: re-asserting a replaced fact should
        make a new memory, not resurrect the old one."""
        old = store.remember("a fact that got replaced")
        new = store.remember("its replacement")
        store.supersede(old.id, new.id)
        assert store.find_by_content_hash("a fact that got replaced") is None

    def test_an_empty_query_never_matches(self, store):
        store.remember("some content")
        assert store.find_by_content_hash("   ") is None
