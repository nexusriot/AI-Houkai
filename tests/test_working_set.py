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

    def test_a_non_idempotent_write_is_not_matched_later(self, fake_store):
        """The hash is only recorded when the caller opts in, so an existing
        plain row does not silently start absorbing repeats."""
        fake_store.remember("no hash recorded")
        fake_store.remember("no hash recorded", idempotent=True)
        assert fake_store.count() == 2

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
