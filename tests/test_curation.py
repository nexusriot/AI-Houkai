"""Curation operations graduated from ai-houkai-service (D).

The service implemented these by reaching through the library's private API
(`store._get_by_id`, `store.collection.update`, `store._journal`). Merge in
particular could not be done correctly from outside: `forget` does not clean up
incoming links, so folding one memory into another without re-pointing them
silently strands every relationship that pointed at the absorbed memory.
"""

from __future__ import annotations

import dataclasses
import json
import time

import pytest
from typer.testing import CliRunner

import ai_houkai.mcp_server.server as srv
from ai_houkai.cli.main import app
from ai_houkai.maintenance.scheduler import MaintenanceScheduler
from ai_houkai.memory_system.curation import MergeError, TrashEntry
from ai_houkai.memory_system.store import content_hash
from ai_houkai.testing import make_store


@pytest.fixture()
def mcp_store(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_HOUKAI_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("AI_HOUKAI_COLLECTION", "curation")
    monkeypatch.setattr(srv, "_store", None)
    yield
    if srv._store is not None:
        srv._store.client.close()
        srv._store = None


class TestMerge:
    def test_combines_text_and_deletes_the_source(self, fake_store):
        a = fake_store.remember("first half")
        b = fake_store.remember("second half")
        merged = fake_store.merge(a.id, b.id)
        assert merged.text == "first half\n\nsecond half"
        assert merged.id == a.id
        assert fake_store.get(b.id) is None

    def test_custom_separator(self, fake_store):
        a = fake_store.remember("left")
        b = fake_store.remember("right")
        assert fake_store.merge(a.id, b.id, separator=" | ").text == "left | right"

    def test_transfers_outgoing_links(self, fake_store):
        a = fake_store.remember("target")
        b = fake_store.remember("absorbed")
        c = fake_store.remember("b's neighbour")
        fake_store.link(b.id, c.id, rel="refines")
        merged = fake_store.merge(a.id, b.id)
        assert {(l.to, l.rel) for l in merged.links} == {(c.id, "refines")}

    def test_repoints_incoming_links(self, fake_store):
        """The reason merge belongs in the store.

        forget() leaves incoming edges dangling, so a naive
        combine-text-then-forget silently strands every relationship that
        pointed at the absorbed memory.
        """
        a = fake_store.remember("target")
        b = fake_store.remember("absorbed")
        pointer = fake_store.remember("points at the absorbed one")
        fake_store.link(pointer.id, b.id, rel="refines")

        fake_store.merge(a.id, b.id)
        links = fake_store.get(pointer.id).links
        assert [(l.to, l.rel) for l in links] == [(a.id, "refines")]

    def test_does_not_create_a_self_loop(self, fake_store):
        a = fake_store.remember("target")
        b = fake_store.remember("absorbed")
        fake_store.link(a.id, b.id, rel="related")
        merged = fake_store.merge(a.id, b.id)
        assert all(l.to != a.id for l in merged.links)

    def test_deduplicates_transferred_links(self, fake_store):
        a = fake_store.remember("target")
        b = fake_store.remember("absorbed")
        shared = fake_store.remember("both point here")
        fake_store.link(a.id, shared.id, rel="refines")
        fake_store.link(b.id, shared.id, rel="refines")
        merged = fake_store.merge(a.id, b.id)
        assert [(l.to, l.rel) for l in merged.links] == [(shared.id, "refines")]

    def test_keeps_the_dedup_hash_in_step(self, fake_store):
        """A merged memory must answer to its NEW text, not its pre-merge one.

        The same invariant edit() holds. Left stale, the hash makes the next
        idempotent write of the pre-merge text look like a repeat, so it is
        absorbed into the merged row and silently lost.
        """
        a = fake_store.remember("first half", idempotent=True)
        b = fake_store.remember("second half")
        merged = fake_store.merge(a.id, b.id)

        assert fake_store.get(a.id).content_hash == content_hash(merged.text)

        again = fake_store.remember("first half", idempotent=True)
        assert again.id != a.id, "the pre-merge text must no longer match"

        same = fake_store.remember(merged.text, idempotent=True)
        assert same.id == a.id, "the merged text should match the merged row"

    def test_rejects_self_merge(self, fake_store):
        a = fake_store.remember("only one")
        with pytest.raises(MergeError, match="itself"):
            fake_store.merge(a.id, a.id)

    def test_missing_memory(self, fake_store):
        a = fake_store.remember("exists")
        with pytest.raises(MergeError) as e:
            fake_store.merge(a.id, "nope")
        assert e.value.not_found is True

    def test_is_journaled_on_both_sides(self, fake_store):
        a = fake_store.remember("target")
        b = fake_store.remember("absorbed")
        pointer = fake_store.remember("pointer")
        fake_store.link(pointer.id, b.id, rel="refines")
        fake_store.merge(a.id, b.id)

        edits = [e for e in fake_store.journal.read() if e.op == "edit"]
        touched = {e.id for e in edits}
        assert a.id in touched and pointer.id in touched
        merge_entry = next(e for e in edits if e.id == a.id)
        assert merge_entry.meta.get("merged_from") == b.id

    def test_merged_text_is_findable(self, fake_store):
        """A merged memory that kept its pre-merge vector would be unfindable
        by the half it absorbed."""
        a = fake_store.remember("alpha content")
        b = fake_store.remember("beta content")
        fake_store.merge(a.id, b.id)
        hits = fake_store.recall("alpha content\n\nbeta content", k=1)
        assert hits and hits[0][0].id == a.id


class TestVersions:
    def test_returns_pre_edit_states_oldest_first(self, fake_store):
        m = fake_store.remember("v1 text", tags=["a"])
        fake_store.edit(m.id, text="v2 text")
        fake_store.edit(m.id, text="v3 text")
        got = fake_store.versions(m.id)
        assert [v.text for v in got] == ["v1 text", "v2 text"]
        assert got[0].tags == ["a"]

    def test_excludes_the_live_state(self, fake_store):
        m = fake_store.remember("only version")
        assert fake_store.versions(m.id) == []

    def test_ignores_other_memories(self, fake_store):
        a = fake_store.remember("a v1")
        b = fake_store.remember("b v1")
        fake_store.edit(b.id, text="b v2")
        assert fake_store.versions(a.id) == []
        assert len(fake_store.versions(b.id)) == 1


class TestTagCuration:
    def test_list_tags_counts_and_sorts(self, fake_store):
        fake_store.remember("one", tags=["common", "rare"])
        fake_store.remember("two", tags=["common"])
        assert fake_store.list_tags() == [("common", 2), ("rare", 1)]

    def test_rename(self, fake_store):
        fake_store.remember("one", tags=["old", "keep"])
        fake_store.remember("two", tags=["unrelated"])
        res = fake_store.rename_tag("old", "new")
        assert res.changed == 1 and res.tag == "new"
        assert dict(fake_store.list_tags()) == {"new": 1, "keep": 1, "unrelated": 1}

    def test_rename_deduplicates_on_collision(self, fake_store):
        m = fake_store.remember("both", tags=["old", "new"])
        fake_store.rename_tag("old", "new")
        assert fake_store.get(m.id).tags == ["new"]

    def test_merge_tags(self, fake_store):
        fake_store.remember("one", tags=["v1", "keep"])
        fake_store.remember("two", tags=["v2"])
        res = fake_store.merge_tags(["v1", "v2"], "version")
        assert res.changed == 2
        assert dict(fake_store.list_tags()) == {"version": 2, "keep": 1}

    def test_delete(self, fake_store):
        m = fake_store.remember("one", tags=["doomed", "keep"])
        res = fake_store.delete_tag("doomed")
        assert res.changed == 1
        assert fake_store.get(m.id).tags == ["keep"]

    def test_no_match_changes_nothing(self, fake_store):
        fake_store.remember("one", tags=["a"])
        assert fake_store.rename_tag("absent", "x").changed == 0
        assert fake_store.delete_tag("absent").changed == 0

    def test_rejects_a_comma(self, fake_store):
        """Tags are stored comma-joined; one with a comma splits into two."""
        fake_store.remember("one", tags=["a"])
        with pytest.raises(ValueError, match="must not contain commas"):
            fake_store.rename_tag("a", "b,c")

    def test_covers_superseded_memories(self, fake_store):
        """A curation pass that skipped them would leave the old spelling alive
        in rows a later restore brings back."""
        old = fake_store.remember("old", tags=["typo"])
        new = fake_store.remember("new")
        fake_store.supersede(old_id=old.id, new_id=new.id)
        assert fake_store.rename_tag("typo", "fixed").changed == 1
        assert fake_store.get(old.id).tags == ["fixed"]

    def test_is_journaled(self, fake_store):
        m = fake_store.remember("one", tags=["old"])
        fake_store.rename_tag("old", "new")
        entry = [e for e in fake_store.journal.read()
                 if e.op == "edit" and e.id == m.id][-1]
        assert entry.actor == "curation"
        assert entry.before["tags"] == ["old"]
        assert entry.after["tags"] == ["new"]


class TestFindPath:
    def test_direct_edge(self, fake_store):
        a = fake_store.remember("a")
        b = fake_store.remember("b")
        fake_store.link(a.id, b.id, rel="refines")
        assert fake_store.find_path(a.id, b.id) == [(a.id, ""), (b.id, "refines")]

    def test_is_undirected(self, fake_store):
        """"How are these related?" does not care which way the arrow points."""
        a = fake_store.remember("a")
        b = fake_store.remember("b")
        fake_store.link(a.id, b.id, rel="refines")
        assert [mid for mid, _ in fake_store.find_path(b.id, a.id)] == [b.id, a.id]

    def test_multi_hop_shortest(self, fake_store):
        a = fake_store.remember("a")
        b = fake_store.remember("b")
        c = fake_store.remember("c")
        fake_store.link(a.id, b.id, rel="refines")
        fake_store.link(b.id, c.id, rel="refines")
        assert [mid for mid, _ in fake_store.find_path(a.id, c.id)] == \
            [a.id, b.id, c.id]

    def test_no_path(self, fake_store):
        a = fake_store.remember("island a")
        b = fake_store.remember("island b")
        assert fake_store.find_path(a.id, b.id) == []

    def test_respects_max_depth(self, fake_store):
        ids = [fake_store.remember(f"chain {i}").id for i in range(5)]
        for i in range(4):
            fake_store.link(ids[i], ids[i + 1], rel="refines")
        assert fake_store.find_path(ids[0], ids[4], max_depth=2) == []
        assert fake_store.find_path(ids[0], ids[4], max_depth=4)

    def test_same_node(self, fake_store):
        a = fake_store.remember("solo")
        assert fake_store.find_path(a.id, a.id) == [(a.id, "")]

    def test_unknown_id(self, fake_store):
        a = fake_store.remember("real")
        assert fake_store.find_path(a.id, "ghost") == []


class TestTrash:
    def test_roundtrip_preserves_identity(self, fake_store):
        other = fake_store.remember("link target")
        m = fake_store.remember("recoverable", tags=["t"], importance=0.9)
        fake_store.link(m.id, other.id, rel="refines")

        assert fake_store.trash(m.id) is True
        assert fake_store.get(m.id) is None

        restored = fake_store.trash_restore(m.id)
        assert restored.id == m.id
        assert restored.tags == ["t"]
        assert restored.importance == 0.9
        assert [(l.to, l.rel) for l in restored.links] == [(other.id, "refines")]
        assert fake_store.get(m.id) is not None

    def test_restored_memory_is_findable(self, fake_store):
        """Restore re-embeds from the text — the vector is not in the trash."""
        m = fake_store.remember("distinctive restored phrase")
        fake_store.trash(m.id)
        fake_store.trash_restore(m.id)
        hits = fake_store.recall("distinctive restored phrase", k=1)
        assert hits and hits[0][0].id == m.id

    def test_list_and_purge(self, fake_store):
        a = fake_store.remember("first")
        b = fake_store.remember("second")
        fake_store.trash(a.id)
        fake_store.trash(b.id)
        assert [e.memory_id for e in fake_store.trash_list()] == [a.id, b.id]

        assert fake_store.trash_purge(a.id) == 1
        assert [e.memory_id for e in fake_store.trash_list()] == [b.id]
        assert fake_store.trash_purge() == 1
        assert fake_store.trash_list() == []

    def test_purge_makes_restore_impossible(self, fake_store):
        m = fake_store.remember("doomed")
        fake_store.trash(m.id)
        fake_store.trash_purge(m.id)
        assert fake_store.trash_restore(m.id) is None

    def test_trashing_a_missing_memory(self, fake_store):
        assert fake_store.trash("nope") is False

    def test_restore_something_not_in_the_trash(self, fake_store):
        assert fake_store.trash_restore("nope") is None

    def test_empty_trash_reads_clean(self, fake_store):
        assert fake_store.trash_list() == []
        assert fake_store.trash_purge() == 0


class TestMcpTools:
    def test_merge(self, mcp_store):
        a = srv.remember(text="target half")
        b = srv.remember(text="absorbed half")
        out = srv.merge(target_id=a["id"], other_id=b["id"])
        assert out["ok"] is True
        assert out["text"] == "target half\n\nabsorbed half"
        assert srv.get(memory_id=b["id"])["found"] is False

    def test_merge_reports_a_missing_memory(self, mcp_store):
        a = srv.remember(text="only one")
        out = srv.merge(target_id=a["id"], other_id="ghost")
        assert out["ok"] is False and out["not_found"] is True

    def test_versions(self, mcp_store):
        m = srv.remember(text="v1")
        srv.edit(memory_id=m["id"], text="v2")
        assert [v["text"] for v in srv.versions(memory_id=m["id"])] == ["v1"]

    def test_tag_tools(self, mcp_store):
        srv.remember(text="one", tags=["old", "keep"])
        assert {t["tag"] for t in srv.list_tags()} == {"old", "keep"}
        assert srv.rename_tag(old="old", new="new")["changed"] == 1
        assert srv.merge_tags(sources=["new"], into="keep")["changed"] == 1
        assert srv.delete_tag(tag="keep")["changed"] == 1
        assert srv.list_tags() == []

    def test_rename_tag_rejects_a_comma(self, mcp_store):
        srv.remember(text="one", tags=["a"])
        out = srv.rename_tag(old="a", new="b,c")
        assert out["ok"] is False and "comma" in out["error"]

    def test_find_path(self, mcp_store):
        a = srv.remember(text="path a")
        b = srv.remember(text="path b")
        srv.link(src_id=a["id"], dst_id=b["id"], rel="refines")
        out = srv.find_path(from_id=a["id"], to_id=b["id"])
        assert out["found"] is True and out["length"] == 1
        assert [h["id"] for h in out["path"]] == [a["id"], b["id"]]

        assert srv.find_path(from_id=a["id"], to_id="ghost")["found"] is False

    def test_trash_tools(self, mcp_store):
        m = srv.remember(text="trashable")
        assert srv.trash(memory_id=m["id"])["trashed"] is True
        assert [e["memory_id"] for e in srv.trash_list()] == [m["id"]]
        assert srv.trash_restore(memory_id=m["id"])["restored"] is True
        assert srv.get(memory_id=m["id"])["found"] is True

        srv.trash(memory_id=m["id"])
        assert srv.trash_purge()["purged"] == 1
        assert srv.trash_restore(memory_id=m["id"])["restored"] is False


class TestCli:
    def _run(self, tmp_path, *args):
        return CliRunner().invoke(app, ["--store", str(tmp_path / "chroma"), *args])

    def _ids(self, tmp_path):
        listing = self._run(tmp_path, "list", "--format", "json")
        return {m["text"]: m["id"] for m in json.loads(listing.stdout)}

    def test_merge_and_versions(self, tmp_path):
        self._run(tmp_path, "remember", "cli target")
        self._run(tmp_path, "remember", "cli absorbed")
        ids = self._ids(tmp_path)

        res = self._run(tmp_path, "merge", ids["cli target"],
                        ids["cli absorbed"], "-y")
        assert res.exit_code == 0, res.stdout
        assert "Merged." in res.stdout

        out = json.loads(self._run(
            tmp_path, "versions", ids["cli target"], "--json").stdout)
        assert [v["text"] for v in out] == ["cli target"]

    def test_tags_group(self, tmp_path):
        self._run(tmp_path, "remember", "cli tagged", "--tag", "old")
        assert json.loads(self._run(
            tmp_path, "tags", "list", "--json").stdout) == [
            {"tag": "old", "count": 1}]

        assert self._run(tmp_path, "tags", "rename", "old", "new").exit_code == 0
        assert json.loads(self._run(
            tmp_path, "tags", "list", "--json").stdout)[0]["tag"] == "new"

        assert self._run(tmp_path, "tags", "delete", "new", "-y").exit_code == 0
        assert json.loads(self._run(tmp_path, "tags", "list", "--json").stdout) == []

    def test_path_command(self, tmp_path):
        self._run(tmp_path, "remember", "cli path a")
        self._run(tmp_path, "remember", "cli path b")
        ids = self._ids(tmp_path)
        self._run(tmp_path, "link", ids["cli path a"], ids["cli path b"],
                  "--rel", "refines")
        out = json.loads(self._run(
            tmp_path, "path", ids["cli path a"], ids["cli path b"],
            "--json").stdout)
        assert out["found"] is True and out["length"] == 1

    def test_path_with_no_route_exits_nonzero(self, tmp_path):
        self._run(tmp_path, "remember", "cli island a")
        self._run(tmp_path, "remember", "cli island b")
        ids = self._ids(tmp_path)
        res = self._run(tmp_path, "path", ids["cli island a"],
                        ids["cli island b"])
        assert res.exit_code == 1

    def test_trash_group(self, tmp_path):
        self._run(tmp_path, "remember", "cli trashable")
        mid = self._ids(tmp_path)["cli trashable"]

        assert self._run(tmp_path, "trash", "put", mid).exit_code == 0
        assert json.loads(self._run(tmp_path, "list", "--format", "json").stdout
                          or "[]") == []
        entries = json.loads(self._run(tmp_path, "trash", "list", "--json").stdout)
        assert [e["memory_id"] for e in entries] == [mid]

        assert self._run(tmp_path, "trash", "restore", mid[:8]).exit_code == 0
        assert self._ids(tmp_path) == {"cli trashable": mid}

    def test_trash_purge_is_confirmed(self, tmp_path):
        self._run(tmp_path, "remember", "cli purgeable")
        mid = self._ids(tmp_path)["cli purgeable"]
        self._run(tmp_path, "trash", "put", mid)
        res = self._run(tmp_path, "trash", "purge", "-y")
        assert res.exit_code == 0 and "Purged 1" in res.stdout
        assert json.loads(self._run(tmp_path, "trash", "list", "--json").stdout) == []


class TestTrashRetention:
    """`trash_purge_expired` — the piece that makes trash a recoverable delete
    rather than a permanent archive.

    Without it the trash file grows without bound, which is why the maintenance
    scheduler drives it on the same tick as the TTL purge.
    """

    def _aged(self, store, text, days_ago):
        """Trash a memory and backdate its deleted_at."""
        mem = store.remember(text)
        store.trash(mem.id)
        entries = store._read_trash()
        for e in entries:
            if e.memory_id == mem.id:
                object.__setattr__(e, "deleted_at", time.time() - days_ago * 86_400)
        store._write_trash(entries)
        return mem

    def test_drops_only_entries_past_the_cutoff(self, store):
        old = self._aged(store, "trashed long ago", days_ago=40)
        recent = self._aged(store, "trashed yesterday", days_ago=1)

        assert store.trash_purge_expired(30) == 1
        remaining = {e.memory_id for e in store.trash_list()}
        assert remaining == {recent.id}
        assert old.id not in remaining

    def test_zero_ttl_is_a_noop_not_purge_everything(self, store):
        """A misconfigured or unset retention must never mean 'delete it all'."""
        self._aged(store, "should survive", days_ago=999)
        assert store.trash_purge_expired(0) == 0
        assert store.trash_purge_expired(-5) == 0
        assert len(store.trash_list()) == 1

    def test_empty_trash_is_a_noop(self, store):
        assert store.trash_purge_expired(30) == 0

    def test_restorable_right_up_to_the_cutoff(self, store):
        kept = self._aged(store, "just inside retention", days_ago=29)
        store.trash_purge_expired(30)
        assert store.trash_restore(kept.id) is not None

    def test_explicit_now_is_honoured(self, store):
        self._aged(store, "aged five days", days_ago=5)
        # Ten days later the same 7-day retention should sweep it.
        assert store.trash_purge_expired(7, now=time.time() + 10 * 86_400) == 1

    def test_scheduler_drives_retention(self, store, tmp_path):
        """Retention has to hold without anyone running a purge by hand."""
        self._aged(store, "swept by the scheduler", days_ago=99)
        sched = MaintenanceScheduler(
            store=store, decay_every=None, reflect_every=None,
            purge_every=1, trash_ttl_days=30,
            state_path=str(tmp_path / "state.json"),
        )
        result = sched.tick()
        assert result.ran_purge is True
        assert result.trash_purged == 1
        assert store.trash_list() == []
        assert "trash retention" in result.summary()

    def test_scheduler_respects_a_disabled_retention(self, store, tmp_path):
        self._aged(store, "kept forever", days_ago=999)
        sched = MaintenanceScheduler(
            store=store, decay_every=None, reflect_every=None,
            purge_every=1, trash_ttl_days=0,
            state_path=str(tmp_path / "state.json"),
        )
        assert sched.tick().trash_purged == 0
        assert len(store.trash_list()) == 1


class TestTrashRetentionSurfaces:
    def test_mcp_older_than_days(self, mcp_store):
        created = srv.remember(text="mcp retention subject")
        srv.trash(memory_id=created["id"])
        entries = srv.get_store()._read_trash()
        object.__setattr__(entries[0], "deleted_at", time.time() - 90 * 86_400)
        srv.get_store()._write_trash(entries)

        assert srv.trash_purge(older_than_days=30) == {"purged": 1}
        assert srv.trash_list() == []

    def test_mcp_rejects_both_arguments(self, mcp_store):
        out = srv.trash_purge(memory_id="x", older_than_days=1)
        assert out["purged"] == 0 and "not both" in out["error"]

    def test_cli_older_than(self, tmp_path):
        runner = CliRunner()
        base = ["--store", str(tmp_path / "chroma")]
        runner.invoke(app, base + ["remember", "cli retention subject"])
        listing = runner.invoke(app, base + ["list", "--format", "json"])
        mid = json.loads(listing.stdout)[0]["id"]
        runner.invoke(app, base + ["trash", "put", mid])

        res = runner.invoke(app, base + ["trash", "purge", "--older-than", "30", "-y"])
        assert res.exit_code == 0
        # Trashed just now, so a 30-day cutoff must not touch it.
        assert "Purged 0 entries" in res.stdout

    def test_cli_rejects_id_with_older_than(self, tmp_path):
        runner = CliRunner()
        base = ["--store", str(tmp_path / "chroma")]
        runner.invoke(app, base + ["remember", "cli conflict subject"])
        res = runner.invoke(app, base + ["trash", "purge", "abc123",
                                        "--older-than", "5", "-y"])
        assert res.exit_code == 1


class TestTrashCollectionScoping:
    """The trash file sits beside the store path and is shared by every
    collection opened on it — entries must be scoped, or a restore from
    collection B materializes collection A's memory into B."""

    def test_other_collection_cannot_see_or_restore(self, tmp_path):
        col_a = make_store(tmp_path / "chroma", collection="col_a")
        col_b = make_store(tmp_path / "chroma", collection="col_b")
        try:
            m = col_a.remember("belongs to a")
            col_a.trash(m.id)

            assert col_b.trash_list() == []
            assert col_b.trash_restore(m.id) is None
            assert col_b.get(m.id) is None

            restored = col_a.trash_restore(m.id)
            assert restored is not None and restored.id == m.id
        finally:
            col_a.client.close()
            col_b.client.close()

    def test_purge_leaves_other_collections_entries(self, tmp_path):
        col_a = make_store(tmp_path / "chroma", collection="col_a")
        col_b = make_store(tmp_path / "chroma", collection="col_b")
        try:
            a = col_a.remember("a's trash")
            b = col_b.remember("b's trash")
            col_a.trash(a.id)
            col_b.trash(b.id)

            assert col_b.trash_purge() == 1          # empties only b's trash
            assert [e.memory_id for e in col_a.trash_list()] == [a.id]
            assert col_a.trash_restore(a.id) is not None
        finally:
            col_a.client.close()
            col_b.client.close()

    def test_purge_expired_is_scoped(self, tmp_path):
        col_a = make_store(tmp_path / "chroma", collection="col_a")
        col_b = make_store(tmp_path / "chroma", collection="col_b")
        try:
            a = col_a.remember("old in a")
            b = col_b.remember("old in b")
            col_a.trash(a.id)
            col_b.trash(b.id)
            entries = col_a._read_trash()
            for e in entries:
                object.__setattr__(e, "deleted_at", time.time() - 90 * 86_400)
            col_a._write_trash(entries)

            assert col_a.trash_purge_expired(30) == 1
            assert [e.memory_id for e in col_b.trash_list()] == [b.id]
        finally:
            col_a.client.close()
            col_b.client.close()

    def test_legacy_untagged_entry_visible_everywhere(self, tmp_path):
        """Entries written before the collection field existed have "" and
        must stay recoverable from any collection."""
        store = make_store(tmp_path / "chroma", collection="whatever")
        try:
            m = store.remember("template row")
            legacy = TrashEntry(
                memory_id="11111111-2222-3333-4444-555555555555",
                deleted_at=time.time(), actor="lib",
                memory={**m.to_dict(),
                        "id": "11111111-2222-3333-4444-555555555555"},
                collection="")
            store._write_trash([legacy])
            assert [e.memory_id for e in store.trash_list()] == [legacy.memory_id]
            assert store.trash_restore(legacy.memory_id) is not None
        finally:
            store.client.close()


class TestTrashRestoreSafety:
    def test_restore_refuses_to_clobber_live_id(self, fake_store):
        """An export→trash→import round-trip can resurrect a trashed id; the
        pre-fix restore silently no-opped on the live row while destroying
        the trash entry."""
        m = fake_store.remember("original text", tags=["keep"])
        exported = fake_store.export(fake_store.path + "/dump.ahkai")
        fake_store.trash(m.id)
        fake_store.import_(exported.path)
        assert fake_store.get(m.id) is not None

        assert fake_store.trash_restore(m.id) is None
        # The snapshot must still be recoverable, not destroyed.
        assert [e.memory_id for e in fake_store.trash_list()] == [m.id]

    def test_duplicate_entries_restore_newest_and_keep_older(self, fake_store):
        m = fake_store.remember("version ONE")
        fake_store.trash(m.id)
        entries = fake_store._read_trash()
        newer = dataclasses.replace(
            entries[0], deleted_at=entries[0].deleted_at + 10,
            memory={**entries[0].memory, "text": "version TWO"})
        fake_store._write_trash(entries + [newer])

        restored = fake_store.trash_restore(m.id)
        assert restored.text == "version TWO"
        # The older snapshot stays parked, still recoverable after a forget.
        assert len(fake_store.trash_list()) == 1
        fake_store.forget(m.id)
        assert fake_store.trash_restore(m.id).text == "version ONE"


class TestTrashCorruption:
    def test_truncated_gzip_member_keeps_earlier_entries(self, fake_store):
        """A crash mid-append truncates the gzip member itself — EOFError,
        not bad JSON — and must not make the whole trash unreadable."""
        a = fake_store.remember("safe entry")
        b = fake_store.remember("casualty entry")
        fake_store.trash(a.id)
        size_after_first = fake_store.trash_path.stat().st_size
        fake_store.trash(b.id)

        # Cut into the middle of the second gzip member — the shape a crash
        # mid-append leaves behind.
        raw = fake_store.trash_path.read_bytes()
        cut = size_after_first + (len(raw) - size_after_first) // 2
        fake_store.trash_path.write_bytes(raw[:cut])

        assert [e.memory_id for e in fake_store.trash_list()] == [a.id]
        assert fake_store.trash_restore(a.id) is not None
        assert fake_store.trash_purge() >= 0
