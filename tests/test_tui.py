"""Tests for the TUI — data-layer view models plus a few Textual pilot runs."""

from __future__ import annotations

import asyncio

import pytest
from rich.text import Text

from ai_houkai.tui.data import (
    Navigator,
    detail_markup,
    mem_row,
    neighbors_view,
    recent_view,
    search_view,
)

from ai_houkai.tui.app import HoukaiTui  # noqa: E402

textual = pytest.importorskip("textual")
pytest_asyncio = pytest.importorskip("pytest_asyncio")


@pytest.fixture()
def seeded(store):
    a = store.remember("Deploys go through make release", type="procedural",
                       importance=0.9, tags=["ops"])
    b = store.remember("The staging box runs Debian 12", type="semantic")
    c = store.remember("Fixed a flaky test in the journal suite", type="episodic")
    store.link(a.id, b.id, rel="refines")
    return store, (a, b, c)


class TestViews:
    def test_mem_row_shape_and_snippet(self, seeded):
        _, (a, _, _) = seeded
        row = mem_row(a, extra="x")
        assert row[0] == a.id[:8]
        assert row[1] == "procedural"
        assert row[2] == "0.90"
        assert row[4] == "x"
        long = mem_row(a.__class__(id="i" * 36, text="word " * 50, type="episodic"))
        assert len(long[5]) <= 60
        assert long[5].endswith("…")

    def test_recent_view(self, seeded):
        store, mems = seeded
        view = recent_view(store)
        assert view.kind == "recent"
        assert len(view.rows) == 3
        assert set(view.memories) == {m.id[:8] for m in mems}

    def test_search_view_scores_in_extra(self, seeded):
        store, _ = seeded
        view = search_view(store, "deployment release")
        assert view.kind == "search"
        assert view.rows
        float(view.rows[0][4])  # extra column is a parseable score
        assert "make release" in view.rows[0][5]

    def test_neighbors_view_rel_in_extra(self, seeded):
        store, (a, b, _) = seeded
        view = neighbors_view(store, a)
        assert [r[0] for r in view.rows] == [b.id[:8]]
        assert view.rows[0][4] == "refines"

    def test_neighbors_view_parallel_links_one_row_per_target(self, seeded):
        # Two edges to the same target must collapse into one row — the app
        # keys DataTable rows by id8, and a repeated key raises DuplicateKey.
        store, (a, b, _) = seeded
        store.link(a.id, b.id, rel="related")
        view = neighbors_view(store, store._get_by_id(a.id))
        assert [r[0] for r in view.rows] == [b.id[:8]]
        assert set(view.rows[0][4].split(",")) == {"refines", "related"}
        assert view.memories[b.id[:8]].id == b.id

    def test_detail_markup_contents(self, seeded):
        store, (a, b, _) = seeded
        a = store._get_by_id(a.id)  # refetch: links were added after remember()
        text = detail_markup(a)
        assert a.id[:8] in text
        assert "make release" in text
        assert "#ops" in text
        assert f"--refines--> [cyan]{b.id[:8]}" in text


class TestNavigator:
    def test_stack_push_back_bottoms_out(self, seeded):
        store, (a, _, _) = seeded
        nav = Navigator(store)
        assert nav.current.kind == "recent"      # lazily opened
        nav.open_search("deploy")
        nav.open_neighbors(a)
        assert [v.kind for v in nav.stack] == ["recent", "search", "neighbors"]
        assert " > " in nav.breadcrumb
        nav.back()
        assert nav.current.kind == "search"
        nav.back()
        nav.back()                                # bottom — stays on recent
        assert nav.current.kind == "recent"

    def test_open_recent_resets_stack(self, seeded):
        store, (a, _, _) = seeded
        nav = Navigator(store)
        nav.open_search("deploy")
        nav.open_neighbors(a)
        nav.open_recent()
        assert len(nav.stack) == 1


@pytest.mark.asyncio
async def test_app_lists_memories_and_shows_detail(seeded):
    store, mems = seeded
    app = HoukaiTui(store=store, collection="test")
    async with app.run_test() as pilot:
        table = app.query_one("#list")
        assert table.row_count == 3
        detail = app.query_one("#detail")
        assert str(detail.render()) != ""


@pytest.mark.asyncio
async def test_app_neighbors_and_back(seeded):
    store, (a, b, _) = seeded
    app = HoukaiTui(store=store, collection="test")
    async with app.run_test() as pilot:
        table = app.query_one("#list")
        # move cursor to memory `a` (row order is recency, c/b/a or similar)
        for i in range(table.row_count):
            key = table.coordinate_to_cell_key((i, 0)).row_key
            if str(key.value) == a.id[:8]:
                table.move_cursor(row=i)
                break
        await pilot.press("n")
        assert app.nav.current.kind == "neighbors"
        assert table.row_count == 1
        await pilot.press("b")
        assert app.nav.current.kind == "recent"
        assert table.row_count == 3


@pytest.mark.asyncio
async def test_app_search_flow(seeded):
    store, _ = seeded
    app = HoukaiTui(store=store, collection="test")
    async with app.run_test() as pilot:
        await pilot.press("/")
        search = app.query_one("#search")
        assert app.focused is search
        await pilot.press(*"deploy")
        await pilot.press("enter")
        assert app.nav.current.kind == "search"
        table = app.query_one("#list")
        assert table.row_count > 0


class TestSearchDoesNotTouch:
    def test_search_view_leaves_access_stats_alone(self, seeded):
        """TUI browsing is read-only: recall via the search box must not bump
        access_count/last_accessed (it would feed decay reinforcement)."""
        store, (a, _, _) = seeded
        before = store._get_by_id(a.id)
        assert before.access_count == 0
        search_view(store, "deployment release")
        after = store._get_by_id(a.id)
        assert after.access_count == 0
        assert after.last_accessed == before.last_accessed


@pytest.mark.asyncio
async def test_nuke_double_press_nukes(seeded):
    store, _ = seeded
    app = HoukaiTui(store=store, collection="test")
    async with app.run_test() as pilot:
        await pilot.press("X")
        assert app._nuke_pending is True
        await pilot.press("X")
        assert store.count() == 0


@pytest.mark.asyncio
async def test_nuke_confirmation_expires(seeded, monkeypatch):
    """The armed nuke state must expire with the warning toast — a stray X
    long after the prompt disappeared must not wipe the store."""
    store, _ = seeded
    monkeypatch.setattr(HoukaiTui, "NUKE_CONFIRM_SECONDS", 0.5)
    app = HoukaiTui(store=store, collection="test")
    async with app.run_test() as pilot:
        await pilot.press("X")
        assert app._nuke_pending is True
        await asyncio.sleep(1.0)
        await pilot.pause()
        assert app._nuke_pending is False         # expired with the toast
        await pilot.press("X")                    # stray X only re-arms
        assert store.count() == 3
        assert app._nuke_pending is True


@pytest.mark.asyncio
async def test_nuke_disarmed_by_navigation(seeded):
    store, _ = seeded
    app = HoukaiTui(store=store, collection="test")
    async with app.run_test() as pilot:
        await pilot.press("X")
        assert app._nuke_pending is True
        await pilot.press("r")                    # navigating disarms
        assert app._nuke_pending is False
        await pilot.press("X")                    # this X re-arms, not nukes
        assert store.count() == 3


class TestMarkupSafety:
    """Memory text reaches Rich-markup contexts (DataTable cells, the detail
    Static). Un-escaped, a stored `[/bold]` crashes the whole app with
    MarkupError and `arr[i]` is silently eaten from the display."""

    def test_snippet_survives_markup_round_trip(self, store):
        store.remember("indexing arr[i] fails on the [/bold] row")
        view = recent_view(store)
        snippet = view.rows[0][5]
        assert Text.from_markup(snippet).plain == \
            "indexing arr[i] fails on the [/bold] row"

    def test_detail_markup_escapes_text_tags_and_source(self, store):
        m = store.remember("list[int] beats [/dim] here",
                           tags=["a[b]"], source="scraper[3]")
        rendered = Text.from_markup(detail_markup(m)).plain
        assert "list[int] beats [/dim] here" in rendered
        assert "#a[b]" in rendered
        assert "scraper[3]" in rendered

    def test_search_title_escapes_query(self, store):
        store.remember("anything")
        view = search_view(store, "weird [/red] query")
        assert "[/red]" in Text.from_markup(view.title).plain
