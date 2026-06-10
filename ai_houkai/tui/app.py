"""houkai tui — Textual memory browser with link-graph navigation.

Keys
    /          focus the search box (semantic recall)
    escape     clear search / unfocus
    enter      (in search box) run the search
    n          open the selected memory's neighbors (walk the graph)
    b          back one view (breadcrumb stack)
    r          reload the recent view
    q          quit
"""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import DataTable, Footer, Header, Input, Static

from ai_houkai.memory_system.store import Memory, MemoryStore
from ai_houkai.tui.data import Navigator, View, detail_markup

_COLUMNS = ("ID", "TYPE", "IMP", "AGE", "REL/SCORE", "TEXT")


class HoukaiTui(App):
    """Browse memories, search them, and walk the link graph."""

    TITLE = "AI-Houkai"

    CSS = """
    #body { height: 1fr; }
    #list { width: 2fr; }
    #detail-scroll { width: 1fr; border-left: solid $primary; padding: 0 1; }
    #search { dock: bottom; display: none; }
    #search.visible { display: block; }
    #crumb { dock: top; height: 1; color: $text-muted; padding: 0 1; }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("/", "focus_search", "Search"),
        Binding("n", "neighbors", "Neighbors"),
        Binding("b", "back", "Back"),
        Binding("r", "recent", "Recent"),
        Binding("escape", "dismiss_search", show=False),
    ]

    def __init__(self, store: MemoryStore, collection: str = "") -> None:
        super().__init__()
        self.store = store
        self.nav = Navigator(store)
        self.sub_title = collection

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("", id="crumb")
        with Horizontal(id="body"):
            yield DataTable(id="list", cursor_type="row", zebra_stripes=True)
            with VerticalScroll(id="detail-scroll"):
                yield Static("", id="detail")
        yield Input(placeholder="semantic search… (enter to run, esc to close)",
                    id="search")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#list", DataTable)
        table.add_columns(*_COLUMNS)
        self._show_view(self.nav.open_recent())
        table.focus()

    def _show_view(self, view: View) -> None:
        table = self.query_one("#list", DataTable)
        table.clear()
        for row in view.rows:
            table.add_row(*row, key=row[0])
        self.query_one("#crumb", Static).update(self.nav.breadcrumb)
        if view.rows:
            table.move_cursor(row=0)
            self._show_detail(view.rows[0][0])
        else:
            self.query_one("#detail", Static).update("[dim]nothing here[/]")

    def _selected_memory(self) -> Memory | None:
        table = self.query_one("#list", DataTable)
        if table.cursor_row is None or table.row_count == 0:
            return None
        row_key = table.coordinate_to_cell_key((table.cursor_row, 0)).row_key
        return self.nav.current.memories.get(str(row_key.value))

    def _show_detail(self, id8: str) -> None:
        mem = self.nav.current.memories.get(id8)
        if mem is not None:
            self.query_one("#detail", Static).update(detail_markup(mem))

    def on_data_table_row_highlighted(
        self, event: DataTable.RowHighlighted
    ) -> None:
        if event.row_key is not None and event.row_key.value is not None:
            self._show_detail(str(event.row_key.value))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        query = event.value.strip()
        self.action_dismiss_search()
        if query:
            self._show_view(self.nav.open_search(query))

    def action_focus_search(self) -> None:
        box = self.query_one("#search", Input)
        box.add_class("visible")
        box.focus()

    def action_dismiss_search(self) -> None:
        box = self.query_one("#search", Input)
        box.value = ""
        box.remove_class("visible")
        self.query_one("#list", DataTable).focus()

    def action_neighbors(self) -> None:
        mem = self._selected_memory()
        if mem is None:
            return
        view = self.nav.open_neighbors(mem)
        if not view.rows:
            self.nav.back()
            self.notify("No links on this memory.", severity="warning")
            return
        self._show_view(view)

    def action_back(self) -> None:
        self._show_view(self.nav.back())

    def action_recent(self) -> None:
        self._show_view(self.nav.open_recent())
