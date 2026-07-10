"""View-model helpers for the TUI — pure functions over MemoryStore.

Kept free of any textual import so the navigation and formatting logic
is unit-testable without a terminal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ai_houkai.cli.output import fmt_age, short_id
from ai_houkai.memory_system.store import Memory, MemoryStore

# One list row: (id8, type, importance, age, extra, snippet)
Row = tuple[str, str, str, str, str, str]


def _snippet(text: str, width: int = 60) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 1] + "…"


def mem_row(mem: Memory, *, extra: str = "") -> Row:
    return (
        short_id(mem.id),
        mem.type,
        f"{mem.importance:.2f}",
        fmt_age(mem.created_at),
        extra,
        _snippet(mem.text),
    )


@dataclass
class View:
    """One screen of the browser: what the list shows and why."""
    kind: str                       # "recent" | "search" | "neighbors"
    title: str
    rows: list[Row]
    memories: dict[str, Memory]     # id8 → Memory


def recent_view(store: MemoryStore, limit: int = 200) -> View:
    mems = store.list_recent(limit=limit)
    return View(
        kind="recent",
        title=f"Recent ({len(mems)})",
        rows=[mem_row(m) for m in mems],
        memories={short_id(m.id): m for m in mems},
    )


def search_view(store: MemoryStore, query: str, k: int = 50) -> View:
    # touch=False: browsing the TUI must not inflate access stats — recall
    # reinforcement (frequency_weight) would otherwise reward every keystroke.
    results = store.recall(query, k=k, touch=False)
    return View(
        kind="search",
        title=f"Search: {query!r} ({len(results)})",
        rows=[mem_row(m, extra=f"{score:.3f}") for m, score in results],
        memories={short_id(m.id): m for m, _ in results},
    )


def neighbors_view(store: MemoryStore, mem: Memory, depth: int = 1) -> View:
    results = store.neighbors(mem.id, direction="both", depth=depth)
    # neighbors() yields one (memory, rel) pair per edge, so parallel links
    # to the same target (A→B related + A→B refines) would repeat an id8 —
    # a duplicate DataTable row key, which crashes the app. One row per
    # target, rels joined.
    by_id: dict[str, tuple[Memory, list[str]]] = {}
    for m, rel in results:
        entry = by_id.setdefault(short_id(m.id), (m, []))
        if rel not in entry[1]:
            entry[1].append(rel)
    return View(
        kind="neighbors",
        title=f"Neighbors of {short_id(mem.id)} ({len(by_id)})",
        rows=[mem_row(m, extra=",".join(rels)) for m, rels in by_id.values()],
        memories={id8: m for id8, (m, _) in by_id.items()},
    )


def detail_markup(mem: Memory) -> str:
    """Rich console markup for the detail pane."""
    lines = [
        f"[bold cyan]{short_id(mem.id)}[/]  [magenta]{mem.type}[/]  "
        f"imp [yellow]{mem.importance:.2f}[/]  {fmt_age(mem.created_at)} old",
    ]
    if mem.tags:
        lines.append("tags: " + " ".join(f"[green]#{t}[/]" for t in mem.tags))
    if mem.source:
        lines.append(f"source: [dim]{mem.source}[/]")
    if mem.superseded_by:
        lines.append(f"[red]superseded by {short_id(mem.superseded_by)}[/]")
    lines.append("")
    lines.append(mem.text)
    if mem.links:
        lines.append("")
        lines.append("[bold]Links[/] (press n to walk):")
        for lnk in mem.links:
            lines.append(f"  --{lnk.rel}--> [cyan]{short_id(lnk.to)}[/]")
    return "\n".join(lines)


@dataclass
class Navigator:
    """Breadcrumb stack of Views; the TUI renders the top of the stack."""
    store: MemoryStore
    stack: list[View] = field(default_factory=list)

    def open_recent(self, limit: int = 200) -> View:
        self.stack = [recent_view(self.store, limit=limit)]
        return self.current

    def open_search(self, query: str) -> View:
        self.stack.append(search_view(self.store, query))
        return self.current

    def open_neighbors(self, mem: Memory) -> View:
        self.stack.append(neighbors_view(self.store, mem))
        return self.current

    def back(self) -> View:
        if len(self.stack) > 1:
            self.stack.pop()
        return self.current

    @property
    def current(self) -> View:
        if not self.stack:
            self.open_recent()
        return self.stack[-1]

    @property
    def breadcrumb(self) -> str:
        return " > ".join(v.title for v in self.stack)
