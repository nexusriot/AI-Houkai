// Package tui implements `houkai tui` — a Bubble Tea memory browser with
// link-graph navigation.
//
// This file holds the view-model helpers — pure functions over MemoryStore,
// kept free of any bubbletea import so the navigation and formatting logic
// is unit-testable without a terminal.
package tui

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/nexusriot/ai-houkai/internal/memory"
)

// Row is one list row: id8, type, importance, age, extra (rel/score), snippet.
type Row struct {
	ID8     string
	Type    string
	Imp     string
	Age     string
	Extra   string
	Snippet string
}

func shortID(id string) string {
	if len(id) > 8 {
		return id[:8]
	}
	return id
}

func fmtAge(ts float64) string {
	t := time.Unix(int64(ts), 0)
	d := time.Since(t)
	switch {
	case d < time.Minute:
		return "just now"
	case d < time.Hour:
		return fmt.Sprintf("%dm ago", int(d.Minutes()))
	case d < 24*time.Hour:
		return fmt.Sprintf("%dh ago", int(d.Hours()))
	case d < 7*24*time.Hour:
		return fmt.Sprintf("%dd ago", int(d.Hours()/24))
	default:
		return t.Format("2006-01-02")
	}
}

func snippet(text string, width int) string {
	flat := strings.Join(strings.Fields(text), " ")
	if len(flat) <= width {
		return flat
	}
	return flat[:width-1] + "…"
}

func memRow(m memory.Memory, extra string) Row {
	return Row{
		ID8:     shortID(m.ID),
		Type:    string(m.Type),
		Imp:     fmt.Sprintf("%.2f", m.Importance),
		Age:     fmtAge(m.CreatedAt),
		Extra:   extra,
		Snippet: snippet(m.Text, 60),
	}
}

// View is one screen of the browser: what the list shows and why.
type View struct {
	Kind     string // "recent" | "search" | "neighbors"
	Title    string
	Rows     []Row
	Memories map[string]memory.Memory // id8 → Memory
}

func recentView(ctx context.Context, store *memory.MemoryStore, limit int) (View, error) {
	mems, err := store.ListRecent(ctx, limit, false)
	if err != nil {
		return View{}, err
	}
	v := View{
		Kind:     "recent",
		Title:    fmt.Sprintf("Recent (%d)", len(mems)),
		Memories: map[string]memory.Memory{},
	}
	for _, m := range mems {
		v.Rows = append(v.Rows, memRow(m, ""))
		v.Memories[shortID(m.ID)] = m
	}
	return v, nil
}

func searchView(ctx context.Context, store *memory.MemoryStore, query string, k int) (View, error) {
	results, err := store.Recall(ctx, query, k, memory.RecallOpts{Overfetch: 3})
	if err != nil {
		return View{}, err
	}
	v := View{
		Kind:     "search",
		Title:    fmt.Sprintf("Search: %q (%d)", query, len(results)),
		Memories: map[string]memory.Memory{},
	}
	for _, r := range results {
		v.Rows = append(v.Rows, memRow(r.Memory, fmt.Sprintf("%.3f", r.Score)))
		v.Memories[shortID(r.ID)] = r.Memory
	}
	return v, nil
}

func neighborsView(ctx context.Context, store *memory.MemoryStore, m memory.Memory) (View, error) {
	results, err := store.Neighbors(ctx, m.ID, "", "both", 1)
	if err != nil {
		return View{}, err
	}
	v := View{
		Kind:     "neighbors",
		Title:    fmt.Sprintf("Neighbors of %s (%d)", shortID(m.ID), len(results)),
		Memories: map[string]memory.Memory{},
	}
	for _, nb := range results {
		v.Rows = append(v.Rows, memRow(nb.Memory, nb.Rel))
		v.Memories[shortID(nb.ID)] = nb.Memory
	}
	return v, nil
}

// DetailText renders the detail pane for one memory (plain text; the app
// layer styles it).
func DetailText(m memory.Memory) string {
	var b strings.Builder
	fmt.Fprintf(&b, "%s  %s  imp %.2f  %s old\n",
		shortID(m.ID), m.Type, m.Importance, fmtAge(m.CreatedAt))
	if len(m.Tags) > 0 {
		parts := make([]string, len(m.Tags))
		for i, t := range m.Tags {
			parts[i] = "#" + t
		}
		fmt.Fprintf(&b, "tags: %s\n", strings.Join(parts, " "))
	}
	if m.Source != "" {
		fmt.Fprintf(&b, "source: %s\n", m.Source)
	}
	if m.SupersededBy != "" {
		fmt.Fprintf(&b, "superseded by %s\n", shortID(m.SupersededBy))
	}
	b.WriteString("\n")
	b.WriteString(m.Text)
	if len(m.Links) > 0 {
		b.WriteString("\n\nLinks (press n to walk):\n")
		for _, l := range m.Links {
			fmt.Fprintf(&b, "  --%s--> %s\n", l.Rel, shortID(l.To))
		}
	}
	return b.String()
}

// Navigator is a breadcrumb stack of Views; the TUI renders the top of the
// stack.
type Navigator struct {
	store *memory.MemoryStore
	stack []View
}

func NewNavigator(store *memory.MemoryStore) *Navigator {
	return &Navigator{store: store}
}

func (n *Navigator) OpenRecent(ctx context.Context, limit int) (View, error) {
	v, err := recentView(ctx, n.store, limit)
	if err != nil {
		return View{}, err
	}
	n.stack = []View{v}
	return v, nil
}

func (n *Navigator) OpenSearch(ctx context.Context, query string) (View, error) {
	v, err := searchView(ctx, n.store, query, 50)
	if err != nil {
		return View{}, err
	}
	n.stack = append(n.stack, v)
	return v, nil
}

func (n *Navigator) OpenNeighbors(ctx context.Context, m memory.Memory) (View, error) {
	v, err := neighborsView(ctx, n.store, m)
	if err != nil {
		return View{}, err
	}
	n.stack = append(n.stack, v)
	return v, nil
}

func (n *Navigator) Back() View {
	if len(n.stack) > 1 {
		n.stack = n.stack[:len(n.stack)-1]
	}
	return n.Current()
}

// Pop removes the top view without rendering (used when a neighbors view
// turns out empty).
func (n *Navigator) Pop() {
	if len(n.stack) > 1 {
		n.stack = n.stack[:len(n.stack)-1]
	}
}

func (n *Navigator) Current() View {
	if len(n.stack) == 0 {
		v, err := n.OpenRecent(context.Background(), 200)
		if err != nil {
			return View{Kind: "recent", Title: "Recent (0)", Memories: map[string]memory.Memory{}}
		}
		return v
	}
	return n.stack[len(n.stack)-1]
}

func (n *Navigator) Breadcrumb() string {
	titles := make([]string, len(n.stack))
	for i, v := range n.stack {
		titles[i] = v.Title
	}
	return strings.Join(titles, " > ")
}
