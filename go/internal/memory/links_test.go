package memory

import (
	"context"
	"errors"
	"testing"
)

func TestAddLinkIdempotent(t *testing.T) {
	m := &Memory{}
	addLink(m, "x", RelRelated)
	addLink(m, "x", RelRelated)
	if len(m.Links) != 1 {
		t.Errorf("addLink should be idempotent by (to, rel); got %d links", len(m.Links))
	}
}

func TestAddLinkDifferentRel(t *testing.T) {
	m := &Memory{}
	addLink(m, "x", RelRelated)
	addLink(m, "x", RelSupersedes)
	if len(m.Links) != 2 {
		t.Errorf("different rel to same target should add new link; got %d", len(m.Links))
	}
}

func TestRemoveLinksAllRels(t *testing.T) {
	m := &Memory{Links: []Link{
		{To: "x", Rel: RelRelated},
		{To: "x", Rel: RelSupersedes},
		{To: "y", Rel: RelRelated},
	}}
	removed := removeLinks(m, "x", "")
	if removed != 2 {
		t.Errorf("want 2 removed, got %d", removed)
	}
	if len(m.Links) != 1 || m.Links[0].To != "y" {
		t.Errorf("unexpected remaining links: %+v", m.Links)
	}
}

func TestRemoveLinksSpecificRel(t *testing.T) {
	m := &Memory{Links: []Link{
		{To: "x", Rel: RelRelated},
		{To: "x", Rel: RelSupersedes},
	}}
	removed := removeLinks(m, "x", RelRelated)
	if removed != 1 {
		t.Errorf("want 1 removed, got %d", removed)
	}
	if len(m.Links) != 1 || m.Links[0].Rel != RelSupersedes {
		t.Errorf("wrong link remaining: %+v", m.Links)
	}
}

func TestLinkValidation(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	a, _, _, _ := mustRemember(t, store, "left endpoint", RememberOpts{})
	b, _, _, _ := mustRemember(t, store, "right endpoint", RememberOpts{})

	if err := store.Link(ctx, a, b, "friend_of"); !IsValidationError(err) {
		t.Errorf("unknown rel: err = %v, want ValidationError", err)
	}
	if err := store.Link(ctx, a, a, RelRelated); !IsValidationError(err) {
		t.Errorf("self link: err = %v, want ValidationError", err)
	}
	// A dangling destination is rejected — graph walkers skip unresolvable
	// targets, so the edge would be stored but unreachable.
	if err := store.Link(ctx, a, "no-such-id", RelRelated); !errors.Is(err, ErrNotFound) {
		t.Errorf("dangling dst: err = %v, want ErrNotFound", err)
	}
	// Nothing above may have written an edge.
	got, _ := store.GetByID(ctx, a)
	if len(got.Links) != 0 {
		t.Errorf("rejected links must not be stored, got %v", got.Links)
	}
}

func TestNeighborsDirectionValidation(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	a, _, _, _ := mustRemember(t, store, "solo node", RememberOpts{})
	if _, err := store.Neighbors(ctx, a, "", "sideways", 1); !IsValidationError(err) {
		t.Errorf("bad direction: err = %v, want ValidationError", err)
	}
}

func TestSubgraphEmptyAndUnknownSeeds(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	g, err := store.Subgraph(ctx, nil, 2)
	if err != nil {
		t.Fatalf("Subgraph(nil): %v", err)
	}
	if len(g.Nodes) != 0 || len(g.Edges) != 0 {
		t.Errorf("Subgraph(nil) = %d nodes / %d edges, want empty", len(g.Nodes), len(g.Edges))
	}

	g, err = store.Subgraph(ctx, []string{"no-such-id"}, 2)
	if err != nil {
		t.Fatalf("Subgraph(unknown): %v", err)
	}
	if len(g.Nodes) != 0 || len(g.Edges) != 0 {
		t.Errorf("Subgraph(unknown seed) should be empty, got %d nodes", len(g.Nodes))
	}
}

func TestSubgraphMultipleSeeds(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	// Two disconnected components: a→b and c (isolated).
	a, _, _, _ := mustRemember(t, store, "component one root", RememberOpts{})
	b, _, _, _ := mustRemember(t, store, "component one leaf", RememberOpts{})
	c, _, _, _ := mustRemember(t, store, "component two isolate", RememberOpts{})
	if err := store.Link(ctx, a, b, RelRelated); err != nil {
		t.Fatal(err)
	}

	g, err := store.Subgraph(ctx, []string{a, c}, 1)
	if err != nil {
		t.Fatalf("Subgraph: %v", err)
	}
	nodes := map[string]bool{}
	for _, n := range g.Nodes {
		nodes[n.ID] = true
	}
	// EVERY seed is expanded — c must appear even though a was seeded first.
	for _, want := range []string{a, b, c} {
		if !nodes[want] {
			t.Errorf("node %s missing from multi-seed subgraph (got %v)", want, nodes)
		}
	}
	if len(g.Edges) != 1 || g.Edges[0].From != a || g.Edges[0].To != b {
		t.Errorf("edges = %v, want exactly a→b", g.Edges)
	}
}

func TestSubgraphDepthAndSeedRevisit(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	// Chain a→b→c. Seeding [c, a] with depth 2 must still expand a's chain
	// even though c was already visited as a leaf.
	a, _, _, _ := mustRemember(t, store, "chain head", RememberOpts{})
	b, _, _, _ := mustRemember(t, store, "chain middle", RememberOpts{})
	c, _, _, _ := mustRemember(t, store, "chain tail", RememberOpts{})
	if err := store.Link(ctx, a, b, RelRefines); err != nil {
		t.Fatal(err)
	}
	if err := store.Link(ctx, b, c, RelRefines); err != nil {
		t.Fatal(err)
	}

	g, err := store.Subgraph(ctx, []string{c, a}, 2)
	if err != nil {
		t.Fatalf("Subgraph: %v", err)
	}
	if len(g.Nodes) != 3 {
		t.Errorf("nodes = %d, want 3 (a, b, c)", len(g.Nodes))
	}
	if len(g.Edges) != 2 {
		t.Errorf("edges = %v, want a→b and b→c", g.Edges)
	}

	// depth 1 from a stops at b: the a→b edge is present, b→c is not walked.
	g, err = store.Subgraph(ctx, []string{a}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(g.Nodes) != 2 || len(g.Edges) != 1 {
		t.Errorf("depth-1 subgraph = %d nodes / %d edges, want 2/1", len(g.Nodes), len(g.Edges))
	}
}

func TestNeighborsReportsParallelEdges(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	a, _, _, _ := mustRemember(t, store, "edge source", RememberOpts{})
	b, _, _, _ := mustRemember(t, store, "edge target", RememberOpts{})
	if err := store.Link(ctx, a, b, RelRelated); err != nil {
		t.Fatal(err)
	}
	if err := store.Link(ctx, a, b, RelRefines); err != nil {
		t.Fatal(err)
	}

	rels := func(rs []NeighborResult) map[string]int {
		out := map[string]int{}
		for _, r := range rs {
			out[r.Rel]++
		}
		return out
	}

	out, err := store.Neighbors(ctx, a, "", "out", 1)
	if err != nil {
		t.Fatal(err)
	}
	if got := rels(out); len(out) != 2 || got[RelRelated] != 1 || got[RelRefines] != 1 {
		t.Errorf("Neighbors(out) rels = %v, want one pair per parallel edge", got)
	}

	in, err := store.Neighbors(ctx, b, "", "in", 1)
	if err != nil {
		t.Fatal(err)
	}
	if got := rels(in); len(in) != 2 || got[RelRelated] != 1 || got[RelRefines] != 1 {
		t.Errorf("Neighbors(in) rels = %v, want one pair per parallel edge", got)
	}

	// A rel filter keeps only the matching parallel edge.
	filtered, err := store.Neighbors(ctx, a, RelRefines, "out", 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(filtered) != 1 || filtered[0].Rel != RelRefines {
		t.Errorf("Neighbors(out, refines) = %v, want the single refines pair", filtered)
	}
}

func TestUnlinkJournalsRemovedRelsAndUndoRestores(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	a, _, _, _ := mustRemember(t, store, "unlink src", RememberOpts{})
	b, _, _, _ := mustRemember(t, store, "unlink dst", RememberOpts{})
	if err := store.Link(ctx, a, b, RelRelated); err != nil {
		t.Fatal(err)
	}
	if err := store.Link(ctx, a, b, RelRefines); err != nil {
		t.Fatal(err)
	}

	// rel="" removes BOTH parallel edges in one call.
	n, err := store.Unlink(ctx, a, b, "")
	if err != nil || n != 2 {
		t.Fatalf("Unlink = %d, %v; want 2, nil", n, err)
	}

	entries, _ := store.Journal().Read(ReadOpts{Op: "unlink", MemoryID: a})
	if len(entries) != 1 {
		t.Fatalf("unlink journal entries = %d, want 1", len(entries))
	}
	raw, ok := entries[0].Meta["removed_rels"].([]any)
	if !ok || len(raw) != 2 {
		t.Fatalf("meta removed_rels = %v, want the 2 removed rels", entries[0].Meta["removed_rels"])
	}
	got := map[string]bool{}
	for _, r := range raw {
		got[r.(string)] = true
	}
	if !got[RelRelated] || !got[RelRefines] {
		t.Errorf("removed_rels = %v, want {related, refines}", got)
	}

	// Undo restores exactly the removed edges.
	ok2, err := store.Undo(ctx, entries[0])
	if err != nil || !ok2 {
		t.Fatalf("Undo(unlink) = %v, %v; want true, nil", ok2, err)
	}
	src, _ := store.GetByID(ctx, a)
	if len(src.Links) != 2 {
		t.Fatalf("links after undo = %v, want both edges back", src.Links)
	}
	back := map[string]bool{}
	for _, l := range src.Links {
		if l.To != b {
			t.Errorf("unexpected link target %q", l.To)
		}
		back[l.Rel] = true
	}
	if !back[RelRelated] || !back[RelRefines] {
		t.Errorf("restored rels = %v, want {related, refines}", back)
	}
}

func TestUndoLegacyUnlinkEntryWithoutRemovedRels(t *testing.T) {
	// Journal entries written before removed_rels existed carry only "rel";
	// Undo must fall back to it gracefully.
	store := newTestStore(t)
	ctx := context.Background()
	a, _, _, _ := mustRemember(t, store, "legacy src", RememberOpts{})
	b, _, _, _ := mustRemember(t, store, "legacy dst", RememberOpts{})
	if err := store.Link(ctx, a, b, RelContradicts); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Unlink(ctx, a, b, RelContradicts); err != nil {
		t.Fatal(err)
	}

	legacy := JournalEntry{
		Op: "unlink", ID: a,
		Meta: map[string]any{"src_id": a, "dst_id": b, "rel": RelContradicts},
	}
	ok, err := store.Undo(ctx, legacy)
	if err != nil || !ok {
		t.Fatalf("Undo(legacy unlink) = %v, %v; want true, nil", ok, err)
	}
	src, _ := store.GetByID(ctx, a)
	if len(src.Links) != 1 || src.Links[0].Rel != RelContradicts {
		t.Errorf("legacy undo links = %v, want the single contradicts edge", src.Links)
	}

	// An endpoint that has since been forgotten is a graceful no-op.
	if _, err := store.Forget(ctx, b); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Unlink(ctx, a, b, ""); err != nil {
		t.Fatal(err)
	}
	if ok, err := store.Undo(ctx, legacy); ok || err != nil {
		t.Errorf("Undo with forgotten endpoint = %v, %v; want false, nil", ok, err)
	}
}
