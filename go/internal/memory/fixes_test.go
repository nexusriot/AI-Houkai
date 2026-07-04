package memory

// Regression tests for the Python-parity review fixes: importance-pointer
// semantics, supersede/restore invariants, subgraph traversal, undoable
// unlink/edit, validation vocabularies, and journal accuracy.

import (
	"context"
	"errors"
	"path/filepath"
	"testing"
)

// importance: nil = unset, explicit 0 kept, clamped [0,1]

func TestRememberExplicitZeroImportanceKept(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	m, _, _, err := store.Remember(ctx, "worthless trivia", RememberOpts{Importance: Float32Ptr(0)})
	if err != nil {
		t.Fatal(err)
	}
	if m.Importance != 0 {
		t.Errorf("explicit importance 0 = %v, want 0 (must not fall back to the 0.5 default)", m.Importance)
	}
	// And it round-trips through storage.
	got, err := store.GetByID(ctx, m.ID)
	if err != nil {
		t.Fatal(err)
	}
	if got.Importance != 0 {
		t.Errorf("stored importance = %v, want 0", got.Importance)
	}
}

func TestRememberImportanceClamped(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	hi, _, _, err := store.Remember(ctx, "over the top", RememberOpts{Importance: Float32Ptr(3.5)})
	if err != nil {
		t.Fatal(err)
	}
	if hi.Importance != 1 {
		t.Errorf("importance 3.5 clamped to %v, want 1", hi.Importance)
	}
	lo, _, _, err := store.Remember(ctx, "below the floor", RememberOpts{Importance: Float32Ptr(-2)})
	if err != nil {
		t.Fatal(err)
	}
	if lo.Importance != 0 {
		t.Errorf("importance -2 clamped to %v, want 0", lo.Importance)
	}
}

// default type semantic + text strip

func TestRememberDefaultsToSemanticAndStripsText(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	m, _, _, err := store.Remember(ctx, "  padded text\n", RememberOpts{})
	if err != nil {
		t.Fatal(err)
	}
	if m.Type != Semantic {
		t.Errorf("default type = %q, want %q (matches Python)", m.Type, Semantic)
	}
	if m.Text != "padded text" {
		t.Errorf("text = %q, want %q (surrounding whitespace stripped)", m.Text, "padded text")
	}
}

// validation vocabularies

func TestRememberValidation(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	cases := []struct {
		name string
		opts RememberOpts
	}{
		{"bad type", RememberOpts{Type: "opinions"}},
		{"bad polarity", RememberOpts{Polarity: 5}},
		{"bad on_conflict", RememberOpts{OnConflict: "explode"}},
		{"comma in tag", RememberOpts{Tags: []string{"a,b"}}},
	}
	for _, tc := range cases {
		_, _, _, err := store.Remember(ctx, "some text", tc.opts)
		if !IsValidationError(err) {
			t.Errorf("%s: err = %v, want ValidationError", tc.name, err)
		}
	}
	if c, _ := store.Count(ctx); c != 0 {
		t.Errorf("rejected writes must not be stored, count = %d", c)
	}
}

func TestRecallValidation(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	if _, err := store.Recall(ctx, "q", 3, RecallOpts{Mode: "hybird"}); !IsValidationError(err) {
		t.Errorf("bad mode: err = %v, want ValidationError", err)
	}
	if _, err := store.Recall(ctx, "q", 3, RecallOpts{Fusion: "borda"}); !IsValidationError(err) {
		t.Errorf("bad fusion: err = %v, want ValidationError", err)
	}
	if _, err := store.Recall(ctx, "q", 3, RecallOpts{Type: "opinions"}); !IsValidationError(err) {
		t.Errorf("bad type filter: err = %v, want ValidationError", err)
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

func TestImportOnConflictValidation(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	_, err := store.Import(ctx, filepath.Join(t.TempDir(), "missing.ahkai"), ImportOpts{OnConflict: "merge"})
	if !IsValidationError(err) {
		t.Errorf("bad import policy: err = %v, want ValidationError", err)
	}
}

// supersede invariants

func TestSupersedeRejectsSelfDanglingAndCycle(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	a, _, _, _ := mustRemember(t, store, "older fact", RememberOpts{})
	b, _, _, _ := mustRemember(t, store, "newer fact", RememberOpts{})

	if err := store.Supersede(ctx, a, a); !IsValidationError(err) {
		t.Errorf("self supersede: err = %v, want ValidationError", err)
	}
	if err := store.Supersede(ctx, a, "no-such-id"); !errors.Is(err, ErrNotFound) {
		t.Errorf("dangling new_id: err = %v, want ErrNotFound", err)
	}
	if err := store.Supersede(ctx, "no-such-id", b); !errors.Is(err, ErrNotFound) {
		t.Errorf("dangling old_id: err = %v, want ErrNotFound", err)
	}

	if err := store.Supersede(ctx, a, b); err != nil {
		t.Fatalf("Supersede: %v", err)
	}
	// b is now a's superseder; the reverse supersede would create a cycle.
	if err := store.Supersede(ctx, b, a); !IsValidationError(err) {
		t.Errorf("cycle: err = %v, want ValidationError", err)
	}
}

func TestSupersedeIdempotentAndLinkDirection(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	a, _, _, _ := mustRemember(t, store, "obsolete rule", RememberOpts{})
	b, _, _, _ := mustRemember(t, store, "current rule", RememberOpts{})

	if err := store.Supersede(ctx, a, b); err != nil {
		t.Fatalf("Supersede: %v", err)
	}
	// Re-superseding by the same id is a no-op, not an error.
	if err := store.Supersede(ctx, a, b); err != nil {
		t.Errorf("idempotent re-supersede: %v, want nil", err)
	}

	oldMem, _ := store.GetByID(ctx, a)
	newMem, _ := store.GetByID(ctx, b)
	if oldMem.SupersededBy != b {
		t.Errorf("old superseded_by = %q, want %q", oldMem.SupersededBy, b)
	}
	// Exactly ONE supersedes edge, and it lives on the NEW memory pointing at
	// the old one (matching Python — the old memory carries no edge).
	if len(oldMem.Links) != 0 {
		t.Errorf("old memory should carry no links, got %v", oldMem.Links)
	}
	if len(newMem.Links) != 1 || newMem.Links[0].To != a || newMem.Links[0].Rel != RelSupersedes {
		t.Errorf("new memory links = %v, want exactly [{%s supersedes}]", newMem.Links, a)
	}
}

func TestRestoreRemovesSupersederEdge(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	a, _, _, _ := mustRemember(t, store, "restore me", RememberOpts{})
	b, _, _, _ := mustRemember(t, store, "the replacement", RememberOpts{})
	if err := store.Supersede(ctx, a, b); err != nil {
		t.Fatal(err)
	}

	ok, err := store.Restore(ctx, a)
	if err != nil || !ok {
		t.Fatalf("Restore = %v, %v; want true, nil", ok, err)
	}
	oldMem, _ := store.GetByID(ctx, a)
	newMem, _ := store.GetByID(ctx, b)
	if oldMem.SupersededBy != "" || oldMem.SupersededAt != 0 {
		t.Errorf("restore left supersede markers: by=%q at=%v", oldMem.SupersededBy, oldMem.SupersededAt)
	}
	if len(newMem.Links) != 0 {
		t.Errorf("restore left an orphan supersedes edge: %v", newMem.Links)
	}

	// Restoring a non-superseded (or missing) memory reports false, not error.
	if ok, err := store.Restore(ctx, a); ok || err != nil {
		t.Errorf("Restore(not superseded) = %v, %v; want false, nil", ok, err)
	}
	if ok, err := store.Restore(ctx, "no-such-id"); ok || err != nil {
		t.Errorf("Restore(missing) = %v, %v; want false, nil", ok, err)
	}
}

func TestUndoSupersedeLeavesNoOrphanEdges(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	a, _, _, _ := mustRemember(t, store, "undo the supersede", RememberOpts{})
	b, _, _, _ := mustRemember(t, store, "its replacement", RememberOpts{})
	if err := store.Supersede(ctx, a, b); err != nil {
		t.Fatal(err)
	}
	entries, _ := store.Journal().Read(ReadOpts{Op: "supersede", MemoryID: a})
	if len(entries) != 1 {
		t.Fatalf("journal supersede entries = %d, want 1", len(entries))
	}
	ok, err := store.Undo(ctx, entries[0])
	if err != nil || !ok {
		t.Fatalf("Undo(supersede) = %v, %v; want true, nil", ok, err)
	}
	oldMem, _ := store.GetByID(ctx, a)
	newMem, _ := store.GetByID(ctx, b)
	if oldMem.SupersededBy != "" {
		t.Errorf("undo left superseded_by = %q", oldMem.SupersededBy)
	}
	if len(newMem.Links) != 0 {
		t.Errorf("undo left an orphan supersedes edge: %v", newMem.Links)
	}
	// A second undo of the same entry is a no-op (memory no longer superseded).
	if ok, err := store.Undo(ctx, entries[0]); ok || err != nil {
		t.Errorf("second Undo = %v, %v; want false, nil", ok, err)
	}
}

// per-call on_conflict + raise rollback

func TestRememberPerCallOnConflictOverridesStorePolicy(t *testing.T) {
	// Store policy is ignore; the per-call raise must still detect + roll back.
	store := newConflictStore(t, PolicyIgnore, 0.1)
	ctx := context.Background()

	first, stored, _, err := store.Remember(ctx, "the API gateway is nginx", RememberOpts{Type: Semantic})
	if err != nil || !stored {
		t.Fatalf("first Remember: %v", err)
	}
	_, stored, conflicts, err := store.Remember(ctx, "the API gateway is nginx",
		RememberOpts{Type: Semantic, OnConflict: PolicyRaise})
	var ce *ConflictError
	if stored || !errors.As(err, &ce) || len(conflicts) == 0 {
		t.Fatalf("per-call raise: stored=%v err=%v conflicts=%d, want rejection", stored, err, len(conflicts))
	}
	// Rollback: only the first memory remains, untouched.
	if c, _ := store.Count(ctx); c != 1 {
		t.Errorf("count after rollback = %d, want 1", c)
	}
	got, _ := store.GetByID(ctx, first.ID)
	if got.SupersededBy != "" || len(got.Links) != 0 {
		t.Errorf("raise rollback disturbed the existing memory: %+v", got)
	}
}

func TestRememberConflictSupersedeLinksNewToOld(t *testing.T) {
	store := newConflictStore(t, PolicySupersede, 0.1)
	ctx := context.Background()
	oldMem, _, _, err := store.Remember(ctx, "the primary region is us-east-1", RememberOpts{Type: Semantic})
	if err != nil {
		t.Fatal(err)
	}
	newMem, stored, _, err := store.Remember(ctx, "the primary region is us-east-1", RememberOpts{Type: Semantic})
	if err != nil || !stored {
		t.Fatalf("supersede policy Remember: stored=%v err=%v", stored, err)
	}
	// The NEW memory carries the supersedes edge; the old one carries none —
	// possible only because the new memory is added BEFORE doSupersede runs.
	gotNew, _ := store.GetByID(ctx, newMem.ID)
	gotOld, _ := store.GetByID(ctx, oldMem.ID)
	if len(gotNew.Links) != 1 || gotNew.Links[0].To != oldMem.ID || gotNew.Links[0].Rel != RelSupersedes {
		t.Errorf("new memory links = %v, want [{%s supersedes}]", gotNew.Links, oldMem.ID)
	}
	if len(gotOld.Links) != 0 {
		t.Errorf("old memory should carry no links, got %v", gotOld.Links)
	}
}

// subgraph

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

// neighbors: one pair per parallel edge, both directions

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

// unlink journals removed_rels; undo restores exactly those

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

// import journal: vectors_preserved reflects actual reuse

func TestImportJournalsVectorsPreserved(t *testing.T) {
	ctx := context.Background()

	src := newTestStore(t)
	if _, _, _, err := src.Remember(ctx, "exported knowledge", RememberOpts{}); err != nil {
		t.Fatal(err)
	}

	// Vectors included → the file's embedding is reused → preserved=true.
	withVec := filepath.Join(t.TempDir(), "with.ahkai")
	if _, err := src.Export(ctx, withVec, ExportOpts{IncludeVectors: true}); err != nil {
		t.Fatal(err)
	}
	dst1 := newTestStore(t)
	if _, err := dst1.Import(ctx, withVec, ImportOpts{OnConflict: ImportSkip}); err != nil {
		t.Fatal(err)
	}
	entries, _ := dst1.Journal().Read(ReadOpts{Op: "import"})
	if len(entries) != 1 {
		t.Fatalf("import entries = %d, want 1", len(entries))
	}
	if v, _ := entries[0].Meta["vectors_preserved"].(bool); !v {
		t.Errorf("vectors_preserved = %v, want true (file vector reused)", entries[0].Meta["vectors_preserved"])
	}

	// Vectors stripped → the row is re-embedded → preserved=false.
	noVec := filepath.Join(t.TempDir(), "without.ahkai")
	if _, err := src.Export(ctx, noVec, ExportOpts{IncludeVectors: false}); err != nil {
		t.Fatal(err)
	}
	dst2 := newTestStore(t)
	if _, err := dst2.Import(ctx, noVec, ImportOpts{OnConflict: ImportSkip}); err != nil {
		t.Fatal(err)
	}
	// Sibling TempDir stores share a default journal path, so read the LATEST
	// import entry (dst1's import is also in the file).
	entries, _ = dst2.Journal().Read(ReadOpts{Op: "import"})
	if len(entries) == 0 {
		t.Fatal("no import entries journaled")
	}
	last := entries[len(entries)-1]
	if v, _ := last.Meta["vectors_preserved"].(bool); v {
		t.Errorf("vectors_preserved = true, want false (row was re-embedded)")
	}
}

// edit: journaled, undoable, validated

func TestEditFieldsInPlace(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	a, _, _, _ := mustRemember(t, store, "the original wording", RememberOpts{Tags: []string{"keep"}})
	b, _, _, _ := mustRemember(t, store, "a linked neighbour", RememberOpts{})
	if err := store.Link(ctx, a, b, RelRelated); err != nil {
		t.Fatal(err)
	}
	before, _ := store.GetByID(ctx, a)

	newText := "the corrected wording"
	newType := Procedural
	pol := -1
	src := "review"
	m, err := store.Edit(ctx, a, EditOpts{
		Text: &newText, Type: &newType, Importance: Float32Ptr(0),
		Polarity: &pol, Source: &src,
	})
	if err != nil {
		t.Fatalf("Edit: %v", err)
	}
	if m.ID != a {
		t.Errorf("Edit changed the id: %q → %q", a, m.ID)
	}
	got, _ := store.GetByID(ctx, a)
	if got.Text != newText || got.Type != Procedural || got.Importance != 0 ||
		got.Polarity != -1 || got.Source != "review" {
		t.Errorf("edited memory = %+v", got)
	}
	// Links, tags, and created_at survive the re-embed.
	if len(got.Links) != 1 || got.Links[0].To != b {
		t.Errorf("edit lost links: %v", got.Links)
	}
	if len(got.Tags) != 1 || got.Tags[0] != "keep" {
		t.Errorf("edit lost tags: %v", got.Tags)
	}
	if got.CreatedAt != before.CreatedAt {
		t.Errorf("edit changed created_at: %v → %v", before.CreatedAt, got.CreatedAt)
	}

	// The text change was re-embedded: recall by the NEW text finds it.
	hits, err := store.Recall(ctx, newText, 1, RecallOpts{})
	if err != nil || len(hits) != 1 || hits[0].ID != a {
		t.Errorf("recall after edit = %v (err %v), want the edited memory", hits, err)
	}
}

func TestEditValidationAndNotFound(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	a, _, _, _ := mustRemember(t, store, "validate edits", RememberOpts{})

	badType := MemoryType("opinions")
	if _, err := store.Edit(ctx, a, EditOpts{Type: &badType}); !IsValidationError(err) {
		t.Errorf("bad type: err = %v, want ValidationError", err)
	}
	badPol := 7
	if _, err := store.Edit(ctx, a, EditOpts{Polarity: &badPol}); !IsValidationError(err) {
		t.Errorf("bad polarity: err = %v, want ValidationError", err)
	}
	empty := "   "
	if _, err := store.Edit(ctx, a, EditOpts{Text: &empty}); !IsValidationError(err) {
		t.Errorf("empty text: err = %v, want ValidationError", err)
	}
	if _, err := store.Edit(ctx, a, EditOpts{Tags: []string{"a,b"}}); !IsValidationError(err) {
		t.Errorf("comma tag: err = %v, want ValidationError", err)
	}
	if _, err := store.Edit(ctx, "no-such-id", EditOpts{Importance: Float32Ptr(1)}); !errors.Is(err, ErrNotFound) {
		t.Errorf("missing id: err = %v, want ErrNotFound", err)
	}
	// Importance is clamped, and tags nil-vs-empty distinguishes clear/keep.
	m, err := store.Edit(ctx, a, EditOpts{Importance: Float32Ptr(9)})
	if err != nil || m.Importance != 1 {
		t.Errorf("Edit importance 9 → %v (err %v), want clamped to 1", m.Importance, err)
	}
}

func TestEditNoOpSkipsJournal(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	a, _, _, _ := mustRemember(t, store, "steady state", RememberOpts{})

	sameText := "steady state"
	if _, err := store.Edit(ctx, a, EditOpts{Text: &sameText}); err != nil {
		t.Fatalf("no-op Edit: %v", err)
	}
	entries, _ := store.Journal().Read(ReadOpts{Op: "edit", MemoryID: a})
	if len(entries) != 0 {
		t.Errorf("no-op edit journaled %d entries, want 0", len(entries))
	}
}

func TestEditJournaledAndUndoable(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	a, _, _, _ := mustRemember(t, store, "before the edit", RememberOpts{})

	after := "after the edit"
	if _, err := store.Edit(ctx, a, EditOpts{Text: &after, Importance: Float32Ptr(0.9)}); err != nil {
		t.Fatal(err)
	}
	entries, _ := store.Journal().Read(ReadOpts{Op: "edit", MemoryID: a})
	if len(entries) != 1 {
		t.Fatalf("edit journal entries = %d, want 1", len(entries))
	}
	if entries[0].Before == nil || entries[0].After == nil {
		t.Fatalf("edit entry must carry before+after snapshots: %+v", entries[0])
	}

	ok, err := store.Undo(ctx, entries[0])
	if err != nil || !ok {
		t.Fatalf("Undo(edit) = %v, %v; want true, nil", ok, err)
	}
	got, _ := store.GetByID(ctx, a)
	if got.Text != "before the edit" {
		t.Errorf("undo text = %q, want the original", got.Text)
	}
	if got.Importance != 0.5 {
		t.Errorf("undo importance = %v, want the original 0.5", got.Importance)
	}
}
