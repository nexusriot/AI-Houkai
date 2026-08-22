package memory

import (
	"context"
	"errors"
	"hash/fnv"
	"math"
	"path/filepath"
	"testing"
	"time"

	"github.com/nexusriot/ai-houkai/internal/vector"
)

// stubEmbedder deterministically hashes a text into a unit vector of fixed dim.
// Identical inputs produce identical vectors; different inputs produce different
// (mostly orthogonal) ones.
type stubEmbedder struct{ dim int }

func (e *stubEmbedder) Dim() int { return e.dim }

func (e *stubEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, t := range texts {
		v := make([]float32, e.dim)
		h := fnv.New64a()
		h.Write([]byte(t))
		seed := h.Sum64()
		// Generate dim values from the seed via a simple LCG.
		for j := 0; j < e.dim; j++ {
			seed = seed*6364136223846793005 + 1442695040888963407
			v[j] = float32(int64(seed>>33)%1000) / 1000.0
		}
		// L2-normalise so cosine similarity is well-behaved.
		var norm float64
		for _, x := range v {
			norm += float64(x) * float64(x)
		}
		norm = math.Sqrt(norm)
		if norm > 0 {
			for j := range v {
				v[j] = float32(float64(v[j]) / norm)
			}
		}
		out[i] = v
	}
	return out, nil
}

func newTestStore(t *testing.T) *MemoryStore {
	t.Helper()
	dir := t.TempDir()
	backend, err := vector.NewChromem(filepath.Join(dir, "s"), "test", 16)
	if err != nil {
		t.Fatalf("NewChromem: %v", err)
	}
	t.Cleanup(func() { _ = backend.Close() })

	cfg := DefaultStoreConfig(dir, "test")
	return NewMemoryStore(backend, &stubEmbedder{dim: 16}, cfg)
}

func TestRememberRecall(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	m, stored, _, err := store.Remember(ctx, "the cat sat on the mat", RememberOpts{
		Type: Semantic, Tags: []string{"animals"}, Importance: Float32Ptr(0.7),
	})
	if err != nil || !stored {
		t.Fatalf("Remember: stored=%v err=%v", stored, err)
	}
	if m.ID == "" {
		t.Error("Remember returned empty ID")
	}

	// Same query text should rank itself first.
	hits, err := store.Recall(ctx, "the cat sat on the mat", 5, RecallOpts{})
	if err != nil {
		t.Fatalf("Recall: %v", err)
	}
	if len(hits) == 0 || hits[0].ID != m.ID {
		t.Errorf("expected own memory back, got %+v", hits)
	}
	if hits[0].AccessCount < 1 {
		t.Errorf("Recall should have touched access_count, got %d", hits[0].AccessCount)
	}
}

func TestRecallTypeFilter(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	_, _, _, _ = store.Remember(ctx, "episode one", RememberOpts{Type: Episodic})
	semID, _, _, _ := mustRemember(t, store, "semantic fact", RememberOpts{Type: Semantic})

	hits, err := store.Recall(ctx, "fact", 5, RecallOpts{Type: Semantic})
	if err != nil {
		t.Fatalf("Recall: %v", err)
	}
	for _, h := range hits {
		if h.Type != Semantic {
			t.Errorf("type filter leaked %q", h.Type)
		}
	}
	if len(hits) != 1 || hits[0].ID != semID {
		t.Errorf("want semantic memory only, got %+v", hits)
	}
}

func TestRecallSourceFilter(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	gitID, _, _, _ := mustRemember(t, store, "deploy via git", RememberOpts{Source: "git"})
	_, _, _, _ = store.Remember(ctx, "deploy via runbook", RememberOpts{Source: "runbook"})

	hits, err := store.Recall(ctx, "deploy", 5, RecallOpts{Source: "git"})
	if err != nil {
		t.Fatalf("Recall: %v", err)
	}
	if len(hits) != 1 || hits[0].ID != gitID {
		t.Fatalf("source filter: want only git memory, got %+v", hits)
	}
	if hits[0].Source != "git" {
		t.Errorf("source leaked: %q", hits[0].Source)
	}
}

func TestRecallSinceUntilFilter(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	oldID, _, _, _ := mustRemember(t, store, "ancient deploy note", RememberOpts{})
	newID, _, _, _ := mustRemember(t, store, "recent deploy note", RememberOpts{})

	// Backdate the first memory to ~100 days ago.
	old, err := store.GetByID(ctx, oldID)
	if err != nil {
		t.Fatalf("GetByID: %v", err)
	}
	cutoff := float64(time.Now().Add(-50 * 24 * time.Hour).Unix())
	old.CreatedAt = float64(time.Now().Add(-100 * 24 * time.Hour).Unix())
	if err := store.UpdateMemory(ctx, old, false); err != nil {
		t.Fatalf("UpdateMemory: %v", err)
	}

	// since=cutoff keeps only the recent memory.
	hits, _ := store.Recall(ctx, "deploy note", 10, RecallOpts{Since: cutoff})
	if len(hits) != 1 || hits[0].ID != newID {
		t.Fatalf("since filter: want only recent memory, got %+v", hits)
	}

	// until=cutoff keeps only the backdated memory.
	hits, _ = store.Recall(ctx, "deploy note", 10, RecallOpts{Until: cutoff})
	if len(hits) != 1 || hits[0].ID != oldID {
		t.Fatalf("until filter: want only old memory, got %+v", hits)
	}
}

func TestForget(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	id, _, _, _ := mustRemember(t, store, "delete me", RememberOpts{})

	ok, err := store.Forget(ctx, id)
	if err != nil || !ok {
		t.Fatalf("Forget: ok=%v err=%v", ok, err)
	}
	ok, _ = store.Forget(ctx, id)
	if ok {
		t.Error("Forget on missing id should return false")
	}
}

func TestSupersedeAndRestore(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	oldID, _, _, _ := mustRemember(t, store, "old fact", RememberOpts{Type: Semantic})
	newID, _, _, _ := mustRemember(t, store, "new fact", RememberOpts{Type: Semantic})

	if err := store.Supersede(ctx, oldID, newID); err != nil {
		t.Fatalf("Supersede: %v", err)
	}
	old, err := store.GetByID(ctx, oldID)
	if err != nil {
		t.Fatalf("GetByID: %v", err)
	}
	if old.SupersededBy != newID {
		t.Errorf("superseded_by = %q, want %q", old.SupersededBy, newID)
	}

	// Default Recall excludes superseded.
	hits, _ := store.Recall(ctx, "old fact", 10, RecallOpts{})
	for _, h := range hits {
		if h.ID == oldID {
			t.Error("superseded memory leaked into default Recall")
		}
	}

	// IncludeSuperseded brings it back.
	hits, _ = store.Recall(ctx, "old fact", 10, RecallOpts{IncludeSuperseded: true})
	found := false
	for _, h := range hits {
		if h.ID == oldID {
			found = true
		}
	}
	if !found {
		t.Error("IncludeSuperseded=true should surface the old memory")
	}

	restored, err := store.Restore(ctx, oldID)
	if err != nil {
		t.Fatalf("Restore: %v", err)
	}
	if !restored {
		t.Fatal("Restore = false, want true for a superseded memory")
	}
	old, _ = store.GetByID(ctx, oldID)
	if old.SupersededBy != "" {
		t.Errorf("Restore should clear SupersededBy, got %q", old.SupersededBy)
	}
}

func TestLinkUnlinkNeighbors(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	a, _, _, _ := mustRemember(t, store, "node a", RememberOpts{})
	b, _, _, _ := mustRemember(t, store, "node b", RememberOpts{})

	if err := store.Link(ctx, a, b, RelRelated); err != nil {
		t.Fatalf("Link: %v", err)
	}
	nbs, err := store.Neighbors(ctx, a, "", "out", 1)
	if err != nil {
		t.Fatalf("Neighbors: %v", err)
	}
	if len(nbs) != 1 || nbs[0].ID != b {
		t.Errorf("want neighbor %s, got %+v", b, nbs)
	}

	removed, err := store.Unlink(ctx, a, b, "")
	if err != nil || removed != 1 {
		t.Fatalf("Unlink: removed=%d err=%v", removed, err)
	}
	nbs, _ = store.Neighbors(ctx, a, "", "out", 1)
	if len(nbs) != 0 {
		t.Errorf("after Unlink, want 0 neighbors, got %d", len(nbs))
	}
}

func TestPrefixResolution(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	id, _, _, _ := mustRemember(t, store, "find me by prefix", RememberOpts{})

	got, err := store.GetByID(ctx, id[:8])
	if err != nil {
		t.Fatalf("GetByID(prefix): %v", err)
	}
	if got.ID != id {
		t.Errorf("prefix resolved to wrong id: got %s want %s", got.ID, id)
	}
}

func TestStats(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	_, _, _, _ = mustRemember(t, store, "one", RememberOpts{Type: Episodic, Tags: []string{"x"}})
	_, _, _, _ = mustRemember(t, store, "two", RememberOpts{Type: Semantic, Tags: []string{"x"}})

	s, err := store.Stats(ctx)
	if err != nil {
		t.Fatalf("Stats: %v", err)
	}
	if s["active"].(int) != 2 {
		t.Errorf("active: got %v, want 2", s["active"])
	}
	if s["total"].(int) != 2 {
		t.Errorf("total: got %v, want 2", s["total"])
	}
	// top_tags must be an object {tag:count}, matching Python's dict shape.
	if tt, ok := s["top_tags"].(map[string]int); !ok || tt["x"] != 2 {
		t.Errorf("top_tags should be a {tag:count} map with x=2, got %#v", s["top_tags"])
	}
}

func mustRemember(t *testing.T, s *MemoryStore, text string, opts RememberOpts) (string, bool, []Conflict, error) {
	t.Helper()
	m, stored, conflicts, err := s.Remember(context.Background(), text, opts)
	if err != nil {
		t.Fatalf("Remember(%q): %v", text, err)
	}
	return m.ID, stored, conflicts, err
}

func TestNuke(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	for i := 0; i < 3; i++ {
		store.Remember(ctx, "memory number "+string(rune('a'+i)), RememberOpts{Type: Semantic})
	}
	if c, _ := store.Count(ctx); c != 3 {
		t.Fatalf("setup: want 3, got %d", c)
	}
	deleted, err := store.Nuke(ctx)
	if err != nil {
		t.Fatalf("Nuke: %v", err)
	}
	if deleted != 3 {
		t.Errorf("Nuke deleted %d, want 3", deleted)
	}
	if c, _ := store.Count(ctx); c != 0 {
		t.Errorf("after nuke count = %d, want 0", c)
	}
	// Nuking an empty collection returns 0, not an error.
	if d, err := store.Nuke(ctx); err != nil || d != 0 {
		t.Errorf("empty nuke: got (%d, %v), want (0, nil)", d, err)
	}
}

func TestFindConflictsGlobalAllPairs(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	// 5 identical same-type memories → every pair is a duplicate conflict.
	for i := 0; i < 5; i++ {
		store.Remember(ctx, "the release pipeline is healthy", RememberOpts{Type: Semantic})
	}
	got, err := store.FindConflicts(ctx, "", 0) // 0 → store default threshold
	if err != nil {
		t.Fatalf("FindConflicts: %v", err)
	}
	if len(got) != 10 { // C(5,2)
		t.Errorf("global scan should find all 10 pairs, got %d", len(got))
	}
	// A superseded member must not appear in any pair.
	all, _ := store.ListRecent(ctx, 0, true, true)
	_ = store.Supersede(ctx, all[0].ID, all[1].ID)
	got2, _ := store.FindConflicts(ctx, "", 0)
	for _, c := range got2 {
		if c.A.ID == all[0].ID || c.B.ID == all[0].ID {
			t.Errorf("superseded memory %s should be excluded from global conflicts", all[0].ID)
		}
	}
}

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
