package memory

import (
	"context"
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
