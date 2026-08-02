package memory

import (
	"context"
	"database/sql"
	"path/filepath"
	"testing"

	"github.com/nexusriot/ai-houkai/internal/sidecar"
	"github.com/nexusriot/ai-houkai/internal/vector"
)

// The derived SQLite sidecar index (E). Two invariants matter more than any
// speedup: it is off by default, and it is a cache rather than a source of
// truth — every read has a scan fallback, and an index that disagrees with the
// backend is disabled rather than trusted.

func newIndexedStore(t *testing.T) (*MemoryStore, string) {
	t.Helper()
	dir := t.TempDir()
	backend, err := vector.NewChromem(filepath.Join(dir, "s"), "idx", 16)
	if err != nil {
		t.Fatalf("NewChromem: %v", err)
	}
	cfg := DefaultStoreConfig(filepath.Join(dir, "s"), "idx")
	cfg.JournalEnabled = false
	store := NewMemoryStore(backend, &stubEmbedder{dim: 16}, cfg)
	idxPath := filepath.Join(dir, "idx.sqlite3")
	if err := store.EnableIndex(context.Background(), idxPath); err != nil {
		t.Fatalf("EnableIndex: %v", err)
	}
	t.Cleanup(func() { _ = store.backend.Close() })
	return store, idxPath
}

func TestIndexOffByDefault(t *testing.T) {
	store := newTestStore(t)
	if store.Index() != nil {
		t.Error("the sidecar index must be opt-in")
	}
}

func TestIndexWriteThrough(t *testing.T) {
	ctx := context.Background()
	store, _ := newIndexedStore(t)

	m, _, _, err := store.Remember(ctx, "indexed subject", RememberOpts{
		Type: Procedural, Tags: []string{"a", "b"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if n, _ := store.Index().Count(); n != 1 {
		t.Fatalf("indexed rows = %d, want 1", n)
	}
	if got := store.Index().TagCounts(false); got["a"] != 1 || got["b"] != 1 {
		t.Errorf("tag counts = %v", got)
	}
	if got := store.Index().TypeCounts(false); got["procedural"] != 1 {
		t.Errorf("type counts = %v", got)
	}

	// An edit must replace the indexed text, not add a second copy.
	newText := "edited subject wording"
	if _, err := store.Edit(ctx, m.ID, EditOpts{Text: &newText}); err != nil {
		t.Fatal(err)
	}
	hits := store.Index().SearchLexical("wording", 50)
	if len(hits) != 1 || hits[0] != m.ID {
		t.Errorf("fts hits after edit = %v, want exactly [%s]", hits, m.ID)
	}

	if _, err := store.Forget(ctx, m.ID); err != nil {
		t.Fatal(err)
	}
	if n, _ := store.Index().Count(); n != 0 {
		t.Errorf("indexed rows after forget = %d, want 0", n)
	}
}

func TestIndexTracksEdges(t *testing.T) {
	ctx := context.Background()
	store, _ := newIndexedStore(t)
	a, _, _, _ := store.Remember(ctx, "edge source", RememberOpts{})
	b, _, _, _ := store.Remember(ctx, "edge target", RememberOpts{})

	if err := store.Link(ctx, a.ID, b.ID, "refines"); err != nil {
		t.Fatal(err)
	}
	in := store.Index().Incoming(b.ID, "")
	if len(in) != 1 || in[0][0] != a.ID || in[0][1] != "refines" {
		t.Fatalf("incoming = %v", in)
	}
	if filtered := store.Index().Incoming(b.ID, "related"); len(filtered) != 0 {
		t.Errorf("rel filter ignored: %v", filtered)
	}

	// Forgetting the source must drop the edge too: a dangling dst would make
	// reverse lookups report a memory that is gone.
	if _, err := store.Forget(ctx, a.ID); err != nil {
		t.Fatal(err)
	}
	if in := store.Index().Incoming(b.ID, ""); len(in) != 0 {
		t.Errorf("dangling edge survived: %v", in)
	}
}

func TestNeighborsInUsesTheIndex(t *testing.T) {
	ctx := context.Background()
	store, _ := newIndexedStore(t)
	hub, _, _, _ := store.Remember(ctx, "the hub", RememberOpts{})
	a, _, _, _ := store.Remember(ctx, "points at the hub a", RememberOpts{})
	b, _, _, _ := store.Remember(ctx, "points at the hub b", RememberOpts{})
	store.Remember(ctx, "unrelated bystander", RememberOpts{})
	store.Link(ctx, a.ID, hub.ID, "refines")
	store.Link(ctx, b.ID, hub.ID, "refines")

	got, err := store.Neighbors(ctx, hub.ID, "", "in", 1)
	if err != nil {
		t.Fatal(err)
	}
	ids := map[string]bool{}
	for _, n := range got {
		ids[n.Memory.ID] = true
	}
	if len(ids) != 2 || !ids[a.ID] || !ids[b.ID] {
		t.Errorf("incoming neighbours = %v", ids)
	}
}

func TestNeighborsMatchTheScanFallback(t *testing.T) {
	// The index must not change the answer, only how it is found.
	ctx := context.Background()
	indexed, _ := newIndexedStore(t)
	plain := newTestStore(t)

	for _, store := range []*MemoryStore{indexed, plain} {
		hub, _, _, _ := store.Remember(ctx, "parity hub", RememberOpts{})
		src, _, _, _ := store.Remember(ctx, "parity source", RememberOpts{})
		other, _, _, _ := store.Remember(ctx, "parity bystander", RememberOpts{})
		store.Link(ctx, src.ID, hub.ID, "refines")
		store.Link(ctx, hub.ID, other.ID, "related")

		got, err := store.Neighbors(ctx, hub.ID, "", "both", 1)
		if err != nil {
			t.Fatal(err)
		}
		seen := map[string]string{}
		for _, n := range got {
			seen[n.Memory.Text] = n.Rel
		}
		if seen["parity source"] != "refines" || seen["parity bystander"] != "related" {
			t.Errorf("neighbours = %v", seen)
		}
	}
}

func TestListRecentCursor(t *testing.T) {
	ctx := context.Background()
	store, _ := newIndexedStore(t)
	for i := 0; i < 6; i++ {
		if _, _, _, err := store.Remember(ctx, string(rune('a'+i))+" page subject",
			RememberOpts{}); err != nil {
			t.Fatal(err)
		}
	}
	page1, err := store.ListRecentPage(ctx, ListRecentOpts{Limit: 2})
	if err != nil || len(page1) != 2 {
		t.Fatalf("page1 = %d (%v)", len(page1), err)
	}
	page2, err := store.ListRecentPage(ctx, ListRecentOpts{
		Limit: 2, Before: page1[len(page1)-1].CreatedAt})
	if err != nil || len(page2) != 2 {
		t.Fatalf("page2 = %d (%v)", len(page2), err)
	}
	for _, a := range page1 {
		for _, b := range page2 {
			if a.ID == b.ID {
				t.Errorf("pages overlap on %s", a.ID)
			}
		}
	}
}

func TestListRecentCursorWithoutIndex(t *testing.T) {
	ctx := context.Background()
	store := newTestStore(t)
	for i := 0; i < 4; i++ {
		store.Remember(ctx, string(rune('a'+i))+" fallback page", RememberOpts{})
	}
	page1, _ := store.ListRecentPage(ctx, ListRecentOpts{Limit: 2})
	page2, _ := store.ListRecentPage(ctx, ListRecentOpts{
		Limit: 2, Before: page1[len(page1)-1].CreatedAt})
	if len(page1) != 2 || len(page2) != 2 {
		t.Fatalf("pages = %d/%d, want 2/2", len(page1), len(page2))
	}
	for _, a := range page1 {
		for _, b := range page2 {
			if a.ID == b.ID {
				t.Errorf("pages overlap on %s", a.ID)
			}
		}
	}
}

func TestPurgeExpiredUsesTheIndex(t *testing.T) {
	ctx := context.Background()
	store, _ := newIndexedStore(t)
	live, _, _, _ := store.Remember(ctx, "outlives the purge", RememberOpts{})
	expiry := 1.0
	dead, _, _, _ := store.Remember(ctx, "expires immediately",
		RememberOpts{ExpiresAt: &expiry})

	purged, err := store.PurgeExpired(ctx, 0, false)
	if err != nil {
		t.Fatal(err)
	}
	if len(purged) != 1 || purged[0].ID != dead.ID {
		t.Fatalf("purged = %v", purged)
	}
	if _, err := store.GetByID(ctx, live.ID); err != nil {
		t.Errorf("live memory was purged: %v", err)
	}
}

func TestLexicalIndexReachesOutsideThePool(t *testing.T) {
	// overfetch=1, k=1 makes the vector pool exactly one row wide, so in a
	// 61-memory corpus the only way this memory can be scored is the lexical
	// channel unioning it in.
	ctx := context.Background()
	store, _ := newIndexedStore(t)
	if !store.Index().FTS {
		t.Skip("this SQLite build lacks FTS5")
	}
	target, _, _, _ := store.Remember(ctx, "the quetzalcoatlus deployment checklist",
		RememberOpts{})
	for i := 0; i < 60; i++ {
		store.Remember(ctx, "unrelated filler memory "+string(rune('a'+i%26)),
			RememberOpts{})
	}

	got, err := store.Recall(ctx, "quetzalcoatlus", 1, RecallOpts{
		Mode: ModeHybrid, Overfetch: 1, LexicalIndex: LexicalFTS,
		Weights: HybridWeights{
			Cosine: 0.2, Lexical: 0.6, Recency: 0.1, Importance: 0.1,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].ID != target.ID {
		t.Fatalf("recall = %v, want the lexical match %s", got, target.ID)
	}
}

func TestLexicalIndexIsANoopWithoutAnIndex(t *testing.T) {
	ctx := context.Background()
	store := newTestStore(t)
	store.Remember(ctx, "no index here", RememberOpts{})
	if _, err := store.Recall(ctx, "index", 3, RecallOpts{
		Mode: ModeHybrid, LexicalIndex: LexicalFTS,
	}); err != nil {
		t.Errorf("fts without an index should be a no-op, got %v", err)
	}
}

func TestIndexCountMismatchDisablesIt(t *testing.T) {
	ctx := context.Background()
	store, idxPath := newIndexedStore(t)
	store.Remember(ctx, "indexed row", RememberOpts{})

	// Simulate an index that fell behind (writes while the sidecar was gone).
	db, err := sql.Open("sqlite", idxPath)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec("DELETE FROM memories"); err != nil {
		t.Fatal(err)
	}
	db.Close()

	idx, err := sidecar.Open(idxPath, "idx")
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()
	if idx.Verify(1) {
		t.Fatal("a stale index must not verify")
	}
	if idx.Healthy() {
		t.Error("a stale index must be disabled")
	}
}

func TestDisabledIndexStillServesReverseLinks(t *testing.T) {
	ctx := context.Background()
	store, _ := newIndexedStore(t)
	a, _, _, _ := store.Remember(ctx, "disabled-path source", RememberOpts{})
	b, _, _, _ := store.Remember(ctx, "disabled-path target", RememberOpts{})
	store.Link(ctx, a.ID, b.ID, "refines")

	store.Index().Disable("test")
	got, err := store.Neighbors(ctx, b.ID, "", "in", 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].Memory.ID != a.ID {
		t.Errorf("scan fallback = %v", got)
	}
}

func TestReindexRestoresHealth(t *testing.T) {
	ctx := context.Background()
	store, _ := newIndexedStore(t)
	store.Remember(ctx, "restore one", RememberOpts{})
	store.Remember(ctx, "restore two", RememberOpts{})
	store.Index().Disable("test")

	res, err := store.Reindex(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if !res.Enabled || res.Indexed != 2 || !res.Healthy {
		t.Fatalf("reindex = %+v", res)
	}
	if n, _ := store.Index().Count(); n != 2 {
		t.Errorf("indexed rows = %d, want 2", n)
	}
}

func TestReindexWithoutAnIndex(t *testing.T) {
	store := newTestStore(t)
	res, err := store.Reindex(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if res.Enabled || res.Error == "" {
		t.Errorf("reindex without an index = %+v", res)
	}
}

func TestFTSQueryQuotesTokens(t *testing.T) {
	for _, tc := range []struct{ in, want string }{
		{"deploy runbook", `"deploy" OR "runbook"`},
		{"", ""},
		{"   ", ""},
		// Bare punctuation would be read as FTS5 operators (or a syntax error).
		{"foo-bar", `"foo" OR "bar"`},
		{`say "hi"`, `"say" OR "hi"`},
		{"NEAR AND OR", `"NEAR" OR "AND" OR "OR"`},
	} {
		if got := sidecar.FTSQuery(tc.in); got != tc.want {
			t.Errorf("FTSQuery(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

func TestOperatorSoupDoesNotBreakSearch(t *testing.T) {
	ctx := context.Background()
	store, _ := newIndexedStore(t)
	store.Remember(ctx, "harmless subject", RememberOpts{})
	if got := store.Index().SearchLexical(`*"^ NEAR( AND`, 10); len(got) != 0 {
		t.Errorf("operator soup returned %v", got)
	}
}
