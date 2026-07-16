package memory

import (
	"context"
	"testing"
)

// batchCountEmbedder tallies Embed *invocations* (not texts) so a test can
// assert that RememberMany batches N documents into ceil(N/batchSize) calls.
type batchCountEmbedder struct {
	dim   int
	calls *int
}

func (e batchCountEmbedder) Dim() int { return e.dim }
func (e batchCountEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	*e.calls++
	out := make([][]float32, len(texts))
	for i := range out {
		v := make([]float32, e.dim)
		v[0] = 1.0
		out[i] = v
	}
	return out, nil
}

func TestRememberManyStoresAllInOrder(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	mems, err := store.RememberMany(ctx, []RememberItem{
		{Text: "alpha fact"},
		{Text: "beta fact", RememberOpts: RememberOpts{Type: Procedural, Tags: []string{"x"}}},
		{Text: "gamma fact", RememberOpts: RememberOpts{Importance: Float32Ptr(0.9)}},
	}, 128, "")
	if err != nil {
		t.Fatalf("RememberMany: %v", err)
	}
	if len(mems) != 3 {
		t.Fatalf("want 3 stored, got %d", len(mems))
	}
	if mems[0].Text != "alpha fact" || mems[1].Text != "beta fact" || mems[2].Text != "gamma fact" {
		t.Errorf("input order not preserved: %q %q %q", mems[0].Text, mems[1].Text, mems[2].Text)
	}
	if mems[1].Type != Procedural || len(mems[1].Tags) != 1 || mems[1].Tags[0] != "x" {
		t.Errorf("beta fields wrong: %+v", mems[1])
	}
	if mems[2].Importance < 0.89 || mems[2].Importance > 0.91 {
		t.Errorf("gamma importance = %v, want ~0.9", mems[2].Importance)
	}
	if n, _ := store.Count(ctx); n != 3 {
		t.Errorf("count = %d, want 3", n)
	}
}

func TestRememberManyEmpty(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	mems, err := store.RememberMany(ctx, nil, 128, "")
	if err != nil {
		t.Fatalf("RememberMany(nil): %v", err)
	}
	if len(mems) != 0 {
		t.Errorf("want 0, got %d", len(mems))
	}
	if n, _ := store.Count(ctx); n != 0 {
		t.Errorf("count = %d, want 0", n)
	}
}

func TestRememberManyBatchesEmbedCalls(t *testing.T) {
	calls := 0
	store := newStoreWithEmbedder(t, batchCountEmbedder{dim: 16, calls: &calls})
	ctx := context.Background()
	items := make([]RememberItem, 10)
	for i := range items {
		items[i] = RememberItem{Text: string(rune('a'+i)) + " item"}
	}
	if _, err := store.RememberMany(ctx, items, 4, PolicyIgnore); err != nil {
		t.Fatalf("RememberMany: %v", err)
	}
	if calls != 3 { // ceil(10 / 4)
		t.Errorf("Embed invocations = %d, want 3 (ceil(10/4))", calls)
	}
	if n, _ := store.Count(ctx); n != 10 {
		t.Errorf("count = %d, want 10", n)
	}
}

func TestRememberManyBatchSizeMustBePositive(t *testing.T) {
	store := newTestStore(t)
	if _, err := store.RememberMany(context.Background(), []RememberItem{{Text: "x"}}, 0, ""); err == nil {
		t.Fatal("expected error for batch_size=0")
	}
}

func TestRememberManyValidationAbortsBeforeWrite(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	_, err := store.RememberMany(ctx, []RememberItem{
		{Text: "ok one"},
		{Text: "bad", RememberOpts: RememberOpts{Type: "not-a-type"}},
	}, 128, "")
	if err == nil {
		t.Fatal("expected validation error for bad type")
	}
	if n, _ := store.Count(ctx); n != 0 {
		t.Errorf("no partial write expected, count = %d", n)
	}
}

func TestRememberManyRaiseRejected(t *testing.T) {
	store := newTestStore(t)
	if _, err := store.RememberMany(context.Background(), []RememberItem{{Text: "x"}}, 128, PolicyRaise); err == nil {
		t.Fatal("expected error: raise is not supported by RememberMany")
	}
}

func TestRememberManyWarnStoresAll(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	// Identical text → stubEmbedder yields identical vectors → a duplicate conflict.
	mems, err := store.RememberMany(ctx, []RememberItem{
		{Text: "duplicate fact"},
		{Text: "duplicate fact"},
	}, 128, PolicyWarn)
	if err != nil {
		t.Fatalf("RememberMany warn: %v", err)
	}
	if len(mems) != 2 {
		t.Fatalf("want 2 stored, got %d", len(mems))
	}
	if n, _ := store.Count(ctx); n != 2 {
		t.Errorf("count = %d, want 2 (warn stores all)", n)
	}
}

func TestRememberManySupersedeEarlierWins(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	mems, err := store.RememberMany(ctx, []RememberItem{
		{Text: "duplicate fact"},
		{Text: "duplicate fact"},
	}, 128, PolicySupersede)
	if err != nil {
		t.Fatalf("RememberMany supersede: %v", err)
	}
	first, second := mems[0], mems[1]
	got2, err := store.GetByID(ctx, second.ID)
	if err != nil {
		t.Fatalf("GetByID(second): %v", err)
	}
	if got2.SupersededBy != first.ID {
		t.Errorf("second.SupersededBy = %q, want %q (earlier wins)", got2.SupersededBy, first.ID)
	}
	got1, err := store.GetByID(ctx, first.ID)
	if err != nil {
		t.Fatalf("GetByID(first): %v", err)
	}
	if got1.SupersededBy != "" {
		t.Errorf("first.SupersededBy = %q, want empty (no cycle)", got1.SupersededBy)
	}
}

func TestRememberManyTTLSetsExpiresAt(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	ttl := 3600.0
	mems, err := store.RememberMany(ctx, []RememberItem{
		{Text: "temp", RememberOpts: RememberOpts{TTLSeconds: &ttl}},
	}, 128, "")
	if err != nil {
		t.Fatalf("RememberMany ttl: %v", err)
	}
	if mems[0].ExpiresAt <= 0 {
		t.Errorf("ExpiresAt = %v, want > 0", mems[0].ExpiresAt)
	}
}

func TestRememberManyJournalsPerIDAndUndo(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	mems, err := store.RememberMany(ctx, []RememberItem{{Text: "j1"}, {Text: "j2"}}, 128, "")
	if err != nil {
		t.Fatalf("RememberMany: %v", err)
	}
	// Exactly one "remember" entry per id.
	for _, m := range mems {
		hist, err := store.History(ctx, m.ID)
		if err != nil {
			t.Fatalf("History: %v", err)
		}
		n := 0
		for _, e := range hist {
			if e.Op == "remember" && e.ID == m.ID {
				n++
			}
		}
		if n != 1 {
			t.Errorf("id %s: %d remember entries, want 1", m.ID, n)
		}
	}
	// Undo the second id's entry → only that memory disappears.
	hist, _ := store.History(ctx, mems[1].ID)
	var entry JournalEntry
	for _, e := range hist {
		if e.Op == "remember" && e.ID == mems[1].ID {
			entry = e
		}
	}
	ok, err := store.Undo(ctx, entry)
	if err != nil || !ok {
		t.Fatalf("Undo: ok=%v err=%v", ok, err)
	}
	if _, err := store.GetByID(ctx, mems[1].ID); err == nil {
		t.Error("mems[1] should be gone after undo")
	}
	if _, err := store.GetByID(ctx, mems[0].ID); err != nil {
		t.Errorf("mems[0] should survive undo: %v", err)
	}
}
