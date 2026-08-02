package memory

import (
	"context"
	"testing"
	"time"
)

func f64ptr(v float64) *float64 { return &v }

func TestRememberTTLSetsExpiry(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	before := nowFloat()
	m, _, _, err := store.Remember(ctx, "milk", RememberOpts{TTLSeconds: f64ptr(100)})
	if err != nil {
		t.Fatalf("Remember: %v", err)
	}
	if m.ExpiresAt < before+99 || m.ExpiresAt > before+101 {
		t.Errorf("expires_at = %v, want ~now+100", m.ExpiresAt)
	}
}

func TestRememberExpiresAtAbsolute(t *testing.T) {
	store := newTestStore(t)
	ts := nowFloat() + 500
	m, _, _, err := store.Remember(context.Background(), "bread", RememberOpts{ExpiresAt: f64ptr(ts)})
	if err != nil {
		t.Fatalf("Remember: %v", err)
	}
	if m.ExpiresAt != ts {
		t.Errorf("expires_at = %v, want %v", m.ExpiresAt, ts)
	}
}

func TestRememberTTLValidation(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	if _, _, _, err := store.Remember(ctx, "x", RememberOpts{ExpiresAt: f64ptr(nowFloat() + 10), TTLSeconds: f64ptr(10)}); err == nil {
		t.Error("want error when both expires_at and ttl_seconds set")
	}
	if _, _, _, err := store.Remember(ctx, "x", RememberOpts{TTLSeconds: f64ptr(0)}); err == nil {
		t.Error("want error for ttl_seconds <= 0")
	}
	if _, _, _, err := store.Remember(ctx, "x", RememberOpts{ExpiresAt: f64ptr(-1)}); err == nil {
		t.Error("want error for negative expires_at")
	}
}

func TestRecallHidesExpired(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	live, _, _, _ := store.Remember(ctx, "deployment pipeline runbook", RememberOpts{TTLSeconds: f64ptr(1000)})
	exp, _, _, _ := store.Remember(ctx, "deployment pipeline hotfix", RememberOpts{ExpiresAt: f64ptr(1.0)})

	hits, err := store.Recall(ctx, "deployment pipeline", 10, RecallOpts{})
	if err != nil {
		t.Fatalf("Recall: %v", err)
	}
	ids := idSet(hits)
	if !ids[live.ID] {
		t.Error("live memory missing from recall")
	}
	if ids[exp.ID] {
		t.Error("expired memory should be hidden from recall")
	}

	hits, _ = store.Recall(ctx, "deployment pipeline", 10, RecallOpts{IncludeExpired: true})
	if !idSet(hits)[exp.ID] {
		t.Error("include_expired should surface the expired memory")
	}
}

func TestRecallExpiredNotUnderfetchedOnFastPath(t *testing.T) {
	// IncludeSuperseded=true would trip the semantic fast path (fetch exactly
	// k); expiry filtering must still force the overfetch pool.
	store := newTestStore(t)
	ctx := context.Background()
	store.Remember(ctx, "alpha topic note one", RememberOpts{ExpiresAt: f64ptr(1.0)})
	live, _, _, _ := store.Remember(ctx, "alpha topic note two", RememberOpts{})
	hits, err := store.Recall(ctx, "alpha topic note", 1, RecallOpts{IncludeSuperseded: true})
	if err != nil {
		t.Fatalf("Recall: %v", err)
	}
	if len(hits) != 1 || hits[0].ID != live.ID {
		t.Errorf("want [%s], got %v", live.ID, idsOf(hits))
	}
}

func TestListRecentHidesExpired(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	store.Remember(ctx, "live one", RememberOpts{})
	exp, _, _, _ := store.Remember(ctx, "expired one", RememberOpts{ExpiresAt: f64ptr(1.0)})

	mems, _ := store.ListRecent(ctx, 50, false, false)
	if memIDSet(mems)[exp.ID] {
		t.Error("list_recent should hide expired by default")
	}
	mems, _ = store.ListRecent(ctx, 50, false, true)
	if !memIDSet(mems)[exp.ID] {
		t.Error("include_expired should surface the expired memory")
	}
}

func TestEditExpiresAt(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	m, _, _, _ := store.Remember(ctx, "editable memory", RememberOpts{})
	if _, err := store.Edit(ctx, m.ID, EditOpts{ExpiresAt: f64ptr(1.0)}); err != nil {
		t.Fatalf("Edit: %v", err)
	}
	got, _ := store.GetByID(ctx, m.ID)
	if got.ExpiresAt != 1.0 {
		t.Errorf("expires_at = %v, want 1.0", got.ExpiresAt)
	}
	hits, _ := store.Recall(ctx, "editable memory", 5, RecallOpts{})
	if idSet(hits)[m.ID] {
		t.Error("expired-via-edit memory should be hidden from recall")
	}
	// Clear it.
	store.Edit(ctx, m.ID, EditOpts{ExpiresAt: f64ptr(0)})
	got, _ = store.GetByID(ctx, m.ID)
	if got.ExpiresAt != 0 {
		t.Errorf("expires_at = %v, want 0 after clear", got.ExpiresAt)
	}
}

func TestPurgeExpired(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	live, _, _, _ := store.Remember(ctx, "keep me", RememberOpts{TTLSeconds: f64ptr(1000)})
	exp, _, _, _ := store.Remember(ctx, "drop me", RememberOpts{ExpiresAt: f64ptr(1.0)})

	// Dry-run reports but deletes nothing.
	purged, err := store.PurgeExpired(ctx, 0, true)
	if err != nil {
		t.Fatalf("PurgeExpired dry: %v", err)
	}
	if len(purged) != 1 || purged[0].ID != exp.ID {
		t.Fatalf("dry-run want [drop], got %v", idsOfMem(purged))
	}
	if n, _ := store.Count(ctx); n != 2 {
		t.Fatalf("dry-run deleted something: count=%d", n)
	}

	// Real purge.
	purged, _ = store.PurgeExpired(ctx, 0, false)
	if len(purged) != 1 || purged[0].ID != exp.ID {
		t.Fatalf("want [drop], got %v", idsOfMem(purged))
	}
	if _, err := store.GetByID(ctx, exp.ID); err == nil {
		t.Error("expired memory still present after purge")
	}
	if _, err := store.GetByID(ctx, live.ID); err != nil {
		t.Error("live memory was wrongly purged")
	}
}

func TestPurgeHonorsCustomNow(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	m, _, _, _ := store.Remember(ctx, "future expiry", RememberOpts{TTLSeconds: f64ptr(100)})
	if got, _ := store.PurgeExpired(ctx, 0, false); len(got) != 0 {
		t.Fatalf("nothing should be expired now, got %d", len(got))
	}
	got, _ := store.PurgeExpired(ctx, nowFloat()+200, false)
	if len(got) != 1 || got[0].ID != m.ID {
		t.Errorf("want [%s] expired at now+200, got %v", m.ID, idsOfMem(got))
	}
}

func TestExpiresAtSerializationRoundTrip(t *testing.T) {
	ts := time.Now().Unix() + 42
	m := Memory{ID: "i", Text: "t", Type: Semantic, ExpiresAt: float64(ts)}
	// metadata round-trip
	if got := MetadataToMemory(m.ID, m.Text, MemoryToMetadata(m)); got.ExpiresAt != float64(ts) {
		t.Errorf("metadata round-trip expires_at = %v, want %v", got.ExpiresAt, ts)
	}
	// dict round-trip
	if got := MemoryFromDict(m.ToDict()); got.ExpiresAt != float64(ts) {
		t.Errorf("dict round-trip expires_at = %v, want %v", got.ExpiresAt, ts)
	}
}

func TestExpiresAtMigrationMissingKey(t *testing.T) {
	// A pre-TTL row has no "expires_at" metadata key → defaults to 0.
	m := MetadataToMemory("x", "text", map[string]string{"type": "semantic"})
	if m.ExpiresAt != 0 {
		t.Errorf("missing expires_at should default to 0, got %v", m.ExpiresAt)
	}
}

func idSet(hits []MemoryWithScore) map[string]bool {
	s := map[string]bool{}
	for _, h := range hits {
		s[h.ID] = true
	}
	return s
}

func idsOf(hits []MemoryWithScore) []string {
	out := make([]string, len(hits))
	for i, h := range hits {
		out[i] = h.ID
	}
	return out
}

func memIDSet(mems []Memory) map[string]bool {
	s := map[string]bool{}
	for _, m := range mems {
		s[m.ID] = true
	}
	return s
}

func idsOfMem(mems []Memory) []string {
	out := make([]string, len(mems))
	for i, m := range mems {
		out[i] = m.ID
	}
	return out
}
