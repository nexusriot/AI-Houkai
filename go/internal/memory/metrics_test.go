package memory

import (
	"context"
	"testing"
)

func TestMetricsFreshStore(t *testing.T) {
	store := newTestStore(t)
	m, err := store.Metrics(context.Background())
	if err != nil {
		t.Fatalf("Metrics: %v", err)
	}
	calls := m["calls"].(map[string]int)
	for _, k := range []string{"remember", "recall", "forget", "edit", "supersede"} {
		if calls[k] != 0 {
			t.Errorf("fresh %s count = %d, want 0", k, calls[k])
		}
	}
	if m["count"].(int) != 0 {
		t.Errorf("fresh count = %v, want 0", m["count"])
	}
	lat := m["recall_latency_ms"].(map[string]any)
	if lat["count"].(int) != 0 {
		t.Errorf("fresh recall latency count = %v, want 0", lat["count"])
	}
}

func TestMetricsCountersTrackOps(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	a, _, _, _ := store.Remember(ctx, "first", RememberOpts{})
	b, _, _, _ := store.Remember(ctx, "second", RememberOpts{})
	store.Recall(ctx, "first", 2, RecallOpts{})
	store.Recall(ctx, "second", 2, RecallOpts{})
	store.Recall(ctx, "third", 2, RecallOpts{})
	imp := float32(0.9)
	store.Edit(ctx, a.ID, EditOpts{Importance: &imp})
	store.Supersede(ctx, a.ID, b.ID)
	store.Forget(ctx, b.ID)

	calls := mustMetrics(t, store)["calls"].(map[string]int)
	// PurgeExpired isn't run here; forget is called once directly.
	if calls["remember"] != 2 {
		t.Errorf("remember = %d, want 2", calls["remember"])
	}
	if calls["recall"] != 3 {
		t.Errorf("recall = %d, want 3", calls["recall"])
	}
	if calls["edit"] != 1 {
		t.Errorf("edit = %d, want 1", calls["edit"])
	}
	if calls["supersede"] != 1 {
		t.Errorf("supersede = %d, want 1", calls["supersede"])
	}
	if calls["forget"] != 1 {
		t.Errorf("forget = %d, want 1", calls["forget"])
	}
}

func TestMetricsRecallLatencyRecorded(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	store.Remember(ctx, "something to find", RememberOpts{})
	store.Recall(ctx, "something", 1, RecallOpts{})
	store.Recall(ctx, "something", 1, RecallOpts{})
	lat := mustMetrics(t, store)["recall_latency_ms"].(map[string]any)
	if lat["count"].(int) != 2 {
		t.Errorf("recall latency count = %v, want 2", lat["count"])
	}
	if lat["avg"].(float64) < 0 || lat["max"].(float64) < 0 {
		t.Errorf("latency avg/max must be non-negative: %v", lat)
	}
}

func TestMetricsEmptyRecallStillCounts(t *testing.T) {
	store := newTestStore(t)
	// recall on an empty store returns nil early but must still be counted.
	store.Recall(context.Background(), "nothing here", 3, RecallOpts{})
	m := mustMetrics(t, store)
	if m["calls"].(map[string]int)["recall"] != 1 {
		t.Errorf("empty recall not counted: %v", m["calls"])
	}
	if m["recall_latency_ms"].(map[string]any)["count"].(int) != 1 {
		t.Error("empty recall latency not recorded")
	}
}

func TestMetricsMutatorCountersTrackOps(t *testing.T) {
	// Previously-uncounted mutators must now increment (parity with Python).
	store := newTestStore(t)
	ctx := context.Background()
	a, _, _, _ := store.Remember(ctx, "alpha", RememberOpts{})
	b, _, _, _ := store.Remember(ctx, "beta", RememberOpts{})
	store.Link(ctx, a.ID, b.ID, "related")
	store.Unlink(ctx, a.ID, b.ID, "")
	c, _, _, _ := store.Remember(ctx, "gamma", RememberOpts{})
	store.Supersede(ctx, c.ID, a.ID)
	store.Restore(ctx, c.ID)
	store.PurgeExpired(ctx, 0, true)

	calls := mustMetrics(t, store)["calls"].(map[string]int)
	for _, k := range []string{"link", "unlink", "restore", "purge_expired"} {
		if calls[k] != 1 {
			t.Errorf("%s counter = %d, want 1", k, calls[k])
		}
	}
}

func TestMetricsRecallLatencyPercentiles(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	store.Remember(ctx, "something to find", RememberOpts{})
	for i := 0; i < 5; i++ {
		store.Recall(ctx, "something", 1, RecallOpts{})
	}
	lat := mustMetrics(t, store)["recall_latency_ms"].(map[string]any)
	p50 := lat["p50"].(float64)
	p95 := lat["p95"].(float64)
	p99 := lat["p99"].(float64)
	max := lat["max"].(float64)
	if !(p50 <= p95 && p95 <= p99 && p99 <= max) {
		t.Errorf("percentiles must be monotonic and bounded by max: p50=%v p95=%v p99=%v max=%v", p50, p95, p99, max)
	}
}

func mustMetrics(t *testing.T, store *MemoryStore) map[string]any {
	t.Helper()
	m, err := store.Metrics(context.Background())
	if err != nil {
		t.Fatalf("Metrics: %v", err)
	}
	return m
}
