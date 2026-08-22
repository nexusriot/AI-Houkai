package mcpserver

import (
	"context"
	"testing"
)

func TestRememberManyTool(t *testing.T) {
	store := newTestStore(t)
	s := New(store, "/store", "test")
	ctx := context.Background()
	out := callTool(t, s, "remember_many", map[string]any{
		"items": []any{
			map[string]any{"text": "mcp batch one"},
			map[string]any{"text": "mcp batch two", "type": "procedural", "tags": []any{"t"}},
		},
	})
	if stored, _ := out["stored"].(float64); stored != 2 {
		t.Fatalf("stored = %v, want 2 (payload %v)", out["stored"], out)
	}
	if ids, _ := out["ids"].([]any); len(ids) != 2 {
		t.Errorf("ids len = %d, want 2", len(ids))
	}
	if c, _ := store.Count(ctx); c != 2 {
		t.Errorf("count = %d, want 2", c)
	}
}

func TestRememberManyToolRaiseRejected(t *testing.T) {
	store := newTestStore(t)
	s := New(store, "/store", "test")
	out := callTool(t, s, "remember_many", map[string]any{
		"items":       []any{map[string]any{"text": "x"}},
		"on_conflict": "raise",
	})
	if stored, _ := out["stored"].(float64); stored != 0 {
		t.Errorf("stored = %v, want 0", out["stored"])
	}
	if out["error"] == nil {
		t.Errorf("expected an error field for on_conflict=raise, got %v", out)
	}
}

func TestRememberManyToolBadItem(t *testing.T) {
	store := newTestStore(t)
	s := New(store, "/store", "test")
	out := callTool(t, s, "remember_many", map[string]any{
		"items": []any{map[string]any{"type": "semantic"}}, // missing text
	})
	if out["error"] == nil {
		t.Errorf("expected an error field for a missing 'text', got %v", out)
	}
}

// The batch tool reported `stored: len(items)`, so a replayed batch told the
// agent it had written every fact again. It counts rows created.
func TestMCPBatchStoredCountsOnlyNewRows(t *testing.T) {
	store := newTestStore(t)
	s := New(store, "/store", "test")
	items := []any{
		map[string]any{"text": "batch fact one"},
		map[string]any{"text": "batch fact two"},
	}

	first := callTool(t, s, "remember_many",
		map[string]any{"items": items, "idempotent": true})
	if first["stored"] != float64(2) {
		t.Fatalf("first batch stored = %v, want 2", first["stored"])
	}

	again := callTool(t, s, "remember_many",
		map[string]any{"items": items, "idempotent": true})
	if again["stored"] != float64(0) {
		t.Errorf("replay stored = %v, want 0", again["stored"])
	}
	ids, _ := again["ids"].([]any)
	if len(ids) != 2 {
		t.Errorf("replay ids = %v, want one per input", ids)
	}
}

func TestMCPBatchIntraDuplicatesCountOnce(t *testing.T) {
	store := newTestStore(t)
	s := New(store, "/store", "test")

	out := callTool(t, s, "remember_many", map[string]any{
		"items": []any{
			map[string]any{"text": "same text"},
			map[string]any{"text": "same text"},
		},
		"idempotent": true,
	})
	if out["stored"] != float64(1) {
		t.Errorf("stored = %v, want 1", out["stored"])
	}
	ids, _ := out["ids"].([]any)
	if len(ids) != 2 || ids[0] != ids[1] {
		t.Errorf("ids = %v, want two entries sharing one id", ids)
	}
}
