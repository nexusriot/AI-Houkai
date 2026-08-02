package mcpserver

import (
	"testing"

	"github.com/nexusriot/ai-houkai/internal/memory"
)

// Store capabilities that previously had no MCP surface (A3): get, subgraph,
// restore, undo, nuke, ready — plus the recall knobs from A1.

func TestGetTool(t *testing.T) {
	store := newTestStore(t)
	s := New(store, "/store", "test")
	rem := callTool(t, s, "remember", map[string]any{
		"text": "fetchable by id", "tags": []any{"x"}, "importance": float64(0.7),
	})
	id := rem["id"].(string)

	got := callTool(t, s, "get", map[string]any{"memory_id": id})
	if got["found"] != true {
		t.Fatalf("get(%s) found = %v, want true", id, got["found"])
	}
	if got["text"] != "fetchable by id" {
		t.Errorf("text = %v", got["text"])
	}
	if got["importance"].(float64) != 0.7 {
		t.Errorf("importance = %v", got["importance"])
	}
	// A plain read: no access-count bump.
	if got["access_count"].(float64) != 0 {
		t.Errorf("access_count = %v, want 0 (get must not touch)", got["access_count"])
	}

	missing := callTool(t, s, "get", map[string]any{"memory_id": "no-such-id"})
	if missing["found"] != false {
		t.Errorf("get(missing) found = %v, want false", missing["found"])
	}
}

func TestSubgraphTool(t *testing.T) {
	store := newTestStore(t)
	s := New(store, "/store", "test")
	a := callTool(t, s, "remember", map[string]any{"text": "graph node a"})["id"].(string)
	b := callTool(t, s, "remember", map[string]any{"text": "graph node b"})["id"].(string)
	c := callTool(t, s, "remember", map[string]any{"text": "graph node c"})["id"].(string)
	callTool(t, s, "link", map[string]any{"src_id": a, "dst_id": b, "rel": "refines"})
	callTool(t, s, "link", map[string]any{"src_id": b, "dst_id": c, "rel": "refines"})

	one := callTool(t, s, "subgraph", map[string]any{"memory_ids": []any{a}, "depth": float64(1)})
	if n := len(one["nodes"].([]any)); n != 2 {
		t.Errorf("depth=1 nodes = %d, want 2", n)
	}
	two := callTool(t, s, "subgraph", map[string]any{"memory_ids": []any{a}, "depth": float64(2)})
	if n := len(two["nodes"].([]any)); n != 3 {
		t.Errorf("depth=2 nodes = %d, want 3", n)
	}
	edges := two["edges"].([]any)
	if len(edges) != 2 {
		t.Fatalf("edges = %d, want 2", len(edges))
	}
	first := edges[0].(map[string]any)
	if first["src"] != a || first["dst"] != b || first["rel"] != "refines" {
		t.Errorf("edge[0] = %v", first)
	}
}

func TestSubgraphToolRequiresIDs(t *testing.T) {
	store := newTestStore(t)
	s := New(store, "/store", "test")
	out := callTool(t, s, "subgraph", map[string]any{"memory_ids": []any{}})
	if out["error"] == nil {
		t.Errorf("empty memory_ids should error, got %v", out)
	}
}

func TestRestoreTool(t *testing.T) {
	store := newTestStore(t)
	s := New(store, "/store", "test")
	old := callTool(t, s, "remember", map[string]any{"text": "the old policy"})["id"].(string)
	fresh := callTool(t, s, "remember", map[string]any{"text": "the new policy"})["id"].(string)
	callTool(t, s, "supersede", map[string]any{"old_id": old, "new_id": fresh})

	out := callTool(t, s, "restore", map[string]any{"memory_id": old})
	if out["restored"] != true {
		t.Fatalf("restore = %v, want true", out)
	}
	if out["was_superseded_by"] != fresh {
		t.Errorf("was_superseded_by = %v, want %s", out["was_superseded_by"], fresh)
	}
	if got := callTool(t, s, "get", map[string]any{"memory_id": old}); got["superseded_by"] != "" {
		t.Errorf("still superseded: %v", got["superseded_by"])
	}
}

func TestRestoreToolOnLiveMemoryIsFalse(t *testing.T) {
	store := newTestStore(t)
	s := New(store, "/store", "test")
	id := callTool(t, s, "remember", map[string]any{"text": "never superseded"})["id"].(string)
	if out := callTool(t, s, "restore", map[string]any{"memory_id": id}); out["restored"] != false {
		t.Errorf("restore on a live memory = %v, want false", out["restored"])
	}
}

func TestUndoTool(t *testing.T) {
	store := newJournaledStore(t)
	s := New(store, "/store", "test")
	id := callTool(t, s, "remember", map[string]any{"text": "about to be edited"})["id"].(string)
	callTool(t, s, "edit", map[string]any{"memory_id": id, "text": "edited text"})

	out := callTool(t, s, "undo", map[string]any{})
	if out["ok"] != true || out["op"] != "edit" {
		t.Fatalf("undo = %v, want ok/edit", out)
	}
	if got := callTool(t, s, "get", map[string]any{"memory_id": id}); got["text"] != "about to be edited" {
		t.Errorf("text after undo = %v", got["text"])
	}
}

func TestUndoToolTargetsOneMemory(t *testing.T) {
	store := newJournaledStore(t)
	s := New(store, "/store", "test")
	first := callTool(t, s, "remember", map[string]any{"text": "first subject"})["id"].(string)
	second := callTool(t, s, "remember", map[string]any{"text": "second subject"})["id"].(string)
	callTool(t, s, "edit", map[string]any{"memory_id": first, "importance": float64(0.9)})
	callTool(t, s, "edit", map[string]any{"memory_id": second, "importance": float64(0.1)})

	// The newest entry overall belongs to `second`; ask for `first`.
	out := callTool(t, s, "undo", map[string]any{"memory_id": first})
	if out["ok"] != true || out["id"] != first {
		t.Fatalf("undo(memory_id=first) = %v", out)
	}
	if got := callTool(t, s, "get", map[string]any{"memory_id": first}); got["importance"].(float64) != 0.5 {
		t.Errorf("first importance = %v, want 0.5", got["importance"])
	}
	if got := callTool(t, s, "get", map[string]any{"memory_id": second}); got["importance"].(float64) != 0.1 {
		t.Errorf("second importance = %v, want 0.1 (untouched)", got["importance"])
	}
}

func TestUndoToolByExactTS(t *testing.T) {
	store := newJournaledStore(t)
	s := New(store, "/store", "test")
	id := callTool(t, s, "remember", map[string]any{"text": "ts targeted"})["id"].(string)
	entries, err := store.Journal().Read(memory.ReadOpts{})
	if err != nil || len(entries) == 0 {
		t.Fatalf("journal read: %v (%d entries)", err, len(entries))
	}
	ts := entries[len(entries)-1].TS

	out := callTool(t, s, "undo", map[string]any{"ts": ts})
	if out["ok"] != true || out["op"] != "remember" {
		t.Fatalf("undo(ts) = %v", out)
	}
	if got := callTool(t, s, "get", map[string]any{"memory_id": id}); got["found"] != false {
		t.Errorf("memory survived undo of its remember: %v", got)
	}
}

func TestUndoToolWithNothingToUndo(t *testing.T) {
	store := newJournaledStore(t)
	s := New(store, "/store", "test")
	if out := callTool(t, s, "undo", map[string]any{}); out["error"] == nil {
		t.Errorf("undo on an empty journal should error, got %v", out)
	}
}

func TestNukeToolRequiresConfirmPhrase(t *testing.T) {
	store := newTestStore(t)
	s := New(store, "/store", "test")
	callTool(t, s, "remember", map[string]any{"text": "should survive a refused nuke"})

	refused := callTool(t, s, "nuke", map[string]any{"confirm": "yes"})
	if refused["ok"] != false || refused["deleted"].(float64) != 0 {
		t.Fatalf("nuke without the phrase = %v, want refused", refused)
	}
	if n := callTool(t, s, "stats", map[string]any{})["total"].(float64); n != 1 {
		t.Fatalf("count after refused nuke = %v, want 1", n)
	}

	done := callTool(t, s, "nuke", map[string]any{"confirm": "DELETE ALL"})
	if done["ok"] != true || done["deleted"].(float64) != 1 {
		t.Fatalf("confirmed nuke = %v", done)
	}
	if n := callTool(t, s, "stats", map[string]any{})["total"].(float64); n != 0 {
		t.Errorf("count after nuke = %v, want 0", n)
	}
}

func TestReadyTool(t *testing.T) {
	store := newTestStore(t)
	s := New(store, "/store", "test")
	out := callTool(t, s, "ready", map[string]any{})
	if out["ready"] != true {
		t.Fatalf("ready = %v, want true", out)
	}
	checks := out["checks"].(map[string]any)
	if checks["store"].(map[string]any)["ok"] != true {
		t.Errorf("store check = %v", checks["store"])
	}
	embedder := checks["embedder"].(map[string]any)
	if embedder["ok"] != true || embedder["dim"].(float64) != 16 {
		t.Errorf("embedder check = %v", embedder)
	}
}

func TestEvalRecallTool(t *testing.T) {
	store := newTestStore(t)
	s := New(store, "/store", "test")
	a := callTool(t, s, "remember", map[string]any{"text": "the deploy runbook lives in ops"})["id"].(string)
	callTool(t, s, "remember", map[string]any{"text": "postgres vacuum runs nightly"})

	out := callTool(t, s, "eval_recall", map[string]any{
		"cases": []any{map[string]any{
			"query": "the deploy runbook lives in ops", "relevant_ids": []any{a},
		}},
	})
	if out["n"].(float64) != 1 {
		t.Fatalf("n = %v", out["n"])
	}
	if out["recall_at_k"].(float64) != 1.0 || out["mrr"].(float64) != 1.0 {
		t.Errorf("scores = recall %v mrr %v, want 1.0/1.0",
			out["recall_at_k"], out["mrr"])
	}

	// Read-only: evaluating must not bump access tracking, or a second run of
	// the same gold set would score against a mutated store.
	if got := callTool(t, s, "get", map[string]any{"memory_id": a}); got["access_count"].(float64) != 0 {
		t.Errorf("access_count = %v, want 0", got["access_count"])
	}
}

func TestEvalRecallToolRejectsMalformedInput(t *testing.T) {
	store := newTestStore(t)
	s := New(store, "/store", "test")
	for _, tc := range []struct {
		name  string
		cases []any
	}{
		{"empty", []any{}},
		{"bare string", []any{"not an object"}},
		{"no query", []any{map[string]any{"relevant_ids": []any{"x"}}}},
		{"no ids", []any{map[string]any{"query": "q"}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			out := callTool(t, s, "eval_recall", map[string]any{"cases": tc.cases})
			if out["error"] == nil {
				t.Errorf("expected an error, got %v", out)
			}
		})
	}
}

func TestEvalRecallToolMixedK(t *testing.T) {
	store := newTestStore(t)
	s := New(store, "/store", "test")
	a := callTool(t, s, "remember", map[string]any{"text": "mixed k subject one"})["id"].(string)
	b := callTool(t, s, "remember", map[string]any{"text": "mixed k subject two"})["id"].(string)

	uniform := callTool(t, s, "eval_recall", map[string]any{
		"cases": []any{
			map[string]any{"query": "mixed k subject one", "relevant_ids": []any{a}},
			map[string]any{"query": "mixed k subject two", "relevant_ids": []any{b}},
		},
		"k": float64(4),
	})
	if uniform["k"].(float64) != 4 {
		t.Errorf("uniform k = %v, want 4", uniform["k"])
	}

	mixed := callTool(t, s, "eval_recall", map[string]any{
		"cases": []any{
			map[string]any{"query": "mixed k subject one", "relevant_ids": []any{a}, "k": float64(2)},
			map[string]any{"query": "mixed k subject two", "relevant_ids": []any{b}, "k": float64(5)},
		},
	})
	if mixed["k"].(float64) != -1 {
		t.Errorf("mixed k = %v, want -1", mixed["k"])
	}
}
