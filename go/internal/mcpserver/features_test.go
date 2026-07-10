package mcpserver

import (
	"context"
	"encoding/json"
	"fmt"
	"path/filepath"
	"testing"
	"time"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/nexusriot/ai-houkai/internal/memory"
	"github.com/nexusriot/ai-houkai/internal/vector"
)

// newJournaledStore builds a store with the journal ON (the shared newTestStore
// disables it) so history/state_at have a log to replay.
func newJournaledStore(t *testing.T) *memory.MemoryStore {
	t.Helper()
	dir := t.TempDir()
	backend, err := vector.NewChromem(filepath.Join(dir, "s"), "test", 16)
	if err != nil {
		t.Fatalf("NewChromem: %v", err)
	}
	t.Cleanup(func() { _ = backend.Close() })
	cfg := memory.DefaultStoreConfig(dir, "test")
	cfg.JournalEnabled = true
	cfg.JournalPath = filepath.Join(dir, "journal.log")
	return memory.NewMemoryStore(backend, &hashEmbedder{dim: 16}, cfg)
}

// callToolArray invokes a tool whose payload is a JSON array.
func callToolArray(t *testing.T, s interface {
	HandleMessage(context.Context, json.RawMessage) mcp.JSONRPCMessage
}, name string, args map[string]any) []map[string]any {
	t.Helper()
	req := map[string]any{
		"jsonrpc": "2.0", "id": 3, "method": "tools/call",
		"params": map[string]any{"name": name, "arguments": args},
	}
	raw, _ := json.Marshal(req)
	rb, _ := json.Marshal(s.HandleMessage(context.Background(), raw))
	var envelope struct {
		Result struct {
			Content []struct {
				Text string `json:"text"`
			} `json:"content"`
		} `json:"result"`
	}
	if err := json.Unmarshal(rb, &envelope); err != nil {
		t.Fatalf("unmarshal %s response: %v", name, err)
	}
	if len(envelope.Result.Content) == 0 {
		t.Fatalf("tool %s returned no content: %s", name, rb)
	}
	var rows []map[string]any
	if err := json.Unmarshal([]byte(envelope.Result.Content[0].Text), &rows); err != nil {
		t.Fatalf("unmarshal %s array %q: %v", name, envelope.Result.Content[0].Text, err)
	}
	return rows
}

func TestMetricsTool(t *testing.T) {
	store := newTestStore(t)
	s := New(store, "/store", "test")
	callTool(t, s, "remember", map[string]any{"text": "counted memory"})
	callToolArray(t, s, "recall", map[string]any{"query": "counted", "k": float64(2)})
	out := callTool(t, s, "metrics", map[string]any{})
	calls := out["calls"].(map[string]any)
	if calls["remember"].(float64) != 1 || calls["recall"].(float64) != 1 {
		t.Errorf("metrics calls = %v", calls)
	}
}

func TestRememberTTLAndPurgeTool(t *testing.T) {
	store := newTestStore(t)
	s := New(store, "/store", "test")
	rem := callTool(t, s, "remember", map[string]any{"text": "ephemeral mcp", "expires_at": float64(1.0)})
	id := rem["id"].(string)
	if rem["expires_at"] == nil {
		t.Error("remember response missing expires_at")
	}
	// Hidden from recall.
	for _, r := range callToolArray(t, s, "recall", map[string]any{"query": "ephemeral", "k": float64(5)}) {
		if r["id"] == id {
			t.Error("expired memory should be hidden from recall")
		}
	}
	// Visible with include_expired.
	found := false
	for _, r := range callToolArray(t, s, "recall", map[string]any{"query": "ephemeral", "k": float64(5), "include_expired": true}) {
		if r["id"] == id {
			found = true
		}
	}
	if !found {
		t.Error("include_expired should surface the expired memory")
	}
	// Purgeable.
	purged := callTool(t, s, "purge_expired", map[string]any{})
	if purged["purged"].(float64) != 1 {
		t.Errorf("purge removed %v, want 1", purged["purged"])
	}
}

func TestRecallExplainTool(t *testing.T) {
	store := newTestStore(t)
	s := New(store, "/store", "test")
	callTool(t, s, "remember", map[string]any{"text": "explain this via mcp"})
	rows := callToolArray(t, s, "recall", map[string]any{
		"query": "explain", "k": float64(1), "mode": "hybrid", "explain": true,
	})
	if len(rows) == 0 {
		t.Fatal("no results")
	}
	expl, ok := rows[0]["explain"].(map[string]any)
	if !ok || expl["mode"] != "hybrid" {
		t.Fatalf("hit missing hybrid explain: %v", rows[0])
	}
}

func TestHistoryTool(t *testing.T) {
	store := newJournaledStore(t)
	s := New(store, "/store", "test")
	rem := callTool(t, s, "remember", map[string]any{"text": "mcp v1"})
	id := rem["id"].(string)
	callTool(t, s, "edit", map[string]any{"memory_id": id, "text": "mcp v2"})
	hist := callToolArray(t, s, "history", map[string]any{"memory_id": id})
	ops := make([]string, len(hist))
	for i, e := range hist {
		ops[i] = e["op"].(string)
	}
	if len(ops) != 2 || ops[0] != "remember" || ops[1] != "edit" {
		t.Errorf("history ops = %v, want [remember edit]", ops)
	}
}

func TestStateAtAndGetAtTool(t *testing.T) {
	store := newJournaledStore(t)
	s := New(store, "/store", "test")
	rem := callTool(t, s, "remember", map[string]any{"text": "mcp point in time"})
	id := rem["id"].(string)
	time.Sleep(20 * time.Millisecond)
	ts := fmt.Sprintf("%f", float64(time.Now().UnixNano())/1e9)

	state := callTool(t, s, "state_at", map[string]any{"ts": ts})
	if state["count"].(float64) < 1 {
		t.Errorf("state_at count = %v, want >=1", state["count"])
	}
	one := callTool(t, s, "get_at", map[string]any{"memory_id": id, "ts": ts})
	if one["ok"] != true || one["text"] != "mcp point in time" {
		t.Errorf("get_at = %v", one)
	}
}
