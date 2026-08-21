package mcpserver

// Regression tests for the 2026-08 functional-bug review on the MCP surface:
// the bitemporal write path on remember, the pinned/trust/valid-time edit
// knobs, and journal_tail's negative-n slice panic.

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/nexusriot/ai-houkai/internal/memory"
)

func toolResultJSON(t *testing.T, res *mcp.CallToolResult) map[string]any {
	t.Helper()
	if len(res.Content) == 0 {
		t.Fatal("empty tool result")
	}
	txt, ok := res.Content[0].(mcp.TextContent)
	if !ok {
		t.Fatalf("content[0] is %T, want TextContent", res.Content[0])
	}
	var out map[string]any
	if err := json.Unmarshal([]byte(txt.Text), &out); err != nil {
		t.Fatalf("result not JSON: %v (%q)", err, txt.Text)
	}
	return out
}

func TestMCPRememberValidityInterval(t *testing.T) {
	store := newTestStore(t)
	handler := rememberHandler(store)
	var req mcp.CallToolRequest
	req.Params.Arguments = map[string]any{
		"text":        "true only in january",
		"valid_from":  float64(100),
		"valid_until": float64(200),
	}
	res, err := handler(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	out := toolResultJSON(t, res)
	id, _ := out["id"].(string)
	if id == "" {
		t.Fatalf("remember failed: %v", out)
	}
	m, err := store.GetByID(context.Background(), id)
	if err != nil || m.ValidFrom != 100 || m.ValidUntil != 200 {
		t.Fatalf("validity = [%v, %v) err=%v, want [100, 200)",
			m.ValidFrom, m.ValidUntil, err)
	}
}

func TestMCPEditPinnedTrustValidity(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	m, _, _, _ := store.Remember(ctx, "edit all the new knobs", memory.RememberOpts{})

	handler := editHandler(store)
	var req mcp.CallToolRequest
	req.Params.Arguments = map[string]any{
		"memory_id":   m.ID,
		"pinned":      true,
		"trust":       "untrusted",
		"valid_from":  float64(10),
		"valid_until": float64(20),
	}
	res, err := handler(ctx, req)
	if err != nil {
		t.Fatal(err)
	}
	out := toolResultJSON(t, res)
	if ok, _ := out["ok"].(bool); !ok {
		t.Fatalf("edit failed: %v", out)
	}
	got, _ := store.GetByID(ctx, m.ID)
	if !got.Pinned {
		t.Error("pinned not applied")
	}
	if got.Trust != memory.TrustLevel("untrusted") {
		t.Errorf("trust = %q, want untrusted", got.Trust)
	}
	if got.ValidFrom != 10 || got.ValidUntil != 20 {
		t.Errorf("validity = [%v, %v), want [10, 20)", got.ValidFrom, got.ValidUntil)
	}
}

func TestMCPJournalTailNegativeNDoesNotPanic(t *testing.T) {
	store := newJournaledStore(t)
	ctx := context.Background()
	store.Remember(ctx, "one journaled write", memory.RememberOpts{})

	var req mcp.CallToolRequest
	req.Params.Arguments = map[string]any{"n": float64(-1)}
	res, err := journalTailHandler(store)(ctx, req)
	if err != nil {
		t.Fatal(err)
	}
	txt := res.Content[0].(mcp.TextContent).Text
	if txt != "[]" {
		t.Fatalf("n=-1 → %q, want []", txt)
	}
}
