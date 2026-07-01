package mcpserver

import (
	"context"
	"encoding/json"
	"hash/fnv"
	"math"
	"path/filepath"
	"testing"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/nexusriot/ai-houkai/internal/memory"
	"github.com/nexusriot/ai-houkai/internal/vector"
)

// hashEmbedder deterministically hashes each text into a fixed-dim,
// L2-normalised vector so identical texts produce identical vectors. Lets this
// package (outside `memory`) build a real *memory.MemoryStore without the
// memory package's test-only stub.
type hashEmbedder struct{ dim int }

func (e *hashEmbedder) Dim() int { return e.dim }

func (e *hashEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, t := range texts {
		v := make([]float32, e.dim)
		h := fnv.New64a()
		_, _ = h.Write([]byte(t))
		seed := h.Sum64()
		for j := 0; j < e.dim; j++ {
			seed = seed*6364136223846793005 + 1442695040888963407
			v[j] = float32(int64(seed>>33)%1000) / 1000.0
		}
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

func newTestStore(t *testing.T) *memory.MemoryStore {
	t.Helper()
	dir := t.TempDir()
	backend, err := vector.NewChromem(filepath.Join(dir, "s"), "test", 16)
	if err != nil {
		t.Fatalf("NewChromem: %v", err)
	}
	t.Cleanup(func() { _ = backend.Close() })
	cfg := memory.DefaultStoreConfig(dir, "test")
	cfg.JournalEnabled = false
	return memory.NewMemoryStore(backend, &hashEmbedder{dim: 16}, cfg)
}

func TestNewRegistersServer(t *testing.T) {
	store := newTestStore(t)
	// New wires up all 16 tools and must not panic or return nil.
	s := New(store, "/some/store/path", "test")
	if s == nil {
		t.Fatal("New returned nil *server.MCPServer")
	}
}

func TestOptFloat32Present(t *testing.T) {
	var req mcp.CallToolRequest
	req.Params.Arguments = map[string]any{
		// JSON numbers decode to float64; optFloat32 only reads float64.
		"diversity": float64(0.75),
	}
	got := optFloat32(req, "diversity")
	if got == nil {
		t.Fatal("optFloat32(present) = nil, want non-nil")
	}
	if *got != float32(0.75) {
		t.Fatalf("optFloat32 = %v, want 0.75", *got)
	}
}

func TestOptFloat32Absent(t *testing.T) {
	var req mcp.CallToolRequest
	req.Params.Arguments = map[string]any{"other": float64(1)}
	if got := optFloat32(req, "diversity"); got != nil {
		t.Fatalf("optFloat32(absent) = %v, want nil", *got)
	}
}

func TestOptFloat32WrongType(t *testing.T) {
	// A non-float64 value (e.g. a string or int) must be treated as absent,
	// because optFloat32 only unwraps float64 (the JSON number type).
	var req mcp.CallToolRequest
	req.Params.Arguments = map[string]any{
		"diversity":  "0.5",  // string, not float64
		"min_cosine": int(1), // int, not float64
	}
	if got := optFloat32(req, "diversity"); got != nil {
		t.Fatalf("optFloat32(string) = %v, want nil", *got)
	}
	if got := optFloat32(req, "min_cosine"); got != nil {
		t.Fatalf("optFloat32(int) = %v, want nil", *got)
	}
}

func TestOptFloat32NilArguments(t *testing.T) {
	// GetArguments returns an empty map when Arguments is nil, so this must
	// not panic and must report absent.
	var req mcp.CallToolRequest
	if got := optFloat32(req, "diversity"); got != nil {
		t.Fatalf("optFloat32(nil args) = %v, want nil", *got)
	}
}

func TestPackResultJSONWithCompressedGroups(t *testing.T) {
	pack := memory.PackResult{
		Items: []memory.PackedMemory{
			{
				Memory: memory.Memory{
					ID:         "id-1",
					Text:       "hello world",
					Type:       memory.Semantic,
					Tags:       []string{"greeting"},
					Importance: 0.8,
				},
				Score:  0.9,
				Tokens: 3,
			},
		},
		Text:       "- hello world",
		UsedTokens: 3,
		Budget:     800,
		Truncated:  true,
		CompressedGroups: []memory.CompressedGroup{
			{
				Memories: []memory.Memory{
					{ID: "a"}, {ID: "b"},
				},
				Text:   "2 similar memories",
				Tokens: 5,
			},
		},
	}

	out := packResultJSON(pack)

	// Top-level keys.
	if out["text"] != "- hello world" {
		t.Errorf("text = %v", out["text"])
	}
	if out["used_tokens"] != 3 {
		t.Errorf("used_tokens = %v, want 3", out["used_tokens"])
	}
	if out["budget"] != 800 {
		t.Errorf("budget = %v, want 800", out["budget"])
	}
	if out["truncated"] != true {
		t.Errorf("truncated = %v, want true", out["truncated"])
	}

	items, ok := out["items"].([]map[string]any)
	if !ok {
		t.Fatalf("items is %T, want []map[string]any", out["items"])
	}
	if len(items) != 1 {
		t.Fatalf("len(items) = %d, want 1", len(items))
	}
	if items[0]["id"] != "id-1" {
		t.Errorf("items[0].id = %v, want id-1", items[0]["id"])
	}
	if items[0]["text"] != "hello world" {
		t.Errorf("items[0].text = %v", items[0]["text"])
	}
	if items[0]["type"] != string(memory.Semantic) {
		t.Errorf("items[0].type = %v, want semantic", items[0]["type"])
	}
	if items[0]["tokens"] != 3 {
		t.Errorf("items[0].tokens = %v, want 3", items[0]["tokens"])
	}
	if items[0]["score"] != float32(0.9) {
		t.Errorf("items[0].score = %v, want 0.9", items[0]["score"])
	}

	groups, ok := out["compressed_groups"].([]map[string]any)
	if !ok {
		t.Fatalf("compressed_groups is %T, want []map[string]any", out["compressed_groups"])
	}
	if len(groups) != 1 {
		t.Fatalf("len(compressed_groups) = %d, want 1", len(groups))
	}
	if groups[0]["count"] != 2 {
		t.Errorf("group.count = %v, want 2", groups[0]["count"])
	}
	if groups[0]["text"] != "2 similar memories" {
		t.Errorf("group.text = %v", groups[0]["text"])
	}
	ids, ok := groups[0]["ids"].([]string)
	if !ok || len(ids) != 2 || ids[0] != "a" || ids[1] != "b" {
		t.Errorf("group.ids = %v, want [a b]", groups[0]["ids"])
	}
}

func TestPackResultJSONNoCompressedGroups(t *testing.T) {
	// With no compressed groups the key must be omitted entirely.
	pack := memory.PackResult{
		Items:      nil,
		Text:       "",
		UsedTokens: 0,
		Budget:     100,
		Truncated:  false,
	}
	out := packResultJSON(pack)
	if _, present := out["compressed_groups"]; present {
		t.Error("compressed_groups present, want omitted when empty")
	}
	items, ok := out["items"].([]map[string]any)
	if !ok {
		t.Fatalf("items is %T, want []map[string]any", out["items"])
	}
	if len(items) != 0 {
		t.Fatalf("len(items) = %d, want 0", len(items))
	}
}

// callTool drives a registered tool end-to-end through the server's JSON-RPC
// HandleMessage path and returns the parsed inner tool-result JSON.
func callTool(t *testing.T, s interface {
	HandleMessage(context.Context, json.RawMessage) mcp.JSONRPCMessage
}, name string, args map[string]any) map[string]any {
	t.Helper()
	req := map[string]any{
		"jsonrpc": "2.0",
		"id":      1,
		"method":  "tools/call",
		"params": map[string]any{
			"name":      name,
			"arguments": args,
		},
	}
	raw, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}
	resp := s.HandleMessage(context.Background(), raw)

	// Round-trip the response through JSON so we do not depend on the SDK's
	// concrete content types; the tool result text is the inner JSON payload.
	rb, err := json.Marshal(resp)
	if err != nil {
		t.Fatalf("marshal response: %v", err)
	}
	var envelope struct {
		Error *struct {
			Message string `json:"message"`
		} `json:"error"`
		Result struct {
			IsError bool `json:"isError"`
			Content []struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"content"`
		} `json:"result"`
	}
	if err := json.Unmarshal(rb, &envelope); err != nil {
		t.Fatalf("unmarshal response %s: %v", rb, err)
	}
	if envelope.Error != nil {
		t.Fatalf("JSON-RPC error for %s: %s", name, envelope.Error.Message)
	}
	if len(envelope.Result.Content) == 0 {
		t.Fatalf("tool %s returned no content: %s", name, rb)
	}
	var inner map[string]any
	if err := json.Unmarshal([]byte(envelope.Result.Content[0].Text), &inner); err != nil {
		t.Fatalf("unmarshal tool payload %q: %v", envelope.Result.Content[0].Text, err)
	}
	return inner
}

func TestRememberRecallEndToEnd(t *testing.T) {
	store := newTestStore(t)
	s := New(store, "/store", "test")

	// remember
	rem := callTool(t, s, "remember", map[string]any{
		"text":       "the sky is blue today",
		"type":       "semantic",
		"importance": float64(0.7),
	})
	stored, ok := rem["stored"].(bool)
	if !ok || !stored {
		t.Fatalf("remember stored = %v, want true (payload %v)", rem["stored"], rem)
	}
	id, ok := rem["id"].(string)
	if !ok || id == "" {
		t.Fatalf("remember returned no id: %v", rem)
	}

	// recall — the deterministic embedder makes the same text the top hit.
	// recall returns a JSON array, but callTool expects an object; call the
	// tool and decode the array form directly here instead.
	results := recallArray(t, s, "the sky is blue today")
	if len(results) == 0 {
		t.Fatal("recall returned no results")
	}
	// The exact-match text should be present among the hits.
	found := false
	for _, r := range results {
		if r["id"] == id && r["text"] == "the sky is blue today" {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("recall did not return the remembered memory; got %v", results)
	}
}

// recallArray invokes the recall tool (whose payload is a JSON array) and
// returns the decoded rows.
func recallArray(t *testing.T, s interface {
	HandleMessage(context.Context, json.RawMessage) mcp.JSONRPCMessage
}, query string) []map[string]any {
	t.Helper()
	req := map[string]any{
		"jsonrpc": "2.0",
		"id":      2,
		"method":  "tools/call",
		"params": map[string]any{
			"name":      "recall",
			"arguments": map[string]any{"query": query, "k": float64(5)},
		},
	}
	raw, _ := json.Marshal(req)
	resp := s.HandleMessage(context.Background(), raw)
	rb, _ := json.Marshal(resp)
	var envelope struct {
		Result struct {
			Content []struct {
				Text string `json:"text"`
			} `json:"content"`
		} `json:"result"`
	}
	if err := json.Unmarshal(rb, &envelope); err != nil {
		t.Fatalf("unmarshal recall response: %v", err)
	}
	if len(envelope.Result.Content) == 0 {
		t.Fatalf("recall returned no content: %s", rb)
	}
	var rows []map[string]any
	if err := json.Unmarshal([]byte(envelope.Result.Content[0].Text), &rows); err != nil {
		t.Fatalf("unmarshal recall array %q: %v", envelope.Result.Content[0].Text, err)
	}
	return rows
}

func TestRememberMissingTextErrors(t *testing.T) {
	store := newTestStore(t)
	s := New(store, "/store", "test")
	// Omitting the required "text" arg makes the handler return an error
	// payload ({"error": ...}) rather than {stored:true}.
	out := callTool(t, s, "remember", map[string]any{"type": "semantic"})
	if _, ok := out["error"]; !ok {
		t.Fatalf("remember(no text) = %v, want an error payload", out)
	}
	if _, ok := out["stored"]; ok {
		t.Fatalf("remember(no text) unexpectedly reported stored: %v", out)
	}
}
