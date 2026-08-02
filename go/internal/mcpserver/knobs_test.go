package mcpserver

import (
	"testing"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/nexusriot/ai-houkai/internal/memory"
)

// The graph-proximity weight and graph-walk expansion reach the MCP recall /
// recall_pack tools (A1). Before this they were HTTP- and library-only, so an
// agent could not use them at all.

func TestWeightsFromReqAbsentIsZeroValue(t *testing.T) {
	var req mcp.CallToolRequest
	req.Params.Arguments = map[string]any{}
	if got := weightsFromReq(req); got != (memory.HybridWeights{}) {
		t.Errorf("weightsFromReq(absent) = %+v, want zero value", got)
	}
}

func TestWeightsFromReqKeepsCoreWeights(t *testing.T) {
	// Go struct literals zero-init, so setting only Graph would leave the core
	// weights at 0 and Recall's validation would reject the call.
	var req mcp.CallToolRequest
	req.Params.Arguments = map[string]any{"graph": float64(0.15)}
	got := weightsFromReq(req)
	if got.Graph != 0.15 {
		t.Errorf("Graph = %v, want 0.15", got.Graph)
	}
	def := memory.DefaultWeights()
	if got.Cosine != def.Cosine || got.Lexical != def.Lexical ||
		got.Recency != def.Recency || got.Importance != def.Importance {
		t.Errorf("core weights zeroed: %+v", got)
	}
}

func TestExpandFromReqAbsentIsNil(t *testing.T) {
	var req mcp.CallToolRequest
	req.Params.Arguments = map[string]any{"query": "x"}
	if got := expandFromReq(req); got != nil {
		t.Errorf("expandFromReq(absent) = %+v, want nil", got)
	}
}

func TestExpandFromReqDefaultsMatchPython(t *testing.T) {
	var req mcp.CallToolRequest
	req.Params.Arguments = map[string]any{"expand_rerank": true}
	got := expandFromReq(req)
	if got == nil {
		t.Fatal("expandFromReq returned nil")
	}
	if got.Depth != 1 || got.Cap != 5 || got.Score != 0.70 || got.Decay != 1.0 {
		t.Errorf("defaults drifted: %+v", got)
	}
	if len(got.Rels) != 2 || got.Rels[0] != "refines" || got.Rels[1] != "example_of" {
		t.Errorf("default rels = %v", got.Rels)
	}
	if !got.Rerank {
		t.Error("Rerank not honoured")
	}
}

func TestExpandFromReqReadsEveryField(t *testing.T) {
	var req mcp.CallToolRequest
	req.Params.Arguments = map[string]any{
		"expand_rels":   []any{"contradicts"},
		"expand_depth":  float64(3),
		"expand_cap":    float64(9),
		"expand_score":  float64(0.4),
		"expand_decay":  float64(0.5),
		"expand_rerank": false,
	}
	got := expandFromReq(req)
	if got == nil {
		t.Fatal("expandFromReq returned nil")
	}
	if len(got.Rels) != 1 || got.Rels[0] != "contradicts" {
		t.Errorf("rels = %v", got.Rels)
	}
	if got.Depth != 3 || got.Cap != 9 || got.Score != 0.4 || got.Decay != 0.5 || got.Rerank {
		t.Errorf("fields = %+v", got)
	}
}

func TestRecallToolExpansionPullsLinkedNeighbour(t *testing.T) {
	store := newTestStore(t)
	s := New(store, "/store", "test")
	a := callTool(t, s, "remember", map[string]any{"text": "alpha topic about compilers"})["id"].(string)
	b := callTool(t, s, "remember", map[string]any{"text": "an entirely separate note on gardening"})["id"].(string)
	callTool(t, s, "link", map[string]any{"src_id": a, "dst_id": b, "rel": "refines"})

	plain := callToolArray(t, s, "recall", map[string]any{"query": "compilers", "k": float64(1)})
	for _, r := range plain {
		if r["id"] == b {
			t.Fatal("unexpanded recall already returned the neighbour")
		}
	}

	expanded := callToolArray(t, s, "recall", map[string]any{
		"query": "compilers", "k": float64(1), "expand_rels": []any{"refines"},
	})
	found := false
	for _, r := range expanded {
		if r["id"] == b {
			found = true
		}
	}
	if !found {
		t.Errorf("expansion did not surface the linked neighbour: %v", expanded)
	}
}

func TestRecallPackHeaderOverride(t *testing.T) {
	store := newTestStore(t)
	s := New(store, "/store", "test")
	callTool(t, s, "remember", map[string]any{"text": "packing header subject matter"})

	out := callTool(t, s, "recall_pack", map[string]any{
		"query": "packing header subject matter", "header": "# Custom",
	})
	text := out["text"].(string)
	if len(text) < 8 || text[:8] != "# Custom" {
		t.Errorf("header override ignored: %q", text)
	}

	// Empty string must mean "no header", not "use the default".
	bare := callTool(t, s, "recall_pack", map[string]any{
		"query": "packing header subject matter", "header": "",
	})
	if bt := bare["text"].(string); len(bt) > 0 && bt[0] == '#' {
		t.Errorf("empty header should suppress the heading, got %q", bt)
	}
}

func TestAutoContextTouchFalseIsReadOnly(t *testing.T) {
	store := newTestStore(t)
	s := New(store, "/store", "test")
	id := callTool(t, s, "remember", map[string]any{"text": "read-only fan-out subject"})["id"].(string)

	callTool(t, s, "auto_context", map[string]any{
		"task": "read-only fan-out subject", "touch": false,
	})
	if got := callTool(t, s, "get", map[string]any{"memory_id": id}); got["access_count"].(float64) != 0 {
		t.Errorf("access_count = %v, want 0 with touch=false", got["access_count"])
	}

	callTool(t, s, "auto_context", map[string]any{"task": "read-only fan-out subject"})
	if got := callTool(t, s, "get", map[string]any{"memory_id": id}); got["access_count"].(float64) == 0 {
		t.Error("default auto_context should bump access_count")
	}
}
