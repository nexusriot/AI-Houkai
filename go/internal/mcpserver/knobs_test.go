package mcpserver

import (
	"context"
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

// The bitemporal write path: valid_from/valid_until must reach the store.
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

// The pinned/trust/valid-time edit knobs must all be applied.
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

func TestRememberToolWiresOnConflict(t *testing.T) {
	store := newTestStore(t) // store policy: ignore
	s := New(store, "/store", "test")
	ctx := context.Background()

	first := callTool(t, s, "remember", map[string]any{
		"text": "the ingress controller is traefik", "type": "semantic",
	})
	if stored, _ := first["stored"].(bool); !stored {
		t.Fatalf("first remember: %v", first)
	}

	// The per-call on_conflict must reach the store: with the store's own
	// policy (ignore) this duplicate would be stored silently.
	second := callTool(t, s, "remember", map[string]any{
		"text": "the ingress controller is traefik", "type": "semantic",
		"on_conflict": "raise",
	})
	if stored, _ := second["stored"].(bool); stored {
		t.Fatalf("on_conflict=raise stored the duplicate: %v", second)
	}
	if second["conflicts"] == nil {
		t.Errorf("on_conflict=raise should report conflicts, got %v", second)
	}
	if c, _ := store.Count(ctx); c != 1 {
		t.Errorf("count after rejected duplicate = %d, want 1 (rolled back)", c)
	}
}

func TestRememberToolExplicitZeroImportance(t *testing.T) {
	store := newTestStore(t)
	s := New(store, "/store", "test")

	out := callTool(t, s, "remember", map[string]any{
		"text": "explicitly worthless", "importance": float64(0),
	})
	if imp, ok := out["importance"].(float64); !ok || imp != 0 {
		t.Errorf("importance = %v, want 0 (explicit zero must not fall back to 0.5)", out["importance"])
	}

	// And omitting importance still yields the store default.
	out = callTool(t, s, "remember", map[string]any{"text": "default importance"})
	if imp, ok := out["importance"].(float64); !ok || imp != 0.5 {
		t.Errorf("default importance = %v, want 0.5", out["importance"])
	}
}

func TestRememberToolDefaultsTypeSemantic(t *testing.T) {
	store := newTestStore(t)
	s := New(store, "/store", "test")
	ctx := context.Background()

	out := callTool(t, s, "remember", map[string]any{"text": "untyped memory"})
	id, _ := out["id"].(string)
	if id == "" {
		t.Fatalf("remember payload: %v", out)
	}
	m, err := store.GetByID(ctx, id)
	if err != nil {
		t.Fatal(err)
	}
	if m.Type != memory.Semantic {
		t.Errorf("default type = %q, want semantic", m.Type)
	}
}

func TestEditTool(t *testing.T) {
	store := newTestStore(t)
	s := New(store, "/store", "test")
	ctx := context.Background()

	created := callTool(t, s, "remember", map[string]any{
		"text": "MCP edit target", "tags": []any{"orig"},
	})
	id, _ := created["id"].(string)

	out := callTool(t, s, "edit", map[string]any{
		"memory_id":  id,
		"text":       "MCP edit target, revised",
		"importance": float64(0),
		"tags":       []any{"newtag"},
		"source":     "review",
	})
	if ok, _ := out["ok"].(bool); !ok {
		t.Fatalf("edit failed: %v", out)
	}
	m, err := store.GetByID(ctx, id)
	if err != nil {
		t.Fatal(err)
	}
	if m.Text != "MCP edit target, revised" || m.Importance != 0 || m.Source != "review" {
		t.Errorf("edited memory = %+v", m)
	}
	if len(m.Tags) != 1 || m.Tags[0] != "newtag" {
		t.Errorf("tags = %v, want [newtag]", m.Tags)
	}

	// Validation errors surface as ok:false, not a transport error.
	bad := callTool(t, s, "edit", map[string]any{"memory_id": id, "type": "opinions"})
	if ok, _ := bad["ok"].(bool); ok {
		t.Errorf("edit with bad type should fail, got %v", bad)
	}
	missing := callTool(t, s, "edit", map[string]any{"memory_id": "no-such-id", "text": "x"})
	if ok, _ := missing["ok"].(bool); ok {
		t.Errorf("edit of missing id should fail, got %v", missing)
	}
}

// An idempotent repeat reported {stored:false, conflicts:[]} — an agent reading
// that sees a rejected write with no reason given, when in fact the store found
// the existing row and bumped it. Report the row and say nothing was created.
func TestMCPIdempotentRepeatReportsTheExistingRow(t *testing.T) {
	store := newTestStore(t)
	s := New(store, "/store", "test")

	first := callTool(t, s, "remember",
		map[string]any{"text": "repeat me", "idempotent": true})
	if first["stored"] != true {
		t.Fatalf("first write = %v, want stored:true", first)
	}

	second := callTool(t, s, "remember",
		map[string]any{"text": "repeat me", "idempotent": true})
	if second["stored"] != false {
		t.Errorf("repeat stored = %v, want false", second["stored"])
	}
	if second["id"] != first["id"] {
		t.Errorf("repeat id = %v, want the existing %v", second["id"], first["id"])
	}
	if c, ok := second["conflicts"]; ok {
		t.Errorf("repeat reported conflicts: %v", c)
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
