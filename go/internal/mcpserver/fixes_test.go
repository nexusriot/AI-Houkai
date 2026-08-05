package mcpserver

// Regression tests for the Python-parity review fixes on the MCP surface:
// remember wiring on_conflict + pointer importance, the edit tool, and the
// schedule-gated maintenance_tick.

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/nexusriot/ai-houkai/internal/maintenance"
	"github.com/nexusriot/ai-houkai/internal/memory"
)

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

func TestMaintenanceTickGatedOnSchedule(t *testing.T) {
	store := newTestStore(t)
	s := New(store, "/store", "test")

	statePath := filepath.Join(t.TempDir(), "state.json")
	// Both jobs recorded as freshly run: with day/week gates nothing is due.
	if err := (maintenance.State{LastDecayAt: 1, LastReflectAt: 1}).Save(statePath); err != nil {
		t.Fatal(err)
	}
	// Pretend "now" is close to the recorded runs by gating with an enormous
	// interval instead of faking the clock.
	SetMaintenance(maintenance.Config{
		DecayRate: 0.1, MinScore: 0.05, Reflect: true,
		DecayEvery: 1e12, ReflectEvery: 1e12,
	}, statePath)
	t.Cleanup(func() {
		maintOpts = struct {
			cfg       maintenance.Config
			statePath string
			set       bool
		}{}
	})

	out := callTool(t, s, "maintenance_tick", map[string]any{})
	if ran, _ := out["ran_decay"].(bool); ran {
		t.Errorf("ran_decay = true, want false (gated by decay_every)")
	}
	if ran, _ := out["ran_reflect"].(bool); ran {
		t.Errorf("ran_reflect = true, want false (gated by reflect_every)")
	}

	// Ungate → both jobs run and the state file is stamped.
	SetMaintenance(maintenance.Config{
		DecayRate: 0.1, MinScore: 0.05, Reflect: true,
		DecayEvery: 0, ReflectEvery: 0,
	}, statePath)
	out = callTool(t, s, "maintenance_tick", map[string]any{})
	if ran, _ := out["ran_decay"].(bool); !ran {
		t.Errorf("ran_decay = false, want true (ungated)")
	}
	if ran, _ := out["ran_reflect"].(bool); !ran {
		t.Errorf("ran_reflect = false, want true (ungated)")
	}
	st := maintenance.LoadState(statePath)
	if st.LastDecayAt <= 1 || st.LastReflectAt <= 1 {
		t.Errorf("state not stamped after ungated tick: %+v", st)
	}
}

// docs/DESIGN.md documents maintenance_tick as returning ran_purge and purged;
// this port dropped both, so a client could not tell whether TTL reclamation
// ran. trash_purged reports the retention sweep that rides the same job.
func TestMaintenanceTickReportsPurgeCounts(t *testing.T) {
	ctx := context.Background()
	store := newTestStore(t)
	s := New(store, "/store", "test")

	m, _, _, err := store.Remember(ctx, "bin me", memory.RememberOpts{})
	if err != nil {
		t.Fatalf("Remember: %v", err)
	}
	if _, err := store.Trash(ctx, m.ID); err != nil {
		t.Fatalf("Trash: %v", err)
	}

	SetMaintenance(maintenance.Config{
		DecayRate: 0.1, MinScore: 0.05,
		DecayEvery: -1, ReflectEvery: -1, PurgeEvery: 0,
		TrashTTLDays: 30,
	}, "")
	t.Cleanup(func() {
		maintOpts = struct {
			cfg       maintenance.Config
			statePath string
			set       bool
		}{}
	})

	out := callTool(t, s, "maintenance_tick", map[string]any{})
	for _, key := range []string{"ran_purge", "purged", "trash_purged"} {
		if _, ok := out[key]; !ok {
			t.Errorf("maintenance_tick result missing documented key %q: %v", key, out)
		}
	}
	if ran, _ := out["ran_purge"].(bool); !ran {
		t.Errorf("ran_purge = false, want true (purge_every=0 → always due)")
	}
	// The entry is seconds old, so a 30-day retention must keep it: the tick
	// enforces retention, it does not empty the bin. (The sweep itself is
	// covered in maintenance, which controls the clock.)
	if n, _ := out["trash_purged"].(float64); n != 0 {
		t.Errorf("trash_purged = %v, want 0 — entry is inside retention", n)
	}
	entries, err := store.TrashList()
	if err != nil {
		t.Fatalf("TrashList: %v", err)
	}
	if len(entries) != 1 {
		t.Fatalf("trash = %d entries, want the one we binned", len(entries))
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
