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
