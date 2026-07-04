package cli

// Command-level regression tests for the review fixes: pointer importance on
// remember, semantic default type, and tag/bump going through the journaled
// store.Edit path (so `journal undo` can reverse them).

import (
	"context"
	"hash/fnv"
	"math"
	"path/filepath"
	"testing"

	"github.com/nexusriot/ai-houkai/internal/memory"
	"github.com/nexusriot/ai-houkai/internal/vector"
	"github.com/spf13/cobra"
)

// cliTestEmbedder deterministically hashes each text into a fixed-dim,
// L2-normalised vector (same scheme as the other packages' test embedders).
type cliTestEmbedder struct{ dim int }

func (e *cliTestEmbedder) Dim() int { return e.dim }

func (e *cliTestEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
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

func newCmdTestStore(t *testing.T) *memory.MemoryStore {
	t.Helper()
	dir := t.TempDir()
	backend, err := vector.NewChromem(filepath.Join(dir, "s"), "test", 16)
	if err != nil {
		t.Fatalf("NewChromem: %v", err)
	}
	t.Cleanup(func() { _ = backend.Close() })
	cfg := memory.DefaultStoreConfig(dir, "test")
	cfg.JournalPath = filepath.Join(dir, "journal.log")
	return memory.NewMemoryStore(backend, &cliTestEmbedder{dim: 16}, cfg)
}

// runCmd executes a CLI command with the store + config injected the same way
// the root command's PersistentPreRunE does.
func runCmd(t *testing.T, store *memory.MemoryStore, cmd *cobra.Command, args ...string) error {
	t.Helper()
	ctx := context.WithValue(context.Background(), storeKey, store)
	ctx = context.WithValue(ctx, cfgKey, defaultConfig())
	cmd.SetArgs(args)
	cmd.SilenceUsage = true
	return cmd.ExecuteContext(ctx)
}

func TestRememberCmdExplicitZeroImportance(t *testing.T) {
	store := newCmdTestStore(t)
	if err := runCmd(t, store, newRememberCmd(), "-i", "0", "explicitly worthless"); err != nil {
		t.Fatalf("remember: %v", err)
	}
	mems, err := store.ListRecent(context.Background(), 1, false)
	if err != nil || len(mems) != 1 {
		t.Fatalf("ListRecent: %v (%d)", err, len(mems))
	}
	if mems[0].Importance != 0 {
		t.Errorf("importance = %v, want 0 (explicit -i 0 must not fall back to 0.5)", mems[0].Importance)
	}
}

func TestRememberCmdDefaultTypeSemantic(t *testing.T) {
	store := newCmdTestStore(t)
	if err := runCmd(t, store, newRememberCmd(), "an untyped note"); err != nil {
		t.Fatalf("remember: %v", err)
	}
	mems, _ := store.ListRecent(context.Background(), 1, false)
	if len(mems) != 1 || mems[0].Type != memory.Semantic {
		t.Errorf("default type = %v, want semantic (config default_type)", mems)
	}
	// And the default importance comes from config (0.5) when -i is omitted.
	if mems[0].Importance != 0.5 {
		t.Errorf("default importance = %v, want 0.5", mems[0].Importance)
	}
}

func TestTagCmdJournaledAndUndoable(t *testing.T) {
	store := newCmdTestStore(t)
	ctx := context.Background()
	m, _, _, err := store.Remember(ctx, "taggable memory", memory.RememberOpts{Tags: []string{"old"}})
	if err != nil {
		t.Fatal(err)
	}

	if err := runCmd(t, store, newTagCmd(), m.ID, "--add", "fresh", "--remove", "old"); err != nil {
		t.Fatalf("tag: %v", err)
	}
	got, _ := store.GetByID(ctx, m.ID)
	if len(got.Tags) != 1 || got.Tags[0] != "fresh" {
		t.Fatalf("tags = %v, want [fresh]", got.Tags)
	}

	// The change went through store.Edit → journaled and undoable.
	entries, _ := store.Journal().Read(memory.ReadOpts{Op: "edit", MemoryID: m.ID})
	if len(entries) != 1 {
		t.Fatalf("edit journal entries = %d, want 1", len(entries))
	}
	if ok, err := store.Undo(ctx, entries[0]); err != nil || !ok {
		t.Fatalf("Undo(tag edit) = %v, %v", ok, err)
	}
	got, _ = store.GetByID(ctx, m.ID)
	if len(got.Tags) != 1 || got.Tags[0] != "old" {
		t.Errorf("tags after undo = %v, want [old]", got.Tags)
	}
}

func TestBumpCmdJournaledAndClamped(t *testing.T) {
	store := newCmdTestStore(t)
	ctx := context.Background()
	m, _, _, err := store.Remember(ctx, "bump me", memory.RememberOpts{Importance: memory.Float32Ptr(0.5)})
	if err != nil {
		t.Fatal(err)
	}

	// Relative bump beyond 1.0 clamps.
	if err := runCmd(t, store, newBumpCmd(), m.ID, "+0.8"); err != nil {
		t.Fatalf("bump: %v", err)
	}
	got, _ := store.GetByID(ctx, m.ID)
	if got.Importance != 1 {
		t.Errorf("importance after +0.8 = %v, want clamped 1", got.Importance)
	}

	// Absolute set to 0 is honoured (explicit zero).
	if err := runCmd(t, store, newBumpCmd(), m.ID, "=0"); err != nil {
		t.Fatalf("bump =0: %v", err)
	}
	got, _ = store.GetByID(ctx, m.ID)
	if got.Importance != 0 {
		t.Errorf("importance after =0 = %v, want 0", got.Importance)
	}

	entries, _ := store.Journal().Read(memory.ReadOpts{Op: "edit", MemoryID: m.ID})
	if len(entries) != 2 {
		t.Errorf("edit journal entries = %d, want 2 (one per bump)", len(entries))
	}
}
