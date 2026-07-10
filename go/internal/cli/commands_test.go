package cli

// Command-level regression tests for the review fixes: pointer importance on
// remember, semantic default type, and tag/bump going through the journaled
// store.Edit path (so `journal undo` can reverse them).

import (
	"context"
	"hash/fnv"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
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
	mems, err := store.ListRecent(context.Background(), 1, false, false)
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
	mems, _ := store.ListRecent(context.Background(), 1, false, false)
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

// TestImportCmdRuntimeErrorFormatting locks in the CLI error handling: a
// runtime failure (here, importing a missing file) must be printed exactly
// once and must NOT be followed by the command's usage text.
//
// Two bugs used to break this: newImportCmd printed the error by hand *and*
// returned it (cobra printed it a second time), and the root command never set
// SilenceUsage, so cobra also dumped the full usage after every runtime error.
func TestImportCmdRuntimeErrorFormatting(t *testing.T) {
	dir := t.TempDir()
	// Force a key-free provider and an isolated store so PersistentPreRunE
	// succeeds without network or real credentials; the missing-file error
	// then surfaces from RunE, which is the path under test.
	t.Setenv("AI_HOUKAI_EMBED_PROVIDER", "ollama")
	t.Setenv("AI_HOUKAI_PATH", filepath.Join(dir, "store"))

	// The manual print (the bug) wrote straight to os.Stderr, bypassing cobra's
	// writer, so capture the real os.Stdout/os.Stderr via a pipe to see both it
	// and cobra's own print.
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("pipe: %v", err)
	}
	origOut, origErr := os.Stdout, os.Stderr
	os.Stdout, os.Stderr = w, w

	root := NewRootCmd()
	root.SetArgs([]string{"import", filepath.Join(dir, "missing.ahkai"), "-y"})
	runErr := root.Execute()

	os.Stdout, os.Stderr = origOut, origErr
	_ = w.Close()
	out, _ := io.ReadAll(r)
	s := string(out)

	if runErr == nil {
		t.Fatal("importing a missing file should fail")
	}
	if n := strings.Count(s, "Error:"); n != 1 {
		t.Errorf("want the error printed exactly once, got %d occurrences:\n%s", n, s)
	}
	if strings.Contains(s, "Usage:") {
		t.Errorf("usage must be silenced for a runtime error, but it was dumped:\n%s", s)
	}
}

func TestRememberCmdTTL(t *testing.T) {
	store := newCmdTestStore(t)
	if err := runCmd(t, store, newRememberCmd(), "--ttl", "100", "ephemeral note"); err != nil {
		t.Fatalf("remember --ttl: %v", err)
	}
	mems, _ := store.ListRecent(context.Background(), 1, false, true)
	if len(mems) != 1 || mems[0].ExpiresAt <= 0 {
		t.Errorf("--ttl should set expires_at, got %+v", mems)
	}
}

func TestPurgeCmd(t *testing.T) {
	store := newCmdTestStore(t)
	ctx := context.Background()
	exp := float64(1.0)
	store.Remember(ctx, "expired doc", memory.RememberOpts{ExpiresAt: &exp})
	store.Remember(ctx, "live doc", memory.RememberOpts{})

	// Dry-run: nothing deleted.
	if err := runCmd(t, store, newPurgeCmd()); err != nil {
		t.Fatalf("purge dry-run: %v", err)
	}
	if n, _ := store.Count(ctx); n != 2 {
		t.Fatalf("dry-run should not delete, count=%d", n)
	}
	// Apply.
	if err := runCmd(t, store, newPurgeCmd(), "--apply", "--yes"); err != nil {
		t.Fatalf("purge apply: %v", err)
	}
	if n, _ := store.Count(ctx); n != 1 {
		t.Errorf("after purge count=%d, want 1 (live doc)", n)
	}
}
