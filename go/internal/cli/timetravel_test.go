package cli

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/nexusriot/ai-houkai/internal/memory"
	"github.com/nexusriot/ai-houkai/internal/vector"
)

// The time-travel commands exist precisely for memories that may no longer be
// live, so their id prefixes resolve against the journal (archives included),
// falling back to the live store only when the journal has no match.

func TestHistoryAndGetAtResolvePrefixOfForgottenMemory(t *testing.T) {
	store := newCmdTestStore(t)
	ctx := context.Background()
	m, _, _, err := store.Remember(ctx, "deleted but historied", memory.RememberOpts{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Forget(ctx, m.ID); err != nil {
		t.Fatal(err)
	}

	if err := runCmd(t, store, newHistoryCmd(), m.ID[:8]); err != nil {
		t.Fatalf("history by prefix after forget: %v", err)
	}
	if err := runCmd(t, store, newGetAtCmd(), m.ID[:8], "now"); err != nil {
		// get-at "now" on a forgotten memory legitimately reports
		// non-existence — the resolution itself must not be the failure.
		if got := err.Error(); got != "memory did not exist at that time" {
			t.Fatalf("get-at by prefix after forget: %v", err)
		}
	}
}

func TestUndoLastResolvesPrefixOfForgottenMemory(t *testing.T) {
	store := newCmdTestStore(t)
	ctx := context.Background()
	m, _, _, err := store.Remember(ctx, "undo me by prefix", memory.RememberOpts{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Forget(ctx, m.ID); err != nil {
		t.Fatal(err)
	}

	if err := runCmd(t, store, newJournalUndoLastCmd(), "--id", m.ID[:8], "-y"); err != nil {
		t.Fatalf("undo-last --id prefix after forget: %v", err)
	}
	if _, err := store.GetByID(ctx, m.ID); err != nil {
		t.Fatalf("undo of the forget did not resurrect the memory: %v", err)
	}
}

func TestResolveJournalIDFallsBackToLiveStore(t *testing.T) {
	// With journaling disabled the journal has no matches; the live resolver
	// fills in (and its own not-found error surfaces for garbage).
	dir := t.TempDir()
	backend, err := vector.NewChromem(filepath.Join(dir, "s"), "test", 16)
	if err != nil {
		t.Fatalf("NewChromem: %v", err)
	}
	t.Cleanup(func() { _ = backend.Close() })
	cfg := memory.DefaultStoreConfig(dir, "test")
	cfg.JournalEnabled = false
	store := memory.NewMemoryStore(backend, &cliTestEmbedder{dim: 16}, cfg)
	ctx := context.Background()
	m, _, _, _ := store.Remember(ctx, "live only", memory.RememberOpts{})

	if err := runCmd(t, store, newHistoryCmd(), m.ID[:8]); err != nil {
		t.Fatalf("live fallback: %v", err)
	}
	if err := runCmd(t, store, newHistoryCmd(), "deadbeef"); err == nil {
		t.Fatal("unknown prefix must error")
	}
}
