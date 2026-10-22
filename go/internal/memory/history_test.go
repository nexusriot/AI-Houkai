package memory

import (
	"context"
	"path/filepath"
	"testing"
	"time"

	"github.com/nexusriot/ai-houkai/internal/vector"
)

func TestJournalOwnershipSurvivesReopenAndCanonicalPaths(t *testing.T) {
	for _, alias := range []bool{false, true} {
		name := "same-path"
		if alias {
			name = "canonical-alias"
		}
		t.Run(name, func(t *testing.T) {
			ctx := context.Background()
			path := filepath.Join(t.TempDir(), "db")
			open := func(path string) *MemoryStore {
				be, err := vector.NewChromem(path, "reopen", 16)
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(func() { be.Close() })
				return NewMemoryStore(be, &stubEmbedder{dim: 16}, DefaultStoreConfig(path, "reopen"))
			}
			first := open(path)
			m, _, _, err := first.Remember(ctx, "persistent journal subject", RememberOpts{})
			if err != nil {
				t.Fatal(err)
			}
			entries, err := first.Journal().Read(ReadOpts{})
			if err != nil || len(entries) != 1 {
				t.Fatalf("journal: %v %v", entries, err)
			}
			if err := first.Journal().rotate(); err != nil {
				t.Fatal(err)
			}
			if err := first.backend.Close(); err != nil {
				t.Fatal(err)
			}
			if alias {
				path += "/../db"
			}
			reopened := open(path)
			if reopened.Journal().Scope != entries[0].Scope {
				t.Fatal("scope changed after reopen")
			}
			state, err := reopened.StateAt(ctx, nowFloat()+1)
			if err != nil || len(state) != 1 || state[0].ID != m.ID {
				t.Fatalf("reopened history: %v %v", state, err)
			}
			if ok, err := reopened.Undo(ctx, entries[0]); !ok || err != nil {
				t.Fatalf("reopened undo: %v %v", ok, err)
			}
			if _, err := reopened.GetByID(ctx, m.ID); err != ErrNotFound {
				t.Fatalf("undo left memory: %v", err)
			}
		})
	}
}

func TestJournalIsolatesStoresAndCollections(t *testing.T) {
	for _, separatePaths := range []bool{false, true} {
		name := "collections"
		if separatePaths {
			name = "stores"
		}
		t.Run(name, func(t *testing.T) {
			ctx := context.Background()
			dir := t.TempDir()
			makeStore := func(path, collection string) *MemoryStore {
				be, err := vector.NewChromem(path, collection, 16)
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(func() { be.Close() })
				return NewMemoryStore(be, &stubEmbedder{dim: 16}, DefaultStoreConfig(path, collection))
			}
			a := makeStore(filepath.Join(dir, "first"), "alpha")
			otherPath, otherCollection := a.cfg.Path, "bravo"
			if separatePaths {
				otherPath, otherCollection = filepath.Join(dir, "second"), "alpha"
			}
			b := makeStore(otherPath, otherCollection)
			other, _, _, _ := b.Remember(ctx, "private bravo fact", RememberOpts{})
			own, _, _, _ := a.Remember(ctx, "private alpha fact", RememberOpts{})
			if a.Journal().Path != b.Journal().Path {
				t.Fatal("fixture does not share a journal")
			}
			if entries, _ := a.Journal().Read(ReadOpts{Limit: 1}); len(entries) != 1 || entries[0].ID != own.ID {
				t.Fatal("foreign events consumed scoped read limit")
			}
			assertOwn := func() {
				t.Helper()
				state, err := a.StateAt(ctx, nowFloat()+1)
				if err != nil || len(state) != 1 || state[0].ID != own.ID {
					t.Fatalf("foreign replay: %+v, %v", state, err)
				}
			}
			assertOwn()
			if history, _ := a.History(ctx, other.ID, true); len(history) != 0 {
				t.Fatal("foreign history")
			}
			b.Forget(ctx, other.ID)
			entries, _ := b.Journal().Read(ReadOpts{Op: "forget"})
			if ok, err := a.Undo(ctx, entries[0]); ok || err != nil {
				t.Fatalf("foreign undo: %v, %v", ok, err)
			}
			if entry, _ := a.Journal().FindByTS(entries[0].TS, 1e-9); entry != nil {
				t.Fatal("timestamp lookup exposed foreign entry")
			}
			b.Remember(ctx, "another bravo fact", RememberOpts{})
			b.Nuke(ctx)
			assertOwn()
			if err := a.Journal().rotate(); err != nil {
				t.Fatal(err)
			}
			assertOwn()
			raw := NewJournal(a.Journal().Path, 64, 90)
			legacy := JournalEntry{TS: nowFloat(), Op: "forget", ID: own.ID, Before: own.ToDict()}
			raw.Append(legacy)
			if ok, err := a.Undo(ctx, legacy); ok || err != nil {
				t.Fatalf("unscoped undo: %v, %v", ok, err)
			}
			assertOwn()
			if all, _ := raw.Read(ReadOpts{IncludeArchives: true}); len(all) < 3 {
				t.Fatal("raw recovery reader lost entries")
			}
			if scoped, _ := a.Journal().Read(ReadOpts{IncludeArchives: true}); len(scoped) != 1 {
				t.Fatal("scoped reader admitted unowned events")
			}
		})
	}
}

// newJournaledStore builds a store with the journal written inside the test's
// temp dir, so History/StateAt have a clean, isolated log to replay.
func newJournaledStore(t *testing.T) *MemoryStore {
	t.Helper()
	dir := t.TempDir()
	be := newChromemForTest(t, dir)
	cfg := DefaultStoreConfig(dir, "test")
	cfg.JournalEnabled = true
	cfg.JournalPath = filepath.Join(dir, "journal.log")
	return NewMemoryStore(be, &stubEmbedder{dim: 16}, cfg)
}

func opsOf(entries []JournalEntry) []string {
	out := make([]string, len(entries))
	for i, e := range entries {
		out[i] = string(e.Op)
	}
	return out
}

func TestHistoryTimeline(t *testing.T) {
	store := newJournaledStore(t)
	ctx := context.Background()
	m, _, _, _ := store.Remember(ctx, "draft", RememberOpts{})
	text := "draft v2"
	store.Edit(ctx, m.ID, EditOpts{Text: &text})
	store.Forget(ctx, m.ID)

	hist, err := store.History(ctx, m.ID, true)
	if err != nil {
		t.Fatalf("History: %v", err)
	}
	got := opsOf(hist)
	want := []string{"remember", "edit", "forget"}
	if len(got) != 3 || got[0] != want[0] || got[1] != want[1] || got[2] != want[2] {
		t.Errorf("history ops = %v, want %v", got, want)
	}
	if hist[0].After == nil || hist[0].After["text"] != "draft" {
		t.Errorf("remember entry should snapshot the text: %v", hist[0].After)
	}
}

func TestHistoryIncludesLinkTarget(t *testing.T) {
	store := newJournaledStore(t)
	ctx := context.Background()
	a, _, _, _ := store.Remember(ctx, "source memory", RememberOpts{})
	b, _, _, _ := store.Remember(ctx, "target memory", RememberOpts{})
	if err := store.Link(ctx, a.ID, b.ID, RelRefines); err != nil {
		t.Fatalf("Link: %v", err)
	}
	// The link entry is filed under src (a); b appears only via meta.dst_id.
	hist, _ := store.History(ctx, b.ID, true)
	found := false
	for _, e := range hist {
		if e.Op == "link" {
			found = true
		}
	}
	if !found {
		t.Errorf("history of link target missing the link event: %v", opsOf(hist))
	}
}

func TestHistoryIncludesSupersedeCounterpart(t *testing.T) {
	store := newJournaledStore(t)
	ctx := context.Background()
	old, _, _, _ := store.Remember(ctx, "old fact", RememberOpts{})
	newMem, _, _, _ := store.Remember(ctx, "new fact", RememberOpts{})
	if err := store.Supersede(ctx, old.ID, newMem.ID); err != nil {
		t.Fatalf("Supersede: %v", err)
	}
	// supersede is filed under old; new appears only via meta.new_id.
	hist, _ := store.History(ctx, newMem.ID, true)
	found := false
	for _, e := range hist {
		if e.Op == "supersede" {
			found = true
		}
	}
	if !found {
		t.Errorf("history of superseding memory missing the supersede event: %v", opsOf(hist))
	}
}

func TestStateAtReconstructsLifecycle(t *testing.T) {
	store := newJournaledStore(t)
	ctx := context.Background()

	tBefore := nowFloat()
	time.Sleep(20 * time.Millisecond)
	m, _, _, _ := store.Remember(ctx, "original text", RememberOpts{})
	time.Sleep(20 * time.Millisecond)
	tAfterCreate := nowFloat()
	time.Sleep(20 * time.Millisecond)
	edited := "edited text"
	store.Edit(ctx, m.ID, EditOpts{Text: &edited})
	time.Sleep(20 * time.Millisecond)
	tAfterEdit := nowFloat()
	time.Sleep(20 * time.Millisecond)
	store.Forget(ctx, m.ID)
	time.Sleep(20 * time.Millisecond)
	tAfterForget := nowFloat()

	if got, _ := store.GetAt(ctx, m.ID, tBefore); got != nil {
		t.Error("memory should not exist before creation")
	}
	if got, _ := store.GetAt(ctx, m.ID, tAfterCreate); got == nil || got.Text != "original text" {
		t.Errorf("at tAfterCreate want original text, got %+v", got)
	}
	if got, _ := store.GetAt(ctx, m.ID, tAfterEdit); got == nil || got.Text != "edited text" {
		t.Errorf("at tAfterEdit want edited text, got %+v", got)
	}
	if got, _ := store.GetAt(ctx, m.ID, tAfterForget); got != nil {
		t.Error("memory should be gone after forget")
	}
}

func TestStateAtNukeResetsState(t *testing.T) {
	store := newJournaledStore(t)
	ctx := context.Background()
	store.Remember(ctx, "doomed one", RememberOpts{})
	store.Remember(ctx, "doomed two", RememberOpts{})
	time.Sleep(20 * time.Millisecond)
	store.Nuke(ctx)
	time.Sleep(20 * time.Millisecond)
	mems, err := store.StateAt(ctx, nowFloat())
	if err != nil {
		t.Fatalf("StateAt: %v", err)
	}
	if len(mems) != 0 {
		t.Errorf("state after nuke should be empty, got %d", len(mems))
	}
}

func TestStateAtReplaysLinkDelta(t *testing.T) {
	store := newJournaledStore(t)
	ctx := context.Background()
	a, _, _, _ := store.Remember(ctx, "has links", RememberOpts{})
	b, _, _, _ := store.Remember(ctx, "neighbour", RememberOpts{})
	store.Link(ctx, a.ID, b.ID, RelRelated)
	time.Sleep(20 * time.Millisecond)
	got, _ := store.GetAt(ctx, a.ID, nowFloat())
	if got == nil {
		t.Fatal("reconstructed memory is nil")
	}
	found := false
	for _, l := range got.Links {
		if l.To == b.ID && l.Rel == RelRelated {
			found = true
		}
	}
	if !found {
		t.Errorf("link delta not replayed into reconstructed state: %+v", got.Links)
	}
}
