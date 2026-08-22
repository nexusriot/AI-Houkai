package memory

import (
	"context"
	"path/filepath"
	"testing"
	"time"
)

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
