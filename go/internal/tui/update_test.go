package tui

import (
	"context"
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/nexusriot/ai-houkai/internal/memory"
)

// key builds a single-rune tea.KeyMsg (e.g. 'X', 'r', '/').
func key(r rune) tea.KeyMsg {
	return tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{r}}
}

// seedStore stores n memories and returns the built, laid-out model. A
// WindowSizeMsg is fed first so layout() runs and the model is "ready".
func seedModel(t *testing.T, n int) (*Model, *memory.MemoryStore) {
	t.Helper()
	store := newTestStore(t)
	ctx := context.Background()
	for i := 0; i < n; i++ {
		if _, _, _, err := store.Remember(ctx, memoryText(i), memory.RememberOpts{}); err != nil {
			t.Fatalf("remember: %v", err)
		}
	}
	m := New(store, "test")
	m.Init()
	// Drive a window size so layout() runs.
	updated, _ := m.Update(tea.WindowSizeMsg{Width: 120, Height: 40})
	return updated.(*Model), store
}

func memoryText(i int) string {
	words := []string{"deploy", "release", "rollback", "cache", "pipeline", "runbook"}
	return "memory number " + string(rune('a'+i)) + " about " + words[i%len(words)]
}

func TestUpdateWindowSizeMakesReady(t *testing.T) {
	store := newTestStore(t)
	m := New(store, "coll")
	m.Init()
	if m.ready {
		t.Fatal("model should not be ready before a WindowSizeMsg")
	}
	updated, _ := m.Update(tea.WindowSizeMsg{Width: 100, Height: 30})
	m = updated.(*Model)
	if !m.ready {
		t.Fatal("model should be ready after WindowSizeMsg")
	}
	if m.width != 100 || m.height != 30 {
		t.Errorf("dimensions = %dx%d, want 100x30", m.width, m.height)
	}
}

func TestNukeTwoPressFlow(t *testing.T) {
	m, store := seedModel(t, 3)
	ctx := context.Background()

	if n, _ := store.Count(ctx); n != 3 {
		t.Fatalf("precondition: store count = %d, want 3", n)
	}

	// First X arms the nuke.
	updated, _ := m.Update(key('X'))
	m = updated.(*Model)
	if !m.nukePending {
		t.Fatal("first X should arm nukePending")
	}
	if !strings.Contains(m.status, "Press X again") {
		t.Errorf("armed status = %q, want it to mention 'Press X again'", m.status)
	}
	if n, _ := store.Count(ctx); n != 3 {
		t.Errorf("store should be untouched after first X, count = %d", n)
	}

	// Second X confirms and wipes the store.
	updated, _ = m.Update(key('X'))
	m = updated.(*Model)
	if m.nukePending {
		t.Error("nukePending should be cleared after the confirming X")
	}
	if n, _ := store.Count(ctx); n != 0 {
		t.Fatalf("store should be empty after second X, count = %d", n)
	}
	if !strings.Contains(m.status, "Nuked") {
		t.Errorf("post-nuke status = %q, want it to mention 'Nuked'", m.status)
	}
}

func TestNukeCancelledByOtherKey(t *testing.T) {
	m, store := seedModel(t, 2)
	ctx := context.Background()

	// Arm.
	updated, _ := m.Update(key('X'))
	m = updated.(*Model)
	if !m.nukePending {
		t.Fatal("first X should arm nukePending")
	}

	// A non-X key cancels the pending nuke.
	updated, _ = m.Update(key('r'))
	m = updated.(*Model)
	if m.nukePending {
		t.Fatal("a non-X key should cancel nukePending")
	}

	// A subsequent X now only re-arms (does not nuke), so the store survives.
	updated, _ = m.Update(key('X'))
	m = updated.(*Model)
	if !m.nukePending {
		t.Error("X after a cancel should re-arm, not nuke")
	}
	if n, _ := store.Count(ctx); n != 2 {
		t.Fatalf("store should survive the cancelled nuke, count = %d, want 2", n)
	}
}

func TestNukeEmptyStore(t *testing.T) {
	m, _ := seedModel(t, 0)
	updated, _ := m.Update(key('X'))
	m = updated.(*Model)
	if m.nukePending {
		t.Error("X on an empty store should not arm nukePending")
	}
	if !strings.Contains(m.status, "already empty") {
		t.Errorf("status = %q, want it to mention 'already empty'", m.status)
	}
}

func TestHandleNukeDirect(t *testing.T) {
	m, store := seedModel(t, 4)
	ctx := context.Background()

	// First call arms.
	if _, _ = m.handleNuke(); !m.nukePending {
		t.Fatal("first handleNuke should arm")
	}
	// Second call executes.
	if _, _ = m.handleNuke(); m.nukePending {
		t.Fatal("second handleNuke should clear nukePending")
	}
	if n, _ := store.Count(ctx); n != 0 {
		t.Fatalf("handleNuke did not empty the store, count = %d", n)
	}
	if !strings.Contains(m.status, "Nuked 4") {
		t.Errorf("status = %q, want it to report 'Nuked 4'", m.status)
	}
}

func TestReloadRecentKey(t *testing.T) {
	m, store := seedModel(t, 1)
	ctx := context.Background()

	// Add a memory behind the model's back, then press 'r' to reload.
	if _, _, _, err := store.Remember(ctx, "freshly added after view built", memory.RememberOpts{}); err != nil {
		t.Fatal(err)
	}
	updated, _ := m.Update(key('r'))
	m = updated.(*Model)
	if m.status != "" {
		t.Errorf("reload status = %q, want empty on success", m.status)
	}
	// The reloaded recent view should now hold both memories.
	if got := len(m.nav.Current().Memories); got != 2 {
		t.Errorf("reloaded view has %d memories, want 2", got)
	}
}

func TestSearchOpenAndClose(t *testing.T) {
	m, _ := seedModel(t, 2)

	// '/' opens the search box.
	updated, _ := m.Update(key('/'))
	m = updated.(*Model)
	if !m.searching {
		t.Fatal("'/' should enter search mode")
	}
	if !m.search.Focused() {
		t.Error("search input should be focused after '/'")
	}

	// While searching, a plain rune is typed into the box, not treated as a command.
	updated, _ = m.Update(key('X'))
	m = updated.(*Model)
	if m.nukePending {
		t.Error("X typed into the search box must not arm the nuke")
	}
	if !m.searching {
		t.Error("still in search mode after typing")
	}

	// esc closes the search box and clears it.
	updated, _ = m.Update(tea.KeyMsg{Type: tea.KeyEsc})
	m = updated.(*Model)
	if m.searching {
		t.Error("esc should exit search mode")
	}
	if m.search.Value() != "" {
		t.Errorf("search value = %q, want cleared", m.search.Value())
	}
}

func TestSearchEnterRunsQuery(t *testing.T) {
	m, store := seedModel(t, 1)
	ctx := context.Background()
	if _, _, _, err := store.Remember(ctx, "the quick brown fox jumps", memory.RememberOpts{}); err != nil {
		t.Fatal(err)
	}

	// Open search, type a query, press enter.
	updated, _ := m.Update(key('/'))
	m = updated.(*Model)
	for _, r := range "brown fox" {
		updated, _ = m.Update(key(r))
		m = updated.(*Model)
	}
	updated, _ = m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	m = updated.(*Model)

	if m.searching {
		t.Error("enter should exit search mode")
	}
	if m.nav.Current().Kind != "search" {
		t.Errorf("current view kind = %q, want search", m.nav.Current().Kind)
	}
}

func TestQuitKeyReturnsQuitCmd(t *testing.T) {
	m, _ := seedModel(t, 1)
	_, cmd := m.Update(key('q'))
	if cmd == nil {
		t.Fatal("'q' should return a command (tea.Quit)")
	}
	// tea.Quit() yields a tea.QuitMsg.
	if _, ok := cmd().(tea.QuitMsg); !ok {
		t.Errorf("'q' command did not produce a QuitMsg")
	}
}

func TestBackKeyStaysAtRoot(t *testing.T) {
	m, _ := seedModel(t, 2)
	// 'b' at the root recent view should keep us on the recent view.
	updated, _ := m.Update(key('b'))
	m = updated.(*Model)
	if m.nav.Current().Kind != "recent" {
		t.Errorf("after 'b' at root, kind = %q, want recent", m.nav.Current().Kind)
	}
}
