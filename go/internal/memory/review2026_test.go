package memory

// Regression tests for the 2026-08 functional-bug review: collection-scoped
// trash with a live-id restore guard, crash-safe journal rotation, import
// rename link remapping, and the recall fast path's validity shortfall.

import (
	"compress/gzip"
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/nexusriot/ai-houkai/internal/vector"
)

// newCollectionStore builds a store on a shared dir under its own collection,
// the shape two collections opened on one store path have in production.
func newCollectionStore(t *testing.T, dir, collection string) *MemoryStore {
	t.Helper()
	backend, err := vector.NewChromem(filepath.Join(dir, "s-"+collection), collection, 16)
	if err != nil {
		t.Fatalf("NewChromem: %v", err)
	}
	t.Cleanup(func() { _ = backend.Close() })
	return NewMemoryStore(backend, &stubEmbedder{dim: 16}, DefaultStoreConfig(dir, collection))
}

func TestTrashScopedByCollection(t *testing.T) {
	dir := t.TempDir()
	colA := newCollectionStore(t, dir, "col_a")
	colB := newCollectionStore(t, dir, "col_b")
	ctx := context.Background()

	m, _, _, err := colA.Remember(ctx, "belongs to a", RememberOpts{})
	if err != nil {
		t.Fatal(err)
	}
	if ok, err := colA.Trash(ctx, m.ID); err != nil || !ok {
		t.Fatalf("Trash: ok=%v err=%v", ok, err)
	}

	if entries, _ := colB.TrashList(); len(entries) != 0 {
		t.Fatalf("col_b sees col_a's trash: %v", entries)
	}
	if _, found, _ := colB.TrashRestore(ctx, m.ID); found {
		t.Fatal("col_b restored col_a's memory — cross-collection leak")
	}
	if _, err := colB.GetByID(ctx, m.ID); err == nil {
		t.Fatal("col_a's memory materialized in col_b")
	}
	restored, found, err := colA.TrashRestore(ctx, m.ID)
	if err != nil || !found || restored.ID != m.ID {
		t.Fatalf("col_a restore: found=%v err=%v", found, err)
	}
}

func TestTrashPurgeLeavesOtherCollections(t *testing.T) {
	dir := t.TempDir()
	colA := newCollectionStore(t, dir, "col_a")
	colB := newCollectionStore(t, dir, "col_b")
	ctx := context.Background()

	a, _, _, _ := colA.Remember(ctx, "a's trash", RememberOpts{})
	b, _, _, _ := colB.Remember(ctx, "b's trash", RememberOpts{})
	colA.Trash(ctx, a.ID)
	colB.Trash(ctx, b.ID)

	if n, err := colB.TrashPurge(""); err != nil || n != 1 {
		t.Fatalf("purge-all in col_b: n=%d err=%v (want 1)", n, err)
	}
	entries, _ := colA.TrashList()
	if len(entries) != 1 || entries[0].MemoryID != a.ID {
		t.Fatalf("col_a's entry lost to col_b's purge: %v", entries)
	}
	if _, found, _ := colA.TrashRestore(ctx, a.ID); !found {
		t.Fatal("col_a can no longer restore after col_b purged")
	}
}

func TestTrashPurgeExpiredScopedToCollection(t *testing.T) {
	dir := t.TempDir()
	colA := newCollectionStore(t, dir, "col_a")
	colB := newCollectionStore(t, dir, "col_b")
	ctx := context.Background()

	a, _, _, _ := colA.Remember(ctx, "old in a", RememberOpts{})
	b, _, _, _ := colB.Remember(ctx, "old in b", RememberOpts{})
	colA.Trash(ctx, a.ID)
	colB.Trash(ctx, b.ID)

	// Both entries are ancient; only col_a runs retention.
	future := nowFloat() + 90*86400
	if n, err := colA.TrashPurgeExpired(30, future); err != nil || n != 1 {
		t.Fatalf("TrashPurgeExpired: n=%d err=%v (want 1)", n, err)
	}
	if entries, _ := colB.TrashList(); len(entries) != 1 {
		t.Fatalf("col_b's entry TTL-purged by col_a's maintenance: %v", entries)
	}
}

func TestLegacyUntaggedTrashEntryVisibleEverywhere(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	m, _, _, _ := store.Remember(ctx, "template row", RememberOpts{})
	store.Forget(ctx, m.ID)

	legacy := TrashEntry{
		MemoryID: m.ID, DeletedAt: nowFloat(), Actor: "lib",
		Memory: m.ToDict(), // Collection deliberately left ""
	}
	if err := store.appendTrash(legacy); err != nil {
		t.Fatal(err)
	}
	entries, _ := store.TrashList()
	if len(entries) != 1 {
		t.Fatalf("legacy entry invisible: %v", entries)
	}
	if _, found, err := store.TrashRestore(ctx, m.ID); err != nil || !found {
		t.Fatalf("legacy entry not restorable: found=%v err=%v", found, err)
	}
}

func TestTrashRestoreRefusesLiveID(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	m, _, _, _ := store.Remember(ctx, "original text", RememberOpts{})
	store.Trash(ctx, m.ID)

	// Resurrect the id out-of-band (an import can do this legitimately).
	live := m
	live.Text = "resurrected by import"
	if err := store.UpdateMemory(ctx, live, true); err != nil {
		t.Fatal(err)
	}

	if _, found, _ := store.TrashRestore(ctx, m.ID); found {
		t.Fatal("restore over a live id must be refused")
	}
	entries, _ := store.TrashList()
	if len(entries) != 1 {
		t.Fatal("refused restore must keep the snapshot recoverable")
	}
}

func TestTrashRestorePicksNewestDuplicate(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	m, _, _, _ := store.Remember(ctx, "version ONE", RememberOpts{})
	store.Trash(ctx, m.ID)

	entries, _ := store.TrashList()
	newer := entries[0]
	newer.DeletedAt += 10
	newer.Memory = m.ToDict()
	newer.Memory["text"] = "version TWO"
	if err := store.appendTrash(newer); err != nil {
		t.Fatal(err)
	}

	restored, found, err := store.TrashRestore(ctx, m.ID)
	if err != nil || !found {
		t.Fatalf("restore: found=%v err=%v", found, err)
	}
	if restored.Text != "version TWO" {
		t.Fatalf("restored %q, want the newest snapshot", restored.Text)
	}
	if entries, _ = store.TrashList(); len(entries) != 1 {
		t.Fatal("older snapshot must stay parked")
	}
	store.Forget(ctx, m.ID)
	restored, found, _ = store.TrashRestore(ctx, m.ID)
	if !found || restored.Text != "version ONE" {
		t.Fatalf("second restore got %q found=%v, want version ONE", restored.Text, found)
	}
}

func TestJournalRotateRenamesBeforeCompressing(t *testing.T) {
	dir := t.TempDir()
	j := NewJournal(filepath.Join(dir, "j.log"), 64, 90)
	for i := 0; i < 5; i++ {
		j.Append(JournalEntry{TS: 1000 + float64(i), Op: "remember", ID: "id", Meta: map[string]any{}})
	}
	if err := j.rotate(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(j.Path); !os.IsNotExist(err) {
		t.Fatal("active file must be renamed away, not truncated in place")
	}
	archives, _ := filepath.Glob(filepath.Join(dir, "j-*.log.gz"))
	if len(archives) != 1 {
		t.Fatalf("archives = %v, want exactly one", archives)
	}
	if leftovers, _ := filepath.Glob(filepath.Join(dir, "j-*.log")); len(leftovers) != 0 {
		t.Fatalf("rotated plain file not cleaned up: %v", leftovers)
	}
	entries, err := j.Read(ReadOpts{IncludeArchives: true})
	if err != nil || len(entries) != 5 {
		t.Fatalf("Read after rotate: %d entries err=%v, want 5", len(entries), err)
	}
}

func TestJournalTruncatedArchiveTolerated(t *testing.T) {
	dir := t.TempDir()
	j := NewJournal(filepath.Join(dir, "j.log"), 64, 90)
	for i := 0; i < 5; i++ {
		j.Append(JournalEntry{TS: 1000 + float64(i), Op: "remember", ID: "old", Meta: map[string]any{}})
	}
	if err := j.rotate(); err != nil {
		t.Fatal(err)
	}
	archive, _ := filepath.Glob(filepath.Join(dir, "j-*.log.gz"))
	raw, _ := os.ReadFile(archive[0])
	os.WriteFile(archive[0], raw[:len(raw)/2], 0o644)

	for i := 0; i < 3; i++ {
		j.Append(JournalEntry{TS: 2000 + float64(i), Op: "remember", ID: "new", Meta: map[string]any{}})
	}
	entries, err := j.Read(ReadOpts{IncludeArchives: true})
	if err != nil {
		t.Fatalf("Read must tolerate a truncated archive: %v", err)
	}
	live := 0
	for _, e := range entries {
		if e.ID == "new" {
			live++
		}
	}
	if live != 3 {
		t.Fatalf("active entries lost behind a truncated archive: %d/3", live)
	}
}

func TestJournalPlainRotationWinsOverSameStemArchive(t *testing.T) {
	dir := t.TempDir()
	j := NewJournal(filepath.Join(dir, "j.log"), 64, 90)
	for i := 0; i < 4; i++ {
		j.Append(JournalEntry{TS: 1000 + float64(i), Op: "remember", ID: "id", Meta: map[string]any{}})
	}
	if err := j.rotate(); err != nil {
		t.Fatal(err)
	}
	// Recreate the crash-between-compress-and-unlink shape: both files exist.
	archives, _ := filepath.Glob(filepath.Join(dir, "j-*.log.gz"))
	f, _ := os.Open(archives[0])
	gr, err := gzip.NewReader(f)
	if err != nil {
		t.Fatal(err)
	}
	var sb strings.Builder
	buf := make([]byte, 4096)
	for {
		n, rerr := gr.Read(buf)
		sb.Write(buf[:n])
		if rerr != nil {
			break
		}
	}
	f.Close()
	plain := strings.TrimSuffix(archives[0], ".gz")
	os.WriteFile(plain, []byte(sb.String()), 0o644)

	entries, err := j.Read(ReadOpts{IncludeArchives: true})
	if err != nil || len(entries) != 4 {
		t.Fatalf("entries = %d err=%v, want 4 (no double-yield)", len(entries), err)
	}
}

func TestImportRenameRepointsLinks(t *testing.T) {
	ctx := context.Background()
	src := newTestStore(t)
	tgt := newTestStore(t)

	hub, _, _, _ := src.Remember(ctx, "the hub", RememberOpts{})
	spoke, _, _, _ := src.Remember(ctx, "the spoke", RememberOpts{})
	if err := src.Link(ctx, spoke.ID, hub.ID, RelRefines); err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(t.TempDir(), "linked.ahkai")
	if _, err := src.Export(ctx, out, ExportOpts{}); err != nil {
		t.Fatal(err)
	}

	// Squat the hub's id in the target with an unrelated memory.
	squatter := Memory{
		ID: hub.ID, Text: "unrelated squatter", Type: Semantic,
		CreatedAt: nowFloat(), Tags: []string{}, Links: []Link{},
	}
	if err := tgt.UpdateMemory(ctx, squatter, true); err != nil {
		t.Fatal(err)
	}

	summary, err := tgt.Import(ctx, out, ImportOpts{OnConflict: ImportRename})
	if err != nil {
		t.Fatal(err)
	}
	if summary.Renamed != 1 || summary.Imported != 1 {
		t.Fatalf("summary = %+v, want 1 renamed + 1 imported", summary)
	}

	importedSpoke, err := tgt.GetByID(ctx, spoke.ID)
	if err != nil {
		t.Fatal(err)
	}
	if len(importedSpoke.Links) != 1 {
		t.Fatalf("links = %v, want exactly one", importedSpoke.Links)
	}
	link := importedSpoke.Links[0]
	if link.To == hub.ID {
		t.Fatal("link still points at the squatter, not the renamed hub")
	}
	renamedHub, err := tgt.GetByID(ctx, link.To)
	if err != nil || renamedHub.Text != "the hub" {
		t.Fatalf("renamed hub: %+v err=%v", renamedHub, err)
	}
}

func TestRecallFastPathValidityShortfall(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	from, until := 1.0, 2.0
	retired, _, _, err := store.Remember(ctx, "quarterly revenue target details",
		RememberOpts{ValidFrom: &from, ValidUntil: &until})
	if err != nil {
		t.Fatal(err)
	}
	live, _, _, _ := store.Remember(ctx, "quarterly revenue target", RememberOpts{})

	// The retired row ranks first (exact text match); the fast path fetches
	// exactly k=1, the validity filter drops it, and pre-fix the result was
	// empty while the live row was never fetched.
	hits, err := store.Recall(ctx, "quarterly revenue target details", 1, RecallOpts{
		Mode: ModeSemantic, IncludeSuperseded: true, IncludeExpired: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 1 || hits[0].Memory.ID != live.ID {
		t.Fatalf("hits = %+v, want exactly the live memory", hits)
	}
	_ = retired

	// With every row retired the result is legitimately empty.
	store.Forget(ctx, live.ID)
	hits, err = store.Recall(ctx, "quarterly revenue target details", 1, RecallOpts{
		Mode: ModeSemantic, IncludeSuperseded: true, IncludeExpired: true,
	})
	if err != nil || len(hits) != 0 {
		t.Fatalf("hits = %+v err=%v, want empty", hits, err)
	}
}
