package memory

import (
	"compress/gzip"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

type unavailableImportEmbedder struct{}

func (unavailableImportEmbedder) Dim() int { return 16 }
func (unavailableImportEmbedder) Embed(context.Context, []string) ([][]float32, error) {
	return nil, fmt.Errorf("embedding provider unavailable")
}

func writeImportRows(t *testing.T, rows []map[string]any, dim int) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "import.ahkai")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	gz := gzip.NewWriter(f)
	enc := json.NewEncoder(gz)
	if err := enc.Encode(map[string]any{"format": "ai-houkai/export", "version": 1,
		"source": map[string]any{"embedding_dim": dim}}); err != nil {
		t.Fatal(err)
	}
	for _, row := range rows {
		if err := enc.Encode(row); err != nil {
			t.Fatal(err)
		}
	}
	if err := gz.Close(); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestFailedImportOverwritePreservesOriginal(t *testing.T) {
	for _, failure := range []string{"provider", "vector"} {
		t.Run(failure, func(t *testing.T) {
			s := newJournaledStore(t)
			ctx := context.Background()
			m, _, _, err := s.Remember(ctx, "original", RememberOpts{Tags: []string{"keep"}, Pinned: true})
			if err != nil {
				t.Fatal(err)
			}
			before, _ := s.GetByID(ctx, m.ID)
			row := map[string]any{"id": m.ID, "text": "replacement", "meta": m.ToDict()}
			dim := 0
			if failure == "vector" {
				dim = 16
				row["vector"] = []float32{1, 0}
			}
			path := writeImportRows(t, []map[string]any{row}, dim)
			if failure == "provider" {
				s.embedder = unavailableImportEmbedder{}
			}
			entriesBefore, _ := s.Journal().Read(ReadOpts{})
			sum, err := s.Import(ctx, path, ImportOpts{OnConflict: ImportOverwrite})
			if err != nil || len(sum.Errors) != 1 || sum.Overwritten != 0 {
				t.Fatalf("unexpected import result: %+v, %v", sum, err)
			}
			after, err := s.GetByID(ctx, m.ID)
			if err != nil || !reflect.DeepEqual(before, after) {
				t.Fatalf("failed overwrite changed original: %+v, %v", after, err)
			}
			entriesAfter, _ := s.Journal().Read(ReadOpts{})
			if !reflect.DeepEqual(entriesBefore, entriesAfter) {
				t.Fatal("failed overwrite changed journal")
			}
		})
	}
}

func TestImportErrorPreflightsEveryCollision(t *testing.T) {
	for _, kind := range []string{"existing-first", "existing-last", "duplicate", "duplicate-meta"} {
		for _, dryRun := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/dry=%v", kind, dryRun), func(t *testing.T) {
				s := newJournaledStore(t)
				ctx := context.Background()
				m, _, _, _ := s.Remember(ctx, "existing", RememberOpts{})
				fresh := map[string]any{"id": "fresh", "text": "new fact"}
				collision := map[string]any{"id": m.ID, "text": "collision"}
				rows := []map[string]any{fresh, collision}
				if kind == "existing-first" {
					rows = []map[string]any{collision, fresh}
				}
				if kind == "duplicate" {
					rows = []map[string]any{fresh, fresh}
				}
				if kind == "duplicate-meta" {
					meta := map[string]any{"meta": map[string]any{"id": "fresh", "text": "new fact"}}
					rows = []map[string]any{meta, meta}
				}
				path := writeImportRows(t, rows, 0)
				before, _ := s.Journal().Read(ReadOpts{})
				sum, err := s.Import(ctx, path, ImportOpts{OnConflict: ImportErrorPolicy, DryRun: dryRun})
				var conflict *ImportConflictError
				if !errors.As(err, &conflict) || len(conflict.Collisions) != 1 || sum.Imported != 0 {
					t.Fatalf("expected preflight rejection, got %+v, %v", sum, err)
				}
				if count, _ := s.Count(ctx); count != 1 {
					t.Fatalf("rejected import left %d memories", count)
				}
				after, _ := s.Journal().Read(ReadOpts{})
				if !reflect.DeepEqual(before, after) {
					t.Fatal("rejected import changed journal")
				}
			})
		}
	}
}

func TestUndoEditRejectsDivergedState(t *testing.T) {
	for _, change := range []string{"edit", "link", "supersede", "recall", "missing-after"} {
		t.Run(change, func(t *testing.T) {
			s := newJournaledStore(t)
			ctx := context.Background()
			m, _, _, _ := s.Remember(ctx, "original", RememberOpts{Importance: Float32Ptr(0.1234567)})
			text := "first edit"
			if _, err := s.Edit(ctx, m.ID, EditOpts{Text: &text}); err != nil {
				t.Fatal(err)
			}
			entries, _ := s.Journal().Read(ReadOpts{Op: "edit"})
			e := entries[0]
			switch change {
			case "edit":
				text = "newer edit"
				s.Edit(ctx, m.ID, EditOpts{Text: &text})
			case "link", "supersede":
				other, _, _, _ := s.Remember(ctx, "other", RememberOpts{})
				if change == "link" {
					s.Link(ctx, m.ID, other.ID, RelRelated)
				} else {
					s.Supersede(ctx, m.ID, other.ID)
				}
			case "recall":
				s.Recall(ctx, text, 1, RecallOpts{})
			case "missing-after":
				e.After = nil
			}
			before, _ := s.GetByID(ctx, m.ID)
			ok, err := s.Undo(ctx, e)
			if err != nil || ok != (change == "recall") {
				t.Fatalf("undo = %v, %v", ok, err)
			}
			after, _ := s.GetByID(ctx, m.ID)
			if change == "recall" {
				if after.Text != "original" {
					t.Fatal("recall prevented a safe undo")
				}
			} else if !reflect.DeepEqual(before, after) {
				t.Fatal("stale undo changed memory")
			}
		})
	}
}

func TestExportImportRoundTrip(t *testing.T) {
	src := newTestStore(t)
	dst := newTestStore(t)
	ctx := context.Background()

	m1, _, _, err := src.Remember(ctx, "alpha one", RememberOpts{Type: Semantic, Tags: []string{"a"}})
	if err != nil {
		t.Fatal(err)
	}
	m2, _, _, err := src.Remember(ctx, "beta two", RememberOpts{Type: Semantic, Tags: []string{"b"}})
	if err != nil {
		t.Fatal(err)
	}

	exportPath := filepath.Join(t.TempDir(), "dump.ahkai")
	sum, err := src.Export(ctx, exportPath, ExportOpts{IncludeVectors: true})
	if err != nil {
		t.Fatal(err)
	}
	if sum.Count != 2 {
		t.Fatalf("exported %d, want 2", sum.Count)
	}
	if _, err := os.Stat(exportPath); err != nil {
		t.Fatalf("export missing: %v", err)
	}

	hdr, count, err := PeekExportHeader(exportPath)
	if err != nil {
		t.Fatal(err)
	}
	if hdr.Format != "ai-houkai/export" || hdr.Version != 1 || count != 2 {
		t.Fatalf("bad header: %+v count=%d", hdr, count)
	}

	impSum, err := dst.Import(ctx, exportPath, ImportOpts{OnConflict: ImportSkip})
	if err != nil {
		t.Fatal(err)
	}
	if impSum.Imported != 2 {
		t.Fatalf("imported %d, want 2", impSum.Imported)
	}

	// Re-import should be a no-op under "skip".
	impSum, err = dst.Import(ctx, exportPath, ImportOpts{OnConflict: ImportSkip})
	if err != nil {
		t.Fatal(err)
	}
	if impSum.Skipped != 2 || impSum.Imported != 0 {
		t.Fatalf("re-import: imported=%d skipped=%d (want 0/2)", impSum.Imported, impSum.Skipped)
	}

	if _, err := dst.GetByID(ctx, m1.ID); err != nil {
		t.Errorf("m1 missing: %v", err)
	}
	if _, err := dst.GetByID(ctx, m2.ID); err != nil {
		t.Errorf("m2 missing: %v", err)
	}

	if dst.Journal() != nil {
		entries, _ := dst.Journal().Read(ReadOpts{Op: "import"})
		if len(entries) != 2 {
			t.Errorf("journal: %d import entries, want 2", len(entries))
		}
	}
}

func TestImportOverwrite(t *testing.T) {
	for _, vectors := range []bool{false, true} {
		t.Run(fmt.Sprintf("vectors=%v", vectors), func(t *testing.T) {
			s := newJournaledStore(t)
			ctx := context.Background()
			original, _, _, err := s.Remember(ctx, "original", RememberOpts{Tags: []string{"archived"}, Pinned: true})
			if err != nil {
				t.Fatal(err)
			}
			before, _ := s.GetByID(ctx, original.ID)
			path := filepath.Join(t.TempDir(), "overwrite.ahkai")
			if _, err := s.Export(ctx, path, ExportOpts{IncludeVectors: vectors}); err != nil {
				t.Fatal(err)
			}
			text := "modified locally"
			if _, err := s.Edit(ctx, original.ID, EditOpts{Text: &text, Tags: []string{"local"}}); err != nil {
				t.Fatal(err)
			}
			sum, err := s.Import(ctx, path, ImportOpts{OnConflict: ImportOverwrite})
			if err != nil || sum.Overwritten != 1 || len(sum.Errors) != 0 {
				t.Fatalf("overwrite: %+v, %v", sum, err)
			}
			got, err := s.GetByID(ctx, original.ID)
			if err != nil || !reflect.DeepEqual(before, got) {
				t.Fatalf("overwrite did not restore snapshot: %+v, %v", got, err)
			}
			if count, _ := s.Count(ctx); count != 1 {
				t.Fatalf("overwrite created %d rows", count)
			}
		})
	}
}

func TestUndoRemember(t *testing.T) {
	s := newTestStore(t)
	ctx := context.Background()

	m, _, _, err := s.Remember(ctx, "to undo", RememberOpts{Type: Semantic})
	if err != nil {
		t.Fatal(err)
	}
	entries, _ := s.Journal().Read(ReadOpts{Op: "remember", MemoryID: m.ID})
	if len(entries) != 1 {
		t.Fatalf("journal: %d remember entries, want 1", len(entries))
	}
	ok, err := s.Undo(ctx, entries[0])
	if err != nil || !ok {
		t.Fatalf("undo: ok=%v err=%v", ok, err)
	}
	if _, err := s.GetByID(ctx, m.ID); err == nil {
		t.Errorf("memory still present after undo")
	}
}

func TestJournalRotateAndPrune(t *testing.T) {
	dir := t.TempDir()
	j := NewJournal(filepath.Join(dir, "j.log"), 1, 1) // 1 MB rotate, 1-day retention
	for i := 0; i < 5; i++ {
		j.Append(JournalEntry{TS: float64(i), Op: "remember", Actor: "test", ID: "id"})
	}
	got, err := j.Read(ReadOpts{})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 5 {
		t.Errorf("read %d, want 5", len(got))
	}
	// Find by ts.
	e, err := j.FindByTS(2, 0.01)
	if err != nil || e == nil || e.TS != 2 {
		t.Errorf("FindByTS: got %+v err=%v", e, err)
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

func TestImportOnConflictValidation(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	_, err := store.Import(ctx, filepath.Join(t.TempDir(), "missing.ahkai"), ImportOpts{OnConflict: "merge"})
	if !IsValidationError(err) {
		t.Errorf("bad import policy: err = %v, want ValidationError", err)
	}
}

func TestImportJournalsVectorsPreserved(t *testing.T) {
	ctx := context.Background()

	src := newTestStore(t)
	if _, _, _, err := src.Remember(ctx, "exported knowledge", RememberOpts{}); err != nil {
		t.Fatal(err)
	}

	// Vectors included → the file's embedding is reused → preserved=true.
	withVec := filepath.Join(t.TempDir(), "with.ahkai")
	if _, err := src.Export(ctx, withVec, ExportOpts{IncludeVectors: true}); err != nil {
		t.Fatal(err)
	}
	dst1 := newTestStore(t)
	if _, err := dst1.Import(ctx, withVec, ImportOpts{OnConflict: ImportSkip}); err != nil {
		t.Fatal(err)
	}
	entries, _ := dst1.Journal().Read(ReadOpts{Op: "import"})
	if len(entries) != 1 {
		t.Fatalf("import entries = %d, want 1", len(entries))
	}
	if v, _ := entries[0].Meta["vectors_preserved"].(bool); !v {
		t.Errorf("vectors_preserved = %v, want true (file vector reused)", entries[0].Meta["vectors_preserved"])
	}

	// Vectors stripped → the row is re-embedded → preserved=false.
	noVec := filepath.Join(t.TempDir(), "without.ahkai")
	if _, err := src.Export(ctx, noVec, ExportOpts{IncludeVectors: false}); err != nil {
		t.Fatal(err)
	}
	dst2 := newTestStore(t)
	if _, err := dst2.Import(ctx, noVec, ImportOpts{OnConflict: ImportSkip}); err != nil {
		t.Fatal(err)
	}
	// Sibling TempDir stores share a default journal path, so read the LATEST
	// import entry (dst1's import is also in the file).
	entries, _ = dst2.Journal().Read(ReadOpts{Op: "import"})
	if len(entries) == 0 {
		t.Fatal("no import entries journaled")
	}
	last := entries[len(entries)-1]
	if v, _ := last.Meta["vectors_preserved"].(bool); v {
		t.Errorf("vectors_preserved = true, want false (row was re-embedded)")
	}
}
