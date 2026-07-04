package vector

import (
	"context"
	"path/filepath"
	"testing"
)

// UpdateMetadata is an in-place upsert: it must never pass through a
// delete-then-add window (a crash or failed re-add would lose the document —
// and this path runs on every recall touch). The observable contract: the
// document survives with its content, embedding, and merged metadata intact.
func TestUpdateMetadataPreservesDocument(t *testing.T) {
	b, err := NewChromem(filepath.Join(t.TempDir(), "store"), "test", 3)
	if err != nil {
		t.Fatalf("NewChromem: %v", err)
	}
	defer b.Close()
	ctx := context.Background()

	orig := Item{
		ID:        "doc",
		Content:   "the content",
		Embedding: []float32{0.6, 0.8, 0},
		Metadata:  map[string]string{"keep": "yes", "change": "old"},
	}
	if err := b.Add(ctx, []Item{orig}); err != nil {
		t.Fatalf("Add: %v", err)
	}

	// Repeated metadata patches (as recall touches do).
	for i := 0; i < 3; i++ {
		if err := b.UpdateMetadata(ctx, "doc", map[string]string{"change": "new"}); err != nil {
			t.Fatalf("UpdateMetadata #%d: %v", i, err)
		}
	}

	got, err := b.Get(ctx, []string{"doc"})
	if err != nil || len(got) != 1 {
		t.Fatalf("Get after update: %v (%d items)", err, len(got))
	}
	if got[0].Content != "the content" {
		t.Errorf("content = %q, want preserved", got[0].Content)
	}
	if got[0].Metadata["change"] != "new" || got[0].Metadata["keep"] != "yes" {
		t.Errorf("metadata = %v, want merged {keep:yes, change:new}", got[0].Metadata)
	}
	// The embedding is reused, not re-computed (chromem normalises, so compare
	// against the normalised original direction).
	if len(got[0].Embedding) != 3 {
		t.Fatalf("embedding lost: %v", got[0].Embedding)
	}
	if sim := CosineSim(got[0].Embedding, orig.Embedding); sim < 0.9999 {
		t.Errorf("embedding changed after UpdateMetadata: cosine = %v, want ~1", sim)
	}

	// The document count is stable — no delete/re-add drift.
	if n, _ := b.Count(ctx); n != 1 {
		t.Errorf("count = %d, want 1", n)
	}
}
