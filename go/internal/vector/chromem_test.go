package vector

import (
	"context"
	"math"
	"path/filepath"
	"testing"
)

func TestCosineSim(t *testing.T) {
	cases := []struct {
		name string
		a, b []float32
		want float32
	}{
		{"identical", []float32{1, 0, 0}, []float32{1, 0, 0}, 1.0},
		{"orthogonal", []float32{1, 0, 0}, []float32{0, 1, 0}, 0.0},
		{"opposite", []float32{1, 0, 0}, []float32{-1, 0, 0}, -1.0},
		{"length-mismatch", []float32{1, 0}, []float32{1, 0, 0}, 0.0},
		{"zero-vec", []float32{0, 0, 0}, []float32{1, 0, 0}, 0.0},
	}
	for _, c := range cases {
		got := CosineSim(c.a, c.b)
		if math.Abs(float64(got-c.want)) > 1e-5 {
			t.Errorf("%s: got %.4f, want %.4f", c.name, got, c.want)
		}
	}
}

func TestChromemRoundTrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "store")
	dim := 3

	b, err := NewChromem(path, "test", dim)
	if err != nil {
		t.Fatalf("NewChromem: %v", err)
	}
	defer b.Close()

	ctx := context.Background()
	items := []Item{
		{
			ID:        "a",
			Content:   "first",
			Embedding: []float32{1, 0, 0},
			Metadata:  map[string]string{"k": "v"},
		},
		{
			ID:        "b",
			Content:   "second",
			Embedding: []float32{0, 1, 0},
			Metadata:  map[string]string{"k": "w"},
		},
	}
	if err := b.Add(ctx, items); err != nil {
		t.Fatalf("Add: %v", err)
	}

	n, err := b.Count(ctx)
	if err != nil || n != 2 {
		t.Fatalf("Count: got n=%d err=%v, want 2", n, err)
	}

	// Query closer to "a" should rank a first.
	hits, err := b.Query(ctx, []float32{0.9, 0.1, 0}, 2)
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	if len(hits) == 0 || hits[0].ID != "a" {
		t.Errorf("expected first hit to be 'a', got %+v", hits)
	}

	all, err := b.All(ctx)
	if err != nil || len(all) != 2 {
		t.Fatalf("All: len=%d err=%v, want 2", len(all), err)
	}

	got, err := b.Get(ctx, []string{"a", "missing"})
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if len(got) != 1 || got[0].ID != "a" {
		t.Errorf("Get should silently skip missing IDs, got %+v", got)
	}

	if err := b.UpdateMetadata(ctx, "a", map[string]string{"k": "vv"}); err != nil {
		t.Fatalf("UpdateMetadata: %v", err)
	}
	got, _ = b.Get(ctx, []string{"a"})
	if got[0].Metadata["k"] != "vv" {
		t.Errorf("metadata not updated: %+v", got[0].Metadata)
	}

	if err := b.Delete(ctx, []string{"a"}); err != nil {
		t.Fatalf("Delete: %v", err)
	}
	n, _ = b.Count(ctx)
	if n != 1 {
		t.Errorf("after delete: got n=%d, want 1", n)
	}
}

func TestCollectionManagement(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "store")
	ctx := context.Background()

	b, err := NewChromem(path, "main", 3)
	if err != nil {
		t.Fatalf("NewChromem: %v", err)
	}
	defer b.Close()

	if err := b.Add(ctx, []Item{
		{ID: "a", Content: "alpha", Embedding: []float32{1, 0, 0}, Metadata: map[string]string{"k": "v"}},
		{ID: "b", Content: "beta", Embedding: []float32{0, 1, 0}, Metadata: map[string]string{}},
	}); err != nil {
		t.Fatalf("Add: %v", err)
	}

	if !b.HasCollection("main") {
		t.Error("main collection should exist")
	}
	if counts := b.ListCollections(); counts["main"] != 2 {
		t.Errorf("ListCollections = %v", counts)
	}

	if err := b.CreateCollection("extra"); err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}
	if !b.HasCollection("extra") {
		t.Error("extra collection missing after create")
	}

	copied, err := b.CopyCollection(ctx, "main", "dst")
	if err != nil {
		t.Fatalf("CopyCollection: %v", err)
	}
	if copied != 2 {
		t.Errorf("copied = %d, want 2", copied)
	}
	if counts := b.ListCollections(); counts["dst"] != 2 {
		t.Errorf("dst count = %d, want 2", counts["dst"])
	}

	// Copy preserves content, metadata, and embeddings.
	dstBackend, err := NewChromem(path, "dst", 3)
	if err != nil {
		t.Fatalf("open dst: %v", err)
	}
	items, err := dstBackend.Get(ctx, []string{"a"})
	if err != nil || len(items) != 1 {
		t.Fatalf("Get from dst: %v (%d items)", err, len(items))
	}
	if items[0].Content != "alpha" || items[0].Metadata["k"] != "v" || len(items[0].Embedding) == 0 {
		t.Errorf("copied item mismatch: %+v", items[0])
	}

	if err := b.DeleteCollection("extra"); err != nil {
		t.Fatalf("DeleteCollection: %v", err)
	}
	if b.HasCollection("extra") {
		t.Error("extra collection still present after delete")
	}

	// Copying a missing collection errors.
	if _, err := b.CopyCollection(ctx, "nope", "x"); err == nil {
		t.Error("expected error for missing src collection")
	}
}
