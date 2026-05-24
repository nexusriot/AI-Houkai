package reflect

import (
	"context"
	"testing"

	"github.com/nexusriot/ai-houkai/internal/memory"
	"github.com/nexusriot/ai-houkai/internal/vector"
)

// fakeReflectStore implements Storable for unit tests.
type fakeReflectStore struct {
	items     []vector.Item
	added     []memory.Memory
	links     []struct{ src, dst, rel string }
	forgotten []string
}

func (f *fakeReflectStore) AllRaw(_ context.Context) ([]vector.Item, error) {
	return f.items, nil
}
func (f *fakeReflectStore) Remember(_ context.Context, text string, opts memory.RememberOpts) (memory.Memory, bool, []memory.Conflict, error) {
	m := memory.Memory{
		ID:         "new-" + text[:min(8, len(text))],
		Text:       text,
		Type:       opts.Type,
		Tags:       opts.Tags,
		Importance: opts.Importance,
	}
	f.added = append(f.added, m)
	return m, true, nil, nil
}
func (f *fakeReflectStore) Forget(_ context.Context, id string) (bool, error) {
	f.forgotten = append(f.forgotten, id)
	return true, nil
}
func (f *fakeReflectStore) Link(_ context.Context, src, dst, rel string) error {
	f.links = append(f.links, struct{ src, dst, rel string }{src, dst, rel})
	return nil
}

func episode(id string, vec []float32, imp float32) vector.Item {
	m := memory.Memory{
		ID:         id,
		Type:       memory.Episodic,
		Importance: imp,
		Tags:       []string{"t-" + id},
	}
	return vector.Item{
		ID:        id,
		Content:   "text-" + id,
		Embedding: vec,
		Metadata:  memory.MemoryToMetadata(m),
	}
}

func TestClustersGroupsSimilar(t *testing.T) {
	store := &fakeReflectStore{items: []vector.Item{
		episode("a", []float32{1, 0, 0}, 0.9),
		episode("b", []float32{0.99, 0.01, 0}, 0.5), // very close to a
		episode("c", []float32{0, 1, 0}, 0.5),        // far
	}}
	e := New(store, 0.9, 2, nil)
	clusters, err := e.Clusters(context.Background())
	if err != nil {
		t.Fatalf("Clusters: %v", err)
	}
	if len(clusters) != 1 {
		t.Fatalf("want 1 cluster, got %d", len(clusters))
	}
	if len(clusters[0]) != 2 {
		t.Errorf("cluster size: got %d, want 2", len(clusters[0]))
	}
}

func TestReflectCreatesSemanticAndLinksSources(t *testing.T) {
	store := &fakeReflectStore{items: []vector.Item{
		episode("a", []float32{1, 0, 0}, 0.9),
		episode("b", []float32{0.99, 0.01, 0}, 0.5),
	}}
	e := New(store, 0.9, 2, nil)

	created, err := e.Reflect(context.Background(), false, false)
	if err != nil {
		t.Fatalf("Reflect: %v", err)
	}
	if len(created) != 1 {
		t.Fatalf("want 1 semantic memory created, got %d", len(created))
	}
	if created[0].Type != memory.Semantic {
		t.Errorf("new memory type: got %q, want semantic", created[0].Type)
	}
	if len(store.links) != 2 {
		t.Errorf("want 2 derived_from links, got %d", len(store.links))
	}
	for _, l := range store.links {
		if l.rel != memory.RelDerivedFrom {
			t.Errorf("link rel: got %q, want %q", l.rel, memory.RelDerivedFrom)
		}
	}
	if len(store.forgotten) != 0 {
		t.Error("non-consolidate run should not delete sources")
	}
}

func TestReflectConsolidateDeletesSources(t *testing.T) {
	store := &fakeReflectStore{items: []vector.Item{
		episode("a", []float32{1, 0, 0}, 0.9),
		episode("b", []float32{0.99, 0.01, 0}, 0.5),
	}}
	e := New(store, 0.9, 2, nil)
	if _, err := e.Reflect(context.Background(), false, true); err != nil {
		t.Fatalf("Reflect: %v", err)
	}
	if len(store.forgotten) != 2 {
		t.Errorf("consolidate should forget 2 sources, got %d", len(store.forgotten))
	}
}

func TestReflectDryRunWritesNothing(t *testing.T) {
	store := &fakeReflectStore{items: []vector.Item{
		episode("a", []float32{1, 0, 0}, 0.9),
		episode("b", []float32{0.99, 0.01, 0}, 0.5),
	}}
	e := New(store, 0.9, 2, nil)
	created, err := e.Reflect(context.Background(), true, true)
	if err != nil {
		t.Fatalf("Reflect: %v", err)
	}
	if len(created) != 1 {
		t.Errorf("dry-run should still report 1 candidate, got %d", len(created))
	}
	if len(store.added) != 0 || len(store.links) != 0 || len(store.forgotten) != 0 {
		t.Errorf("dry-run should write nothing; added=%d links=%d forgotten=%d",
			len(store.added), len(store.links), len(store.forgotten))
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
