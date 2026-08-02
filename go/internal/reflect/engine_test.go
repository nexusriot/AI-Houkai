package reflect

import (
	"context"
	"strconv"
	"strings"
	"testing"

	"github.com/nexusriot/ai-houkai/internal/memory"
	"github.com/nexusriot/ai-houkai/internal/vector"
)

// setPolarity rewrites the polarity in an episode item's stored metadata.
func setPolarity(it *vector.Item, p int) {
	it.Metadata["polarity"] = strconv.Itoa(p)
}

// fakeReflectStore implements Storable for unit tests.
type fakeReflectStore struct {
	items      []vector.Item
	added      []memory.Memory
	links      []struct{ src, dst, rel string }
	forgotten  []string
	superseded []struct{ old, new string }
}

func (f *fakeReflectStore) AllRaw(_ context.Context) ([]vector.Item, error) {
	return f.items, nil
}
func (f *fakeReflectStore) Remember(_ context.Context, text string, opts memory.RememberOpts) (memory.Memory, bool, []memory.Conflict, error) {
	m := memory.Memory{
		ID:   "new-" + text[:min(8, len(text))],
		Text: text,
		Type: opts.Type,
		Tags: opts.Tags,
	}
	if opts.Importance != nil {
		m.Importance = *opts.Importance
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
func (f *fakeReflectStore) Supersede(_ context.Context, oldID, newID string) error {
	f.superseded = append(f.superseded, struct{ old, new string }{oldID, newID})
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
		episode("c", []float32{0, 1, 0}, 0.5),       // far
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

func TestClustersSkipsOppositePolarity(t *testing.T) {
	// Two near-identical episodics with opposite polarity must NOT cluster;
	// a neutral third one may join the seed.
	pos := episode("a", []float32{1, 0, 0}, 0.9)
	neg := episode("b", []float32{0.99, 0.01, 0}, 0.8)
	neu := episode("c", []float32{0.98, 0.02, 0}, 0.7)
	setPolarity(&pos, 1)
	setPolarity(&neg, -1)
	setPolarity(&neu, 0)
	store := &fakeReflectStore{items: []vector.Item{pos, neg, neu}}
	e := New(store, 0.9, 2, nil)

	clusters, err := e.Clusters(context.Background())
	if err != nil {
		t.Fatalf("Clusters: %v", err)
	}
	if len(clusters) != 1 {
		t.Fatalf("want 1 cluster (seed+neutral), got %d", len(clusters))
	}
	if len(clusters[0]) != 2 {
		t.Errorf("opposite-polarity member must be excluded; cluster size=%d want 2", len(clusters[0]))
	}
	for _, m := range clusters[0] {
		if m.Polarity == -1 {
			t.Error("negative-polarity memory should not join a positive seed's cluster")
		}
	}
}

func TestReflectNoneLeavesSourcesUntouched(t *testing.T) {
	store := &fakeReflectStore{items: []vector.Item{
		episode("a", []float32{1, 0, 0}, 0.9),
		episode("b", []float32{0.99, 0.01, 0}, 0.5),
	}}
	e := New(store, 0.9, 2, nil)

	created, err := e.Reflect(context.Background(), false, ConsolidateNone)
	if err != nil {
		t.Fatalf("Reflect: %v", err)
	}
	if len(created) != 1 {
		t.Fatalf("want 1 semantic memory created, got %d", len(created))
	}
	if created[0].Type != memory.Semantic {
		t.Errorf("new memory type: got %q, want semantic", created[0].Type)
	}
	// none: sources are left entirely alone — no link, no supersede, no delete.
	if len(store.links) != 0 {
		t.Errorf("none mode should not create links, got %d", len(store.links))
	}
	if len(store.superseded) != 0 {
		t.Errorf("none mode should not supersede, got %d", len(store.superseded))
	}
	if len(store.forgotten) != 0 {
		t.Error("none mode should not delete sources")
	}
}

func TestReflectSoftSupersedesAndLinks(t *testing.T) {
	store := &fakeReflectStore{items: []vector.Item{
		episode("a", []float32{1, 0, 0}, 0.9),
		episode("b", []float32{0.99, 0.01, 0}, 0.5),
	}}
	e := New(store, 0.9, 2, nil)
	if _, err := e.Reflect(context.Background(), false, ConsolidateSoft); err != nil {
		t.Fatalf("Reflect: %v", err)
	}
	if len(store.superseded) != 2 {
		t.Errorf("soft should supersede 2 sources, got %d", len(store.superseded))
	}
	if len(store.links) != 2 {
		t.Errorf("soft should add 2 derived_from links, got %d", len(store.links))
	}
	for _, l := range store.links {
		if l.rel != memory.RelDerivedFrom {
			t.Errorf("link rel: got %q, want %q", l.rel, memory.RelDerivedFrom)
		}
	}
	if len(store.forgotten) != 0 {
		t.Error("soft must not forget sources")
	}
}

func TestReflectHardForgetsSources(t *testing.T) {
	store := &fakeReflectStore{items: []vector.Item{
		episode("a", []float32{1, 0, 0}, 0.9),
		episode("b", []float32{0.99, 0.01, 0}, 0.5),
	}}
	e := New(store, 0.9, 2, nil)
	if _, err := e.Reflect(context.Background(), false, ConsolidateHard); err != nil {
		t.Fatalf("Reflect: %v", err)
	}
	if len(store.forgotten) != 2 {
		t.Errorf("hard should forget 2 sources, got %d", len(store.forgotten))
	}
	if len(store.links) != 0 || len(store.superseded) != 0 {
		t.Errorf("hard should not link/supersede; links=%d superseded=%d", len(store.links), len(store.superseded))
	}
}

func TestReflectDryRunWritesNothing(t *testing.T) {
	store := &fakeReflectStore{items: []vector.Item{
		episode("a", []float32{1, 0, 0}, 0.9),
		episode("b", []float32{0.99, 0.01, 0}, 0.5),
	}}
	e := New(store, 0.9, 2, nil)
	created, err := e.Reflect(context.Background(), true, ConsolidateSoft)
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

// Tiered reflection (F1): the engine only ever clustered episodic memories, so
// summaries were never themselves consolidated and a long-lived store
// accumulated them without bound.

// typedItem builds a vector.Item of an arbitrary type and tag set, so the
// tier tests can seed existing summaries directly.
func typedItem(id string, t memory.MemoryType, vec []float32, tags ...string) vector.Item {
	m := memory.Memory{ID: id, Type: t, Importance: 0.5, Tags: tags}
	return vector.Item{
		ID: id, Content: "text-" + id, Embedding: vec,
		Metadata: memory.MemoryToMetadata(m),
	}
}

func TestReflectDefaultsToEpisodicOnly(t *testing.T) {
	store := &fakeReflectStore{items: []vector.Item{
		typedItem("e1", memory.Episodic, []float32{1, 0, 0}),
		typedItem("e2", memory.Episodic, []float32{1, 0, 0}),
		typedItem("s1", memory.Semantic, []float32{1, 0, 0}),
		typedItem("s2", memory.Semantic, []float32{1, 0, 0}),
	}}
	clusters, err := New(store, 0.9, 2, nil).Clusters(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(clusters) != 1 {
		t.Fatalf("clusters = %d, want 1", len(clusters))
	}
	for _, m := range clusters[0] {
		if m.Type != memory.Episodic {
			t.Errorf("clustered a %s memory by default", m.Type)
		}
	}
}

func TestReflectCanClusterOtherTypes(t *testing.T) {
	store := &fakeReflectStore{items: []vector.Item{
		typedItem("f1", memory.Feedback, []float32{1, 0, 0}),
		typedItem("f2", memory.Feedback, []float32{1, 0, 0}),
	}}
	e := New(store, 0.9, 2, nil)
	e.Types = []memory.MemoryType{memory.Feedback}
	clusters, err := e.Clusters(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(clusters) != 1 {
		t.Fatalf("feedback memories were not clustered: %d", len(clusters))
	}
}

func TestReflectTagsTheTier(t *testing.T) {
	store := &fakeReflectStore{items: []vector.Item{
		typedItem("e1", memory.Episodic, []float32{1, 0, 0}),
		typedItem("e2", memory.Episodic, []float32{1, 0, 0}),
	}}
	made, err := New(store, 0.9, 2, nil).Reflect(context.Background(), false, ConsolidateNone)
	if err != nil || len(made) == 0 {
		t.Fatalf("reflect = %d (%v)", len(made), err)
	}
	if LevelOf(made[0]) != 1 {
		t.Errorf("tags = %v, want level:1", made[0].Tags)
	}
}

func TestReflectMaxLevelCapsTheHierarchy(t *testing.T) {
	// The guard against runaway re-summarisation eating the store.
	seed := func() *fakeReflectStore {
		return &fakeReflectStore{items: []vector.Item{
			typedItem("s1", memory.Semantic, []float32{1, 0, 0}, "reflection", "level:1"),
			typedItem("s2", memory.Semantic, []float32{1, 0, 0}, "reflection", "level:1"),
		}}
	}
	capped := New(seed(), 0.9, 2, nil)
	capped.Types = []memory.MemoryType{memory.Semantic}
	capped.MaxLevel = 1
	if made, _ := capped.Reflect(context.Background(), false, ConsolidateNone); len(made) != 0 {
		t.Errorf("max_level=1 still produced %d summaries", len(made))
	}

	deeper := New(seed(), 0.9, 2, nil)
	deeper.Types = []memory.MemoryType{memory.Semantic}
	deeper.MaxLevel = 2
	made, err := deeper.Reflect(context.Background(), false, ConsolidateNone)
	if err != nil || len(made) == 0 {
		t.Fatalf("max_level=2 reflect = %d (%v)", len(made), err)
	}
	if LevelOf(made[0]) != 2 {
		t.Errorf("tags = %v, want level:2", made[0].Tags)
	}
	// The summary carries its own tier, not its members'.
	levels := 0
	for _, tag := range made[0].Tags {
		if strings.HasPrefix(tag, "level:") {
			levels++
		}
	}
	if levels != 1 {
		t.Errorf("tags = %v, want exactly one level tag", made[0].Tags)
	}
}
