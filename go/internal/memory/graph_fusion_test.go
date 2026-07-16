package memory

import (
	"context"
	"math"
	"path/filepath"
	"testing"

	"github.com/nexusriot/ai-houkai/internal/vector"
)

func graphApproxEq(a, b, tol float32) bool { return float32(math.Abs(float64(a-b))) <= tol }

func TestGraphSpreadEmptyAndNoEdges(t *testing.T) {
	if graphSpread(nil, nil) != nil {
		t.Error("empty pool should return nil")
	}
	mems := []Memory{{ID: "a"}, {ID: "b"}}
	if graphSpread(mems, []float32{0.9, 0.5}) != nil {
		t.Error("pool with no internal edges should return nil")
	}
}

func TestGraphSpreadStarHub(t *testing.T) {
	mems := []Memory{
		{ID: "center", Links: []Link{{To: "l1", Rel: "related"}, {To: "l2", Rel: "related"}, {To: "l3", Rel: "related"}}},
		{ID: "l1"}, {ID: "l2"}, {ID: "l3"}, {ID: "lonely"},
	}
	seeds := []float32{0.5, 0.5, 0.5, 0.5, 0.5}
	spread := graphSpread(mems, seeds)
	if spread == nil {
		t.Fatal("expected non-nil spread for a pool with edges")
	}
	if !graphApproxEq(spread["center"], 1.0, 1e-6) {
		t.Errorf("star hub should have max spread 1.0, got %.4f", spread["center"])
	}
	if spread["center"] <= spread["l1"] {
		t.Error("hub should outrank a leaf")
	}
	if spread["center"] <= spread["lonely"] {
		t.Error("hub should outrank the isolated node")
	}
}

func TestGraphSpreadReverseEdges(t *testing.T) {
	// center stores only OUTGOING links; it still ends up highest, proving
	// reverse (leaf -> center) edges are followed (undirected spread).
	mems := []Memory{
		{ID: "center", Links: []Link{{To: "l1", Rel: "related"}, {To: "l2", Rel: "related"}}},
		{ID: "l1"}, {ID: "l2"},
	}
	spread := graphSpread(mems, []float32{0.3, 0.3, 0.3})
	if !graphApproxEq(spread["center"], 1.0, 1e-6) {
		t.Errorf("center should reach max spread via reverse edges, got %.4f", spread["center"])
	}
	if spread["center"] <= spread["l1"] {
		t.Error("center should outrank its leaves")
	}
}

func seedStar(t *testing.T, store *MemoryStore) (Memory, Memory) {
	t.Helper()
	ctx := context.Background()
	center, _, _, _ := store.Remember(ctx, "kubernetes networking overview", RememberOpts{Type: Semantic})
	l1, _, _, _ := store.Remember(ctx, "kubernetes ingress setup", RememberOpts{Type: Semantic})
	l2, _, _, _ := store.Remember(ctx, "kubernetes service mesh", RememberOpts{Type: Semantic})
	l3, _, _, _ := store.Remember(ctx, "kubernetes dns resolution", RememberOpts{Type: Semantic})
	lonely, _, _, _ := store.Remember(ctx, "kubernetes storage volumes", RememberOpts{Type: Semantic})
	for _, leaf := range []Memory{l1, l2, l3} {
		if err := store.Link(ctx, center.ID, leaf.ID, "related"); err != nil {
			t.Fatal(err)
		}
	}
	return center, lonely
}

func TestGraphFusionWeightedZeroNoop(t *testing.T) {
	store := newTestStore(t)
	seedStar(t, store)
	ctx := context.Background()
	base, _ := store.Recall(ctx, "kubernetes", 5, RecallOpts{Mode: ModeHybrid, Weights: DefaultWeights(), NoTouch: true})
	w := DefaultWeights()
	w.Graph = 0
	withw, _ := store.Recall(ctx, "kubernetes", 5, RecallOpts{Mode: ModeHybrid, Weights: w, NoTouch: true})
	if len(base) != len(withw) {
		t.Fatalf("length mismatch: %d vs %d", len(base), len(withw))
	}
	for i := range base {
		if base[i].ID != withw[i].ID || !graphApproxEq(base[i].Score, withw[i].Score, 1e-6) {
			t.Errorf("graph=0 should be a no-op at %d: %v vs %v", i, base[i], withw[i])
		}
	}
}

func TestGraphFusionWeightedLiftsHub(t *testing.T) {
	store := newTestStore(t)
	center, lonely := seedStar(t, store)
	ctx := context.Background()
	scoreMap := func(g float32) map[string]float32 {
		w := DefaultWeights()
		w.Graph = g
		out, _ := store.Recall(ctx, "kubernetes", 5, RecallOpts{Mode: ModeHybrid, Weights: w, NoTouch: true})
		m := map[string]float32{}
		for _, r := range out {
			m[r.ID] = r.Score
		}
		return m
	}
	base := scoreMap(0)
	boosted := scoreMap(0.5)
	hubDelta := boosted[center.ID] - base[center.ID]
	lonelyDelta := boosted[lonely.ID] - base[lonely.ID]
	if hubDelta <= 0 {
		t.Errorf("hub score should increase with graph>0, delta=%.4f", hubDelta)
	}
	if hubDelta <= lonelyDelta {
		t.Errorf("hub should gain more than the isolated node: hub=%.4f lonely=%.4f", hubDelta, lonelyDelta)
	}
}

func TestGraphFusionExplain(t *testing.T) {
	store := newTestStore(t)
	center, _ := seedStar(t, store)
	ctx := context.Background()
	w := DefaultWeights()
	w.Graph = 0.5
	out, _ := store.Recall(ctx, "kubernetes", 5, RecallOpts{Mode: ModeHybrid, Weights: w, Explain: true, NoTouch: true})
	for _, r := range out {
		if r.ID == center.ID {
			if _, ok := r.Explain["graph"]; !ok {
				t.Error("weighted explain should record a graph term for the hub")
			}
			return
		}
	}
	t.Fatal("hub not present in results")
}

func TestGraphFusionRRFSignal(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	a, _, _, _ := store.Remember(ctx, "postgres replication tuning", RememberOpts{Type: Semantic})
	b, _, _, _ := store.Remember(ctx, "postgres vacuum settings", RememberOpts{Type: Semantic})
	if err := store.Link(ctx, a.ID, b.ID, "related"); err != nil {
		t.Fatal(err)
	}
	w := DefaultWeights()
	w.Graph = 0.3
	out, _ := store.Recall(ctx, "postgres", 5, RecallOpts{Mode: ModeHybrid, Fusion: FusionRRF, Weights: w, Explain: true, NoTouch: true})
	found := false
	for _, r := range out {
		if sig, ok := r.Explain["signals"].(map[string]any); ok {
			if _, ok := sig["graph"]; ok {
				found = true
			}
		}
	}
	if !found {
		t.Error("RRF explain should include a graph signal for a connected node")
	}
}

func seedHubChildren(t *testing.T, store *MemoryStore, n int) Memory {
	t.Helper()
	ctx := context.Background()
	hub, _, _, _ := store.Remember(ctx, "root fact", RememberOpts{Type: Semantic})
	for i := 0; i < n; i++ {
		c, _, _, _ := store.Remember(ctx, "child detail with unrelated words "+string(rune('a'+i)), RememberOpts{Type: Semantic})
		if err := store.Link(ctx, hub.ID, c.ID, "refines"); err != nil {
			t.Fatal(err)
		}
	}
	return hub
}

func TestExpansionLegacyCanExceedK(t *testing.T) {
	store := newTestStore(t)
	seedHubChildren(t, store, 5)
	ctx := context.Background()
	// min_cosine keeps only the exact query hit as a primary; children arrive
	// purely by expansion and (rerank=false) are appended after the top-k cut.
	spec := &ExpandSpec{Rels: []string{"refines"}, Cap: 5, Score: 0.6, Rerank: false}
	out, err := store.Recall(ctx, "root fact", 1, RecallOpts{Mode: ModeSemantic, MinCosine: f32p(0.99), Expand: spec})
	if err != nil {
		t.Fatal(err)
	}
	if len(out) <= 1 {
		t.Fatalf("legacy expansion should append beyond k=1, got %d", len(out))
	}
}

func TestExpansionRerankRespectsK(t *testing.T) {
	store := newTestStore(t)
	seedHubChildren(t, store, 5)
	ctx := context.Background()
	// rerank=true merges expanded nodes into the pool before the top-k cut, so
	// they compete for the k slots and can never overflow k.
	spec := &ExpandSpec{Rels: []string{"refines"}, Cap: 5, Score: 0.6, Rerank: true}
	out, err := store.Recall(ctx, "root fact", 1, RecallOpts{Mode: ModeSemantic, MinCosine: f32p(0.99), Expand: spec})
	if err != nil {
		t.Fatal(err)
	}
	if len(out) > 1 {
		t.Fatalf("rerank expansion must respect k=1, got %d", len(out))
	}
}

func TestExpansionRerankDedupsDuplicate(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	hub, _, _, _ := store.Remember(ctx, "root fact", RememberOpts{Type: Semantic})
	// Two identical children (same text -> same stub vector).
	c1, _, _, _ := store.Remember(ctx, "identical duplicate child body", RememberOpts{Type: Semantic})
	c2, _, _, _ := store.Remember(ctx, "identical duplicate child body", RememberOpts{Type: Semantic})
	for _, c := range []Memory{c1, c2} {
		if err := store.Link(ctx, hub.ID, c.ID, "refines"); err != nil {
			t.Fatal(err)
		}
	}
	dedup := f32p(0.9)
	spec := &ExpandSpec{Rels: []string{"refines"}, Cap: 5, Score: 0.95, Rerank: true}
	out, err := store.Recall(ctx, "root fact", 5, RecallOpts{
		Mode: ModeSemantic, MinCosine: f32p(0.99), DedupThreshold: dedup, Expand: spec,
	})
	if err != nil {
		t.Fatal(err)
	}
	dupCount := 0
	for _, r := range out {
		if r.ID == c1.ID || r.ID == c2.ID {
			dupCount++
		}
	}
	if dupCount != 1 {
		t.Fatalf("dedup in rerank mode should keep exactly one of the duplicate children, got %d", dupCount)
	}
}

func TestRRFRerankDoesNotBuryPrimary(t *testing.T) {
	// RRF scores are ~1/rrfK (tiny); a raw hop score of 0.9 would sort the
	// expanded node above the real hit. The fix rescales the hop score into the
	// pool's own range, so the strongest primary stays rank 0.
	store := newTestStore(t)
	ctx := context.Background()
	hub, _, _, _ := store.Remember(ctx, "distributed tracing spans", RememberOpts{Type: Semantic})
	child, _, _, _ := store.Remember(ctx, "totally unrelated weather almanac", RememberOpts{Type: Semantic})
	if err := store.Link(ctx, hub.ID, child.ID, "refines"); err != nil {
		t.Fatal(err)
	}
	spec := &ExpandSpec{Rels: []string{"refines"}, Cap: 5, Score: 0.9, Rerank: true}
	out, err := store.Recall(ctx, "distributed tracing spans", 5, RecallOpts{
		Mode: ModeHybrid, Fusion: FusionRRF, MinCosine: f32p(0.99), Expand: spec})
	if err != nil {
		t.Fatal(err)
	}
	if len(out) == 0 || out[0].ID != hub.ID {
		t.Fatalf("primary hub should stay rank 0 under rrf+rerank, got %v", out)
	}
}

func TestZeroCoreWeightsRejected(t *testing.T) {
	// HybridWeights{Graph: 0.15} zeroes the core weights in Go (no dataclass
	// defaults); Recall must reject it rather than rank by graph alone.
	store := newTestStore(t)
	ctx := context.Background()
	store.Remember(ctx, "some memory", RememberOpts{Type: Semantic})
	_, err := store.Recall(ctx, "some memory", 5, RecallOpts{
		Mode: ModeHybrid, Weights: HybridWeights{Graph: 0.15}})
	if err == nil {
		t.Fatal("expected a validation error for all-zero core weights")
	}
}

// stripEmbBackend wraps a real backend and, once `strip` is flipped, returns
// items from Get() with their embedding cleared — simulating a stored-vector
// fetch miss (Python's test does the same by monkeypatching _emb_for_ids).
type stripEmbBackend struct {
	vector.Backend
	strip *bool
}

func (b *stripEmbBackend) Get(ctx context.Context, ids []string) ([]vector.Item, error) {
	items, err := b.Backend.Get(ctx, ids)
	if err == nil && *b.strip {
		for i := range items {
			items[i].Embedding = nil
		}
	}
	return items, err
}

func TestExpansionRerankDropsUnfetchableEmbedding(t *testing.T) {
	// F2 (Go): when dedup is on (needEmb) but an expanded node's STORED vector
	// can't be fetched, it must be dropped — not admitted with a free novelty
	// pass. overfetch=1 keeps `child` out of the primary query pool, so its only
	// embedding source is the backend.Get fetch (which we make miss).
	dir := t.TempDir()
	inner, err := vector.NewChromem(filepath.Join(dir, "s"), "test", 16)
	if err != nil {
		t.Fatalf("NewChromem: %v", err)
	}
	t.Cleanup(func() { _ = inner.Close() })
	strip := false
	be := &stripEmbBackend{Backend: inner, strip: &strip}
	store := NewMemoryStore(be, &stubEmbedder{dim: 16}, DefaultStoreConfig(dir, "test"))
	ctx := context.Background()
	hub, _, _, _ := store.Remember(ctx, "cache invalidation policy", RememberOpts{Type: Procedural})
	child, _, _, _ := store.Remember(ctx, "totally unrelated weather almanac", RememberOpts{Type: Episodic})
	if err := store.Link(ctx, hub.ID, child.ID, "refines"); err != nil {
		t.Fatal(err)
	}
	strip = true // from now on backend.Get returns items without embeddings
	out, err := store.Recall(ctx, "cache invalidation policy", 1, RecallOpts{
		Mode: ModeHybrid, Overfetch: 1, MinCosine: f32p(0.99), DedupThreshold: f32p(0.9),
		Expand: &ExpandSpec{Rels: []string{"refines"}, Cap: 5, Score: 0.95, Rerank: true},
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range out {
		if r.ID == child.ID {
			t.Fatal("node with an unfetchable stored embedding must be dropped, not merged")
		}
	}
}

func TestCollectExpansionShieldsSeenIDs(t *testing.T) {
	// F3 (Go): a node listed in seenIDs must be neither re-emitted nor have its
	// existing explain entry clobbered by the graph_expansion stub.
	store := newTestStore(t)
	ctx := context.Background()
	a, _, _, _ := store.Remember(ctx, "seed node", RememberOpts{Type: Semantic})
	b, _, _, _ := store.Remember(ctx, "linked neighbour", RememberOpts{Type: Semantic})
	if err := store.Link(ctx, a.ID, b.ID, "refines"); err != nil {
		t.Fatal(err)
	}
	expl := map[string]map[string]any{b.ID: {"mode": "hybrid", "score": 0.5}}
	seed := []MemoryWithScore{{Memory: a, Score: 1.0}}
	spec := &ExpandSpec{Rels: []string{"refines"}, Cap: 5, Score: 0.7}

	extra, err := store.collectExpansion(ctx, seed, spec, false, expl, []string{b.ID})
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range extra {
		if e.ID == b.ID {
			t.Fatal("a seen id must not be re-emitted by collectExpansion")
		}
	}
	if expl[b.ID]["mode"] != "hybrid" {
		t.Fatalf("shielded node's explain was clobbered: %v", expl[b.ID])
	}
}
