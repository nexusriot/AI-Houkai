package memory

import (
	"context"
	"strings"
	"testing"
)

func f32p(v float32) *float32 { return &v }

// remember3Polarities stores the same text three times with polarity +1/0/-1.
func remember3Polarities(t *testing.T, store *MemoryStore) {
	t.Helper()
	ctx := context.Background()
	for _, p := range []int{1, 0, -1} {
		if _, ok, _, err := store.Remember(ctx, "the sky is blue today", RememberOpts{
			Type: Semantic, Importance: Float32Ptr(0.5), Polarity: p,
		}); err != nil || !ok {
			t.Fatalf("Remember polarity=%d: ok=%v err=%v", p, ok, err)
		}
	}
}

func TestRecallPolarityScoring(t *testing.T) {
	store := newTestStore(t)
	remember3Polarities(t, store)

	out, err := store.Recall(context.Background(), "the sky is blue today", 3, RecallOpts{Mode: ModeHybrid})
	if err != nil {
		t.Fatalf("Recall: %v", err)
	}
	if len(out) != 3 {
		t.Fatalf("expected 3 results, got %d", len(out))
	}
	if out[0].Polarity != 1 {
		t.Errorf("positive polarity should rank first, got polarity=%d", out[0].Polarity)
	}
	if out[2].Polarity != -1 {
		t.Errorf("negative polarity should rank last, got polarity=%d", out[2].Polarity)
	}
}

func TestRecallPolarityWeightZeroDisables(t *testing.T) {
	store := newTestStore(t)
	remember3Polarities(t, store)

	// Weighted, but with polarity weight explicitly zeroed → identical scores.
	w := DefaultWeights()
	w.PolarityWeight = 0
	out, err := store.Recall(context.Background(), "the sky is blue today", 3, RecallOpts{Mode: ModeHybrid, Weights: w})
	if err != nil {
		t.Fatalf("Recall: %v", err)
	}
	for _, r := range out {
		// All three share text/importance/recency, so with polarity off the
		// scores must be equal.
		if r.Score != out[0].Score {
			t.Errorf("polarity weight 0 should give equal scores, got %v vs %v", r.Score, out[0].Score)
		}
	}
}

func TestRecallMinCosineGate(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	_, _, _, _ = store.Remember(ctx, "apples oranges bananas", RememberOpts{Type: Semantic})
	_, _, _, _ = store.Remember(ctx, "quantum chromodynamics lattice", RememberOpts{Type: Semantic})

	// An exact-match query has cosine ~1.0 to the first; the unrelated second
	// sits well below. A high floor keeps only the near-identical hit.
	floor := f32p(0.95)
	out, err := store.Recall(ctx, "apples oranges bananas", 5, RecallOpts{Mode: ModeSemantic, MinCosine: floor})
	if err != nil {
		t.Fatalf("Recall: %v", err)
	}
	if len(out) != 1 {
		t.Fatalf("min_cosine floor should keep only the exact match, got %d", len(out))
	}
	if out[0].Text != "apples oranges bananas" {
		t.Errorf("unexpected surviving hit: %q", out[0].Text)
	}
}

func TestRecallNoTouch(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	m, _, _, _ := store.Remember(ctx, "read only please", RememberOpts{Type: Semantic})

	if _, err := store.Recall(ctx, "read only please", 5, RecallOpts{Mode: ModeSemantic, NoTouch: true}); err != nil {
		t.Fatalf("Recall: %v", err)
	}
	got, _ := store.GetByID(ctx, m.ID)
	if got.AccessCount != 0 {
		t.Errorf("NoTouch recall must not bump access_count, got %d", got.AccessCount)
	}

	if _, err := store.Recall(ctx, "read only please", 5, RecallOpts{Mode: ModeSemantic}); err != nil {
		t.Fatalf("Recall: %v", err)
	}
	got, _ = store.GetByID(ctx, m.ID)
	if got.AccessCount != 1 {
		t.Errorf("touching recall should bump access_count to 1, got %d", got.AccessCount)
	}
}

func TestRecallExplain(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	store.Remember(ctx, "explainable retrieval matters", RememberOpts{Type: Semantic})

	out, err := store.Recall(ctx, "explainable retrieval matters", 1, RecallOpts{Mode: ModeHybrid, Explain: true})
	if err != nil || len(out) != 1 {
		t.Fatalf("Recall: len=%d err=%v", len(out), err)
	}
	if out[0].Explain == nil {
		t.Fatal("Explain map should be populated")
	}
	if out[0].Explain["mode"] != "hybrid" || out[0].Explain["fusion"] != "weighted" {
		t.Errorf("unexpected explain payload: %v", out[0].Explain)
	}
}

func TestRecallRRFExplain(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	store.Remember(ctx, "reciprocal rank fusion test", RememberOpts{Type: Semantic})

	out, err := store.Recall(ctx, "reciprocal rank fusion test", 1, RecallOpts{Mode: ModeHybrid, Fusion: FusionRRF, Explain: true})
	if err != nil || len(out) != 1 {
		t.Fatalf("Recall rrf: len=%d err=%v", len(out), err)
	}
	if out[0].Explain["fusion"] != "rrf" {
		t.Errorf("expected rrf fusion in explain, got %v", out[0].Explain["fusion"])
	}
}

func TestRecallDedupThreshold(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	// Two identical texts → identical embeddings → cosine 1.0 to each other.
	store.Remember(ctx, "duplicate content here", RememberOpts{Type: Semantic})
	store.Remember(ctx, "duplicate content here", RememberOpts{Type: Semantic})

	dedup := f32p(0.99)
	out, err := store.Recall(ctx, "duplicate content here", 5, RecallOpts{Mode: ModeSemantic, DedupThreshold: dedup})
	if err != nil {
		t.Fatalf("Recall: %v", err)
	}
	if len(out) != 1 {
		t.Errorf("dedup should collapse identical memories to 1, got %d", len(out))
	}
}

func TestRecallDiversityBounded(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	for _, txt := range []string{"alpha one", "beta two", "gamma three", "delta four"} {
		store.Remember(ctx, txt, RememberOpts{Type: Semantic})
	}
	div := f32p(0.5)
	out, err := store.Recall(ctx, "alpha one", 2, RecallOpts{Mode: ModeHybrid, Diversity: div})
	if err != nil {
		t.Fatalf("Recall: %v", err)
	}
	if len(out) != 2 {
		t.Errorf("MMR should return exactly k=2, got %d", len(out))
	}
}

// TestSemanticRecallNoUnderfetch guards the #14 fix: a filtered semantic recall
// must still return k results when enough matching memories exist, even though
// the vector query also surfaces non-matching ones.
func TestSemanticRecallNoUnderfetch(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	// 5 tagged + 5 untagged memories, all lexically related to the query.
	for i := 0; i < 5; i++ {
		store.Remember(ctx, "project note keep", RememberOpts{Type: Semantic, Tags: []string{"keep"}})
		store.Remember(ctx, "project note drop", RememberOpts{Type: Semantic, Tags: []string{"drop"}})
	}
	out, err := store.Recall(ctx, "project note", 5, RecallOpts{Mode: ModeSemantic, Tag: "keep"})
	if err != nil {
		t.Fatalf("Recall: %v", err)
	}
	if len(out) != 5 {
		t.Errorf("tagged semantic recall under-fetched: got %d, want 5", len(out))
	}
	for _, r := range out {
		if !containsTag(r.Memory, "keep") {
			t.Errorf("result %q missing required tag", r.Text)
		}
	}
}

func TestExpandMultiHopDecay(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	a, _, _, _ := store.Remember(ctx, "root fact", RememberOpts{Type: Semantic})
	b, _, _, _ := store.Remember(ctx, "hop one fact", RememberOpts{Type: Semantic})
	c, _, _, _ := store.Remember(ctx, "hop two fact", RememberOpts{Type: Semantic})
	// a -refines-> b -refines-> c
	if err := store.Link(ctx, a.ID, b.ID, "refines"); err != nil {
		t.Fatal(err)
	}
	if err := store.Link(ctx, b.ID, c.ID, "refines"); err != nil {
		t.Fatal(err)
	}

	spec := &ExpandSpec{Rels: []string{"refines"}, Depth: 2, Cap: 5, Score: 0.7, Decay: 0.5}
	out, err := store.Recall(ctx, "root fact", 1, RecallOpts{Mode: ModeSemantic, MinCosine: f32p(0.99), Expand: spec})
	if err != nil {
		t.Fatalf("Recall: %v", err)
	}
	// Expect: root (hit) + b (hop 1, score 0.7) + c (hop 2, score 0.35).
	byID := map[string]MemoryWithScore{}
	for _, r := range out {
		byID[r.ID] = r
	}
	if _, ok := byID[b.ID]; !ok {
		t.Fatal("hop-1 neighbour not expanded")
	}
	if _, ok := byID[c.ID]; !ok {
		t.Fatal("hop-2 neighbour not expanded")
	}
	if s := byID[b.ID].Score; s < 0.69 || s > 0.71 {
		t.Errorf("hop-1 score should be ~0.7, got %.3f", s)
	}
	if s := byID[c.ID].Score; s < 0.34 || s > 0.36 {
		t.Errorf("hop-2 score should be ~0.35 (0.7*0.5), got %.3f", s)
	}
}

func TestRecallKZeroReturnsEmpty(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	store.Remember(ctx, "something", RememberOpts{Type: Semantic})
	// k<=0 returns nothing (matches Python), not a defaulted 5.
	for _, k := range []int{0, -1} {
		out, err := store.Recall(ctx, "something", k, RecallOpts{Mode: ModeSemantic})
		if err != nil || len(out) != 0 {
			t.Errorf("Recall(k=%d) = %d results, err=%v; want 0/nil", k, len(out), err)
		}
	}
}

func TestEstimateTokensBankersRounding(t *testing.T) {
	// len 10 → 10/4 = 2.5 → round-half-to-even → 2 (Python's round()).
	if got := EstimateTokens("- (fact) x"); got != 2 {
		t.Errorf("EstimateTokens(len 10) = %d, want 2 (banker's rounding)", got)
	}
	// len 6 → 1.5 → round-half-to-even → 2.
	if got := EstimateTokens("abcdef"); got != 2 {
		t.Errorf("EstimateTokens(len 6) = %d, want 2", got)
	}
	// len 40 → 10.0 → 10.
	if got := EstimateTokens(strings.Repeat("a", 40)); got != 10 {
		t.Errorf("EstimateTokens(len 40) = %d, want 10", got)
	}
}

func TestExtractKeyPhrasesZeroHonored(t *testing.T) {
	if got := ExtractKeyPhrases("the deployment pipeline failed", 0); len(got) != 0 {
		t.Errorf("max_phrases=0 should yield no phrases, got %v", got)
	}
}
