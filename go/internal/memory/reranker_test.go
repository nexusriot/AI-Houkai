package memory

import (
	"context"
	"fmt"
	"path/filepath"
	"testing"

	"github.com/nexusriot/ai-houkai/internal/vector"
)

func newChromemForTest(t *testing.T, dir string) vector.Backend {
	t.Helper()
	be, err := vector.NewChromem(filepath.Join(dir, "s"), "test", 16)
	if err != nil {
		t.Fatalf("NewChromem: %v", err)
	}
	t.Cleanup(func() { _ = be.Close() })
	return be
}

func seedTopics(t *testing.T, store *MemoryStore) []Memory {
	t.Helper()
	ctx := context.Background()
	mems := make([]Memory, 6)
	for i := range mems {
		m, _, _, err := store.Remember(ctx, fmt.Sprintf("topic entry number %d", i), RememberOpts{})
		if err != nil {
			t.Fatalf("Remember: %v", err)
		}
		mems[i] = m
	}
	return mems
}

func TestRerankerReordersResults(t *testing.T) {
	store := newTestStore(t)
	mems := seedTopics(t, store)
	wanted := mems[len(mems)-1].ID

	rr := func(query string, cands []Memory) []float32 {
		out := make([]float32, len(cands))
		for i, m := range cands {
			if m.ID == wanted {
				out[i] = 1.0
			}
		}
		return out
	}
	hits, err := store.Recall(context.Background(), "topic entry", 3, RecallOpts{Reranker: rr})
	if err != nil {
		t.Fatalf("Recall: %v", err)
	}
	if len(hits) == 0 || hits[0].ID != wanted {
		t.Fatalf("reranked top = %v, want %s", idsOf(hits), wanted)
	}
	if hits[0].Score != 1.0 {
		t.Errorf("reranker score should replace the blend: got %v", hits[0].Score)
	}
}

func TestRerankerPromotesCandidateOutsideFirstStageTopK(t *testing.T) {
	store := newTestStore(t)
	seedTopics(t, store)
	// Large overfetch so the pool holds the whole first-stage ranking.
	first, err := store.Recall(context.Background(), "topic entry", 5, RecallOpts{Overfetch: 20})
	if err != nil {
		t.Fatalf("Recall: %v", err)
	}
	if len(first) < 3 {
		t.Fatalf("need >=3 candidates, got %d", len(first))
	}
	target := first[len(first)-1].ID // first-stage LAST → force to top

	rr := func(query string, cands []Memory) []float32 {
		out := make([]float32, len(cands))
		for i, m := range cands {
			if m.ID == target {
				out[i] = 1.0
			}
		}
		return out
	}
	hits, _ := store.Recall(context.Background(), "topic entry", 1, RecallOpts{Overfetch: 20, Reranker: rr})
	if len(hits) != 1 || hits[0].ID != target {
		t.Errorf("want promoted %s, got %v", target, idsOf(hits))
	}
}

func TestRerankerExplainRecordsRerankBlock(t *testing.T) {
	store := newTestStore(t)
	seedTopics(t, store)
	// Ascending scores → last candidate wins.
	rr := func(query string, cands []Memory) []float32 {
		out := make([]float32, len(cands))
		for i := range cands {
			out[i] = float32(i)
		}
		return out
	}
	hits, err := store.Recall(context.Background(), "topic entry", 3, RecallOpts{Reranker: rr, Explain: true})
	if err != nil {
		t.Fatalf("Recall: %v", err)
	}
	blk, ok := hits[0].Explain["rerank"].(map[string]any)
	if !ok {
		t.Fatalf("explain missing rerank block: %v", hits[0].Explain)
	}
	if blk["rank"] != 0 {
		t.Errorf("rerank.rank = %v, want 0", blk["rank"])
	}
	for _, key := range []string{"score", "first_stage_rank", "first_stage_score"} {
		if _, ok := blk[key]; !ok {
			t.Errorf("rerank block missing %q", key)
		}
	}
}

func TestPerStoreDefaultReranker(t *testing.T) {
	dir := t.TempDir()
	be := newChromemForTest(t, dir)
	var wantedID string
	cfg := DefaultStoreConfig(dir, "test")
	cfg.Reranker = func(query string, cands []Memory) []float32 {
		out := make([]float32, len(cands))
		for i, m := range cands {
			if m.ID == wantedID {
				out[i] = 1.0
			}
		}
		return out
	}
	store := NewMemoryStore(be, &stubEmbedder{dim: 16}, cfg)
	mems := seedTopics(t, store)
	wantedID = mems[2].ID
	hits, _ := store.Recall(context.Background(), "topic entry", 1, RecallOpts{}) // no per-call reranker
	if len(hits) != 1 || hits[0].ID != wantedID {
		t.Errorf("store-default reranker not applied: got %v, want %s", idsOf(hits), wantedID)
	}
}

func TestRerankerWrongLengthErrors(t *testing.T) {
	store := newTestStore(t)
	seedTopics(t, store)
	rr := func(query string, cands []Memory) []float32 { return []float32{1.0} } // too few
	if _, err := store.Recall(context.Background(), "topic entry", 3, RecallOpts{Reranker: rr}); err == nil {
		t.Error("want error when reranker returns the wrong number of scores")
	}
}
