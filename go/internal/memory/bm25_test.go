package memory

import "testing"

func TestBM25EmptyCorpus(t *testing.T) {
	if got := bm25Score("anything", nil); got != nil {
		t.Fatalf("expected nil for empty corpus, got %v", got)
	}
}

func TestBM25RanksMatchingDocFirst(t *testing.T) {
	docs := []string{
		"the cat sat on the mat",
		"unrelated quantum physics theory",
		"cats and dogs are common pets",
	}
	scores := bm25Score("cat", docs)
	if len(scores) != 3 {
		t.Fatalf("want 3 scores, got %d", len(scores))
	}
	// Doc 0 contains "cat" — should outscore doc 1 (no match).
	if scores[0] <= scores[1] {
		t.Errorf("doc 0 (%.3f) should outscore doc 1 (%.3f)", scores[0], scores[1])
	}
}

func TestBM25NormalisedToUnitMax(t *testing.T) {
	docs := []string{"alpha beta", "alpha alpha gamma", "delta epsilon"}
	scores := bm25Score("alpha", docs)
	var max float32
	for _, s := range scores {
		if s > max {
			max = s
		}
	}
	if max != 1.0 {
		t.Errorf("expected max score 1.0 after normalisation, got %.3f", max)
	}
}

func TestBM25NoQueryMatchAllZero(t *testing.T) {
	docs := []string{"alpha beta", "gamma delta"}
	scores := bm25Score("xyz", docs)
	for i, s := range scores {
		if s != 0 {
			t.Errorf("score[%d] = %.3f, want 0", i, s)
		}
	}
}
