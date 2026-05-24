package memory

import (
	"testing"
	"time"
)

func TestHybridScoreWeightedSum(t *testing.T) {
	w := HybridWeights{Cosine: 1, Lexical: 0, Recency: 0, Importance: 0}
	// With only cosine weight, recency/importance/lexical shouldn't matter.
	got := hybridScore(0.42, 0.99, 0.99, 0, w, 0.1)
	if got < 0.41 || got > 0.43 {
		t.Errorf("expected ≈0.42, got %.3f", got)
	}
}

func TestHybridScoreRecencyDecays(t *testing.T) {
	w := HybridWeights{Cosine: 0, Lexical: 0, Recency: 1, Importance: 0}
	now := time.Now().Unix()
	old := time.Now().Add(-30 * 24 * time.Hour).Unix()

	fresh := hybridScore(0, 0, 0, float64(now), w, 0.1)
	stale := hybridScore(0, 0, 0, float64(old), w, 0.1)
	if fresh <= stale {
		t.Errorf("fresh (%.3f) should outscore stale (%.3f)", fresh, stale)
	}
}

func TestDefaultWeightsSumToOne(t *testing.T) {
	w := DefaultWeights()
	total := w.Cosine + w.Lexical + w.Recency + w.Importance
	if total < 0.99 || total > 1.01 {
		t.Errorf("default weights should sum to ~1.0, got %.3f", total)
	}
}
