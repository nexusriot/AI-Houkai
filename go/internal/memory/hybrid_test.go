package memory

import (
	"testing"
	"time"
)

func TestRecencyScoreDecays(t *testing.T) {
	w := DefaultWeights()
	now := float64(time.Now().Unix())
	fresh := Memory{CreatedAt: now}
	stale := Memory{CreatedAt: now - 30*24*3600}
	if recencyScore(fresh, w, 0.1, now) <= recencyScore(stale, w, 0.1, now) {
		t.Errorf("fresh memory should outscore stale one on recency")
	}
}

func TestRecencyBasisSelectsTimestamp(t *testing.T) {
	now := float64(time.Now().Unix())
	// Created long ago but accessed just now.
	m := Memory{CreatedAt: now - 30*24*3600, LastAccessed: now}

	created := recencyScore(m, HybridWeights{RecencyBasis: "created"}, 0.1, now)
	accessed := recencyScore(m, HybridWeights{RecencyBasis: "accessed"}, 0.1, now)
	if accessed <= created {
		t.Errorf("accessed-basis recency (%.4f) should exceed created-basis (%.4f)", accessed, created)
	}
	// Empty basis behaves as "created".
	if recencyScore(m, HybridWeights{}, 0.1, now) != created {
		t.Errorf("empty RecencyBasis should behave as created")
	}
}

func TestDefaultWeightsSumToOne(t *testing.T) {
	w := DefaultWeights()
	total := w.Cosine + w.Lexical + w.Recency + w.Importance
	if total < 0.99 || total > 1.01 {
		t.Errorf("default core weights should sum to ~1.0, got %.3f", total)
	}
	if w.PolarityWeight != 0.05 {
		t.Errorf("default polarity weight should be 0.05, got %.3f", w.PolarityWeight)
	}
	if w.RecencyBasis != "created" {
		t.Errorf("default recency basis should be created, got %q", w.RecencyBasis)
	}
}
