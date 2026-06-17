package decay

import (
	"context"
	"testing"
	"time"

	"github.com/nexusriot/ai-houkai/internal/memory"
)

// fakeStore implements Storable for unit tests.
type fakeStore struct {
	mems   []memory.Memory
	forgot []string
}

func (f *fakeStore) ListRecent(_ context.Context, _ int, _ bool) ([]memory.Memory, error) {
	return f.mems, nil
}

func (f *fakeStore) Forget(_ context.Context, id string) (bool, error) {
	f.forgot = append(f.forgot, id)
	for i, m := range f.mems {
		if m.ID == id {
			f.mems = append(f.mems[:i], f.mems[i+1:]...)
			return true, nil
		}
	}
	return false, nil
}

func ts(daysAgo float64) float64 {
	return float64(time.Now().Add(-time.Duration(daysAgo * 24 * float64(time.Hour))).Unix())
}

func TestScoreDecaysWithAge(t *testing.T) {
	e := New(nil, 0.1, 0.05, nil, 0)
	fresh := memory.Memory{Importance: 1.0, LastAccessed: ts(0)}
	old := memory.Memory{Importance: 1.0, LastAccessed: ts(30)}
	if e.Score(fresh) <= e.Score(old) {
		t.Errorf("fresh (%.3f) should outscore old (%.3f)", e.Score(fresh), e.Score(old))
	}
}

func TestPruneRemovesBelowThreshold(t *testing.T) {
	fs := &fakeStore{mems: []memory.Memory{
		{ID: "keep", Type: memory.Semantic, Importance: 1.0, LastAccessed: ts(0)},
		{ID: "drop", Type: memory.Semantic, Importance: 0.1, LastAccessed: ts(60)},
	}}
	e := New(fs, 0.1, 0.05, nil, 0)

	pruned, err := e.Prune(context.Background(), false)
	if err != nil {
		t.Fatalf("Prune: %v", err)
	}
	if len(pruned) != 1 || pruned[0].ID != "drop" {
		t.Errorf("want pruned=[drop], got %v", pruned)
	}
	if len(fs.forgot) != 1 || fs.forgot[0] != "drop" {
		t.Errorf("Forget should have been called on 'drop', got %v", fs.forgot)
	}
}

func TestPruneDryRunDoesNotDelete(t *testing.T) {
	fs := &fakeStore{mems: []memory.Memory{
		{ID: "drop", Type: memory.Semantic, Importance: 0.01, LastAccessed: ts(60)},
	}}
	e := New(fs, 0.1, 0.05, nil, 0)
	pruned, err := e.Prune(context.Background(), true)
	if err != nil {
		t.Fatalf("Prune dry-run: %v", err)
	}
	if len(pruned) != 1 {
		t.Errorf("dry-run should still report 1 candidate, got %d", len(pruned))
	}
	if len(fs.forgot) != 0 {
		t.Errorf("dry-run must not delete; got forgot=%v", fs.forgot)
	}
}

func TestFrequencyWeightOffMatchesRecencyOnly(t *testing.T) {
	// With frequency_weight=0 the access count must not affect the score.
	plain := New(nil, 0.1, 0.05, nil, 0)
	rare := memory.Memory{Importance: 0.5, LastAccessed: ts(10), AccessCount: 0}
	often := memory.Memory{Importance: 0.5, LastAccessed: ts(10), AccessCount: 50}
	if plain.Score(rare) != plain.Score(often) {
		t.Errorf("frequency_weight=0 should ignore access_count: %.4f vs %.4f",
			plain.Score(rare), plain.Score(often))
	}
}

func TestFrequencyWeightReinforcesFrequentRecalls(t *testing.T) {
	reinf := New(nil, 0.1, 0.05, nil, 0.2)
	// Equal importance and age (fresh, so decay ≈ 1); only access count differs.
	rare := memory.Memory{Importance: 0.5, LastAccessed: ts(0), AccessCount: 0}
	often := memory.Memory{Importance: 0.5, LastAccessed: ts(0), AccessCount: 20}
	if reinf.Score(often) <= reinf.Score(rare) {
		t.Errorf("frequently-recalled memory should score higher: often=%.4f rare=%.4f",
			reinf.Score(often), reinf.Score(rare))
	}
	// On a fresh memory reinforcement pushes the score above raw importance.
	if reinf.Score(often) <= often.Importance {
		t.Errorf("reinforced score %.4f should exceed importance %.2f", reinf.Score(often), often.Importance)
	}
}

func TestFrequencyWeightSavesFromPrune(t *testing.T) {
	// Two equal-importance, equal-age memories; only the often-recalled one
	// should survive a prune once reinforcement is on.
	mk := func() []memory.Memory {
		return []memory.Memory{
			{ID: "rare", Type: memory.Semantic, Importance: 0.2, LastAccessed: ts(10), AccessCount: 0},
			{ID: "often", Type: memory.Semantic, Importance: 0.2, LastAccessed: ts(10), AccessCount: 30},
		}
	}

	plain := New(&fakeStore{mems: mk()}, 0.1, 0.1, nil, 0)
	prunedPlain, _ := plain.Prune(context.Background(), true)
	if len(prunedPlain) != 2 {
		t.Fatalf("recency-only: expected both pruned, got %d", len(prunedPlain))
	}

	reinf := New(&fakeStore{mems: mk()}, 0.1, 0.1, nil, 0.3)
	prunedReinf, _ := reinf.Prune(context.Background(), true)
	if len(prunedReinf) != 1 || prunedReinf[0].ID != "rare" {
		t.Fatalf("reinforced: expected only 'rare' pruned, got %v", prunedReinf)
	}
}

func TestPruneProtectsType(t *testing.T) {
	fs := &fakeStore{mems: []memory.Memory{
		{ID: "p", Type: memory.Procedural, Importance: 0.01, LastAccessed: ts(60)},
	}}
	e := New(fs, 0.1, 0.05, nil, 0) // default protects Procedural
	pruned, err := e.Prune(context.Background(), false)
	if err != nil {
		t.Fatalf("Prune: %v", err)
	}
	if len(pruned) != 0 {
		t.Errorf("procedural should be protected, got pruned=%v", pruned)
	}
}
