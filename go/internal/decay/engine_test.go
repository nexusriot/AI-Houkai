package decay

import (
	"context"
	"testing"
	"time"

	"github.com/nexusriot/ai-houkai/internal/memory"
)

// fakeStore implements Storable for unit tests.
type fakeStore struct {
	mems       []memory.Memory
	forgot     []string
	sawInclude bool // last includeSuperseded arg ListRecent was called with
}

func (f *fakeStore) ListRecent(_ context.Context, _ int, includeSuperseded, _ bool) ([]memory.Memory, error) {
	f.sawInclude = includeSuperseded
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
	now := time.Now()
	fresh := memory.Memory{Importance: 1.0, LastAccessed: ts(0)}
	old := memory.Memory{Importance: 1.0, LastAccessed: ts(30)}
	if e.scoreAt(fresh, now) <= e.scoreAt(old, now) {
		t.Errorf("fresh (%.3f) should outscore old (%.3f)", e.scoreAt(fresh, now), e.scoreAt(old, now))
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

func TestPruneConsidersSupersededMemories(t *testing.T) {
	// Prune must ask for superseded memories too, else soft-deleted entries
	// never age out and the store grows without bound.
	fs := &fakeStore{mems: []memory.Memory{
		{ID: "keep", Type: memory.Semantic, Importance: 1.0, LastAccessed: ts(0)},
	}}
	e := New(fs, 0.1, 0.05, nil, 0)
	if _, err := e.Prune(context.Background(), true); err != nil {
		t.Fatalf("Prune: %v", err)
	}
	if !fs.sawInclude {
		t.Error("Prune called ListRecent with includeSuperseded=false; superseded memories will linger forever")
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
	now := time.Now()
	rare := memory.Memory{Importance: 0.5, LastAccessed: ts(10), AccessCount: 0}
	often := memory.Memory{Importance: 0.5, LastAccessed: ts(10), AccessCount: 50}
	if plain.scoreAt(rare, now) != plain.scoreAt(often, now) {
		t.Errorf("frequency_weight=0 should ignore access_count: %.4f vs %.4f",
			plain.scoreAt(rare, now), plain.scoreAt(often, now))
	}
}

func TestFrequencyWeightReinforcesFrequentRecalls(t *testing.T) {
	reinf := New(nil, 0.1, 0.05, nil, 0.2)
	now := time.Now()
	// Equal importance and age (fresh, so decay ≈ 1); only access count differs.
	rare := memory.Memory{Importance: 0.5, LastAccessed: ts(0), AccessCount: 0}
	often := memory.Memory{Importance: 0.5, LastAccessed: ts(0), AccessCount: 20}
	if reinf.scoreAt(often, now) <= reinf.scoreAt(rare, now) {
		t.Errorf("frequently-recalled memory should score higher: often=%.4f rare=%.4f",
			reinf.scoreAt(often, now), reinf.scoreAt(rare, now))
	}
	// On a fresh memory reinforcement pushes the score above raw importance.
	if reinf.scoreAt(often, now) <= often.Importance {
		t.Errorf("reinforced score %.4f should exceed importance %.2f", reinf.scoreAt(often, now), often.Importance)
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

// trashingStore is a Storable that records trash calls, so we can prove the
// engine routes prunes through the recoverable path rather than Forget.
type trashingStore struct {
	fakeStore
	trashed []string
}

func (f *trashingStore) Trash(_ context.Context, id string) (bool, error) {
	f.trashed = append(f.trashed, id)
	for i, m := range f.mems {
		if m.ID == id {
			f.mems = append(f.mems[:i], f.mems[i+1:]...)
			return true, nil
		}
	}
	return false, nil
}

// Decay is driven by tunable constants, so a mis-set MinScore used to destroy
// data outright. README and docs/DESIGN.md §26 both promise pruning routes to
// the trash; Prune called Forget, so that safety property was absent.
func TestPruneRoutesToTrashNotForget(t *testing.T) {
	fs := &trashingStore{fakeStore: fakeStore{mems: []memory.Memory{
		{ID: "old", Importance: 0.5, LastAccessed: ts(400)},
		{ID: "new", Importance: 0.9, LastAccessed: ts(0)},
	}}}
	e := New(fs, 0.1, 0.05, nil, 0)

	pruned, err := e.Prune(context.Background(), false)
	if err != nil {
		t.Fatalf("Prune: %v", err)
	}
	if len(pruned) != 1 || pruned[0].ID != "old" {
		t.Fatalf("pruned = %+v, want just [old]", pruned)
	}
	if len(fs.trashed) != 1 || fs.trashed[0] != "old" {
		t.Fatalf("trashed = %v, want [old]", fs.trashed)
	}
	if len(fs.forgot) != 0 {
		t.Fatalf("forgot = %v, want none — prune must be recoverable", fs.forgot)
	}
}

func TestPruneDryRunTrashesNothing(t *testing.T) {
	fs := &trashingStore{fakeStore: fakeStore{mems: []memory.Memory{
		{ID: "old", Importance: 0.5, LastAccessed: ts(400)},
	}}}
	pruned, err := New(fs, 0.1, 0.05, nil, 0).Prune(context.Background(), true)
	if err != nil {
		t.Fatalf("Prune: %v", err)
	}
	if len(pruned) != 1 {
		t.Fatalf("pruned = %+v, want 1 candidate", pruned)
	}
	if len(fs.trashed) != 0 || len(fs.forgot) != 0 {
		t.Fatalf("dry run mutated the store: trashed=%v forgot=%v",
			fs.trashed, fs.forgot)
	}
}

// A store that predates the trash capability must still prune rather than
// silently keeping everything forever.
func TestPruneFallsBackToForgetWithoutTrash(t *testing.T) {
	fs := &fakeStore{mems: []memory.Memory{
		{ID: "old", Importance: 0.5, LastAccessed: ts(400)},
	}}
	if _, err := New(fs, 0.1, 0.05, nil, 0).Prune(context.Background(), false); err != nil {
		t.Fatalf("Prune: %v", err)
	}
	if len(fs.forgot) != 1 || fs.forgot[0] != "old" {
		t.Fatalf("forgot = %v, want [old]", fs.forgot)
	}
}
