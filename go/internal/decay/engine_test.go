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
	e := New(nil, 0.1, 0.05, nil)
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
	e := New(fs, 0.1, 0.05, nil)

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
	e := New(fs, 0.1, 0.05, nil)
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

func TestPruneProtectsType(t *testing.T) {
	fs := &fakeStore{mems: []memory.Memory{
		{ID: "p", Type: memory.Procedural, Importance: 0.01, LastAccessed: ts(60)},
	}}
	e := New(fs, 0.1, 0.05, nil) // default protects Procedural
	pruned, err := e.Prune(context.Background(), false)
	if err != nil {
		t.Fatalf("Prune: %v", err)
	}
	if len(pruned) != 0 {
		t.Errorf("procedural should be protected, got pruned=%v", pruned)
	}
}
