package memory

import (
	"context"
	"errors"
	"path/filepath"
	"testing"
	"time"

	"github.com/nexusriot/ai-houkai/internal/vector"
)

type failEmbedder struct{}

func (failEmbedder) Dim() int { return 16 }
func (failEmbedder) Embed(context.Context, []string) ([][]float32, error) {
	return nil, errors.New("provider unreachable")
}

// countingEmbedder returns a fixed valid vector and tallies its Embed calls.
type countingEmbedder struct {
	dim   int
	calls *int
}

func (e countingEmbedder) Dim() int { return e.dim }
func (e countingEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	*e.calls += len(texts)
	out := make([][]float32, len(texts))
	for i := range out {
		v := make([]float32, e.dim)
		v[0] = 1.0 // unit vector, well-defined cosine
		out[i] = v
	}
	return out, nil
}

func newStoreWithEmbedder(t *testing.T, e interface {
	Dim() int
	Embed(context.Context, []string) ([][]float32, error)
}) *MemoryStore {
	t.Helper()
	dir := t.TempDir()
	backend, err := vector.NewChromem(filepath.Join(dir, "s"), "test", 16)
	if err != nil {
		t.Fatalf("NewChromem: %v", err)
	}
	t.Cleanup(func() { _ = backend.Close() })
	return NewMemoryStore(backend, e, DefaultStoreConfig(dir, "test"))
}

func TestProbeEmbedding(t *testing.T) {
	store := newTestStore(t)
	probe := store.ProbeEmbedding(context.Background())
	if ok, _ := probe["ok"].(bool); !ok {
		t.Fatalf("probe should succeed: %v", probe)
	}
	if dim, _ := probe["dim"].(int); dim != 16 {
		t.Errorf("probe dim = %v, want 16", probe["dim"])
	}
}

func TestReadinessOK(t *testing.T) {
	store := newTestStore(t)
	r := store.Readiness(context.Background(), 0)
	if ready, _ := r["ready"].(bool); !ready {
		t.Fatalf("store should be ready: %v", r)
	}
	checks := r["checks"].(map[string]any)
	if s := checks["store"].(map[string]any); !s["ok"].(bool) {
		t.Error("store check should be ok")
	}
	if e := checks["embedder"].(map[string]any); !e["ok"].(bool) {
		t.Error("embedder check should be ok")
	}
}

func TestReadinessEmbedderFailure(t *testing.T) {
	store := newStoreWithEmbedder(t, failEmbedder{})
	r := store.Readiness(context.Background(), 0)
	if ready, _ := r["ready"].(bool); ready {
		t.Fatalf("readiness should be false when the embedder is down: %v", r)
	}
	checks := r["checks"].(map[string]any)
	e := checks["embedder"].(map[string]any)
	if e["ok"].(bool) {
		t.Error("embedder check should report failure")
	}
	if e["error"] == nil {
		t.Error("embedder failure should carry an error message")
	}
}

func TestReadinessCacheAvoidsReembedding(t *testing.T) {
	n := 0
	store := newStoreWithEmbedder(t, countingEmbedder{dim: 16, calls: &n})
	ctx := context.Background()
	store.Readiness(ctx, time.Minute)
	store.Readiness(ctx, time.Minute)
	if n != 1 {
		t.Fatalf("a ready result within the TTL must be cached; embed calls=%d", n)
	}
	store.Readiness(ctx, 0)
	if n != 2 {
		t.Fatalf("cacheTTL=0 must always re-probe; embed calls=%d", n)
	}
}
