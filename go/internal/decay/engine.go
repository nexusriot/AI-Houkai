package decay

import (
	"context"
	"math"
	"sort"
	"time"

	"github.com/nexusriot/ai-houkai/internal/memory"
)

// Engine applies time-based decay to prune stale memories.
type Engine struct {
	store        Storable
	DecayRate    float32
	MinScore     float32
	ProtectTypes []memory.MemoryType
}

// Storable is the subset of MemoryStore the Engine needs.
type Storable interface {
	ListRecent(ctx context.Context, limit int, includeSuperseded bool) ([]memory.Memory, error)
	Forget(ctx context.Context, id string) (bool, error)
}

func New(store Storable, decayRate, minScore float32, protectTypes []memory.MemoryType) *Engine {
	if decayRate == 0 {
		decayRate = 0.1
	}
	if minScore == 0 {
		minScore = 0.05
	}
	if protectTypes == nil {
		protectTypes = []memory.MemoryType{memory.Procedural}
	}
	return &Engine{
		store:        store,
		DecayRate:    decayRate,
		MinScore:     minScore,
		ProtectTypes: protectTypes,
	}
}

// Score returns the decay score for a single memory.
func (e *Engine) Score(m memory.Memory) float32 {
	return e.scoreAt(m, time.Now())
}

func (e *Engine) scoreAt(m memory.Memory, now time.Time) float32 {
	lastAccess := time.Unix(int64(m.LastAccessed), 0)
	days := float32(now.Sub(lastAccess).Hours() / 24)
	return m.Importance * float32(math.Exp(float64(-e.DecayRate*days)))
}

type ScoredMemory struct {
	memory.Memory
	DecayScore float32
}

// ScoreAll returns all memories with their decay scores, sorted descending.
func (e *Engine) ScoreAll(ctx context.Context) ([]ScoredMemory, error) {
	mems, err := e.store.ListRecent(ctx, 0, false)
	if err != nil {
		return nil, err
	}
	now := time.Now()
	scored := make([]ScoredMemory, len(mems))
	for i, m := range mems {
		scored[i] = ScoredMemory{Memory: m, DecayScore: e.scoreAt(m, now)}
	}
	sort.Slice(scored, func(i, j int) bool {
		return scored[i].DecayScore > scored[j].DecayScore
	})
	return scored, nil
}

// Prune removes memories below MinScore (unless protected). Returns pruned IDs.
// If dryRun is true, no deletions occur.
func (e *Engine) Prune(ctx context.Context, dryRun bool) ([]memory.Memory, error) {
	mems, err := e.store.ListRecent(ctx, 0, false)
	if err != nil {
		return nil, err
	}
	now := time.Now()
	var pruned []memory.Memory
	for _, m := range mems {
		if e.isProtected(m) {
			continue
		}
		if e.scoreAt(m, now) < e.MinScore {
			if !dryRun {
				_, _ = e.store.Forget(ctx, m.ID)
			}
			pruned = append(pruned, m)
		}
	}
	return pruned, nil
}

func (e *Engine) isProtected(m memory.Memory) bool {
	for _, t := range e.ProtectTypes {
		if m.Type == t {
			return true
		}
	}
	return false
}
