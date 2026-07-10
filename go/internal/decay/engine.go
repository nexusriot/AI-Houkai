package decay

import (
	"context"
	"math"
	"time"

	"github.com/nexusriot/ai-houkai/internal/memory"
)

// Engine applies time-based decay to prune stale memories.
type Engine struct {
	store        Storable
	DecayRate    float32
	MinScore     float32
	ProtectTypes []memory.MemoryType

	// FrequencyWeight is the recall-reinforcement strength: the decay score is
	// multiplied by 1 + FrequencyWeight × ln(1 + access_count), so a
	// frequently-recalled memory ages out more slowly than an untouched one of
	// equal importance and age. 0 (the default) disables reinforcement — scores
	// match the recency-only behaviour. With it enabled the score can exceed
	// importance; MinScore is interpreted against the reinforced value.
	FrequencyWeight float32
}

// Storable is the subset of MemoryStore the Engine needs.
type Storable interface {
	ListRecent(ctx context.Context, limit int, includeSuperseded, includeExpired bool) ([]memory.Memory, error)
	Forget(ctx context.Context, id string) (bool, error)
}

// actorScoped is implemented by *memory.MemoryStore; we use it to attribute
// journal entries to "decay" without making it required for tests.
type actorScoped interface {
	AsActor(name string) func()
}

func New(store Storable, decayRate, minScore float32, protectTypes []memory.MemoryType, frequencyWeight float32) *Engine {
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
		store:           store,
		DecayRate:       decayRate,
		MinScore:        minScore,
		ProtectTypes:    protectTypes,
		FrequencyWeight: frequencyWeight,
	}
}

func (e *Engine) scoreAt(m memory.Memory, now time.Time) float32 {
	lastAccess := time.Unix(int64(m.LastAccessed), 0)
	days := float32(now.Sub(lastAccess).Hours() / 24)
	base := m.Importance * float32(math.Exp(float64(-e.DecayRate*days)))
	if e.FrequencyWeight != 0 {
		count := m.AccessCount
		if count < 0 {
			count = 0
		}
		base *= 1 + e.FrequencyWeight*float32(math.Log1p(float64(count)))
	}
	return base
}

// Prune removes memories below MinScore (unless protected). Returns pruned IDs.
// If dryRun is true, no deletions occur.
func (e *Engine) Prune(ctx context.Context, dryRun bool) ([]memory.Memory, error) {
	// includeSuperseded=true so soft-deleted memories also age out — otherwise
	// every supersede leaves the old memory in the store forever and the
	// collection grows without bound (matches Python's score_all).
	// includeExpired=true so decay also considers TTL-expired rows.
	mems, err := e.store.ListRecent(ctx, 0, true, true)
	if err != nil {
		return nil, err
	}
	if !dryRun {
		if as, ok := e.store.(actorScoped); ok {
			defer as.AsActor("decay")()
		}
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
