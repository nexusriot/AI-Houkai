package reflect

import (
	"context"
	"fmt"
	"math"
	"sort"
	"strings"
	"time"

	"github.com/nexusriot/ai-houkai/internal/memory"
	"github.com/nexusriot/ai-houkai/internal/vector"
)

// Summarizer condenses a cluster of memories into a single string.
type Summarizer func(ctx context.Context, ms []memory.Memory) (string, error)

// Engine clusters episodic memories and condenses them into semantic ones.
type Engine struct {
	store               Storable
	SimilarityThreshold float32
	MinClusterSize      int
	Summarizer          Summarizer
}

// Storable is the subset of MemoryStore the Engine needs.
type Storable interface {
	AllRaw(ctx context.Context) ([]vector.Item, error)
	Remember(ctx context.Context, text string, opts memory.RememberOpts) (memory.Memory, bool, []memory.Conflict, error)
	Forget(ctx context.Context, id string) (bool, error)
	Link(ctx context.Context, srcID, dstID, rel string) error
	Supersede(ctx context.Context, oldID, newID string) error
}

// ConsolidateMode selects what happens to source episodics after a reflection.
//
//	none — leave sources untouched (default)
//	soft — supersede sources by the summary and add a derived_from link
//	hard — permanently forget sources (data is lost)
type ConsolidateMode string

const (
	ConsolidateNone ConsolidateMode = "none"
	ConsolidateSoft ConsolidateMode = "soft"
	ConsolidateHard ConsolidateMode = "hard"
)

// ConsolidateModeFromString maps a user string (including "" and legacy bool
// spellings) to a ConsolidateMode. Unknown values fall back to none.
func ConsolidateModeFromString(s string) ConsolidateMode {
	switch s {
	case "soft", "true", "1", "yes":
		return ConsolidateSoft
	case "hard":
		return ConsolidateHard
	default:
		return ConsolidateNone
	}
}

// actorScoped is implemented by *memory.MemoryStore. We optionally use it
// to attribute journal entries to "reflection" without making it required.
type actorScoped interface {
	AsActor(name string) func()
}

func New(store Storable, similarityThreshold float32, minClusterSize int, summarizer Summarizer) *Engine {
	if similarityThreshold == 0 {
		similarityThreshold = 0.75
	}
	if minClusterSize == 0 {
		minClusterSize = 2
	}
	if summarizer == nil {
		summarizer = defaultSummarizer
	}
	return &Engine{
		store:               store,
		SimilarityThreshold: similarityThreshold,
		MinClusterSize:      minClusterSize,
		Summarizer:          summarizer,
	}
}

// Clusters returns the greedy single-linkage clusters without writing anything.
func (e *Engine) Clusters(ctx context.Context) ([][]memory.Memory, error) {
	items, err := e.store.AllRaw(ctx)
	if err != nil {
		return nil, err
	}

	type episodicItem struct {
		memory.Memory
		embedding []float32
	}

	var eps []episodicItem
	for _, it := range items {
		m := memory.MetadataToMemory(it.ID, it.Content, it.Metadata)
		if m.Type == memory.Episodic && m.SupersededBy == "" {
			eps = append(eps, episodicItem{Memory: m, embedding: it.Embedding})
		}
	}

	// Sort by importance descending so highest-importance seeds first.
	sort.Slice(eps, func(i, j int) bool {
		return eps[i].Importance > eps[j].Importance
	})

	assigned := make([]bool, len(eps))
	var clusters [][]memory.Memory

	for i, seed := range eps {
		if assigned[i] {
			continue
		}
		if len(seed.embedding) == 0 {
			continue
		}
		assigned[i] = true
		cluster := []memory.Memory{seed.Memory}

		for j, other := range eps {
			if assigned[j] || len(other.embedding) == 0 {
				continue
			}
			// Never merge explicitly opposite polarities: a positive and a
			// negative memory about the same event describe contradictory
			// states and must not collapse into one summary.
			if seed.Polarity != 0 && other.Polarity != 0 && seed.Polarity != other.Polarity {
				continue
			}
			sim := vector.CosineSim(seed.embedding, other.embedding)
			if sim >= e.SimilarityThreshold {
				assigned[j] = true
				cluster = append(cluster, other.Memory)
			}
		}
		if len(cluster) >= e.MinClusterSize {
			clusters = append(clusters, cluster)
		}
	}
	return clusters, nil
}

// Reflect condenses clusters into semantic memories. Returns created memories.
//
// consolidate controls what happens to source episodics:
//
//	none — leave sources untouched (no derived_from link)
//	soft — supersede each source by the summary and add a derived_from link
//	hard — permanently forget each source (no link)
//
// If dryRun is true, no writes occur (consolidate is ignored).
func (e *Engine) Reflect(ctx context.Context, dryRun bool, consolidate ConsolidateMode) ([]memory.Memory, error) {
	clusters, err := e.Clusters(ctx)
	if err != nil {
		return nil, err
	}

	// Attribute journal entries written by this reflection pass.
	if !dryRun {
		if as, ok := e.store.(actorScoped); ok {
			defer as.AsActor("reflection")()
		}
	}

	var created []memory.Memory
	for _, cluster := range clusters {
		text, err := e.Summarizer(ctx, cluster)
		if err != nil {
			return created, fmt.Errorf("reflect summarizer: %w", err)
		}

		// Aggregate tags (reflection first, then first-seen order) and mean
		// importance rounded to 3 decimals, matching Python.
		tags := []string{"reflection"}
		seen := map[string]bool{"reflection": true}
		var totalImp float64
		for _, m := range cluster {
			for _, t := range m.Tags {
				if !seen[t] {
					seen[t] = true
					tags = append(tags, t)
				}
			}
			totalImp += float64(m.Importance)
		}
		// float64 accumulation + round-half-to-even to 3 decimals, matching
		// Python's round(sum/len, 3).
		avgImp := float32(math.RoundToEven(totalImp/float64(len(cluster))*1000) / 1000)

		if dryRun {
			created = append(created, memory.Memory{
				Text:       text,
				Type:       memory.Semantic,
				Tags:       tags,
				Importance: avgImp,
				Source:     "reflection/dry-run",
				CreatedAt:  float64(time.Now().Unix()),
			})
			continue
		}

		newMem, _, _, err := e.store.Remember(ctx, text, memory.RememberOpts{
			Type:       memory.Semantic,
			Tags:       tags,
			Importance: memory.Float32Ptr(avgImp),
			Source:     "reflection",
		})
		if err != nil {
			return created, err
		}
		created = append(created, newMem)

		switch consolidate {
		case ConsolidateHard:
			for _, src := range cluster {
				_, _ = e.store.Forget(ctx, src.ID)
			}
		case ConsolidateSoft:
			for _, src := range cluster {
				_ = e.store.Supersede(ctx, src.ID, newMem.ID)
				_ = e.store.Link(ctx, newMem.ID, src.ID, memory.RelDerivedFrom)
			}
		}
	}
	return created, nil
}

func defaultSummarizer(_ context.Context, ms []memory.Memory) (string, error) {
	sort.Slice(ms, func(i, j int) bool {
		return ms[i].Importance > ms[j].Importance
	})
	parts := make([]string, len(ms))
	for i, m := range ms {
		parts[i] = m.Text
	}
	body := strings.Join(parts, " | ")
	prefix := fmt.Sprintf("[Reflection ×%d] ", len(ms))
	full := prefix + body
	// Truncate by runes, not bytes, so a multi-byte character is never split
	// (Python slices by characters).
	if r := []rune(full); len(r) > 512 {
		full = string(r[:512])
	}
	return full, nil
}
