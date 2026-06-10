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
	store              Storable
	SimilarityThreshold float32
	MinClusterSize     int
	Summarizer         Summarizer
}

// Storable is the subset of MemoryStore the Engine needs.
type Storable interface {
	AllRaw(ctx context.Context) ([]vector.Item, error)
	Remember(ctx context.Context, text string, opts memory.RememberOpts) (memory.Memory, bool, []memory.Conflict, error)
	Forget(ctx context.Context, id string) (bool, error)
	Link(ctx context.Context, srcID, dstID, rel string) error
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
			sim := cosineSim(seed.embedding, other.embedding)
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
// If consolidate is true, deletes source episodic memories.
// If dryRun is true, no writes occur.
func (e *Engine) Reflect(ctx context.Context, dryRun, consolidate bool) ([]memory.Memory, error) {
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

		// Aggregate tags and mean importance.
		tagSet := map[string]bool{"reflection": true}
		var totalImp float32
		for _, m := range cluster {
			for _, t := range m.Tags {
				tagSet[t] = true
			}
			totalImp += m.Importance
		}
		tags := make([]string, 0, len(tagSet))
		for t := range tagSet {
			tags = append(tags, t)
		}
		sort.Strings(tags)
		avgImp := totalImp / float32(len(cluster))

		if dryRun {
			created = append(created, memory.Memory{
				Text:       text,
				Type:       memory.Semantic,
				Tags:       tags,
				Importance: avgImp,
				CreatedAt:  float64(time.Now().Unix()),
			})
			continue
		}

		newMem, _, _, err := e.store.Remember(ctx, text, memory.RememberOpts{
			Type:       memory.Semantic,
			Tags:       tags,
			Importance: avgImp,
		})
		if err != nil {
			return created, err
		}
		// Link new semantic memory to sources.
		for _, src := range cluster {
			_ = e.store.Link(ctx, newMem.ID, src.ID, memory.RelDerivedFrom)
		}
		created = append(created, newMem)

		if consolidate {
			for _, src := range cluster {
				_, _ = e.store.Forget(ctx, src.ID)
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
	if len(full) > 512 {
		full = full[:512]
	}
	return full, nil
}

func cosineSim(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0
	}
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return float32(dot / (math.Sqrt(na) * math.Sqrt(nb)))
}
