package memory

import (
	"context"
	"fmt"
	"log"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/nexusriot/ai-houkai/internal/embed"
	"github.com/nexusriot/ai-houkai/internal/vector"
)

// RecallMode selects the scoring strategy.
type RecallMode string

const (
	ModeSemantic RecallMode = "semantic"
	ModeHybrid   RecallMode = "hybrid"
)

// StoreConfig holds MemoryStore creation parameters.
type StoreConfig struct {
	Path              string
	Collection        string
	ConflictPolicy    ConflictPolicy
	ConflictThreshold float32
	ConflictFn        func(a, b Memory) bool
	DefaultImportance float32
	DecayRate         float32 // used for recency in hybrid scoring
}

func DefaultStoreConfig(path, collection string) StoreConfig {
	return StoreConfig{
		Path:              path,
		Collection:        collection,
		ConflictPolicy:    PolicyIgnore,
		ConflictThreshold: 0.80,
		DefaultImportance: 0.5,
		DecayRate:         0.1,
	}
}

// MemoryStore is the main interface to stored memories.
type MemoryStore struct {
	backend  vector.Backend
	embedder embed.Embedder
	cfg      StoreConfig
}

// NewMemoryStore creates a new store backed by the given vector backend + embedder.
func NewMemoryStore(backend vector.Backend, embedder embed.Embedder, cfg StoreConfig) *MemoryStore {
	return &MemoryStore{backend: backend, embedder: embedder, cfg: cfg}
}

// Remember stores a new memory and returns it.
func (s *MemoryStore) Remember(ctx context.Context, text string, opts RememberOpts) (Memory, bool, []Conflict, error) {
	if opts.Type == "" {
		opts.Type = Episodic
	}
	if opts.Importance == 0 {
		opts.Importance = s.cfg.DefaultImportance
	}

	now := float64(time.Now().Unix())
	m := Memory{
		ID:           uuid.NewString(),
		Text:         text,
		Type:         opts.Type,
		Tags:         opts.Tags,
		Importance:   opts.Importance,
		CreatedAt:    now,
		LastAccessed: now,
		AccessCount:  0,
		Source:       opts.Source,
		Links:        []Link{},
		SupersededBy: "",
		SupersededAt: 0,
		Polarity:     opts.Polarity,
	}
	if m.Tags == nil {
		m.Tags = []string{}
	}

	// Conflict check.
	if s.cfg.ConflictPolicy != PolicyIgnore {
		vecs, err := s.embedder.Embed(ctx, []string{text})
		if err != nil {
			return Memory{}, false, nil, err
		}

		hits, err := s.backend.Query(ctx, vecs[0], 12)
		if err != nil {
			return Memory{}, false, nil, err
		}
		candidates := hitsToMemoriesWithScore(hits)
		conflicts := detectConflicts(m, candidates, s.cfg.ConflictThreshold, s.cfg.ConflictFn)

		if len(conflicts) > 0 {
			switch s.cfg.ConflictPolicy {
			case PolicyWarn:
				log.Printf("ai-houkai: %d conflict(s) detected for new memory", len(conflicts))
			case PolicyRaise:
				return Memory{}, false, conflicts, &ConflictError{Conflicts: conflicts}
			case PolicySupersede:
				for _, c := range conflicts {
					_ = s.doSupersede(ctx, c.B.ID, m.ID)
				}
			}
		}
	}

	vecs, err := s.embedder.Embed(ctx, []string{text})
	if err != nil {
		return Memory{}, false, nil, err
	}

	err = s.backend.Add(ctx, []vector.Item{{
		ID:        m.ID,
		Content:   text,
		Embedding: vecs[0],
		Metadata:  MemoryToMetadata(m),
	}})
	if err != nil {
		return Memory{}, false, nil, err
	}
	return m, true, nil, nil
}

// RememberOpts are optional fields for Remember.
type RememberOpts struct {
	Type       MemoryType
	Tags       []string
	Importance float32
	Source     string
	Polarity   int
}

// Recall returns up to k memories matching query.
func (s *MemoryStore) Recall(ctx context.Context, query string, k int, opts RecallOpts) ([]MemoryWithScore, error) {
	if k <= 0 {
		k = 5
	}
	overfetch := k
	if opts.Mode == ModeHybrid {
		overfetch = k * opts.Overfetch
		if overfetch < k {
			overfetch = k * 3
		}
	}

	vecs, err := s.embedder.Embed(ctx, []string{query})
	if err != nil {
		return nil, err
	}

	hits, err := s.backend.Query(ctx, vecs[0], overfetch)
	if err != nil {
		return nil, err
	}

	candidates := hitsToMemoriesWithScore(hits)

	// Metadata filters.
	var filtered []MemoryWithScore
	for _, c := range candidates {
		if !opts.IncludeSuperseded && c.SupersededBy != "" {
			continue
		}
		if opts.Type != "" && c.Type != opts.Type {
			continue
		}
		if c.Importance < opts.MinImportance {
			continue
		}
		if opts.Tag != "" && !containsTag(c.Memory, opts.Tag) {
			continue
		}
		filtered = append(filtered, c)
	}

	// Hybrid rescoring.
	if opts.Mode == ModeHybrid {
		if opts.Weights == (HybridWeights{}) {
			opts.Weights = DefaultWeights()
		}
		texts := make([]string, len(filtered))
		for i, c := range filtered {
			texts[i] = c.Text
		}
		bm25Scores := bm25Score(query, texts)
		for i := range filtered {
			var bm25s float32
			if bm25Scores != nil {
				bm25s = bm25Scores[i]
			}
			filtered[i].Score = hybridScore(filtered[i].Score, bm25s,
				filtered[i].Importance, filtered[i].LastAccessed,
				opts.Weights, s.cfg.DecayRate)
		}
		sort.Slice(filtered, func(i, j int) bool {
			return filtered[i].Score > filtered[j].Score
		})
	}

	if len(filtered) > k {
		filtered = filtered[:k]
	}

	// Touch accessed memories.
	for i := range filtered {
		if err := s.touch(ctx, &filtered[i].Memory); err != nil {
			log.Printf("ai-houkai: touch %s: %v", filtered[i].ID, err)
		}
	}

	// Graph expansion.
	if opts.Expand != nil {
		filtered, err = s.expandResults(ctx, filtered, opts.Expand)
		if err != nil {
			return nil, err
		}
	}

	return filtered, nil
}

// RecallOpts holds optional Recall parameters.
type RecallOpts struct {
	Type              MemoryType
	Tag               string
	MinImportance     float32
	Mode              RecallMode
	Overfetch         int
	Weights           HybridWeights
	IncludeSuperseded bool
	Expand            *ExpandSpec
}

// Forget deletes a memory by ID. Returns true if found and deleted.
func (s *MemoryStore) Forget(ctx context.Context, id string) (bool, error) {
	items, err := s.backend.Get(ctx, []string{id})
	if err != nil || len(items) == 0 {
		return false, err
	}
	return true, s.backend.Delete(ctx, []string{id})
}

// ListRecent returns up to limit memories sorted by created_at desc.
func (s *MemoryStore) ListRecent(ctx context.Context, limit int, includeSuperseded bool) ([]Memory, error) {
	items, err := s.backend.All(ctx)
	if err != nil {
		return nil, err
	}
	var mems []Memory
	for _, it := range items {
		m := MetadataToMemory(it.ID, it.Content, it.Metadata)
		if !includeSuperseded && m.SupersededBy != "" {
			continue
		}
		mems = append(mems, m)
	}
	sort.Slice(mems, func(i, j int) bool {
		return mems[i].CreatedAt > mems[j].CreatedAt
	})
	if limit > 0 && len(mems) > limit {
		mems = mems[:limit]
	}
	return mems, nil
}

// Count returns total stored memories.
func (s *MemoryStore) Count(ctx context.Context) (int, error) {
	return s.backend.Count(ctx)
}

// GetByID returns a single memory or ErrNotFound.
func (s *MemoryStore) GetByID(ctx context.Context, id string) (Memory, error) {
	// Resolve 8-char prefix.
	if len(id) < 36 {
		return s.resolvePrefix(ctx, id)
	}
	items, err := s.backend.Get(ctx, []string{id})
	if err != nil {
		return Memory{}, err
	}
	if len(items) == 0 {
		return Memory{}, ErrNotFound
	}
	return MetadataToMemory(items[0].ID, items[0].Content, items[0].Metadata), nil
}

// UpdateMemory re-persists a Memory (delta update on metadata; re-embeds if text changed).
func (s *MemoryStore) UpdateMemory(ctx context.Context, m Memory, textChanged bool) error {
	if textChanged {
		vecs, err := s.embedder.Embed(ctx, []string{m.Text})
		if err != nil {
			return err
		}
		_ = s.backend.Delete(ctx, []string{m.ID})
		return s.backend.Add(ctx, []vector.Item{{
			ID:        m.ID,
			Content:   m.Text,
			Embedding: vecs[0],
			Metadata:  MemoryToMetadata(m),
		}})
	}
	return s.backend.UpdateMetadata(ctx, m.ID, MemoryToMetadata(m))
}

func (s *MemoryStore) Link(ctx context.Context, srcID, dstID, rel string) error {
	if rel == "" {
		rel = RelRelated
	}
	src, err := s.GetByID(ctx, srcID)
	if err != nil {
		return fmt.Errorf("link src: %w", err)
	}
	addLink(&src, dstID, rel)
	return s.backend.UpdateMetadata(ctx, src.ID, MemoryToMetadata(src))
}

func (s *MemoryStore) Unlink(ctx context.Context, srcID, dstID, rel string) (int, error) {
	src, err := s.GetByID(ctx, srcID)
	if err != nil {
		return 0, fmt.Errorf("unlink src: %w", err)
	}
	removed := removeLinks(&src, dstID, rel)
	if removed > 0 {
		if err := s.backend.UpdateMetadata(ctx, src.ID, MemoryToMetadata(src)); err != nil {
			return 0, err
		}
	}
	return removed, nil
}

type NeighborResult struct {
	Memory
	Rel string `json:"rel"`
}

func (s *MemoryStore) Neighbors(ctx context.Context, memID, rel, direction string, depth int) ([]NeighborResult, error) {
	if depth <= 0 {
		depth = 1
	}
	visited := map[string]bool{memID: true}
	var results []NeighborResult
	queue := []string{memID}

	for d := 0; d < depth && len(queue) > 0; d++ {
		var next []string
		for _, qid := range queue {
			m, err := s.GetByID(ctx, qid)
			if err != nil {
				continue
			}
			// Outgoing links.
			if direction == "out" || direction == "both" {
				for _, l := range m.Links {
					if rel != "" && l.Rel != rel {
						continue
					}
					if visited[l.To] {
						continue
					}
					nb, err := s.GetByID(ctx, l.To)
					if err == nil {
						results = append(results, NeighborResult{Memory: nb, Rel: l.Rel})
						visited[l.To] = true
						next = append(next, l.To)
					}
				}
			}
			// Incoming links (reverse scan).
			if direction == "in" || direction == "both" {
				all, _ := s.backend.All(ctx)
				for _, it := range all {
					other := MetadataToMemory(it.ID, it.Content, it.Metadata)
					if visited[other.ID] {
						continue
					}
					for _, l := range other.Links {
						if l.To == qid {
							if rel != "" && l.Rel != rel {
								continue
							}
							results = append(results, NeighborResult{Memory: other, Rel: l.Rel})
							visited[other.ID] = true
							next = append(next, other.ID)
							break
						}
					}
				}
			}
		}
		queue = next
	}
	return results, nil
}

func (s *MemoryStore) Subgraph(ctx context.Context, ids []string, depth int) (Graph, error) {
	var g Graph
	nodeSet := make(map[string]bool)
	for _, id := range ids {
		nodeSet[id] = true
	}
	nbs, err := s.Neighbors(ctx, ids[0], "", "both", depth)
	if err != nil {
		return g, err
	}
	for _, nb := range nbs {
		nodeSet[nb.ID] = true
	}

	for id := range nodeSet {
		m, err := s.GetByID(ctx, id)
		if err == nil {
			g.Nodes = append(g.Nodes, m)
		}
	}
	for _, node := range g.Nodes {
		for _, l := range node.Links {
			if nodeSet[l.To] {
				g.Edges = append(g.Edges, struct {
					From string
					To   string
					Rel  string
				}{From: node.ID, To: l.To, Rel: l.Rel})
			}
		}
	}
	return g, nil
}

func (s *MemoryStore) FindConflicts(ctx context.Context, memID string, threshold float32) ([]Conflict, error) {
	if threshold == 0 {
		threshold = s.cfg.ConflictThreshold
	}

	var targets []Memory
	if memID != "" {
		m, err := s.GetByID(ctx, memID)
		if err != nil {
			return nil, err
		}
		targets = []Memory{m}
	} else {
		all, err := s.ListRecent(ctx, 0, false)
		if err != nil {
			return nil, err
		}
		targets = all
	}

	seen := make(map[string]bool)
	var conflicts []Conflict
	for _, t := range targets {
		vecs, err := s.embedder.Embed(ctx, []string{t.Text})
		if err != nil {
			continue
		}
		hits, err := s.backend.Query(ctx, vecs[0], 12)
		if err != nil {
			continue
		}
		candidates := hitsToMemoriesWithScore(hits)
		cs := detectConflicts(t, candidates, threshold, s.cfg.ConflictFn)
		for _, c := range cs {
			key := c.A.ID + ":" + c.B.ID
			if key2 := c.B.ID + ":" + c.A.ID; seen[key] || seen[key2] {
				continue
			}
			seen[key] = true
			conflicts = append(conflicts, c)
		}
	}
	return conflicts, nil
}

func (s *MemoryStore) Supersede(ctx context.Context, oldID, newID string) error {
	return s.doSupersede(ctx, oldID, newID)
}

func (s *MemoryStore) doSupersede(ctx context.Context, oldID, newID string) error {
	old, err := s.GetByID(ctx, oldID)
	if err != nil {
		return fmt.Errorf("supersede old: %w", err)
	}
	now := float64(time.Now().Unix())
	old.SupersededBy = newID
	old.SupersededAt = now
	addLink(&old, newID, RelSupersedes)
	if err := s.backend.UpdateMetadata(ctx, old.ID, MemoryToMetadata(old)); err != nil {
		return err
	}
	// Add reverse link on new memory.
	newMem, err := s.GetByID(ctx, newID)
	if err == nil {
		addLink(&newMem, oldID, RelSupersedes)
		_ = s.backend.UpdateMetadata(ctx, newMem.ID, MemoryToMetadata(newMem))
	}
	return nil
}

func (s *MemoryStore) Restore(ctx context.Context, memID string) error {
	m, err := s.GetByID(ctx, memID)
	if err != nil {
		return err
	}
	m.SupersededBy = ""
	m.SupersededAt = 0
	return s.backend.UpdateMetadata(ctx, m.ID, MemoryToMetadata(m))
}

func (s *MemoryStore) touch(ctx context.Context, m *Memory) error {
	m.LastAccessed = float64(time.Now().Unix())
	m.AccessCount++
	return s.backend.UpdateMetadata(ctx, m.ID, MemoryToMetadata(*m))
}

func (s *MemoryStore) resolvePrefix(ctx context.Context, prefix string) (Memory, error) {
	all, err := s.backend.All(ctx)
	if err != nil {
		return Memory{}, err
	}
	var matches []vector.Item
	for _, it := range all {
		if strings.HasPrefix(it.ID, prefix) {
			matches = append(matches, it)
		}
	}
	if len(matches) == 0 {
		return Memory{}, fmt.Errorf("no memory with prefix %q: %w", prefix, ErrNotFound)
	}
	if len(matches) > 1 {
		return Memory{}, fmt.Errorf("ambiguous prefix %q: %d matches", prefix, len(matches))
	}
	return MetadataToMemory(matches[0].ID, matches[0].Content, matches[0].Metadata), nil
}

func (s *MemoryStore) expandResults(ctx context.Context, results []MemoryWithScore, spec *ExpandSpec) ([]MemoryWithScore, error) {
	if spec == nil {
		return results, nil
	}
	seen := make(map[string]bool)
	for _, r := range results {
		seen[r.ID] = true
	}
	for _, r := range results {
		nbs, err := s.Neighbors(ctx, r.ID, "", "out", spec.Depth)
		if err != nil {
			continue
		}
		for _, nb := range nbs {
			if seen[nb.ID] {
				continue
			}
			if spec.Score > 0 && r.Score < spec.Score {
				continue
			}
			relOK := len(spec.Rels) == 0
			for _, rel := range spec.Rels {
				if nb.Rel == rel {
					relOK = true
					break
				}
			}
			if !relOK {
				continue
			}
			seen[nb.ID] = true
			results = append(results, MemoryWithScore{Memory: nb.Memory, Score: r.Score * 0.9})
			if spec.Cap > 0 && len(results) >= spec.Cap {
				return results, nil
			}
		}
	}
	return results, nil
}

func hitsToMemoriesWithScore(hits []vector.Hit) []MemoryWithScore {
	out := make([]MemoryWithScore, len(hits))
	for i, h := range hits {
		out[i] = MemoryWithScore{
			Memory: MetadataToMemory(h.ID, h.Content, h.Metadata),
			Score:  h.Similarity,
		}
	}
	return out
}

func containsTag(m Memory, tag string) bool {
	for _, t := range m.Tags {
		if t == tag {
			return true
		}
	}
	return false
}

// AllRaw returns all items with embeddings — used by ReflectionEngine.
func (s *MemoryStore) AllRaw(ctx context.Context) ([]vector.Item, error) {
	return s.backend.All(ctx)
}

// Stats returns aggregate stats.
func (s *MemoryStore) Stats(ctx context.Context) (map[string]any, error) {
	count, err := s.backend.Count(ctx)
	if err != nil {
		return nil, err
	}

	mems, err := s.ListRecent(ctx, 0, true)
	if err != nil {
		return nil, err
	}

	typeCounts := map[string]int{}
	for _, m := range mems {
		typeCounts[string(m.Type)]++
	}

	tagCounts := map[string]int{}
	var totalImp float32
	for _, m := range mems {
		for _, t := range m.Tags {
			tagCounts[t]++
		}
		totalImp += m.Importance
	}

	var avgImp float32
	if count > 0 {
		avgImp = totalImp / float32(count)
	}

	topTags := topN(tagCounts, 5)
	return map[string]any{
		"count":          count,
		"by_type":        typeCounts,
		"top_tags":       topTags,
		"avg_importance": avgImp,
	}, nil
}

func topN(m map[string]int, n int) []string {
	type kv struct {
		k string
		v int
	}
	var sl []kv
	for k, v := range m {
		sl = append(sl, kv{k, v})
	}
	sort.Slice(sl, func(i, j int) bool { return sl[i].v > sl[j].v })
	var out []string
	for i, x := range sl {
		if i >= n {
			break
		}
		out = append(out, x.k+"="+strconv.Itoa(x.v))
	}
	return out
}
