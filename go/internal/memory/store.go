package memory

import (
	"context"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"sort"
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
	EmbeddingModel    string // recorded in export headers
	ConflictPolicy    ConflictPolicy
	ConflictThreshold float32
	ConflictFn        func(a, b Memory) bool
	DefaultImportance float32
	DecayRate         float32 // used for recency in hybrid scoring

	// ImportanceFn, when set, auto-scores memories remembered without an
	// explicit importance (e.g. ScoreImportance). Nil → DefaultImportance.
	ImportanceFn func(text string, memType MemoryType, tags []string) float32

	// Audit-journal options.
	Actor           string // "cli" | "mcp" | "lib" | "reflection" | "decay" | "import"
	JournalEnabled  bool   // default: true
	JournalPath     string // default: <Path>/../journal.log
	JournalRotateMB int    // default: 64
	JournalKeepDays int    // default: 90
}

func DefaultStoreConfig(path, collection string) StoreConfig {
	return StoreConfig{
		Path:              path,
		Collection:        collection,
		ConflictPolicy:    PolicyIgnore,
		ConflictThreshold: 0.80,
		DefaultImportance: 0.5,
		DecayRate:         0.1,
		Actor:             "lib",
		JournalEnabled:    true,
		JournalRotateMB:   64,
		JournalKeepDays:   90,
	}
}

// MemoryStore is the main interface to stored memories.
type MemoryStore struct {
	backend  vector.Backend
	embedder embed.Embedder
	cfg      StoreConfig
	journal  *Journal
	actor    string
}

// NewMemoryStore creates a new store backed by the given vector backend + embedder.
func NewMemoryStore(backend vector.Backend, embedder embed.Embedder, cfg StoreConfig) *MemoryStore {
	if cfg.Actor == "" {
		cfg.Actor = "lib"
	}
	s := &MemoryStore{backend: backend, embedder: embedder, cfg: cfg, actor: cfg.Actor}
	if cfg.JournalEnabled {
		jp := cfg.JournalPath
		if jp == "" {
			// Default sits next to the store directory.
			jp = filepath.Join(filepath.Dir(cfg.Path), "journal.log")
		}
		s.journal = NewJournal(jp, cfg.JournalRotateMB, cfg.JournalKeepDays)
	}
	return s
}

// Journal returns the underlying audit journal (may be nil if disabled).
func (s *MemoryStore) Journal() *Journal { return s.journal }

// Config returns the store's configuration (collection name, embedding
// model, …). Used by the export header.
func (s *MemoryStore) Config() StoreConfig { return s.cfg }

// AsActor temporarily attributes journal entries to *name*. The returned
// closure restores the previous actor — call it via `defer`.
//
//	defer store.AsActor("reflection")()
func (s *MemoryStore) AsActor(name string) func() {
	prev := s.actor
	s.actor = name
	return func() { s.actor = prev }
}

// nowFloat returns sub-second-precision time, matching Python's time.time().
func nowFloat() float64 {
	return float64(time.Now().UnixNano()) / 1e9
}

func (s *MemoryStore) journalEntry(op, id string, before, after, meta map[string]any) {
	if s.journal == nil {
		return
	}
	if meta == nil {
		meta = map[string]any{}
	}
	s.journal.Append(JournalEntry{
		TS: nowFloat(), Op: op, Actor: s.actor, ID: id,
		Before: before, After: after, Meta: meta,
	})
}

// Remember stores a new memory and returns it.
func (s *MemoryStore) Remember(ctx context.Context, text string, opts RememberOpts) (Memory, bool, []Conflict, error) {
	if opts.Type == "" {
		opts.Type = Episodic
	}
	// Importance 0 means "unset" → auto-score when an ImportanceFn is
	// configured, else keep the configured default.
	if opts.Importance == 0 {
		if s.cfg.ImportanceFn != nil {
			opts.Importance = s.cfg.ImportanceFn(text, opts.Type, opts.Tags)
		} else {
			opts.Importance = s.cfg.DefaultImportance
		}
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

	// Conflict check. A per-call OnConflict overrides the store default.
	policy := s.cfg.ConflictPolicy
	if opts.OnConflict != "" {
		policy = opts.OnConflict
	}
	if policy != PolicyIgnore {
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
			switch policy {
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
	s.journalEntry("remember", m.ID, nil, m.ToDict(), nil)
	return m, true, nil, nil
}

// RememberOpts are optional fields for Remember.
type RememberOpts struct {
	Type       MemoryType
	Tags       []string
	Importance float32
	Source     string
	Polarity   int
	// OnConflict overrides the store's configured conflict policy for this
	// call only (empty = use the store default).
	OnConflict ConflictPolicy
}

// Recall returns up to k memories matching query.
func (s *MemoryStore) Recall(ctx context.Context, query string, k int, opts RecallOpts) ([]MemoryWithScore, error) {
	if opts.Diversity != nil && (*opts.Diversity < 0 || *opts.Diversity > 1) {
		return nil, fmt.Errorf("diversity must be in [0, 1]")
	}
	if opts.DedupThreshold != nil && (*opts.DedupThreshold < 0 || *opts.DedupThreshold > 1) {
		return nil, fmt.Errorf("dedup_threshold must be in [0, 1]")
	}
	if opts.MinCosine != nil && (*opts.MinCosine < -1 || *opts.MinCosine > 1) {
		return nil, fmt.Errorf("min_cosine must be in [-1, 1]")
	}
	count, err := s.backend.Count(ctx)
	if err != nil {
		return nil, err
	}
	// Matches Python: a non-positive k or an empty store returns nothing.
	if k <= 0 || count == 0 {
		return nil, nil
	}

	needEmb := opts.Diversity != nil || opts.DedupThreshold != nil
	overfetch := opts.Overfetch
	if overfetch <= 0 {
		overfetch = 4
	}

	// Unlike Python (which pushes type/importance/source/time into Chroma's
	// where-clause), the Go backend filters entirely client-side, so ANY active
	// filter — not just tag — forces an over-fetch to avoid under-returning.
	noPostFilter := opts.Mode == ModeSemantic && opts.IncludeSuperseded &&
		opts.Tag == "" && opts.Type == "" && opts.MinImportance <= 0 &&
		opts.Source == "" && opts.Since == 0 && opts.Until == 0 &&
		!needEmb && opts.MinCosine == nil

	nFetch := k
	if !noPostFilter {
		nFetch = k * overfetch
		if nFetch > count {
			nFetch = count
		}
	}
	if nFetch < k {
		nFetch = k
	}

	vecs, err := s.embedder.Embed(ctx, []string{query})
	if err != nil {
		return nil, err
	}
	hits, err := s.backend.Query(ctx, vecs[0], nFetch)
	if err != nil {
		return nil, err
	}

	// The "where-clause" filters (type/importance/source/since/until) form the
	// scoring pool; tag/superseded/min_cosine are applied inside the scorers so
	// the BM25 pool matches Python's server-side-filtered document set.
	var pool []cand
	for _, h := range hits {
		m := MetadataToMemory(h.ID, h.Content, h.Metadata)
		if opts.Type != "" && m.Type != opts.Type {
			continue
		}
		if m.Importance < opts.MinImportance {
			continue
		}
		if opts.Source != "" && m.Source != opts.Source {
			continue
		}
		if opts.Since > 0 && m.CreatedAt < opts.Since {
			continue
		}
		if opts.Until > 0 && m.CreatedAt > opts.Until {
			continue
		}
		pool = append(pool, cand{mem: m, cosine: h.Similarity, emb: h.Embedding})
	}

	w := opts.Weights
	if w == (HybridWeights{}) {
		w = DefaultWeights()
	}

	var expl map[string]map[string]any
	if opts.Explain {
		expl = map[string]map[string]any{}
	}

	now := nowFloat()
	var scored []scoredCand
	if opts.Mode == ModeHybrid {
		docs := make([]string, len(pool))
		for i, c := range pool {
			docs[i] = c.mem.Text
		}
		bm25 := bm25Score(query, docs)
		if opts.Fusion == FusionRRF {
			scored = s.scoreRRF(pool, bm25, w, opts.Tag, opts.IncludeSuperseded, opts.MinCosine, expl, now, 60)
		} else {
			scored = s.scoreWeighted(pool, bm25, w, opts.Tag, opts.IncludeSuperseded, opts.MinCosine, expl, now)
		}
	} else {
		scored = scoreSemantic(pool, w, opts.Tag, opts.IncludeSuperseded, opts.MinCosine, expl)
	}

	var chosen []scoredCand
	if needEmb {
		chosen = mmrSelect(scored, k, opts.Diversity, opts.DedupThreshold)
	} else if len(scored) > k {
		chosen = scored[:k]
	} else {
		chosen = scored
	}

	out := make([]MemoryWithScore, 0, len(chosen))
	for _, sc := range chosen {
		mws := MemoryWithScore{Memory: sc.mem, Score: sc.score}
		if expl != nil {
			mws.Explain = expl[sc.mem.ID]
		}
		out = append(out, mws)
	}

	// Touch accessed memories (batched) unless read-only.
	if !opts.NoTouch && len(out) > 0 {
		mems := make([]*Memory, len(out))
		for i := range out {
			mems[i] = &out[i].Memory
		}
		if err := s.touchMany(ctx, mems); err != nil {
			log.Printf("ai-houkai: touch: %v", err)
		}
	}

	// Graph expansion.
	if opts.Expand != nil && opts.Expand.Cap > 0 {
		out, err = s.expandResults(ctx, out, opts.Expand, opts.IncludeSuperseded, expl)
		if err != nil {
			return nil, err
		}
	}

	return out, nil
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

	// Source keeps only memories whose provenance string matches exactly.
	Source string
	// Since/Until bound created_at (Unix seconds, inclusive). 0 = unbounded.
	Since float64
	Until float64

	// Advanced retrieval controls (all optional; zero values = disabled).
	Fusion         FusionMode // "" / "weighted" (default) | "rrf"
	Diversity      *float32   // MMR λ in [0,1]; nil disables MMR re-selection
	DedupThreshold *float32   // cosine hard-dedup floor in [0,1]; nil disables
	MinCosine      *float32   // absolute cosine relevance floor in [-1,1]; nil disables
	NoTouch        bool       // skip the access-count / last_accessed bump (read-only)
	Explain        bool       // populate MemoryWithScore.Explain with per-signal breakdowns
}

// Forget deletes a memory by ID. Returns true if found and deleted.
func (s *MemoryStore) Forget(ctx context.Context, id string) (bool, error) {
	items, err := s.backend.Get(ctx, []string{id})
	if err != nil || len(items) == 0 {
		return false, err
	}
	before := MetadataToMemory(items[0].ID, items[0].Content, items[0].Metadata)
	if err := s.backend.Delete(ctx, []string{id}); err != nil {
		return false, err
	}
	s.journalEntry("forget", id, before.ToDict(), nil, nil)
	return true, nil
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

// Nuke deletes every memory in the current collection (keeping the collection
// itself) and returns the count deleted, journalling a single "nuke" entry.
func (s *MemoryStore) Nuke(ctx context.Context) (int, error) {
	items, err := s.backend.All(ctx)
	if err != nil {
		return 0, err
	}
	if len(items) == 0 {
		return 0, nil
	}
	ids := make([]string, len(items))
	for i, it := range items {
		ids[i] = it.ID
	}
	if err := s.backend.Delete(ctx, ids); err != nil {
		return 0, err
	}
	s.journalEntry("nuke", "*", nil, nil, map[string]any{"count": len(ids)})
	return len(ids), nil
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
	added, err := s.linkRaw(ctx, srcID, dstID, rel)
	if err != nil {
		return err
	}
	if added {
		s.journalEntry("link", srcID, nil, nil, map[string]any{
			"src_id": srcID, "dst_id": dstID, "rel": rel,
		})
	}
	return nil
}

// linkRaw adds a link without journaling. Returns true iff a new edge was inserted.
func (s *MemoryStore) linkRaw(ctx context.Context, srcID, dstID, rel string) (bool, error) {
	if srcID == dstID {
		return false, fmt.Errorf("cannot link a memory to itself")
	}
	src, err := s.GetByID(ctx, srcID)
	if err != nil {
		return false, fmt.Errorf("link src: %w", err)
	}
	for _, l := range src.Links {
		if l.To == dstID && l.Rel == rel {
			return false, nil
		}
	}
	addLink(&src, dstID, rel)
	return true, s.backend.UpdateMetadata(ctx, src.ID, MemoryToMetadata(src))
}

func (s *MemoryStore) Unlink(ctx context.Context, srcID, dstID, rel string) (int, error) {
	removed, err := s.unlinkRaw(ctx, srcID, dstID, rel)
	if err != nil {
		return 0, err
	}
	if removed > 0 {
		s.journalEntry("unlink", srcID, nil, nil, map[string]any{
			"src_id": srcID, "dst_id": dstID, "rel": rel, "removed": removed,
		})
	}
	return removed, nil
}

// unlinkRaw removes link(s) without journaling.
func (s *MemoryStore) unlinkRaw(ctx context.Context, srcID, dstID, rel string) (int, error) {
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

	// Single-memory scan: check the given memory against its top-12 nearest
	// neighbours (mirrors Python's _check_conflicts).
	if memID != "" {
		m, err := s.GetByID(ctx, memID)
		if err != nil {
			return nil, err
		}
		vecs, err := s.embedder.Embed(ctx, []string{m.Text})
		if err != nil {
			return nil, err
		}
		hits, err := s.backend.Query(ctx, vecs[0], 12)
		if err != nil {
			return nil, err
		}
		return detectConflicts(m, hitsToMemoriesWithScore(hits), threshold, s.cfg.ConflictFn), nil
	}

	// Global scan: exact O(n²) all-pairs cosine over stored embeddings, matching
	// Python's find_conflicts(None) — the per-target top-12 approximation missed
	// pairs once many similar same-type memories existed.
	return s.findConflictsAll(ctx, threshold)
}

// findConflictsAll does the exact pairwise conflict scan over all active memories.
func (s *MemoryStore) findConflictsAll(ctx context.Context, threshold float32) ([]Conflict, error) {
	items, err := s.backend.All(ctx)
	if err != nil {
		return nil, err
	}
	type memEmb struct {
		mem Memory
		emb []float32
	}
	pool := make([]memEmb, 0, len(items))
	for _, it := range items {
		m := MetadataToMemory(it.ID, it.Content, it.Metadata)
		if m.SupersededBy != "" || len(it.Embedding) == 0 {
			continue
		}
		pool = append(pool, memEmb{mem: m, emb: it.Embedding})
	}

	var conflicts []Conflict
	for i := 0; i < len(pool); i++ {
		for j := i + 1; j < len(pool); j++ {
			a, b := pool[i].mem, pool[j].mem
			if a.Type != b.Type {
				continue
			}
			sim := vector.CosineSim(pool[i].emb, pool[j].emb)
			if sim < threshold {
				continue
			}
			if !tagsOverlap(a, b) {
				continue
			}
			kind, reason := classifyConflict(a, b, s.cfg.ConflictFn)
			conflicts = append(conflicts, Conflict{Kind: kind, Reason: reason, Similarity: sim, A: a, B: b})
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
	before := old.ToDict()
	now := nowFloat()
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
	s.journalEntry("supersede", oldID, before, old.ToDict(),
		map[string]any{"old_id": oldID, "new_id": newID})
	return nil
}

func (s *MemoryStore) Restore(ctx context.Context, memID string) error {
	m, err := s.GetByID(ctx, memID)
	if err != nil {
		return err
	}
	if m.SupersededBy == "" {
		return nil
	}
	before := m.ToDict()
	superseder := m.SupersededBy
	m.SupersededBy = ""
	m.SupersededAt = 0
	if err := s.backend.UpdateMetadata(ctx, m.ID, MemoryToMetadata(m)); err != nil {
		return err
	}
	// Remove the "supersedes" edge from the superseder, matching Python.
	_, _ = s.unlinkRaw(ctx, superseder, memID, RelSupersedes)
	s.journalEntry("restore", memID, before, m.ToDict(),
		map[string]any{"superseder_id": superseder})
	return nil
}

// touchMany bumps access tracking for several memories, mirroring Python's
// _touch_many (which collapses the writes into one Chroma update). The Go
// backend patches metadata one id at a time; the first error is returned but
// every memory is still attempted.
func (s *MemoryStore) touchMany(ctx context.Context, mems []*Memory) error {
	if len(mems) == 0 {
		return nil
	}
	now := float64(time.Now().Unix())
	var firstErr error
	for _, m := range mems {
		m.LastAccessed = now
		m.AccessCount++
		if err := s.backend.UpdateMetadata(ctx, m.ID, MemoryToMetadata(*m)); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
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

// expandResults performs breadth-first graph expansion of the recalled set,
// honouring spec.Depth (multi-hop) with a per-hop spec.Decay applied to the
// assigned spec.Score, mirroring Python's _expand_links.
func (s *MemoryStore) expandResults(ctx context.Context, out []MemoryWithScore, spec *ExpandSpec, includeSuperseded bool, expl map[string]map[string]any) ([]MemoryWithScore, error) {
	if spec == nil || spec.Cap <= 0 {
		return out, nil
	}
	decay := spec.Decay
	if decay == 0 {
		decay = 1.0
	}
	depth := spec.Depth
	if depth <= 0 {
		depth = 1
	}
	seen := make(map[string]bool, len(out))
	frontier := make([]string, 0, len(out))
	for _, r := range out {
		seen[r.ID] = true
		frontier = append(frontier, r.ID)
	}
	var extra []MemoryWithScore
	added := 0
	for hop := 1; hop <= depth; hop++ {
		if added >= spec.Cap || len(frontier) == 0 {
			break
		}
		hopScore := spec.Score * float32(math.Pow(float64(decay), float64(hop-1)))
		var next []string
		for _, mid := range frontier {
			if added >= spec.Cap {
				break
			}
			src, err := s.GetByID(ctx, mid)
			if err != nil {
				continue
			}
			for _, lnk := range src.Links {
				if !relAllowed(lnk.Rel, spec.Rels) {
					continue
				}
				if seen[lnk.To] {
					continue
				}
				nb, err := s.GetByID(ctx, lnk.To)
				if err != nil {
					continue
				}
				if !includeSuperseded && nb.SupersededBy != "" {
					continue
				}
				mws := MemoryWithScore{Memory: nb, Score: hopScore}
				if expl != nil {
					e := map[string]any{
						"source": "graph_expansion", "rel": lnk.Rel,
						"hop": hop, "score": round4(hopScore),
					}
					expl[nb.ID] = e
					mws.Explain = e
				}
				extra = append(extra, mws)
				seen[lnk.To] = true
				next = append(next, lnk.To)
				added++
				if added >= spec.Cap {
					break
				}
			}
		}
		frontier = next
	}
	return append(out, extra...), nil
}

// relAllowed reports whether rel is one of the expansion relationships. An
// empty rels list matches nothing (mirrors Python, whose ExpandSpec.rels
// defaults to a non-empty tuple).
func relAllowed(rel string, rels []string) bool {
	for _, r := range rels {
		if r == rel {
			return true
		}
	}
	return false
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

// Stats returns aggregate stats. Type/tag counts and average importance are
// computed over ACTIVE (non-superseded) memories, matching the Python CLI.
func (s *MemoryStore) Stats(ctx context.Context) (map[string]any, error) {
	mems, err := s.ListRecent(ctx, 0, true)
	if err != nil {
		return nil, err
	}

	typeCounts := map[string]int{}
	tagCounts := map[string]int{}
	var active, superseded int
	for _, m := range mems {
		if m.SupersededBy != "" {
			superseded++
			continue
		}
		active++
		typeCounts[string(m.Type)]++
		for _, t := range m.Tags {
			tagCounts[t]++
		}
	}

	// Base fields mirror the Python CLI stats dict exactly (avg_importance is
	// only reported inside the health block, not here).
	return map[string]any{
		"store_path":       s.cfg.Path,
		"collection":       s.cfg.Collection,
		"total":            len(mems),
		"active":           active,
		"superseded":       superseded,
		"by_type":          typeCounts,
		"top_tags":         topNMap(tagCounts, 15),
		"store_size_bytes": dirSize(s.cfg.Path),
	}, nil
}

// dirSize sums the byte sizes of all files under path (0 if it doesn't exist).
func dirSize(path string) int64 {
	var total int64
	_ = filepath.Walk(path, func(_ string, info os.FileInfo, err error) error {
		if err == nil && info != nil && !info.IsDir() {
			total += info.Size()
		}
		return nil
	})
	return total
}

// topNMap returns the n most-frequent tags as a {tag: count} map, matching
// Python's `dict(Counter.most_common(n))` JSON shape.
func topNMap(m map[string]int, n int) map[string]int {
	type kv struct {
		k string
		v int
	}
	sl := make([]kv, 0, len(m))
	for k, v := range m {
		sl = append(sl, kv{k, v})
	}
	sort.Slice(sl, func(i, j int) bool {
		if sl[i].v != sl[j].v {
			return sl[i].v > sl[j].v
		}
		return sl[i].k < sl[j].k // stable tie-break for determinism
	})
	out := map[string]int{}
	for i, x := range sl {
		if i >= n {
			break
		}
		out[x.k] = x.v
	}
	return out
}
