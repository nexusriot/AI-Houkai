package memory

import (
	"context"
	"errors"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"sync"
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

// Reranker rescores a recall candidate pool with a stronger (usually
// cross-encoder) relevance model. It receives the query and the candidate
// memories (already filtered + first-stage ranked) and returns one score per
// memory, in the same order; higher = more relevant. Recall re-sorts by the
// returned scores and, in explain mode, records the first-stage score/rank.
type Reranker func(query string, mems []Memory) []float32

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

	// Reranker, when set, is the default second-stage reranker applied to the
	// recall candidate pool (per-call RecallOpts.Reranker overrides it).
	Reranker Reranker

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

	// Process-local runtime metrics (not persisted; reset on restart). Guarded
	// by metricMu since HTTP/MCP handlers may call the store concurrently.
	metricMu      sync.Mutex
	metricStarted time.Time
	metricCalls   map[string]int
	recallCount   int
	recallTotalS  float64
	recallMaxS    float64
}

// NewMemoryStore creates a new store backed by the given vector backend + embedder.
func NewMemoryStore(backend vector.Backend, embedder embed.Embedder, cfg StoreConfig) *MemoryStore {
	if cfg.Actor == "" {
		cfg.Actor = "lib"
	}
	s := &MemoryStore{
		backend: backend, embedder: embedder, cfg: cfg, actor: cfg.Actor,
		metricStarted: time.Now(),
		metricCalls: map[string]int{
			"remember": 0, "recall": 0, "forget": 0, "edit": 0, "supersede": 0,
		},
	}
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

// Remember stores a new memory and returns it. The text is stripped of
// surrounding whitespace before storage (matching Python).
func (s *MemoryStore) Remember(ctx context.Context, text string, opts RememberOpts) (Memory, bool, []Conflict, error) {
	text = strings.TrimSpace(text)
	if opts.Type == "" {
		opts.Type = Semantic
	}
	if err := validateChoice(string(opts.Type), MemoryTypes, "type"); err != nil {
		return Memory{}, false, nil, err
	}
	if err := validatePolarity(opts.Polarity); err != nil {
		return Memory{}, false, nil, err
	}
	if opts.OnConflict != "" {
		if err := validateChoice(string(opts.OnConflict), ConflictPolicies, "on_conflict"); err != nil {
			return Memory{}, false, nil, err
		}
	}
	if err := validateTags(opts.Tags); err != nil {
		return Memory{}, false, nil, err
	}

	// Expiry: absolute epoch (ExpiresAt) or relative (TTLSeconds from now).
	// nil / 0 means "never expires". They are mutually exclusive.
	var expiresAt float64
	if opts.ExpiresAt != nil && opts.TTLSeconds != nil {
		return Memory{}, false, nil, validationErrorf("pass at most one of ExpiresAt, TTLSeconds")
	}
	if opts.TTLSeconds != nil {
		if *opts.TTLSeconds <= 0 {
			return Memory{}, false, nil, validationErrorf("ttl_seconds must be > 0")
		}
		expiresAt = nowFloat() + *opts.TTLSeconds
	} else if opts.ExpiresAt != nil {
		if *opts.ExpiresAt < 0 {
			return Memory{}, false, nil, validationErrorf("expires_at must be >= 0")
		}
		expiresAt = *opts.ExpiresAt
	}

	// Importance nil means "unset" → auto-score when an ImportanceFn is
	// configured, else keep the configured default. An explicit value —
	// including 0 — is honoured, clamped to [0, 1].
	var importance float32
	if opts.Importance == nil {
		if s.cfg.ImportanceFn != nil {
			importance = s.cfg.ImportanceFn(text, opts.Type, opts.Tags)
		} else {
			importance = s.cfg.DefaultImportance
		}
	} else {
		importance = clamp01(*opts.Importance)
	}

	// Sub-second precision, matching Python's time.time(): whole-second
	// timestamps make same-second remembers tie, so ListRecent order flaps.
	now := nowFloat()
	m := Memory{
		ID:           uuid.NewString(),
		Text:         text,
		Type:         opts.Type,
		Tags:         opts.Tags,
		Importance:   importance,
		CreatedAt:    now,
		LastAccessed: now,
		AccessCount:  0,
		Source:       opts.Source,
		Links:        []Link{},
		SupersededBy: "",
		SupersededAt: 0,
		Polarity:     opts.Polarity,
		ExpiresAt:    expiresAt,
	}
	if m.Tags == nil {
		m.Tags = []string{}
	}

	// Embed once — the same vector serves the insert and the conflict scan.
	vecs, err := s.embedder.Embed(ctx, []string{text})
	if err != nil {
		return Memory{}, false, nil, err
	}

	// Add FIRST, then detect conflicts (matching Python): PolicySupersede can
	// link the stored memories both ways, and PolicyRaise rolls the insert
	// back so a failed Add can never leave old memories superseded_by an id
	// that was never created.
	err = s.backend.Add(ctx, []vector.Item{{
		ID:        m.ID,
		Content:   text,
		Embedding: vecs[0],
		Metadata:  MemoryToMetadata(m),
	}})
	if err != nil {
		return Memory{}, false, nil, err
	}
	s.recordCall("remember")
	s.journalEntry("remember", m.ID, nil, m.ToDict(), nil)

	// Conflict check. A per-call OnConflict overrides the store default.
	policy := s.cfg.ConflictPolicy
	if opts.OnConflict != "" {
		policy = opts.OnConflict
	}
	if policy != PolicyIgnore {
		hits, err := s.backend.Query(ctx, vecs[0], 12)
		if err != nil {
			// The memory is stored; surface the scan failure (like Python,
			// where an exception after collection.add leaves the row in).
			return m, true, nil, fmt.Errorf("conflict scan: %w", err)
		}
		conflicts := detectConflicts(m, hitsToMemoriesWithScore(hits), s.cfg.ConflictThreshold, s.cfg.ConflictFn)

		if len(conflicts) > 0 {
			switch policy {
			case PolicyWarn:
				log.Printf("ai-houkai: %d conflict(s) detected for new memory", len(conflicts))
			case PolicySupersede:
				for _, c := range conflicts {
					// A failed supersede must surface (Python's exception
					// propagates); the memory itself IS stored.
					if err := s.doSupersede(ctx, c.B.ID, m.ID); err != nil {
						return m, true, nil, fmt.Errorf("supersede conflict %s: %w", c.B.ID, err)
					}
				}
			case PolicyRaise:
				// Roll back the just-added memory — the caller is told it was
				// not stored, so it must not linger in the backend.
				_, _ = s.Forget(ctx, m.ID)
				return Memory{}, false, conflicts, &ConflictError{Conflicts: conflicts}
			}
		}
	}

	return m, true, nil, nil
}

// RememberOpts are optional fields for Remember.
type RememberOpts struct {
	Type MemoryType
	Tags []string
	// Importance nil = unset (auto-score / store default); an explicit value
	// — including 0 — is honoured and clamped to [0, 1]. Use Float32Ptr.
	Importance *float32
	Source     string
	Polarity   int
	// ExpiresAt / TTLSeconds set an optional TTL: an absolute epoch (ExpiresAt)
	// or a relative lifetime in seconds (TTLSeconds, from now). nil = never
	// expires. Pass at most one.
	ExpiresAt  *float64
	TTLSeconds *float64
	// OnConflict overrides the store's configured conflict policy for this
	// call only (empty = use the store default).
	OnConflict ConflictPolicy
}

// Recall returns up to k memories matching query.
func (s *MemoryStore) Recall(ctx context.Context, query string, k int, opts RecallOpts) ([]MemoryWithScore, error) {
	if opts.Mode == "" {
		opts.Mode = ModeSemantic
	}
	if err := validateChoice(string(opts.Mode), RecallModes, "mode"); err != nil {
		return nil, err
	}
	if opts.Fusion == "" {
		opts.Fusion = FusionWeighted
	}
	if err := validateChoice(string(opts.Fusion), Fusions, "fusion"); err != nil {
		return nil, err
	}
	if opts.Type != "" {
		if err := validateChoice(string(opts.Type), MemoryTypes, "type"); err != nil {
			return nil, err
		}
	}
	if opts.Diversity != nil && (*opts.Diversity < 0 || *opts.Diversity > 1) {
		return nil, validationErrorf("diversity must be in [0, 1]")
	}
	if opts.DedupThreshold != nil && (*opts.DedupThreshold < 0 || *opts.DedupThreshold > 1) {
		return nil, validationErrorf("dedup_threshold must be in [0, 1]")
	}
	if opts.MinCosine != nil && (*opts.MinCosine < -1 || *opts.MinCosine > 1) {
		return nil, validationErrorf("min_cosine must be in [-1, 1]")
	}

	t0 := time.Now()
	s.recordCall("recall")
	defer func() { s.recordRecallLatency(time.Since(t0)) }()

	reranker := opts.Reranker
	if reranker == nil {
		reranker = s.cfg.Reranker
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
	// Expiry filtering and reranking both reorder/drop rows, so they force it too.
	noPostFilter := opts.Mode == ModeSemantic && opts.IncludeSuperseded &&
		opts.IncludeExpired && reranker == nil &&
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

	now := nowFloat()

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
		// Drop expired unless asked to keep them. Post-filter (not a backend
		// query filter), so rows written before TTL existed (ExpiresAt 0) are
		// unaffected.
		if !opts.IncludeExpired && m.ExpiresAt > 0 && m.ExpiresAt <= now {
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

	// Second-stage rerank over the surviving pool, before the top-k cut.
	if reranker != nil && len(scored) > 0 {
		scored, err = applyReranker(reranker, query, scored, expl)
		if err != nil {
			return nil, err
		}
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
	// IncludeExpired keeps memories whose ExpiresAt has passed (hidden by
	// default, like superseded).
	IncludeExpired bool
	// Reranker overrides the store's default reranker for this call: it
	// rescores the first-stage pool before the top-k cut. nil = store default.
	Reranker Reranker
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
	s.recordCall("forget")
	s.journalEntry("forget", id, before.ToDict(), nil, nil)
	return true, nil
}

// EditOpts selects the fields Edit changes. Nil pointers leave a field
// unchanged. Tags: nil = unchanged, an empty non-nil slice clears. Source:
// nil = unchanged, a pointer to "" clears (the Go analogue of Python's
// sentinel-vs-None tri-state).
type EditOpts struct {
	Text       *string
	Type       *MemoryType
	Tags       []string
	Importance *float32 // clamped to [0, 1]
	Polarity   *int
	// ExpiresAt: nil = unchanged; a pointer to 0 clears the TTL.
	ExpiresAt *float64
	Source    *string
}

// Edit updates fields of an existing memory in place, keeping its id.
//
// Text changes re-embed the document. Links, superseded_by, created_at, and
// access tracking are preserved — unlike a Forget+Remember round-trip. The
// change is journaled (op "edit", with before/after snapshots) and reversible
// via Undo. A call that changes nothing is a no-op: no write, no journal
// entry.
//
// Returns ErrNotFound (wrapped) if the memory does not exist and a
// ValidationError on a bad type / polarity / tags / empty text.
func (s *MemoryStore) Edit(ctx context.Context, memoryID string, opts EditOpts) (Memory, error) {
	mem, err := s.GetByID(ctx, memoryID)
	if err != nil {
		return Memory{}, fmt.Errorf("memory_id %q: %w", memoryID, err)
	}
	before := mem.ToDict()

	if opts.Type != nil {
		if err := validateChoice(string(*opts.Type), MemoryTypes, "type"); err != nil {
			return Memory{}, err
		}
		mem.Type = *opts.Type
	}
	if opts.Polarity != nil {
		if err := validatePolarity(*opts.Polarity); err != nil {
			return Memory{}, err
		}
		mem.Polarity = *opts.Polarity
	}
	if opts.Importance != nil {
		mem.Importance = clamp01(*opts.Importance)
	}
	if opts.ExpiresAt != nil {
		if *opts.ExpiresAt < 0 {
			return Memory{}, validationErrorf("expires_at must be >= 0")
		}
		mem.ExpiresAt = *opts.ExpiresAt
	}
	if opts.Tags != nil {
		if err := validateTags(opts.Tags); err != nil {
			return Memory{}, err
		}
		mem.Tags = append([]string{}, opts.Tags...)
	}
	if opts.Source != nil {
		mem.Source = *opts.Source
	}
	textChanged := false
	if opts.Text != nil {
		newText := strings.TrimSpace(*opts.Text)
		if newText == "" {
			return Memory{}, validationErrorf("text must be non-empty")
		}
		textChanged = newText != mem.Text
		mem.Text = newText
	}

	after := mem.ToDict()
	if reflect.DeepEqual(after, before) {
		return mem, nil // nothing to do — keep the journal quiet
	}
	if err := s.UpdateMemory(ctx, mem, textChanged); err != nil {
		return Memory{}, err
	}
	s.recordCall("edit")
	s.journalEntry("edit", mem.ID, before, after, nil)
	return mem, nil
}

// ListRecent returns up to limit memories sorted by created_at desc.
func (s *MemoryStore) ListRecent(ctx context.Context, limit int, includeSuperseded, includeExpired bool) ([]Memory, error) {
	items, err := s.backend.All(ctx)
	if err != nil {
		return nil, err
	}
	now := nowFloat()
	var mems []Memory
	for _, it := range items {
		m := MetadataToMemory(it.ID, it.Content, it.Metadata)
		if !includeSuperseded && m.SupersededBy != "" {
			continue
		}
		if !includeExpired && m.ExpiresAt > 0 && m.ExpiresAt <= now {
			continue
		}
		mems = append(mems, m)
	}
	// Stable: same-created_at rows keep insertion order instead of flapping
	// between calls (export claims byte-identical output for a fixed store).
	sort.SliceStable(mems, func(i, j int) bool {
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

func (s *MemoryStore) recordCall(name string) {
	s.metricMu.Lock()
	s.metricCalls[name]++
	s.metricMu.Unlock()
}

func (s *MemoryStore) recordRecallLatency(d time.Duration) {
	sec := d.Seconds()
	s.metricMu.Lock()
	s.recallCount++
	s.recallTotalS += sec
	if sec > s.recallMaxS {
		s.recallMaxS = sec
	}
	s.metricMu.Unlock()
}

// Metrics returns process-local runtime metrics since this store was created.
// Complements Stats (content aggregates) with operational counters and recall
// latency. Per-instance and in-memory — not persisted, reset on restart.
func (s *MemoryStore) Metrics(ctx context.Context) (map[string]any, error) {
	count, err := s.backend.Count(ctx)
	if err != nil {
		return nil, err
	}
	s.metricMu.Lock()
	calls := make(map[string]int, len(s.metricCalls))
	for k, v := range s.metricCalls {
		calls[k] = v
	}
	n := s.recallCount
	avgMs := 0.0
	if n > 0 {
		avgMs = math.Round(s.recallTotalS/float64(n)*1e3*1e3) / 1e3
	}
	maxMs := math.Round(s.recallMaxS * 1e3 * 1e3) / 1e3
	s.metricMu.Unlock()
	return map[string]any{
		"uptime_seconds": math.Round(time.Since(s.metricStarted).Seconds()*1e3) / 1e3,
		"count":          count,
		"calls":          calls,
		"recall_latency_ms": map[string]any{
			"count": n,
			"avg":   avgMs,
			"max":   maxMs,
		},
	}, nil
}

// PurgeExpired hard-deletes memories whose TTL has passed, returning those
// purged. Expired memories are already hidden from recall/list; this reclaims
// their storage. Unlike decay's Prune it ignores protect-types — an explicit
// TTL is a stronger signal. Each deletion is journaled per-row (actor "purge").
// dryRun reports without deleting. A now of 0 uses the current time.
func (s *MemoryStore) PurgeExpired(ctx context.Context, now float64, dryRun bool) ([]Memory, error) {
	if now == 0 {
		now = nowFloat()
	}
	items, err := s.backend.All(ctx)
	if err != nil {
		return nil, err
	}
	var expired []Memory
	for _, it := range items {
		m := MetadataToMemory(it.ID, it.Content, it.Metadata)
		if m.ExpiresAt > 0 && m.ExpiresAt <= now {
			expired = append(expired, m)
		}
	}
	if !dryRun && len(expired) > 0 {
		restore := s.AsActor("purge")
		defer restore()
		for _, m := range expired {
			if _, err := s.Forget(ctx, m.ID); err != nil {
				return expired, err
			}
		}
	}
	return expired, nil
}

// entryTouches reports whether a journal entry concerns memoryID — as the op's
// subject (entry.ID) or as a link target / supersede counterpart recorded only
// in meta (dst_id / new_id / superseder_id).
func entryTouches(e JournalEntry, memoryID string) bool {
	if e.ID == memoryID {
		return true
	}
	for _, key := range []string{"dst_id", "new_id", "superseder_id"} {
		if v, ok := e.Meta[key].(string); ok && v == memoryID {
			return true
		}
	}
	return false
}

// History returns every journaled event touching memoryID, oldest first.
// Covers the memory as an op's subject and as a link/supersede counterpart.
// Store-wide "nuke" events are omitted (use JournalTail for those). Bounded by
// journal retention.
func (s *MemoryStore) History(ctx context.Context, memoryID string) ([]JournalEntry, error) {
	if s.journal == nil {
		return nil, nil
	}
	all, err := s.journal.Read(ReadOpts{IncludeArchives: true})
	if err != nil {
		return nil, err
	}
	out := []JournalEntry{}
	for _, e := range all {
		if entryTouches(e, memoryID) {
			out = append(out, e)
		}
	}
	return out, nil
}

// replayLink applies a link/unlink journal delta to a reconstructed state map.
func replayLink(state map[string]map[string]any, meta map[string]any, add bool) {
	src, _ := meta["src_id"].(string)
	dst, _ := meta["dst_id"].(string)
	rel, _ := meta["rel"].(string)
	d, ok := state[src]
	if src == "" || !ok {
		return
	}
	links, _ := d["links"].([]any)
	if add {
		for _, l := range links {
			if lm, ok := l.(map[string]any); ok && lm["to"] == dst && lm["rel"] == rel {
				return // already present
			}
		}
		d["links"] = append(links, map[string]any{"to": dst, "rel": rel})
		return
	}
	// unlink: a rel="" unlink removes several rels, recorded in removed_rels.
	removed := map[string]bool{}
	if rr, ok := meta["removed_rels"].([]any); ok {
		for _, r := range rr {
			if rs, ok := r.(string); ok {
				removed[rs] = true
			}
		}
	} else if rel != "" {
		removed[rel] = true
	}
	kept := []any{}
	for _, l := range links {
		lm, _ := l.(map[string]any)
		to, _ := lm["to"].(string)
		lrel, _ := lm["rel"].(string)
		matchAll := len(removed) == 0 && rel == ""
		if to == dst && (matchAll || removed[lrel]) {
			continue
		}
		kept = append(kept, l)
	}
	d["links"] = kept
}

// StateAt reconstructs the live memories as of timestamp by replaying the
// journal. Best-effort audit tool, not a source of record: it sees only
// journaled snapshots, so it is accurate back to the oldest retained entry, a
// "nuke" in the window clears reconstructed state, and memories written with
// journaling disabled are invisible. Reconstructed memories carry no embedding.
func (s *MemoryStore) StateAt(ctx context.Context, timestamp float64) ([]Memory, error) {
	if s.journal == nil {
		return nil, nil
	}
	entries, err := s.journal.Read(ReadOpts{Until: timestamp, IncludeArchives: true})
	if err != nil {
		return nil, err
	}
	state := map[string]map[string]any{}
	for _, e := range entries {
		switch e.Op {
		case "remember", "import", "edit", "supersede", "restore":
			if e.After != nil {
				state[e.ID] = e.After
			}
		case "forget":
			delete(state, e.ID)
		case "nuke":
			state = map[string]map[string]any{}
		case "link":
			replayLink(state, e.Meta, true)
		case "unlink":
			replayLink(state, e.Meta, false)
		case "undo":
			if e.After != nil {
				state[e.ID] = e.After
			} else if ofOp, _ := e.Meta["of_op"].(string); ofOp == "remember" {
				delete(state, e.ID)
			}
		}
	}
	out := make([]Memory, 0, len(state))
	for _, d := range state {
		out = append(out, MemoryFromDict(d))
	}
	return out, nil
}

// GetAt reconstructs a single memory as of timestamp (see StateAt). Returns nil
// if the memory did not exist then.
func (s *MemoryStore) GetAt(ctx context.Context, memoryID string, timestamp float64) (*Memory, error) {
	mems, err := s.StateAt(ctx, timestamp)
	if err != nil {
		return nil, err
	}
	for i := range mems {
		if mems[i].ID == memoryID {
			return &mems[i], nil
		}
	}
	return nil, nil
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

// linkRaw adds a link without journaling. Returns true iff a new edge was
// inserted. Rejects unknown rels, self-links, and dangling endpoints (graph
// walkers skip targets that don't resolve, so a dangling edge would be stored
// but unreachable — matching Python).
func (s *MemoryStore) linkRaw(ctx context.Context, srcID, dstID, rel string) (bool, error) {
	if err := validateChoice(rel, LinkRels, "rel"); err != nil {
		return false, err
	}
	if srcID == dstID {
		return false, validationErrorf("cannot link a memory to itself")
	}
	src, err := s.GetByID(ctx, srcID)
	if err != nil {
		return false, fmt.Errorf("src_id %q: %w", srcID, err)
	}
	dst, err := s.GetByID(ctx, dstID)
	if err != nil {
		return false, fmt.Errorf("dst_id %q: %w", dstID, err)
	}
	if src.ID == dst.ID {
		return false, validationErrorf("cannot link a memory to itself")
	}
	for _, l := range src.Links {
		if l.To == dst.ID && l.Rel == rel {
			return false, nil
		}
	}
	addLink(&src, dst.ID, rel)
	return true, s.backend.UpdateMetadata(ctx, src.ID, MemoryToMetadata(src))
}

func (s *MemoryStore) Unlink(ctx context.Context, srcID, dstID, rel string) (int, error) {
	removedRels, err := s.unlinkRaw(ctx, srcID, dstID, rel)
	if err != nil {
		return 0, err
	}
	if len(removedRels) > 0 {
		// removed_rels is what makes the entry undoable: a rel="" unlink may
		// drop several differently-typed edges at once.
		s.journalEntry("unlink", srcID, nil, nil, map[string]any{
			"src_id": srcID, "dst_id": dstID, "rel": rel,
			"removed": len(removedRels), "removed_rels": removedRels,
		})
	}
	return len(removedRels), nil
}

// unlinkRaw removes matching link(s) without journaling and returns the
// removed rels. A missing src is a no-op (matching Python).
func (s *MemoryStore) unlinkRaw(ctx context.Context, srcID, dstID, rel string) ([]string, error) {
	if rel != "" {
		if err := validateChoice(rel, LinkRels, "rel"); err != nil {
			return nil, err
		}
	}
	src, err := s.GetByID(ctx, srcID)
	if err != nil {
		if errors.Is(err, ErrNotFound) {
			return nil, nil
		}
		return nil, fmt.Errorf("src_id %q: %w", srcID, err)
	}
	// Links store full ids; resolve a dst prefix when possible.
	dst := dstID
	if d, err := s.GetByID(ctx, dstID); err == nil {
		dst = d.ID
	}
	var removedRels []string
	for _, l := range src.Links {
		if l.To == dst && (rel == "" || l.Rel == rel) {
			removedRels = append(removedRels, l.Rel)
		}
	}
	if len(removedRels) > 0 {
		removeLinks(&src, dst, rel)
		if err := s.backend.UpdateMetadata(ctx, src.ID, MemoryToMetadata(src)); err != nil {
			return nil, err
		}
	}
	return removedRels, nil
}

type NeighborResult struct {
	Memory
	Rel string `json:"rel"`
}

// Neighbors returns (memory, rel) pairs reachable from memID via links.
// Two memories may be joined by several differently-typed edges — each rel is
// reported as its own pair, but the node is visited/expanded only once
// (matching Python).
func (s *MemoryStore) Neighbors(ctx context.Context, memID, rel, direction string, depth int) ([]NeighborResult, error) {
	if err := validateChoice(direction, Directions, "direction"); err != nil {
		return nil, err
	}
	if rel != "" {
		if err := validateChoice(rel, LinkRels, "rel"); err != nil {
			return nil, err
		}
	}
	if depth <= 0 {
		depth = 1
	}
	// Links store full ids; resolve a starting prefix when possible.
	root := memID
	if m, err := s.GetByID(ctx, memID); err == nil {
		root = m.ID
	}
	visited := map[string]bool{root: true}
	var results []NeighborResult
	queue := []string{root}

	for d := 0; d < depth && len(queue) > 0; d++ {
		var next []string
		for _, qid := range queue {
			// Outgoing links.
			if direction == "out" || direction == "both" {
				m, err := s.GetByID(ctx, qid)
				if err == nil {
					for _, l := range m.Links {
						if rel != "" && l.Rel != rel {
							continue
						}
						if visited[l.To] {
							continue
						}
						nb, err := s.GetByID(ctx, l.To)
						if err != nil {
							continue
						}
						// Report one pair per parallel edge to this target.
						for _, parallel := range m.Links {
							if parallel.To == l.To && (rel == "" || parallel.Rel == rel) {
								results = append(results, NeighborResult{Memory: nb, Rel: parallel.Rel})
							}
						}
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
					var matched []string
					for _, l := range other.Links {
						if l.To == qid && (rel == "" || l.Rel == rel) {
							matched = append(matched, l.Rel)
						}
					}
					if len(matched) > 0 {
						for _, r := range matched {
							results = append(results, NeighborResult{Memory: other, Rel: r})
						}
						visited[other.ID] = true
						next = append(next, other.ID)
					}
				}
			}
		}
		queue = next
	}
	return results, nil
}

// Subgraph returns the graph of memories reachable from the seed ids within
// depth hops, following OUTGOING links (matching Python's subgraph). Every
// seed is expanded; zero seeds yield an empty graph. A per-node "best
// remaining budget" is tracked instead of a plain visited set so diamonds are
// not truncated: a node first reached with 0 remaining hops is expanded again
// when a shorter path reaches it with budget to spare.
func (s *MemoryStore) Subgraph(ctx context.Context, ids []string, depth int) (Graph, error) {
	var g Graph
	bestRemaining := map[string]int{}
	inNodes := map[string]bool{}
	type edgeKey struct{ from, to, rel string }
	edgeSeen := map[edgeKey]bool{}

	var visit func(mid string, remaining int)
	visit = func(mid string, remaining int) {
		m, err := s.GetByID(ctx, mid)
		if err != nil {
			return
		}
		if prev, ok := bestRemaining[m.ID]; ok && prev >= remaining {
			return
		}
		bestRemaining[m.ID] = remaining
		if !inNodes[m.ID] {
			inNodes[m.ID] = true
			g.Nodes = append(g.Nodes, m)
		}
		if remaining > 0 {
			for _, l := range m.Links {
				k := edgeKey{m.ID, l.To, l.Rel}
				if !edgeSeen[k] {
					edgeSeen[k] = true
					g.Edges = append(g.Edges, struct {
						From string
						To   string
						Rel  string
					}{From: m.ID, To: l.To, Rel: l.Rel})
				}
				visit(l.To, remaining-1)
			}
		}
	}
	for _, mid := range ids {
		visit(mid, depth)
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

// Supersede marks oldID as superseded by newID and adds a single 'supersedes'
// link on the NEW memory pointing at the old one (matching Python — the old
// memory carries no edge). Rejects self-supersede, unknown ids, and cycles;
// re-superseding by the same id is an idempotent no-op.
func (s *MemoryStore) Supersede(ctx context.Context, oldID, newID string) error {
	return s.doSupersede(ctx, oldID, newID)
}

func (s *MemoryStore) doSupersede(ctx context.Context, oldID, newID string) error {
	if oldID == newID {
		return validationErrorf("cannot supersede a memory with itself")
	}
	old, err := s.GetByID(ctx, oldID)
	if err != nil {
		return fmt.Errorf("old_id %q: %w", oldID, err)
	}
	newMem, err := s.GetByID(ctx, newID)
	if err != nil {
		return fmt.Errorf("new_id %q: %w", newID, err)
	}
	if old.ID == newMem.ID {
		return validationErrorf("cannot supersede a memory with itself")
	}
	if newMem.SupersededBy == old.ID {
		return validationErrorf("cycle: %s is already superseded by %s", newMem.ID, old.ID)
	}
	if old.SupersededBy == newMem.ID {
		return nil // idempotent
	}

	before := old.ToDict()
	old.SupersededBy = newMem.ID
	old.SupersededAt = nowFloat()
	if err := s.backend.UpdateMetadata(ctx, old.ID, MemoryToMetadata(old)); err != nil {
		return err
	}
	if _, err := s.linkRaw(ctx, newMem.ID, old.ID, RelSupersedes); err != nil {
		return err
	}
	s.recordCall("supersede")
	s.journalEntry("supersede", old.ID, before, old.ToDict(),
		map[string]any{"old_id": old.ID, "new_id": newMem.ID})
	return nil
}

// Restore clears the superseded_by marker on a memory (undo a supersede) and
// removes the superseder's 'supersedes' edge. Returns false when the memory
// does not exist or is not superseded (matching Python).
func (s *MemoryStore) Restore(ctx context.Context, memID string) (bool, error) {
	m, err := s.GetByID(ctx, memID)
	if err != nil {
		if errors.Is(err, ErrNotFound) {
			return false, nil
		}
		return false, err
	}
	if m.SupersededBy == "" {
		return false, nil
	}
	before := m.ToDict()
	superseder := m.SupersededBy
	m.SupersededBy = ""
	m.SupersededAt = 0
	if err := s.backend.UpdateMetadata(ctx, m.ID, MemoryToMetadata(m)); err != nil {
		return false, err
	}
	// Remove the "supersedes" edge from the superseder, matching Python.
	_, _ = s.unlinkRaw(ctx, superseder, m.ID, RelSupersedes)
	s.journalEntry("restore", m.ID, before, m.ToDict(),
		map[string]any{"superseder_id": superseder})
	return true, nil
}

// touchMany bumps access tracking for several memories, mirroring Python's
// _touch_many (which collapses the writes into one Chroma update). The Go
// backend patches metadata one id at a time; the first error is returned but
// every memory is still attempted.
func (s *MemoryStore) touchMany(ctx context.Context, mems []*Memory) error {
	if len(mems) == 0 {
		return nil
	}
	now := nowFloat()
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
	// includeExpired=true: expired-but-not-yet-purged memories are still in the
	// store, so they count toward totals.
	mems, err := s.ListRecent(ctx, 0, true, true)
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
