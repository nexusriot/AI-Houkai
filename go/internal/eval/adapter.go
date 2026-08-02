package eval

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/nexusriot/ai-houkai/internal/memory"
)

// StoreAdapter bridges *memory.MemoryStore to the Recaller interface.
//
// It lives here rather than in the memory package so eval stays a leaf: the
// store knows nothing about evaluation, and evaluation depends on the store
// only through this one file.
type StoreAdapter struct{ Store *memory.MemoryStore }

// Recall forwards to the store with NoTouch always set — evaluating must never
// perturb access-count or recency, or the second run of the same gold set
// would score differently from the first.
func (a StoreAdapter) Recall(ctx context.Context, query string, k int, o RecallOpts) ([]Hit, error) {
	opts := memory.RecallOpts{
		Mode:           memory.RecallMode(o.Mode),
		Fusion:         memory.FusionMode(o.Fusion),
		Diversity:      o.Diversity,
		DedupThreshold: o.DedupThreshold,
		MinCosine:      o.MinCosine,
		NoTouch:        true,
	}
	if o.Graph != nil {
		// Start from the defaults: a bare Graph would leave the core weights
		// zeroed and Recall rejects that.
		w := memory.DefaultWeights()
		w.Graph = *o.Graph
		opts.Weights = w
	}
	if o.ExpandRerank {
		opts.Expand = &memory.ExpandSpec{
			Rels:  []string{"refines", "example_of"},
			Depth: 1, Cap: 5, Score: 0.70, Decay: 1.0, Rerank: true,
		}
	}
	res, err := a.Store.Recall(ctx, query, k, opts)
	if err != nil {
		return nil, err
	}
	hits := make([]Hit, len(res))
	for i, r := range res {
		hits[i] = Hit{ID: r.ID}
	}
	return hits, nil
}

// LoadGoldset parses a JSONL gold set — one case per line. Blank lines and
// lines starting with '#' are skipped. Byte-compatible with the Python CLI's
// format so one gold set serves both ports.
func LoadGoldset(path string) ([]Case, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var cases []Case
	for i, line := range strings.Split(string(raw), "\n") {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" || strings.HasPrefix(trimmed, "#") {
			continue
		}
		var c Case
		if err := json.Unmarshal([]byte(trimmed), &c); err != nil {
			return nil, fmt.Errorf("%s:%d: invalid JSON — %w", path, i+1, err)
		}
		if c.Query == "" {
			return nil, fmt.Errorf("%s:%d: missing 'query'", path, i+1)
		}
		if len(c.RelevantIDs) == 0 {
			return nil, fmt.Errorf("%s:%d: 'relevant_ids' must be a non-empty list", path, i+1)
		}
		cases = append(cases, c)
	}
	if len(cases) == 0 {
		return nil, fmt.Errorf("%s: no evaluation cases found", path)
	}
	return cases, nil
}
