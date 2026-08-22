// Package eval scores retrieval quality against a gold set.
//
// A port of Python's ai_houkai/eval.py, metric-for-metric, so the two ports
// report the same numbers for the same store and gold set. Dependency-free,
// matching the project's local-first philosophy.
//
// The metric functions also work standalone on any ranked list of ids:
//
//	RecallAtK([]string{"a", "b", "c"}, []string{"b"}, 2)  // -> 1.0
package eval

import (
	"context"
	"math"
	"sort"
)

// Recaller is the subset of MemoryStore that Evaluate needs. Keeping it an
// interface lets tests score a canned ranking without a live store.
type Recaller interface {
	Recall(ctx context.Context, query string, k int, opts RecallOpts) ([]Hit, error)
}

// RecallOpts and Hit mirror the store's shapes without importing it, so the
// eval package stays a leaf (memory imports nothing from here).
type RecallOpts struct {
	Mode           string
	Fusion         string
	Graph          *float32
	Diversity      *float32
	DedupThreshold *float32
	MinCosine      *float32
	ExpandRerank   bool
}

// Hit is one ranked result; only the id matters for ranking metrics.
type Hit struct {
	ID string
}

// RecallAtK is the fraction of relevant ids appearing in the top k.
// Counts distinct matches so a duplicated retrieved id cannot exceed 1.0.
func RecallAtK(retrieved []string, relevant []string, k int) float64 {
	rel := toSet(relevant)
	if len(rel) == 0 {
		return 0
	}
	top := toSet(head(retrieved, k))
	hits := 0
	for id := range top {
		if rel[id] {
			hits++
		}
	}
	return float64(hits) / float64(len(rel))
}

// PrecisionAtK is the fraction of the top k that is relevant.
func PrecisionAtK(retrieved []string, relevant []string, k int) float64 {
	rel := toSet(relevant)
	top := head(retrieved, k)
	if len(top) == 0 {
		return 0
	}
	hits := 0
	for _, id := range top {
		if rel[id] {
			hits++
		}
	}
	return float64(hits) / float64(len(top))
}

// ReciprocalRank is 1/rank of the first relevant id (0 if none retrieved).
func ReciprocalRank(retrieved []string, relevant []string) float64 {
	rel := toSet(relevant)
	for i, id := range retrieved {
		if rel[id] {
			return 1.0 / float64(i+1)
		}
	}
	return 0
}

// AveragePrecision credits each relevant id once, at its first occurrence.
func AveragePrecision(retrieved []string, relevant []string) float64 {
	rel := toSet(relevant)
	if len(rel) == 0 {
		return 0
	}
	seen := map[string]bool{}
	hits := 0
	total := 0.0
	for i, id := range retrieved {
		if rel[id] && !seen[id] {
			seen[id] = true
			hits++
			total += float64(hits) / float64(i+1)
		}
	}
	return total / float64(len(rel))
}

// DCGAtK is discounted cumulative gain (binary relevance) over the top k.
// A relevant id is credited once so duplicates cannot inflate the gain.
func DCGAtK(retrieved []string, relevant []string, k int) float64 {
	rel := toSet(relevant)
	seen := map[string]bool{}
	total := 0.0
	for i, id := range head(retrieved, k) {
		if rel[id] && !seen[id] {
			seen[id] = true
			total += 1.0 / math.Log2(float64(i+2))
		}
	}
	return total
}

// NDCGAtK is normalised DCG@k (binary relevance) in [0, 1].
func NDCGAtK(retrieved []string, relevant []string, k int) float64 {
	rel := toSet(relevant)
	if len(rel) == 0 {
		return 0
	}
	ideal := len(rel)
	if k < ideal {
		ideal = k
	}
	idcg := 0.0
	for i := 1; i <= ideal; i++ {
		idcg += 1.0 / math.Log2(float64(i+1))
	}
	if idcg == 0 {
		return 0
	}
	return DCGAtK(retrieved, relevant, k) / idcg
}

// Case is one query with its known-relevant memory ids. K and Mode are zero /
// empty by default so they fall back to Evaluate's defaults; set them per case
// only to override.
type Case struct {
	Query       string   `json:"query"`
	RelevantIDs []string `json:"relevant_ids"`
	K           int      `json:"k,omitempty"`
	Mode        string   `json:"mode,omitempty"`
}

// CaseResult is the per-case breakdown.
type CaseResult struct {
	Query        string   `json:"query"`
	K            int      `json:"k"`
	RecallAtK    float64  `json:"recall_at_k"`
	PrecisionAtK float64  `json:"precision_at_k"`
	RR           float64  `json:"rr"`
	AP           float64  `json:"ap"`
	NDCGAtK      float64  `json:"ndcg_at_k"`
	Retrieved    []string `json:"retrieved"`
}

// Result is the aggregate over all cases. K is -1 when the cases used mixed k.
type Result struct {
	N            int          `json:"n"`
	K            int          `json:"k"`
	RecallAtK    float64      `json:"recall_at_k"`
	PrecisionAtK float64      `json:"precision_at_k"`
	MRR          float64      `json:"mrr"`
	MAP          float64      `json:"map"`
	NDCGAtK      float64      `json:"ndcg_at_k"`
	PerCase      []CaseResult `json:"per_case"`
}

// Options configures a run.
type Options struct {
	DefaultK    int    // default 5
	DefaultMode string // default "hybrid"
	Recall      RecallOpts
}

// Evaluate runs each case through the store and aggregates ranking metrics.
//
// Recall is invoked read-only (the caller's RecallOpts are forwarded with
// touch disabled by the adapter) so evaluating does not perturb access-count
// or recency.
func Evaluate(ctx context.Context, store Recaller, cases []Case, opts Options) (Result, error) {
	if opts.DefaultK == 0 {
		opts.DefaultK = 5
	}
	if opts.DefaultMode == "" {
		opts.DefaultMode = "hybrid"
	}

	var sumRecall, sumPrecision, sumRR, sumAP, sumNDCG float64
	perCase := make([]CaseResult, 0, len(cases))
	ksUsed := map[int]bool{}

	for _, c := range cases {
		k := c.K
		if k == 0 {
			k = opts.DefaultK
		}
		mode := c.Mode
		if mode == "" {
			mode = opts.DefaultMode
		}
		ksUsed[k] = true

		ro := opts.Recall
		ro.Mode = mode
		hits, err := store.Recall(ctx, c.Query, k, ro)
		if err != nil {
			return Result{}, err
		}
		ids := make([]string, len(hits))
		for i, h := range hits {
			ids[i] = h.ID
		}

		cr := CaseResult{
			Query:        c.Query,
			K:            k,
			RecallAtK:    RecallAtK(ids, c.RelevantIDs, k),
			PrecisionAtK: PrecisionAtK(ids, c.RelevantIDs, k),
			RR:           ReciprocalRank(ids, c.RelevantIDs),
			AP:           AveragePrecision(ids, c.RelevantIDs),
			NDCGAtK:      NDCGAtK(ids, c.RelevantIDs, k),
			Retrieved:    ids,
		}
		perCase = append(perCase, cr)
		sumRecall += cr.RecallAtK
		sumPrecision += cr.PrecisionAtK
		sumRR += cr.RR
		sumAP += cr.AP
		sumNDCG += cr.NDCGAtK
	}

	n := len(cases)
	denom := float64(n)
	if n == 0 {
		denom = 1
	}
	// Label the result with the k actually used when uniform, else -1 ("mixed").
	resultK := -1
	if len(ksUsed) == 1 {
		keys := make([]int, 0, 1)
		for k := range ksUsed {
			keys = append(keys, k)
		}
		sort.Ints(keys)
		resultK = keys[0]
	}

	return Result{
		N: n, K: resultK,
		RecallAtK:    sumRecall / denom,
		PrecisionAtK: sumPrecision / denom,
		MRR:          sumRR / denom,
		MAP:          sumAP / denom,
		NDCGAtK:      sumNDCG / denom,
		PerCase:      perCase,
	}, nil
}

func toSet(items []string) map[string]bool {
	out := make(map[string]bool, len(items))
	for _, s := range items {
		out[s] = true
	}
	return out
}

func head(items []string, k int) []string {
	if k < 0 {
		k = 0
	}
	if k > len(items) {
		k = len(items)
	}
	return items[:k]
}
