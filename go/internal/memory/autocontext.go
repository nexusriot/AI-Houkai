package memory

import (
	"context"
	"sort"
)

// stopWords mirrors Python's _STOP_WORDS: common function words filtered out of
// key-phrase extraction so the fan-out queries carry actual signal.
var stopWords = map[string]bool{
	"a": true, "an": true, "the": true, "is": true, "are": true, "was": true,
	"were": true, "be": true, "been": true, "being": true, "have": true,
	"has": true, "had": true, "do": true, "does": true, "did": true, "will": true,
	"would": true, "could": true, "should": true, "may": true, "might": true,
	"shall": true, "can": true, "need": true, "must": true, "to": true, "for": true,
	"of": true, "in": true, "on": true, "at": true, "by": true, "from": true,
	"with": true, "about": true, "and": true, "or": true, "but": true, "if": true,
	"then": true, "so": true, "as": true, "that": true, "this": true, "these": true,
	"those": true, "it": true, "its": true, "i": true, "my": true, "we": true,
	"our": true, "you": true, "your": true, "they": true, "their": true,
	"what": true, "how": true, "when": true, "where": true, "who": true,
	"which": true, "why": true, "not": true, "just": true, "now": true,
	"also": true, "than": true, "only": true, "any": true, "all": true, "each": true,
}

// ExtractKeyPhrases extracts up to maxPhrases key phrases from task without any
// NLP library: it filters stop words and short tokens, then prefers bigrams
// (more specific) over single words. Used by AutoContextPack to fan out recall
// over several query angles derived from one task description.
func ExtractKeyPhrases(task string, maxPhrases int) []string {
	var words []string
	for _, w := range tokenize(task) {
		if !stopWords[w] && len([]rune(w)) > 2 {
			words = append(words, w)
		}
	}
	var phrases []string
	for i := 0; i+1 < len(words); i++ {
		phrases = append(phrases, words[i]+" "+words[i+1])
	}
	phrases = append(phrases, words...)

	seen := map[string]bool{}
	var unique []string
	for _, p := range phrases {
		if !seen[p] {
			seen[p] = true
			unique = append(unique, p)
		}
	}
	if maxPhrases >= 0 && len(unique) > maxPhrases {
		unique = unique[:maxPhrases]
	}
	return unique
}

// AutoContextOpts holds optional AutoContextPack parameters.
type AutoContextOpts struct {
	TokenBudget  int              // default 800
	MaxPhrases   int              // default 3
	Mode         RecallMode       // default hybrid
	MinCosine    *float32         // absolute relevance floor per fan-out query
	Header       *string          // nil → "## Relevant memory"; "" → no header
	TokenCounter func(string) int // default EstimateTokens
	// NoTouch skips the access-count / last_accessed bump on every fan-out
	// recall, making the whole call read-only.
	NoTouch           bool
	Compress          bool
	CompressThreshold float32
	CompressMinGroup  int
	// LexicalIndex and MinTrust apply to every fan-out query, exactly as they do
	// in RecallPack. The trust floor matters more here than anywhere else: this
	// is the entry point an agent calls *without* choosing a query, so it is the
	// one most likely to pull scraped material into a context block unattended.
	LexicalIndex LexicalIndexMode
	MinTrust     TrustLevel
}

// AutoContextPack fans out recall over the task plus its extracted key phrases,
// deduplicates by memory id (keeping the highest score seen), then packs the
// results greedily within the token budget via the shared packer (so
// compression works here too). Unlike recall_pack it does not offer RRF fusion,
// because RRF scores are rank-relative to each query's own pool and cannot be
// compared across the fan-out queries.
func (s *MemoryStore) AutoContextPack(ctx context.Context, task string, opts AutoContextOpts) (PackResult, error) {
	if opts.TokenBudget == 0 {
		opts.TokenBudget = 800
	}
	// MaxPhrases is honored as-is (including an explicit 0 = task-only, matching
	// Python). Callers/handlers apply the default of 3 for an *absent* value; a
	// negative value is treated as "no cap" by ExtractKeyPhrases below, so clamp.
	if opts.MaxPhrases < 0 {
		opts.MaxPhrases = 0
	}
	if opts.Mode == "" {
		opts.Mode = ModeHybrid
	}
	countFn := opts.TokenCounter
	if countFn == nil {
		countFn = EstimateTokens
	}
	header := defaultPackHeader
	if opts.Header != nil {
		header = *opts.Header
	}

	queries := append([]string{task}, ExtractKeyPhrases(task, opts.MaxPhrases)...)

	best := map[string]MemoryWithScore{}
	var order []string // first-seen order, for deterministic stable sorting
	for _, q := range queries {
		res, err := s.Recall(ctx, q, 10, RecallOpts{
			Mode: opts.Mode, MinCosine: opts.MinCosine, NoTouch: opts.NoTouch,
			LexicalIndex: opts.LexicalIndex, MinTrust: opts.MinTrust,
		})
		if err != nil {
			return PackResult{}, err
		}
		for _, r := range res {
			if cur, ok := best[r.ID]; !ok {
				best[r.ID] = r
				order = append(order, r.ID)
			} else if r.Score > cur.Score {
				best[r.ID] = r
			}
		}
	}

	ranked := make([]MemoryWithScore, len(order))
	for i, id := range order {
		ranked[i] = best[id]
	}
	sort.SliceStable(ranked, func(i, j int) bool { return ranked[i].Score > ranked[j].Score })

	return packRanked(ranked, opts.TokenBudget, countFn, header,
		opts.Compress, opts.CompressThreshold, opts.CompressMinGroup), nil
}
