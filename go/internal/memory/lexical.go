package memory

import (
	"context"
	"sort"

	"github.com/nexusriot/ai-houkai/internal/vector"
)

// lexicalMaxTokens caps how many query tokens a corpus-lexical recall probes.
// Each probe is a content scan inside the store, so a long query must not turn
// into a long series of them; the longest tokens are the most selective, so
// they go first.
const lexicalMaxTokens = 4

// lexicalMinTokenLen skips tokens so short they match almost everything —
// probing them costs a scan to return the whole collection.
const lexicalMinTokenLen = 3

// unionLexical merges full-corpus lexical candidates into a query result.
//
// The vector over-fetch pool is chosen by embedding distance alone, so a memory
// carrying the query's exact tokens but embedding weakly never enters it — and
// therefore can never be surfaced by the BM25 term, at any corpus size. This
// pulls those candidates in via the backend's document-content predicate and
// lets the existing pool-relative BM25 score them.
//
// Candidates keep their real cosine similarity against the query vector.
// Fabricating one is wrong in both directions: a neutral value invents vector
// evidence the candidate never earned, and a worst-case value buries it below
// anything the lexical weight could recover, making the channel decorative.
func (s *MemoryStore) unionLexical(
	ctx context.Context, hits []vector.Hit, query string,
	queryVec []float32, nFetch int,
) []vector.Hit {
	present := make(map[string]bool, len(hits))
	for _, h := range hits {
		present[h.ID] = true
	}
	for _, it := range s.lexicalCandidates(ctx, query, nFetch) {
		if present[it.ID] {
			continue
		}
		present[it.ID] = true
		hits = append(hits, vector.Hit{
			Item:       it,
			Similarity: vector.CosineSim(queryVec, it.Embedding),
		})
	}
	return hits
}

// lexicalCandidates returns items whose text contains any sufficiently long
// query token, capped at lexicalMaxTokens probes.
func (s *MemoryStore) lexicalCandidates(
	ctx context.Context, query string, limit int,
) []vector.Item {
	seen := map[string]bool{}
	var tokens []string
	for _, t := range tokenize(query) {
		if len([]rune(t)) >= lexicalMinTokenLen && !seen[t] {
			seen[t] = true
			tokens = append(tokens, t)
		}
	}
	// Longest first: the most selective probes should run before the cap bites.
	sort.SliceStable(tokens, func(i, j int) bool {
		return len(tokens[i]) > len(tokens[j])
	})
	if len(tokens) > lexicalMaxTokens {
		tokens = tokens[:lexicalMaxTokens]
	}

	found := map[string]bool{}
	var out []vector.Item
	for _, token := range tokens {
		items, err := s.backend.SearchDocuments(ctx, token, limit)
		if err != nil {
			continue
		}
		for _, it := range items {
			if !found[it.ID] {
				found[it.ID] = true
				out = append(out, it)
			}
		}
		if len(out) >= limit {
			return out[:limit]
		}
	}
	return out
}
