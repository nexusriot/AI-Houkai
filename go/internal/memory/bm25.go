package memory

import (
	"math"
	"strings"
)

const (
	bm25K1 = 1.2
	bm25B  = 0.75
)

type bm25Doc struct {
	tokens []string
	tf     map[string]int
}

// bm25Score scores query tokens against a slice of documents (the over-fetch pool).
// Returns a slice of scores parallel to docs.
func bm25Score(query string, docs []string) []float32 {
	if len(docs) == 0 {
		return nil
	}

	tokenise := func(s string) []string {
		return strings.Fields(strings.ToLower(s))
	}

	// Build corpus.
	corpus := make([]bm25Doc, len(docs))
	var totalLen float64
	for i, d := range docs {
		toks := tokenise(d)
		tf := make(map[string]int, len(toks))
		for _, t := range toks {
			tf[t]++
		}
		corpus[i] = bm25Doc{tokens: toks, tf: tf}
		totalLen += float64(len(toks))
	}
	avgLen := totalLen / float64(len(docs))

	// IDF over this local pool.
	df := make(map[string]int)
	for _, doc := range corpus {
		seen := make(map[string]bool)
		for t := range doc.tf {
			if !seen[t] {
				df[t]++
				seen[t] = true
			}
		}
	}
	idf := func(term string) float64 {
		n := float64(len(docs))
		d := float64(df[term])
		return math.Log((n-d+0.5)/(d+0.5) + 1)
	}

	queryTokens := tokenise(query)
	scores := make([]float32, len(docs))
	for i, doc := range corpus {
		dl := float64(len(doc.tokens))
		var score float64
		for _, qt := range queryTokens {
			freq := float64(doc.tf[qt])
			if freq == 0 {
				continue
			}
			num := freq * (bm25K1 + 1)
			den := freq + bm25K1*(1-bm25B+bm25B*dl/avgLen)
			score += idf(qt) * num / den
		}
		scores[i] = float32(score)
	}

	// Normalise to [0, 1] based on pool max.
	var maxS float32
	for _, s := range scores {
		if s > maxS {
			maxS = s
		}
	}
	if maxS > 0 {
		for i := range scores {
			scores[i] /= maxS
		}
	}
	return scores
}
