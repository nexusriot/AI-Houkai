package memory

import (
	"strings"
)

var negationWords = map[string]bool{
	"not": true, "never": true, "no": true, "dont": true, "doesnt": true,
	"didnt": true, "cant": true, "wont": true, "shouldnt": true,
	"wouldnt": true, "couldnt": true, "isnt": true, "arent": true,
	"wasnt": true, "werent": true, "havent": true, "hasnt": true,
	"hadnt": true,
}

// negationParity counts negation words mod 2 in a text.
func negationParity(text string) int {
	// Strip apostrophes for simple normalisation.
	s := strings.ReplaceAll(strings.ToLower(text), "'", "")
	count := 0
	for _, tok := range strings.Fields(s) {
		if negationWords[tok] {
			count++
		}
	}
	return count % 2
}

func negationDiff(a, b Memory) bool {
	return negationParity(a.Text) != negationParity(b.Text)
}

func tagsOverlap(a, b Memory) bool {
	if len(a.Tags) == 0 && len(b.Tags) == 0 {
		return true
	}
	set := make(map[string]bool, len(a.Tags))
	for _, t := range a.Tags {
		set[t] = true
	}
	for _, t := range b.Tags {
		if set[t] {
			return true
		}
	}
	return false
}

// detectConflicts returns conflicts between `a` and a set of candidates.
func detectConflicts(a Memory, candidates []MemoryWithScore, threshold float32, customFn func(a, b Memory) bool) []Conflict {
	var out []Conflict
	for _, c := range candidates {
		b := c.Memory
		if b.ID == a.ID {
			continue
		}
		if b.Type != a.Type {
			continue
		}
		if c.Score < threshold {
			continue
		}
		if !tagsOverlap(a, b) {
			continue
		}
		if negationDiff(a, b) {
			out = append(out, Conflict{
				Kind:       KindContradiction,
				Reason:     "negation_diff",
				Similarity: c.Score,
				A:          a,
				B:          b,
			})
		} else if customFn != nil && customFn(a, b) {
			out = append(out, Conflict{
				Kind:       KindContradiction,
				Reason:     "custom_fn",
				Similarity: c.Score,
				A:          a,
				B:          b,
			})
		} else {
			out = append(out, Conflict{
				Kind:       KindDuplicate,
				Reason:     "similarity",
				Similarity: c.Score,
				A:          a,
				B:          b,
			})
		}
	}
	return out
}
