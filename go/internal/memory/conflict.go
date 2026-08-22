package memory

// negationWords mirrors Python's _NEG (apostrophes already stripped by the
// shared tokenizer, so "don't" → "dont").
var negationWords = map[string]bool{
	"not": true, "never": true, "no": true, "dont": true, "doesnt": true,
	"wont": true, "shouldnt": true, "cant": true, "without": true,
	"avoid": true, "neither": true, "nor": true, "nothing": true,
	"nobody": true, "nowhere": true, "none": true,
}

// negationParity counts negation words mod 2 in a text.
func negationParity(text string) int {
	count := 0
	for _, tok := range tokenize(text) {
		if negationWords[tok] {
			count++
		}
	}
	return count % 2
}

func negationDiff(a, b Memory) bool {
	return negationParity(a.Text) != negationParity(b.Text)
}

// tagsOverlap reports whether the tag guard allows a/b to be considered a
// conflict. Matching Python, a pair is excluded ONLY when BOTH memories carry
// tags and those tag sets are disjoint; if either side is untagged there is no
// tag signal to exclude on, so the pair proceeds.
func tagsOverlap(a, b Memory) bool {
	if len(a.Tags) == 0 || len(b.Tags) == 0 {
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

// classifyConflict decides the (kind, reason) for a same-type, above-threshold,
// tag-compatible pair, in Python's precedence order: polarity_diff → negation_diff
// → custom_fn → duplicate/similarity.
func classifyConflict(a, b Memory, customFn func(a, b Memory) bool) (ConflictKind, string) {
	if a.Polarity != 0 && b.Polarity != 0 && a.Polarity != b.Polarity {
		return KindContradiction, "polarity_diff"
	}
	if negationDiff(a, b) {
		return KindContradiction, "negation_diff"
	}
	if customFn != nil && customFn(a, b) {
		return KindContradiction, "custom_fn"
	}
	return KindDuplicate, "similarity"
}

// detectConflicts returns conflicts between `a` and a set of candidates.
// Superseded candidates are skipped (matching Python), so a soft-deleted memory
// never resurfaces as a conflict.
func detectConflicts(a Memory, candidates []MemoryWithScore, threshold float32, customFn func(a, b Memory) bool) []Conflict {
	var out []Conflict
	for _, c := range candidates {
		b := c.Memory
		if b.ID == a.ID {
			continue
		}
		if b.SupersededBy != "" {
			continue
		}
		// A lapsed row is hidden from recall/list and waiting for PurgeExpired,
		// so it must not clash with a new write: under on_conflict="raise" it
		// would reject the write with a conflict the caller cannot see or
		// resolve, and under "supersede" it would re-label a memory that is
		// already on its way out. findByContentHash skips expired rows for the
		// same reason.
		if b.ExpiresAt > 0 && b.ExpiresAt <= nowFloat() {
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
		kind, reason := classifyConflict(a, b, customFn)
		out = append(out, Conflict{Kind: kind, Reason: reason, Similarity: c.Score, A: a, B: b})
	}
	return out
}
