package memory

import (
	"regexp"
	"strings"
)

// wordRE approximates Python's Unicode-aware \b\w+\b: runs of letters, digits
// and underscore across all scripts (Go's \w is ASCII-only, so we spell out the
// Unicode classes to keep CJK runs as single tokens, matching Python's re).
var wordRE = regexp.MustCompile(`[\p{L}\p{N}_]+`)

// cjkRE matches a single Hiragana/Katakana/CJK-ideograph/Hangul character.
// These scripts are written without spaces, so a run of them collapses into a
// single \w+ token; we additionally emit character bigrams so lexical
// (BM25/Jaccard) matching works for non-Latin queries — a standard,
// dependency-free IR technique for CJK. Ranges mirror Python's _CJK_RE.
var cjkRE = regexp.MustCompile(`[\x{3040}-\x{30FF}\x{3400}-\x{4DBF}\x{4E00}-\x{9FFF}\x{F900}-\x{FAFF}\x{AC00}-\x{D7A3}]`)

// tokenize lowercases, strips apostrophes ("don't" → "dont"), splits on word
// boundaries, and — when CJK is present — appends character bigrams of each
// multi-char CJK run. Shared by BM25, Jaccard similarity, key-phrase
// extraction and negation counting so they all agree on token boundaries.
func tokenize(text string) []string {
	// Lowercase, then strip both ASCII and typographic apostrophes so
	// "don't"/"don’t" → "dont", matching Python's _tokenize.
	normalized := strings.ToLower(text)
	normalized = strings.ReplaceAll(normalized, "’", "")
	normalized = strings.ReplaceAll(normalized, "'", "")
	tokens := wordRE.FindAllString(normalized, -1)
	if !cjkRE.MatchString(normalized) {
		return tokens
	}
	extra := []string{}
	for _, tok := range tokens {
		var chars []string
		for _, r := range tok {
			c := string(r)
			if cjkRE.MatchString(c) {
				chars = append(chars, c)
			}
		}
		// Bigrams of multi-char CJK runs; a single CJK char is already its own
		// token, so we don't re-emit it (avoids double-counting in tf).
		for i := 0; i+1 < len(chars); i++ {
			extra = append(extra, chars[i]+chars[i+1])
		}
	}
	return append(tokens, extra...)
}
