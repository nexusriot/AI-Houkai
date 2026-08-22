// Package ingest provides chunking for bulk ingestion (`houkai ingest`).
//
// ChunkText splits free-form text (notes, markdown, transcripts) into
// memory-sized chunks:
//
//  1. Split on blank lines into paragraphs.
//  2. A markdown heading is glued to the paragraph that follows it, so the
//     stored memory keeps its context ("## Deploy\nUse make release").
//  3. Paragraphs longer than maxChars are re-split on sentence boundaries
//     and greedily re-packed up to maxChars.
//  4. Blocks shorter than minChars are dropped (separators, stray list
//     bullets, noise). A short *fragment* of a split paragraph is folded
//     into a neighbour instead, because it is content rather than noise.
//
// Deterministic and dependency-free; embedding and storage happen at the
// caller (one Remember per chunk).
package ingest

import (
	"regexp"
	"strings"
)

var (
	sentenceSplit = regexp.MustCompile(`(?:[.!?])\s+`)
	heading       = regexp.MustCompile(`^#{1,6}\s+\S`)
	blankLine     = regexp.MustCompile(`\n\s*\n`)
)

// splitSentences splits on sentence-ending punctuation followed by
// whitespace, keeping the punctuation with the preceding sentence
// (Go's regexp has no lookbehind, so we locate boundaries manually).
func splitSentences(paragraph string) []string {
	locs := sentenceSplit.FindAllStringIndex(paragraph, -1)
	var out []string
	start := 0
	for _, loc := range locs {
		// loc[0] is the punctuation character; keep it in the sentence.
		out = append(out, paragraph[start:loc[0]+1])
		start = loc[1]
	}
	out = append(out, paragraph[start:])
	return out
}

// splitLong greedily packs sentences into chunks of at most maxChars.
//
// Fragments below minChars are folded into a neighbour rather than left for the
// caller's noise filter to delete. Greedy packing leaves a short fragment
// whenever the next sentence will not fit, and that fragment is the tail (or
// head) of a real paragraph — content, not a separator. maxChars is already a
// soft target, since a single over-long sentence is kept whole, so growing a
// chunk is the lesser evil against silently losing text.
func splitLong(paragraph string, maxChars, minChars int) []string {
	var chunks []string
	current := ""
	for _, sent := range splitSentences(paragraph) {
		sent = strings.TrimSpace(sent)
		if sent == "" {
			continue
		}
		candidate := sent
		if current != "" {
			candidate = current + " " + sent
		}
		if len(candidate) <= maxChars || current == "" {
			current = candidate
		} else {
			chunks = append(chunks, current)
			current = sent
		}
	}
	if current != "" {
		chunks = append(chunks, current)
	}

	// A single sentence longer than maxChars is kept whole — splitting
	// mid-sentence would destroy the embedding's meaning.
	if minChars <= 0 || len(chunks) < 2 {
		return chunks
	}

	var folded []string
	for _, chunk := range chunks {
		if len(folded) > 0 && len(chunk) < minChars {
			folded[len(folded)-1] += " " + chunk
		} else {
			folded = append(folded, chunk)
		}
	}
	// A runt in first position has no predecessor to join, so fold it forward.
	if len(folded) > 1 && len(folded[0]) < minChars {
		folded[1] = folded[0] + " " + folded[1]
		folded = folded[1:]
	}
	return folded
}

// ChunkText splits text into memory-sized chunks (see package doc).
func ChunkText(text string, maxChars, minChars int) []string {
	if maxChars <= 0 {
		maxChars = 500
	}
	if minChars <= 0 {
		minChars = 30
	}
	normalized := strings.ReplaceAll(text, "\r\n", "\n")
	var blocks []string
	for _, b := range blankLine.Split(normalized, -1) {
		b = strings.TrimSpace(b)
		if b != "" {
			blocks = append(blocks, b)
		}
	}

	// Glue headings onto the following block.
	var merged []string
	pendingHeading := ""
	for _, block := range blocks {
		if heading.MatchString(block) && !strings.Contains(block, "\n") {
			if pendingHeading != "" {
				pendingHeading += "\n" + block
			} else {
				pendingHeading = block
			}
			continue
		}
		if pendingHeading != "" {
			block = pendingHeading + "\n" + block
			pendingHeading = ""
		}
		merged = append(merged, block)
	}
	if pendingHeading != "" {
		merged = append(merged, pendingHeading)
	}

	var chunks []string
	for _, block := range merged {
		if len(block) <= maxChars {
			chunks = append(chunks, block)
		} else {
			chunks = append(chunks, splitLong(block, maxChars, minChars)...)
		}
	}

	var out []string
	for _, c := range chunks {
		if len(c) >= minChars {
			out = append(out, c)
		}
	}
	return out
}
