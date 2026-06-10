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
//  4. Chunks shorter than minChars are dropped (separators, stray list
//     bullets, noise).
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
func splitLong(paragraph string, maxChars int) []string {
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
	return chunks
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
			chunks = append(chunks, splitLong(block, maxChars)...)
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
