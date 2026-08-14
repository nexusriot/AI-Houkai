package ingest

import (
	"fmt"
	"strings"
	"testing"
)

func TestSplitsOnBlankLines(t *testing.T) {
	chunks := ChunkText(
		"First paragraph about deployment process.\n\n"+
			"Second paragraph about testing conventions.", 0, 0)
	want := []string{
		"First paragraph about deployment process.",
		"Second paragraph about testing conventions.",
	}
	if len(chunks) != 2 || chunks[0] != want[0] || chunks[1] != want[1] {
		t.Errorf("got %#v, want %#v", chunks, want)
	}
}

func TestHeadingGluedToNextParagraph(t *testing.T) {
	chunks := ChunkText("## Deploy\n\nUse make release to push a new version out.", 0, 0)
	if len(chunks) != 1 {
		t.Fatalf("got %d chunks, want 1", len(chunks))
	}
	if !strings.HasPrefix(chunks[0], "## Deploy\n") || !strings.Contains(chunks[0], "make release") {
		t.Errorf("bad chunk: %q", chunks[0])
	}
}

func TestShortChunksDropped(t *testing.T) {
	chunks := ChunkText("ok\n\nThis paragraph is long enough to survive the cut.", 0, 0)
	if len(chunks) != 1 || chunks[0] != "This paragraph is long enough to survive the cut." {
		t.Errorf("got %#v", chunks)
	}
}

func TestLongParagraphSplitOnSentences(t *testing.T) {
	var parts []string
	for i := 0; i < 10; i++ {
		parts = append(parts, fmt.Sprintf("Sentence number %d talks about something moderately interesting.", i))
	}
	para := strings.Join(parts, " ")
	chunks := ChunkText(para, 150, 0)
	if len(chunks) <= 1 {
		t.Fatalf("expected multiple chunks, got %d", len(chunks))
	}
	for _, c := range chunks {
		if len(c) > 150 {
			t.Errorf("chunk over max_chars: %d", len(c))
		}
	}
	if !strings.Contains(chunks[len(chunks)-1], "Sentence number 9") {
		t.Error("content lost during split")
	}
}

func TestSingleOversizedSentenceKeptWhole(t *testing.T) {
	sent := strings.TrimSpace(strings.Repeat("word ", 60)) + "."
	chunks := ChunkText(sent, 100, 0)
	if len(chunks) != 1 {
		t.Errorf("got %d chunks, want 1", len(chunks))
	}
}

func TestCRLFNormalised(t *testing.T) {
	chunks := ChunkText("Paragraph one is right here today.\r\n\r\nParagraph two follows after it.", 0, 0)
	if len(chunks) != 2 {
		t.Errorf("got %d chunks, want 2", len(chunks))
	}
}

func TestEmptyInput(t *testing.T) {
	if got := ChunkText("", 0, 0); len(got) != 0 {
		t.Errorf("got %#v", got)
	}
	if got := ChunkText("\n\n\n", 0, 0); len(got) != 0 {
		t.Errorf("got %#v", got)
	}
}

func TestTrailingHeadingKept(t *testing.T) {
	chunks := ChunkText("A real paragraph with enough words in it.\n\n## Dangling heading at the end", 0, 0)
	if len(chunks) == 0 || chunks[len(chunks)-1] != "## Dangling heading at the end" {
		t.Errorf("got %#v", chunks)
	}
}

// minChars drops noise, not the tail of a real paragraph. Greedy sentence
// packing leaves a short fragment whenever the next sentence will not fit, and
// applying the noise filter to it deletes ingested text with no warning:
// separators and stray bullets are noise, the last sentence of a paragraph is
// content.
func TestChunkTextKeepsShortSplitFragments(t *testing.T) {
	tail := "Tiny tail."
	para := strings.Repeat("A", 38) + ". " + strings.Repeat("B", 38) + ". " +
		strings.Repeat("C", 96) + ". " + tail

	chunks := ChunkText(para, 100, 30)
	joined := strings.Join(chunks, " ")
	for _, want := range []string{
		strings.Repeat("A", 38) + ".",
		strings.Repeat("B", 38) + ".",
		strings.Repeat("C", 96) + ".",
		tail,
	} {
		if !strings.Contains(joined, want) {
			t.Errorf("chunking lost %q; got %q", want, chunks)
		}
	}
}

// A runt can also come first, when one huge sentence follows a tiny one.
func TestChunkTextKeepsAShortLeadingFragment(t *testing.T) {
	chunks := ChunkText("Hi. "+strings.Repeat("Z", 300)+".", 100, 30)
	if !strings.Contains(strings.Join(chunks, " "), "Hi.") {
		t.Errorf("leading fragment lost; got %q", chunks)
	}
}

// The filter must keep doing its job on standalone short blocks.
func TestChunkTextStillDropsNoiseBlocks(t *testing.T) {
	body := strings.Repeat("R", 80)
	chunks := ChunkText("---\n\n"+body+"\n\n*\n", 500, 30)
	if len(chunks) != 1 || chunks[0] != body {
		t.Errorf("noise blocks should still be dropped; got %q", chunks)
	}
}
