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
