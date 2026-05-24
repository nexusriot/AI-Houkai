package memory

import "testing"

func TestNegationParity(t *testing.T) {
	cases := []struct {
		text string
		want int
	}{
		{"the sky is blue", 0},
		{"the sky is not blue", 1},
		{"I never said it wasn't true", 0},  // two negations → parity 0
		{"don't say no to me", 0},           // two negations
		{"I can't not love it", 0},          // two
		{"I cannot decide", 0},              // "cannot" isn't in our list
	}
	for _, c := range cases {
		if got := negationParity(c.text); got != c.want {
			t.Errorf("negationParity(%q) = %d, want %d", c.text, got, c.want)
		}
	}
}

func TestTagsOverlap(t *testing.T) {
	a := Memory{Tags: []string{"go", "memory"}}
	b := Memory{Tags: []string{"rust", "memory"}}
	c := Memory{Tags: []string{"python"}}
	empty := Memory{}

	if !tagsOverlap(a, b) {
		t.Error("a and b share 'memory' — should overlap")
	}
	if tagsOverlap(a, c) {
		t.Error("a and c share no tags — should not overlap")
	}
	if !tagsOverlap(empty, empty) {
		t.Error("two empty-tag memories conventionally overlap")
	}
	if tagsOverlap(a, empty) {
		t.Error("one empty / one non-empty → should not overlap")
	}
}

func TestDetectConflictsContradictionVsDuplicate(t *testing.T) {
	a := Memory{ID: "a", Type: Semantic, Text: "the build is green", Tags: []string{"ci"}}
	contradict := Memory{ID: "b", Type: Semantic, Text: "the build is not green", Tags: []string{"ci"}}
	duplicate := Memory{ID: "c", Type: Semantic, Text: "the build is green now", Tags: []string{"ci"}}
	differentType := Memory{ID: "d", Type: Episodic, Text: "the build is not green", Tags: []string{"ci"}}
	lowSim := Memory{ID: "e", Type: Semantic, Text: "the build is not green", Tags: []string{"ci"}}
	noTagOverlap := Memory{ID: "f", Type: Semantic, Text: "the build is not green", Tags: []string{"db"}}

	candidates := []MemoryWithScore{
		{Memory: contradict, Score: 0.9},
		{Memory: duplicate, Score: 0.9},
		{Memory: differentType, Score: 0.95}, // filtered: type mismatch
		{Memory: lowSim, Score: 0.5},         // filtered: below threshold
		{Memory: noTagOverlap, Score: 0.95},  // filtered: tag mismatch
	}
	got := detectConflicts(a, candidates, 0.8, nil)
	if len(got) != 2 {
		t.Fatalf("want 2 conflicts after filtering, got %d", len(got))
	}

	kinds := map[ConflictKind]int{}
	for _, c := range got {
		kinds[c.Kind]++
	}
	if kinds[KindContradiction] != 1 {
		t.Errorf("want 1 contradiction, got %d", kinds[KindContradiction])
	}
	if kinds[KindDuplicate] != 1 {
		t.Errorf("want 1 duplicate, got %d", kinds[KindDuplicate])
	}
}

func TestDetectConflictsSkipsSelf(t *testing.T) {
	a := Memory{ID: "a", Type: Semantic, Text: "same text", Tags: []string{"x"}}
	got := detectConflicts(a, []MemoryWithScore{{Memory: a, Score: 1.0}}, 0.5, nil)
	if len(got) != 0 {
		t.Errorf("self-match should be skipped, got %d conflicts", len(got))
	}
}
