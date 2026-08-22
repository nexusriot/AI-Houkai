package memory

import (
	"context"
	"errors"
	"testing"
	"time"
)

func TestNegationParity(t *testing.T) {
	cases := []struct {
		text string
		want int
	}{
		{"the sky is blue", 0},
		{"the sky is not blue", 1},
		{"I never said it wasn't true", 1}, // only "never" ∈ _NEG ("wasnt" is not), matching Python
		{"don't say no to me", 0},          // dont + no → two negations
		{"I can't not love it", 0},         // cant + not → two
		{"I cannot decide", 0},             // "cannot" isn't in _NEG
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
	// Matching Python: the tag guard only excludes a pair when BOTH sides are
	// tagged and disjoint, so a tagged/untagged pair is allowed to conflict.
	if !tagsOverlap(a, empty) {
		t.Error("one empty / one non-empty → tag guard should not exclude")
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

func TestDetectConflictsPolarityDiff(t *testing.T) {
	a := Memory{ID: "a", Type: Semantic, Text: "coffee is good for you", Tags: []string{"health"}, Polarity: 1}
	// Opposite polarity, same (non-negated) text → polarity_diff contradiction,
	// taking priority over the negation heuristic (which would see no diff).
	opp := Memory{ID: "b", Type: Semantic, Text: "coffee is good for you", Tags: []string{"health"}, Polarity: -1}
	// Same polarity → not a polarity contradiction (falls through to duplicate).
	same := Memory{ID: "c", Type: Semantic, Text: "coffee is good for you", Tags: []string{"health"}, Polarity: 1}
	// Neutral polarity → never a polarity contradiction.
	neutral := Memory{ID: "d", Type: Semantic, Text: "coffee is good for you", Tags: []string{"health"}, Polarity: 0}

	got := detectConflicts(a, []MemoryWithScore{
		{Memory: opp, Score: 0.95},
		{Memory: same, Score: 0.95},
		{Memory: neutral, Score: 0.95},
	}, 0.8, nil)

	byID := map[string]Conflict{}
	for _, c := range got {
		byID[c.B.ID] = c
	}
	if byID["b"].Reason != "polarity_diff" || byID["b"].Kind != KindContradiction {
		t.Errorf("opposite polarity should be a polarity_diff contradiction, got %+v", byID["b"])
	}
	if byID["c"].Reason == "polarity_diff" {
		t.Errorf("same polarity must not be polarity_diff, got %+v", byID["c"])
	}
	if byID["d"].Reason == "polarity_diff" {
		t.Errorf("neutral polarity must not be polarity_diff, got %+v", byID["d"])
	}
}

func TestDetectConflictsSkipsSuperseded(t *testing.T) {
	a := Memory{ID: "a", Type: Semantic, Text: "the build is green", Tags: []string{"ci"}}
	superseded := Memory{ID: "b", Type: Semantic, Text: "the build is green", Tags: []string{"ci"}, SupersededBy: "z"}
	got := detectConflicts(a, []MemoryWithScore{{Memory: superseded, Score: 1.0}}, 0.8, nil)
	if len(got) != 0 {
		t.Errorf("superseded candidate must be skipped, got %d conflicts", len(got))
	}
}

func TestDetectConflictsTaggedVsUntagged(t *testing.T) {
	// One side tagged, the other untagged → the tag guard must NOT exclude it
	// (matches Python; only both-tagged-and-disjoint is excluded).
	a := Memory{ID: "a", Type: Semantic, Text: "the build is green", Tags: []string{"ci"}}
	untagged := Memory{ID: "b", Type: Semantic, Text: "the build is green"}
	got := detectConflicts(a, []MemoryWithScore{{Memory: untagged, Score: 0.95}}, 0.8, nil)
	if len(got) != 1 {
		t.Errorf("tagged-vs-untagged should still conflict, got %d", len(got))
	}
}

func TestDetectConflictsSkipsSelf(t *testing.T) {
	a := Memory{ID: "a", Type: Semantic, Text: "same text", Tags: []string{"x"}}
	got := detectConflicts(a, []MemoryWithScore{{Memory: a, Score: 1.0}}, 0.5, nil)
	if len(got) != 0 {
		t.Errorf("self-match should be skipped, got %d conflicts", len(got))
	}
}

// A lapsed memory must not be reported as a conflict. Expired rows are hidden
// from recall/list and findByContentHash skips them, so a re-assertion creates a
// fresh memory rather than resurrecting a dead one. Without the same filter here
// an invisible row rejects a legitimate write under on_conflict="raise" — a
// conflict the caller cannot inspect, resolve, or even see.
func TestDetectConflictsSkipsExpired(t *testing.T) {
	past := float64(time.Now().Add(-time.Hour).Unix())
	a := Memory{ID: "a", Type: Semantic, Text: "the build is green", Tags: []string{"ci"}}
	lapsed := Memory{ID: "b", Type: Semantic, Text: "the build is green",
		Tags: []string{"ci"}, ExpiresAt: past}
	got := detectConflicts(a, []MemoryWithScore{{Memory: lapsed, Score: 1.0}}, 0.8, nil)
	if len(got) != 0 {
		t.Errorf("expired candidate must be skipped, got %d conflicts", len(got))
	}
}

// Only *lapsed* rows are excluded — a deadline in the future is still live, so
// the filter must not blunt the feature.
func TestDetectConflictsKeepsUnexpired(t *testing.T) {
	future := float64(time.Now().Add(time.Hour).Unix())
	a := Memory{ID: "a", Type: Semantic, Text: "the build is green", Tags: []string{"ci"}}
	live := Memory{ID: "b", Type: Semantic, Text: "the build is green",
		Tags: []string{"ci"}, ExpiresAt: future}
	got := detectConflicts(a, []MemoryWithScore{{Memory: live, Score: 1.0}}, 0.8, nil)
	if len(got) != 1 {
		t.Errorf("a future TTL is still a live conflict, got %d", len(got))
	}
}

func TestRememberPerCallOnConflictOverridesStorePolicy(t *testing.T) {
	// Store policy is ignore; the per-call raise must still detect + roll back.
	store := newConflictStore(t, PolicyIgnore, 0.1)
	ctx := context.Background()

	first, stored, _, err := store.Remember(ctx, "the API gateway is nginx", RememberOpts{Type: Semantic})
	if err != nil || !stored {
		t.Fatalf("first Remember: %v", err)
	}
	_, stored, conflicts, err := store.Remember(ctx, "the API gateway is nginx",
		RememberOpts{Type: Semantic, OnConflict: PolicyRaise})
	var ce *ConflictError
	if stored || !errors.As(err, &ce) || len(conflicts) == 0 {
		t.Fatalf("per-call raise: stored=%v err=%v conflicts=%d, want rejection", stored, err, len(conflicts))
	}
	// Rollback: only the first memory remains, untouched.
	if c, _ := store.Count(ctx); c != 1 {
		t.Errorf("count after rollback = %d, want 1", c)
	}
	got, _ := store.GetByID(ctx, first.ID)
	if got.SupersededBy != "" || len(got.Links) != 0 {
		t.Errorf("raise rollback disturbed the existing memory: %+v", got)
	}
}

func TestRememberConflictSupersedeLinksNewToOld(t *testing.T) {
	store := newConflictStore(t, PolicySupersede, 0.1)
	ctx := context.Background()
	oldMem, _, _, err := store.Remember(ctx, "the primary region is us-east-1", RememberOpts{Type: Semantic})
	if err != nil {
		t.Fatal(err)
	}
	newMem, stored, _, err := store.Remember(ctx, "the primary region is us-east-1", RememberOpts{Type: Semantic})
	if err != nil || !stored {
		t.Fatalf("supersede policy Remember: stored=%v err=%v", stored, err)
	}
	// The NEW memory carries the supersedes edge; the old one carries none —
	// possible only because the new memory is added BEFORE doSupersede runs.
	gotNew, _ := store.GetByID(ctx, newMem.ID)
	gotOld, _ := store.GetByID(ctx, oldMem.ID)
	if len(gotNew.Links) != 1 || gotNew.Links[0].To != oldMem.ID || gotNew.Links[0].Rel != RelSupersedes {
		t.Errorf("new memory links = %v, want [{%s supersedes}]", gotNew.Links, oldMem.ID)
	}
	if len(gotOld.Links) != 0 {
		t.Errorf("old memory should carry no links, got %v", gotOld.Links)
	}
}
