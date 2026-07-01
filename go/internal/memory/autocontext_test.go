package memory

import (
	"context"
	"strings"
	"testing"
)

func TestExtractKeyPhrases(t *testing.T) {
	// Stop words ("the", "is", "a", "of") and short tokens (<=2 chars) drop out;
	// bigrams come before unigrams; result is capped at maxPhrases.
	got := ExtractKeyPhrases("the deployment pipeline is a source of failures", 3)
	if len(got) != 3 {
		t.Fatalf("want 3 phrases, got %d: %v", len(got), got)
	}
	if got[0] != "deployment pipeline" {
		t.Errorf("first phrase should be the leading bigram, got %q", got[0])
	}
	for _, p := range got {
		for _, w := range strings.Fields(p) {
			if stopWords[w] {
				t.Errorf("stop word %q leaked into phrase %q", w, p)
			}
		}
	}
}

func TestAutoContextPackDedupesByID(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	// A memory whose text overlaps both the full task and an extracted phrase,
	// so it surfaces from multiple fan-out queries but must appear once.
	store.Remember(ctx, "deployment pipeline runbook and rollback steps", RememberOpts{Type: Procedural, Importance: 0.9})
	store.Remember(ctx, "unrelated cooking recipe", RememberOpts{Type: Semantic, Importance: 0.2})

	pack, err := store.AutoContextPack(ctx, "the deployment pipeline failed", AutoContextOpts{TokenBudget: 800, MaxPhrases: 3})
	if err != nil {
		t.Fatalf("AutoContextPack: %v", err)
	}
	seen := map[string]int{}
	for _, it := range pack.Items {
		seen[it.Memory.ID]++
	}
	for id, n := range seen {
		if n != 1 {
			t.Errorf("memory %s appears %d times, want 1 (dedup by id)", id, n)
		}
	}
	if len(pack.Items) == 0 {
		t.Error("expected at least the relevant memory in the pack")
	}
}
