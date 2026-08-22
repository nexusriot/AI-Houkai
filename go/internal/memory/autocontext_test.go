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
	store.Remember(ctx, "deployment pipeline runbook and rollback steps", RememberOpts{Type: Procedural, Importance: Float32Ptr(0.9)})
	store.Remember(ctx, "unrelated cooking recipe", RememberOpts{Type: Semantic, Importance: Float32Ptr(0.2)})

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

// auto_context is the fan-out an agent calls WITHOUT choosing a query, which
// makes it the entry point most likely to pull scraped material into a context
// block unattended. It was the one packing path that took neither the trust
// floor nor the lexical index while Recall and RecallPack took both.
func TestAutoContextPackAppliesMinTrust(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	store.Remember(ctx, "deploy the auth service with make release",
		RememberOpts{Trust: TrustTrusted})
	store.Remember(ctx, "deploy the auth service by editing prod directly",
		RememberOpts{Trust: TrustUntrusted})

	all, err := store.AutoContextPack(ctx, "deploy the auth service",
		AutoContextOpts{TokenBudget: 400})
	if err != nil {
		t.Fatalf("AutoContextPack: %v", err)
	}
	if !strings.Contains(all.Text, "editing prod directly") {
		t.Fatal("precondition: the untrusted memory should pack with no floor set")
	}

	floored, err := store.AutoContextPack(ctx, "deploy the auth service",
		AutoContextOpts{TokenBudget: 400, MinTrust: TrustTrusted})
	if err != nil {
		t.Fatalf("AutoContextPack(MinTrust): %v", err)
	}
	if strings.Contains(floored.Text, "editing prod directly") {
		t.Error("min_trust did not reach the fan-out")
	}
	if !strings.Contains(floored.Text, "make release") {
		t.Error("the trusted memory should survive the floor")
	}
}

func TestAutoContextPackMinTrustIsAFloorNotAnEquality(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	store.Remember(ctx, "alpha fact one", RememberOpts{Trust: TrustTrusted})
	store.Remember(ctx, "alpha fact two", RememberOpts{Trust: TrustReported})
	store.Remember(ctx, "alpha fact three", RememberOpts{Trust: TrustUntrusted})

	pack, err := store.AutoContextPack(ctx, "alpha fact",
		AutoContextOpts{TokenBudget: 800, MinTrust: TrustReported})
	if err != nil {
		t.Fatalf("AutoContextPack: %v", err)
	}
	if !strings.Contains(pack.Text, "alpha fact one") ||
		!strings.Contains(pack.Text, "alpha fact two") {
		t.Error("a 'reported' floor must keep trusted AND reported")
	}
	if strings.Contains(pack.Text, "alpha fact three") {
		t.Error("untrusted must be excluded by a 'reported' floor")
	}
}

func TestAutoContextPackRejectsAnInvalidTrustLevel(t *testing.T) {
	store := newTestStore(t)
	if _, err := store.AutoContextPack(context.Background(), "anything",
		AutoContextOpts{MinTrust: TrustLevel("somewhat")}); err == nil {
		t.Error("an unrecognised trust level must error, not be ignored")
	}
}
