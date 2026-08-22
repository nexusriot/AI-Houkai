package memory

import (
	"context"
	"fmt"
	"testing"
)

// Full-corpus lexical recall via the backend's document-content predicate,
// rather than a second index. See docs/DESIGN.md §25 for why.

func TestLexicalCorpusReachesOutsideThePool(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	target, _, _, err := store.Remember(ctx, "the quetzalcoatlus deployment checklist",
		RememberOpts{})
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 60; i++ {
		if _, _, _, err := store.Remember(ctx, "unrelated filler note", RememberOpts{}); err != nil {
			t.Fatal(err)
		}
	}

	// Overfetch 1 with k=1 makes the vector pool one row wide, so in a
	// 61-memory corpus only the lexical channel can surface this.
	w := DefaultWeights()
	w.Cosine, w.Lexical, w.Recency, w.Importance = 0.2, 0.6, 0.1, 0.1
	hits, err := store.Recall(ctx, "quetzalcoatlus", 1, RecallOpts{
		Mode: ModeHybrid, Overfetch: 1, Weights: w,
		LexicalIndex: LexicalCorpus,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 1 || hits[0].ID != target.ID {
		t.Fatalf("corpus-lexical recall = %v, want [%s]", idsOf(hits), target.ID)
	}
}

func TestLexicalPoolIsTheDefault(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	if _, _, _, err := store.Remember(ctx, "default lexical subject", RememberOpts{}); err != nil {
		t.Fatal(err)
	}
	// Without LexicalCorpus the candidate helper must not be consulted at all;
	// asserting on behaviour, the call simply succeeds with pool-only scoring.
	hits, err := store.Recall(ctx, "default lexical subject", 1,
		RecallOpts{Mode: ModeHybrid})
	if err != nil || len(hits) != 1 {
		t.Fatalf("pool-only recall = %v (%v)", idsOf(hits), err)
	}
}

func TestLexicalCandidatesSkipsShortTokensAndCaps(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	if _, _, _, err := store.Remember(ctx,
		"alpha bravo charlie delta echo foxtrot golf hotel", RememberOpts{}); err != nil {
		t.Fatal(err)
	}
	// Two-letter tokens match nearly everything, so they are not probed.
	if got := store.lexicalCandidates(ctx, "an ox", 10); len(got) != 0 {
		t.Errorf("short-token probe returned %d items, want 0", len(got))
	}
	// A long query must not become a long series of scans; the cap bounds it.
	got := store.lexicalCandidates(ctx,
		"alpha bravo charlie delta echo foxtrot golf hotel", 10)
	if len(got) == 0 {
		t.Error("expected at least one candidate for a matching query")
	}
}

func TestLexicalCandidatesNoMatchIsEmpty(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	if _, _, _, err := store.Remember(ctx, "something else entirely", RememberOpts{}); err != nil {
		t.Fatal(err)
	}
	if got := store.lexicalCandidates(ctx, "zzzznomatchzzzz", 10); len(got) != 0 {
		t.Errorf("no-match probe returned %d items, want 0", len(got))
	}
}

func TestSearchDocumentsOnEmptyBackend(t *testing.T) {
	store := newTestStore(t)
	got := store.lexicalCandidates(context.Background(), "anything", 10)
	if len(got) != 0 {
		t.Errorf("empty store returned %d candidates", len(got))
	}
}

// The fast path fetches exactly k and skips the over-fetch pool, so every
// post-query filter must appear in the noPostFilter guard. A missing one
// silently under-returns rather than failing.

func TestFastPathRespectsMinTrust(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	// Unique text per row: identical texts hash to identical vectors, so a
	// corpus of two identical-vector clusters can put every over-fetched row in
	// the same cluster and the test would prove nothing.
	for i := 0; i < 20; i++ {
		if _, _, _, err := store.Remember(ctx,
			fmt.Sprintf("widget note untrusted %d", i),
			RememberOpts{Trust: "untrusted"}); err != nil {
			t.Fatal(err)
		}
	}
	for i := 0; i < 20; i++ {
		if _, _, _, err := store.Remember(ctx,
			fmt.Sprintf("widget note trusted %d", i),
			RememberOpts{Trust: "trusted"}); err != nil {
			t.Fatal(err)
		}
	}

	hits, err := store.Recall(ctx, "widget note", 5, RecallOpts{
		Mode: ModeSemantic, IncludeSuperseded: true, IncludeExpired: true,
		MinTrust: "trusted",
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 5 {
		t.Errorf("fast path returned %d of k=5 with a trust floor set", len(hits))
	}
	for _, h := range hits {
		if h.Trust != "trusted" {
			t.Errorf("returned a %q memory under MinTrust=trusted", h.Trust)
		}
	}
}

func TestFastPathRespectsCorpusLexical(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	for i := 0; i < 30; i++ {
		if _, _, _, err := store.Remember(ctx,
			fmt.Sprintf("filler note %d", i), RememberOpts{}); err != nil {
			t.Fatal(err)
		}
	}
	// LexicalCorpus unions extra candidates in, so the pool must be over-fetched
	// rather than sized to exactly k.
	hits, err := store.Recall(ctx, "filler", 5, RecallOpts{
		Mode: ModeSemantic, IncludeSuperseded: true, IncludeExpired: true,
		LexicalIndex: LexicalCorpus,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 5 {
		t.Errorf("got %d hits, want 5", len(hits))
	}
}

// "corpus" replaced an earlier "fts" spelling. Unvalidated, that name is
// silently read as "pool", so a caller carrying it forward loses full-corpus
// recall with no error. Mode, fusion, type and min_trust are all validated;
// this belongs with them.
func TestRecallRejectsAnUnknownLexicalIndex(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	if _, _, _, err := store.Remember(ctx, "something", RememberOpts{}); err != nil {
		t.Fatal(err)
	}

	for _, bad := range []LexicalIndexMode{"fts", "typo-nonsense"} {
		_, err := store.Recall(ctx, "something", 1, RecallOpts{
			Mode: ModeHybrid, LexicalIndex: bad,
		})
		if err == nil {
			t.Errorf("Recall(lexical_index=%q) succeeded, want a validation error", bad)
		}
	}
}

// "" means "not specified" and must keep defaulting to pool.
func TestRecallAcceptsTheEmptyLexicalIndex(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	if _, _, _, err := store.Remember(ctx, "something", RememberOpts{}); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Recall(ctx, "something", 1, RecallOpts{Mode: ModeHybrid}); err != nil {
		t.Errorf("Recall with no lexical_index: %v", err)
	}
}
