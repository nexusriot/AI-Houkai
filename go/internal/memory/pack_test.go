package memory

import (
	"context"
	"strings"
	"testing"
)

func TestEstimateTokens(t *testing.T) {
	if got := EstimateTokens(strings.Repeat("a", 40)); got != 10 {
		t.Errorf("EstimateTokens(40 chars) = %d, want 10", got)
	}
	if got := EstimateTokens(""); got != 1 {
		t.Errorf("EstimateTokens(\"\") = %d, want 1", got)
	}
	if got := EstimateTokens("a"); got != 1 {
		t.Errorf("EstimateTokens(\"a\") = %d, want 1", got)
	}
}

func seedPack(t *testing.T, store *MemoryStore) {
	t.Helper()
	ctx := context.Background()
	seeds := []struct {
		text string
		typ  MemoryType
		tags []string
		imp  float32
	}{
		{"pytest tmp_path fixture for test isolation", Procedural, []string{"testing"}, 0.9},
		{"Never use EphemeralClient in tests", Procedural, []string{"testing"}, 0.95},
		{"Deploy API with make release", Procedural, []string{"deploy"}, 0.7},
		{"API versioned at /api/v1/", Semantic, []string{"api"}, 0.8},
	}
	for _, s := range seeds {
		if _, _, _, err := store.Remember(ctx, s.text, RememberOpts{
			Type: s.typ, Tags: s.tags, Importance: s.imp,
		}); err != nil {
			t.Fatalf("seed: %v", err)
		}
	}
}

func TestRecallPackCompression(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	// Four identical memories with a short first sentence and a long tail: the
	// full line is expensive, but the compressed first-sentence summary is cheap.
	text := "topic alpha matches. " + strings.Repeat("filler ", 20)
	for i := 0; i < 4; i++ {
		if _, _, _, err := store.Remember(ctx, text, RememberOpts{Type: Semantic}); err != nil {
			t.Fatalf("seed: %v", err)
		}
	}
	// Count tokens as whitespace-separated words for a predictable budget.
	words := func(s string) int { return len(strings.Fields(s)) }

	// Without compression: one item fits, the rest are dropped and lost.
	plain, err := store.RecallPack(ctx, text, PackOpts{
		TokenBudget: 40, TokenCounter: words, Mode: ModeSemantic,
	})
	if err != nil {
		t.Fatalf("RecallPack: %v", err)
	}
	if len(plain.Items) != 1 || !plain.Truncated {
		t.Fatalf("expected 1 item + truncated, got %d items truncated=%v", len(plain.Items), plain.Truncated)
	}
	if len(plain.CompressedGroups) != 0 {
		t.Errorf("compression off should yield no compressed groups")
	}

	// With compression: the three dropped identical memories fold into one line.
	comp, err := store.RecallPack(ctx, text, PackOpts{
		TokenBudget: 40, TokenCounter: words, Mode: ModeSemantic,
		Compress: true, CompressThreshold: 0.30, CompressMinGroup: 2,
	})
	if err != nil {
		t.Fatalf("RecallPack compress: %v", err)
	}
	if len(comp.CompressedGroups) != 1 {
		t.Fatalf("expected 1 compressed group, got %d", len(comp.CompressedGroups))
	}
	if n := len(comp.CompressedGroups[0].Memories); n != 3 {
		t.Errorf("compressed group should hold 3 memories, got %d", n)
	}
	if !strings.Contains(comp.Text, "(compressed)") || !strings.Contains(comp.Text, "[×3 similar]") {
		t.Errorf("compressed text missing markers: %q", comp.Text)
	}
	if comp.UsedTokens > 40 {
		t.Errorf("compression must respect the budget, used %d > 40", comp.UsedTokens)
	}
}

func TestRecallPackReturnsResult(t *testing.T) {
	store := newTestStore(t)
	seedPack(t, store)
	pack, err := store.RecallPack(context.Background(), "testing", PackOpts{TokenBudget: 500})
	if err != nil {
		t.Fatalf("RecallPack: %v", err)
	}
	if pack.Budget != 500 {
		t.Errorf("budget = %d, want 500", pack.Budget)
	}
	if len(pack.Items) == 0 {
		t.Error("expected items in pack")
	}
}

func TestRecallPackEmptyStore(t *testing.T) {
	store := newTestStore(t)
	pack, err := store.RecallPack(context.Background(), "anything", PackOpts{TokenBudget: 500})
	if err != nil {
		t.Fatalf("RecallPack: %v", err)
	}
	if len(pack.Items) != 0 || pack.Text != "" || pack.UsedTokens != 0 || pack.Truncated {
		t.Errorf("expected empty pack, got %+v", pack)
	}
}

func TestRecallPackRespectsBudget(t *testing.T) {
	store := newTestStore(t)
	seedPack(t, store)
	pack, err := store.RecallPack(context.Background(), "testing", PackOpts{TokenBudget: 10, MaxItems: 50})
	if err != nil {
		t.Fatalf("RecallPack: %v", err)
	}
	if pack.UsedTokens > 10 {
		t.Errorf("used %d tokens, budget 10", pack.UsedTokens)
	}
}

func TestRecallPackTinyBudgetTruncates(t *testing.T) {
	store := newTestStore(t)
	seedPack(t, store)
	pack, err := store.RecallPack(context.Background(), "testing", PackOpts{TokenBudget: 1})
	if err != nil {
		t.Fatalf("RecallPack: %v", err)
	}
	if len(pack.Items) != 0 || !pack.Truncated || pack.Text != "" {
		t.Errorf("expected empty truncated pack, got %+v", pack)
	}
}

func TestRecallPackGenerousBudgetNotTruncated(t *testing.T) {
	store := newTestStore(t)
	seedPack(t, store)
	pack, err := store.RecallPack(context.Background(), "testing rules", PackOpts{TokenBudget: 10000, MaxItems: 50})
	if err != nil {
		t.Fatalf("RecallPack: %v", err)
	}
	if pack.Truncated {
		t.Error("expected not truncated")
	}
}

func TestRecallPackHeaderAndLines(t *testing.T) {
	store := newTestStore(t)
	seedPack(t, store)
	header := "## Memory"
	pack, err := store.RecallPack(context.Background(), "testing", PackOpts{TokenBudget: 1000, Header: &header})
	if err != nil {
		t.Fatalf("RecallPack: %v", err)
	}
	if !strings.HasPrefix(pack.Text, "## Memory\n") {
		t.Errorf("text missing header: %q", pack.Text)
	}
	for _, p := range pack.Items {
		if !strings.Contains(pack.Text, p.Memory.Text) {
			t.Errorf("text missing item %q", p.Memory.Text)
		}
	}
}

func TestRecallPackNoHeader(t *testing.T) {
	store := newTestStore(t)
	seedPack(t, store)
	empty := ""
	pack, err := store.RecallPack(context.Background(), "testing", PackOpts{TokenBudget: 1000, Header: &empty})
	if err != nil {
		t.Fatalf("RecallPack: %v", err)
	}
	if strings.HasPrefix(pack.Text, "#") {
		t.Errorf("unexpected header in %q", pack.Text)
	}
	if pack.Text == "" {
		t.Error("expected non-empty body")
	}
}

func TestRecallPackUsedTokensMatchesSum(t *testing.T) {
	store := newTestStore(t)
	seedPack(t, store)
	pack, err := store.RecallPack(context.Background(), "testing", PackOpts{TokenBudget: 1000})
	if err != nil {
		t.Fatalf("RecallPack: %v", err)
	}
	sum := 0
	for _, p := range pack.Items {
		sum += p.Tokens
	}
	if pack.UsedTokens != sum {
		t.Errorf("used_tokens %d != sum %d", pack.UsedTokens, sum)
	}
}

func TestRecallPackCustomTokenCounter(t *testing.T) {
	store := newTestStore(t)
	seedPack(t, store)
	// 1 token per item → budget of 2 admits exactly 2 items.
	pack, err := store.RecallPack(context.Background(), "testing", PackOpts{
		TokenBudget:  2,
		MaxItems:     50,
		TokenCounter: func(string) int { return 1 },
	})
	if err != nil {
		t.Fatalf("RecallPack: %v", err)
	}
	if len(pack.Items) != 2 || pack.UsedTokens != 2 || !pack.Truncated {
		t.Errorf("got items=%d used=%d truncated=%v, want 2/2/true",
			len(pack.Items), pack.UsedTokens, pack.Truncated)
	}
}

func TestRecallPackTypeFilter(t *testing.T) {
	store := newTestStore(t)
	seedPack(t, store)
	pack, err := store.RecallPack(context.Background(), "api", PackOpts{TokenBudget: 1000, Type: Semantic})
	if err != nil {
		t.Fatalf("RecallPack: %v", err)
	}
	for _, p := range pack.Items {
		if p.Memory.Type != Semantic {
			t.Errorf("unexpected type %s", p.Memory.Type)
		}
	}
}

func TestRecallPackExcludesSuperseded(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	old, _, _, err := store.Remember(ctx, "Old testing rule", RememberOpts{Type: Procedural, Tags: []string{"testing"}})
	if err != nil {
		t.Fatal(err)
	}
	niu, _, _, err := store.Remember(ctx, "New testing rule", RememberOpts{Type: Procedural, Tags: []string{"testing"}})
	if err != nil {
		t.Fatal(err)
	}
	if err := store.Supersede(ctx, old.ID, niu.ID); err != nil {
		t.Fatal(err)
	}
	pack, err := store.RecallPack(ctx, "testing rule", PackOpts{TokenBudget: 1000})
	if err != nil {
		t.Fatalf("RecallPack: %v", err)
	}
	for _, id := range pack.IDs() {
		if id == old.ID {
			t.Error("superseded memory present in pack")
		}
	}
}

func TestRecallPackSemanticMode(t *testing.T) {
	store := newTestStore(t)
	seedPack(t, store)
	pack, err := store.RecallPack(context.Background(), "testing", PackOpts{TokenBudget: 1000, Mode: ModeSemantic})
	if err != nil {
		t.Fatalf("RecallPack: %v", err)
	}
	if len(pack.Items) == 0 {
		t.Error("expected items")
	}
}

func TestRecallPackPreservesRankOrder(t *testing.T) {
	store := newTestStore(t)
	seedPack(t, store)
	ctx := context.Background()
	ranked, err := store.Recall(ctx, "testing", 50, RecallOpts{Mode: ModeHybrid, Overfetch: 3})
	if err != nil {
		t.Fatal(err)
	}
	pack, err := store.RecallPack(ctx, "testing", PackOpts{TokenBudget: 10000, MaxItems: 50})
	if err != nil {
		t.Fatal(err)
	}
	pos := func(id string) int {
		for i, r := range ranked {
			if r.ID == id {
				return i
			}
		}
		return -1
	}
	prev := -1
	for _, id := range pack.IDs() {
		p := pos(id)
		if p < prev {
			t.Errorf("pack order violates ranking at id %s", id)
		}
		prev = p
	}
}
