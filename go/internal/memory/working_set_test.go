package memory

import (
	"context"
	"strings"
	"testing"
)

// Pinned tier (F3), idempotent writes (F4) and the trust tier (G).
//
// Importance was doing three jobs at once — ranking, decay survival and the
// MinImportance filter — so protecting a standing instruction meant distorting
// every search it appeared in. Agents re-assert the same fact every session,
// and the only defence was a per-write vector conflict scan that still created
// a row. And anything reaching Remember becomes durable, well-ranked agent
// context later, so a fact scraped from a page and one stated by the user have
// to be distinguishable at recall time.

func TestContentHashNormalises(t *testing.T) {
	if ContentHash("Use ruff for linting") != ContentHash("use  ruff\nfor linting  ") {
		t.Error("case and whitespace must not change the hash")
	}
	if ContentHash("alpha") == ContentHash("beta") {
		t.Error("different text must hash differently")
	}
	// Pinned so the two ports keep agreeing: a hash written by Python must be
	// recognised here (sha256 of the normalised text, truncated to 16 bytes).
	if got := ContentHash("Use ruff for linting"); got != "b23f48edab8281612dce603cbc7e34b8" {
		t.Errorf("ContentHash drifted from the Python port: %s", got)
	}
}

func TestIdempotentRemember(t *testing.T) {
	ctx := context.Background()
	store := newTestStore(t)

	a, stored, _, err := store.Remember(ctx, "Use ruff for linting",
		RememberOpts{Idempotent: true})
	if err != nil || !stored {
		t.Fatalf("first write: stored=%v err=%v", stored, err)
	}
	b, stored, _, err := store.Remember(ctx, "use  ruff for linting",
		RememberOpts{Idempotent: true})
	if err != nil {
		t.Fatal(err)
	}
	if stored {
		t.Error("a repeat must not report a new write")
	}
	if a.ID != b.ID {
		t.Errorf("ids differ: %s vs %s", a.ID, b.ID)
	}
	if n, _ := store.Count(ctx); n != 1 {
		t.Errorf("count = %d, want 1", n)
	}
	if b.AccessCount != 1 {
		t.Errorf("access_count = %d, want the repeat to bump it", b.AccessCount)
	}
}

func TestIdempotentIsOffByDefault(t *testing.T) {
	ctx := context.Background()
	store := newTestStore(t)
	store.Remember(ctx, "plain write", RememberOpts{})
	store.Remember(ctx, "plain write", RememberOpts{})
	if n, _ := store.Count(ctx); n != 2 {
		t.Errorf("count = %d, want 2 — dedup must be opt-in", n)
	}
}

func TestIdempotentSkipsSupersededRows(t *testing.T) {
	// Re-asserting a fact that was explicitly replaced should create a new
	// memory, not resurrect the old one.
	ctx := context.Background()
	store := newTestStore(t)
	old, _, _, _ := store.Remember(ctx, "the policy", RememberOpts{Idempotent: true})
	fresh, _, _, _ := store.Remember(ctx, "the replacement", RememberOpts{})
	if err := store.Supersede(ctx, old.ID, fresh.ID); err != nil {
		t.Fatal(err)
	}
	again, _, _, _ := store.Remember(ctx, "the policy", RememberOpts{Idempotent: true})
	if again.ID == old.ID {
		t.Error("a superseded row absorbed the repeat")
	}
}

func TestIdempotentHashFollowsAnEdit(t *testing.T) {
	// Otherwise an edited memory would still answer to its original text.
	ctx := context.Background()
	store := newTestStore(t)
	m, _, _, _ := store.Remember(ctx, "original wording", RememberOpts{Idempotent: true})
	revised := "revised wording"
	if _, err := store.Edit(ctx, m.ID, EditOpts{Text: &revised}); err != nil {
		t.Fatal(err)
	}
	got, _ := store.GetByID(ctx, m.ID)
	if got.ContentHash != ContentHash(revised) {
		t.Errorf("hash = %s, want the revised text's", got.ContentHash)
	}
	again, _, _, _ := store.Remember(ctx, "original wording", RememberOpts{Idempotent: true})
	if again.ID == m.ID {
		t.Error("the stale hash still matched")
	}
}

func TestPinnedRoundtripsAndEdits(t *testing.T) {
	ctx := context.Background()
	store := newTestStore(t)
	plain, _, _, _ := store.Remember(ctx, "ordinary", RememberOpts{})
	pinned, _, _, _ := store.Remember(ctx, "standing instruction",
		RememberOpts{Pinned: true})

	if got, _ := store.GetByID(ctx, plain.ID); got.Pinned {
		t.Error("pinned must default to false")
	}
	if got, _ := store.GetByID(ctx, pinned.ID); !got.Pinned {
		t.Error("pinned did not round-trip")
	}

	no := false
	if _, err := store.Edit(ctx, pinned.ID, EditOpts{Pinned: &no}); err != nil {
		t.Fatal(err)
	}
	if got, _ := store.GetByID(ctx, pinned.ID); got.Pinned {
		t.Error("unpinning did not take")
	}
}

func TestPinnedPrependedToAPack(t *testing.T) {
	ctx := context.Background()
	store := newTestStore(t)
	store.Remember(ctx, "ALWAYS run make lint", RememberOpts{
		Pinned: true, Type: Procedural,
	})
	store.Remember(ctx, "unrelated gardening note", RememberOpts{})

	pack, err := store.RecallPack(ctx, "gardening", PackOpts{IncludePinned: true})
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(pack.Text, "\n")
	if len(lines) < 2 || !strings.HasPrefix(lines[1], "- (procedural) [pinned]") {
		t.Errorf("pinned memory not first: %q", pack.Text)
	}
}

func TestPinnedNotDuplicatedWhenItAlsoMatches(t *testing.T) {
	ctx := context.Background()
	store := newTestStore(t)
	m, _, _, _ := store.Remember(ctx, "the pinned and matching subject",
		RememberOpts{Pinned: true})
	pack, err := store.RecallPack(ctx, "the pinned and matching subject",
		PackOpts{IncludePinned: true})
	if err != nil {
		t.Fatal(err)
	}
	seen := 0
	for _, p := range pack.Items {
		if p.Memory.ID == m.ID {
			seen++
		}
	}
	if seen != 1 {
		t.Errorf("pinned memory appears %d times", seen)
	}
}

func TestPinnedStillRespectsTheBudget(t *testing.T) {
	// A standing instruction is prioritised, not exempt.
	ctx := context.Background()
	store := newTestStore(t)
	store.Remember(ctx, strings.Repeat("x", 400), RememberOpts{Pinned: true})
	pack, err := store.RecallPack(ctx, "anything", PackOpts{
		TokenBudget: 10, IncludePinned: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if pack.UsedTokens > 10 {
		t.Errorf("used %d tokens against a budget of 10", pack.UsedTokens)
	}
}

func TestTrustDefaultsAndRoundtrips(t *testing.T) {
	ctx := context.Background()
	store := newTestStore(t)
	m, _, _, _ := store.Remember(ctx, "from the user", RememberOpts{})
	if m.Trust != TrustTrusted {
		t.Errorf("default trust = %q, want trusted", m.Trust)
	}
	for _, level := range []TrustLevel{TrustTrusted, TrustReported, TrustUntrusted} {
		made, _, _, err := store.Remember(ctx, "a "+string(level)+" memory",
			RememberOpts{Trust: level})
		if err != nil {
			t.Fatalf("remember(%s): %v", level, err)
		}
		got, _ := store.GetByID(ctx, made.ID)
		if got.Trust != level {
			t.Errorf("trust = %q, want %q", got.Trust, level)
		}
	}
}

func TestTrustRejectsAnUnknownLevel(t *testing.T) {
	ctx := context.Background()
	store := newTestStore(t)
	if _, _, _, err := store.Remember(ctx, "bad",
		RememberOpts{Trust: "probably-fine"}); err == nil {
		t.Error("an unknown trust level must be rejected")
	}
}

func TestMinTrustFiltersRecall(t *testing.T) {
	ctx := context.Background()
	store := newTestStore(t)
	user, _, _, _ := store.Remember(ctx, "stated by the user",
		RememberOpts{Trust: TrustTrusted})
	tool, _, _, _ := store.Remember(ctx, "relayed by a tool",
		RememberOpts{Trust: TrustReported})
	store.Remember(ctx, "scraped from a page", RememberOpts{Trust: TrustUntrusted})

	all, err := store.Recall(ctx, "a memory", 5, RecallOpts{})
	if err != nil || len(all) != 3 {
		t.Fatalf("unfiltered = %d (%v)", len(all), err)
	}
	strict, _ := store.Recall(ctx, "a memory", 5, RecallOpts{MinTrust: TrustTrusted})
	if len(strict) != 1 || strict[0].ID != user.ID {
		t.Errorf("min_trust=trusted = %d results", len(strict))
	}
	moderate, _ := store.Recall(ctx, "a memory", 5, RecallOpts{MinTrust: TrustReported})
	if len(moderate) != 2 {
		t.Errorf("min_trust=reported = %d results", len(moderate))
	}
	ids := map[string]bool{}
	for _, r := range moderate {
		ids[r.ID] = true
	}
	if !ids[user.ID] || !ids[tool.ID] {
		t.Errorf("min_trust=reported dropped the wrong rows: %v", ids)
	}
}

func TestMinTrustRejectsAnUnknownLevel(t *testing.T) {
	ctx := context.Background()
	store := newTestStore(t)
	if _, err := store.Recall(ctx, "q", 5, RecallOpts{MinTrust: "nonsense"}); err == nil {
		t.Error("an unknown min_trust must be rejected")
	}
}

func TestPackMarksUntrustedLines(t *testing.T) {
	// The packed block goes straight into a model's context; a scraped fact
	// must not be indistinguishable there from a user-stated one.
	ctx := context.Background()
	store := newTestStore(t)
	store.Remember(ctx, "scraped claim", RememberOpts{Trust: TrustUntrusted})
	pack, err := store.RecallPack(ctx, "scraped claim", PackOpts{})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(pack.Text, "[untrusted]") {
		t.Errorf("untrusted line not marked: %q", pack.Text)
	}

	store2 := newTestStore(t)
	store2.Remember(ctx, "user stated fact", RememberOpts{})
	pack2, _ := store2.RecallPack(ctx, "user stated fact", PackOpts{})
	if strings.Contains(pack2.Text, "[") {
		t.Errorf("trusted line should carry no mark: %q", pack2.Text)
	}
}

func TestOldRowsReadAsTrusted(t *testing.T) {
	// An existing store must not change behaviour just by being opened.
	m := MetadataToMemory("legacy-1", "written before trust existed",
		map[string]string{
			"type": "semantic", "tags": "", "importance": "0.5",
			"created_at": "1", "last_accessed": "1", "access_count": "0",
			"source": "", "links": "[]", "superseded_by": "",
			"superseded_at": "0", "polarity": "0",
		})
	if m.Trust != TrustTrusted {
		t.Errorf("legacy trust = %q, want trusted", m.Trust)
	}
	if m.Pinned {
		t.Error("legacy pinned must be false")
	}
	if m.ContentHash != "" {
		t.Error("legacy content_hash must be empty")
	}
	if TrustRank(m.Trust) > TrustRank(TrustTrusted) {
		t.Error("a legacy row must survive min_trust=trusted")
	}
}
