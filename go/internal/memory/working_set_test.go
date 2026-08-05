package memory

import (
	"context"
	"fmt"
	"path/filepath"
	"strings"
	"testing"

	"github.com/nexusriot/ai-houkai/internal/vector"
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

// A pinned memory is a standing-instruction slot (docs/DESIGN.md §27).
// Superseding one is how you correct a standing instruction, but the pin was
// dropped and a superseded row is out of the working set — so the slot silently
// emptied and the agent stopped seeing the rule.
func TestSupersedeCarriesThePin(t *testing.T) {
	ctx := context.Background()
	store := newTestStore(t)

	old, _, _, err := store.Remember(ctx, "indent with tabs",
		RememberOpts{Pinned: true})
	if err != nil {
		t.Fatalf("Remember: %v", err)
	}
	fresh, _, _, err := store.Remember(ctx, "indent with four spaces", RememberOpts{})
	if err != nil {
		t.Fatalf("Remember: %v", err)
	}
	if err := store.Supersede(ctx, old.ID, fresh.ID); err != nil {
		t.Fatalf("Supersede: %v", err)
	}

	got, err := store.GetByID(ctx, fresh.ID)
	if err != nil {
		t.Fatalf("GetByID: %v", err)
	}
	if !got.Pinned {
		t.Error("replacement is not pinned — the standing-instruction slot emptied")
	}
	pinned, err := store.PinnedMemories(ctx)
	if err != nil {
		t.Fatalf("PinnedMemories: %v", err)
	}
	if len(pinned) != 1 || pinned[0].ID != fresh.ID {
		t.Fatalf("working set = %+v, want just the replacement", pinned)
	}
}

func TestSupersedeOfAnUnpinnedMemoryPinsNothing(t *testing.T) {
	ctx := context.Background()
	store := newTestStore(t)

	old, _, _, _ := store.Remember(ctx, "a plain fact", RememberOpts{})
	fresh, _, _, _ := store.Remember(ctx, "a corrected plain fact", RememberOpts{})
	if err := store.Supersede(ctx, old.ID, fresh.ID); err != nil {
		t.Fatalf("Supersede: %v", err)
	}

	got, _ := store.GetByID(ctx, fresh.ID)
	if got.Pinned {
		t.Error("replacement was pinned out of nowhere")
	}
}

// Supersede keeps both rows, each with its own provenance — unlike merge, which
// folds two into one and must take the worse label.
func TestSupersedeDoesNotCarryTrust(t *testing.T) {
	ctx := context.Background()
	store := newTestStore(t)

	old, _, _, _ := store.Remember(ctx, "reported claim",
		RememberOpts{Pinned: true, Trust: TrustReported})
	fresh, _, _, _ := store.Remember(ctx, "verified replacement", RememberOpts{})
	if err := store.Supersede(ctx, old.ID, fresh.ID); err != nil {
		t.Fatalf("Supersede: %v", err)
	}

	got, _ := store.GetByID(ctx, fresh.ID)
	if got.Trust != TrustTrusted {
		t.Errorf("trust = %q, want %q", got.Trust, TrustTrusted)
	}
}

// Undoing a supersede must not leave two rows of one chain in the working set.
func TestRestoreHandsThePinBack(t *testing.T) {
	ctx := context.Background()
	store := newTestStore(t)

	old, _, _, _ := store.Remember(ctx, "original rule", RememberOpts{Pinned: true})
	fresh, _, _, _ := store.Remember(ctx, "replacement rule", RememberOpts{})
	if err := store.Supersede(ctx, old.ID, fresh.ID); err != nil {
		t.Fatalf("Supersede: %v", err)
	}
	if ok, err := store.Restore(ctx, old.ID); err != nil || !ok {
		t.Fatalf("Restore = %v, %v", ok, err)
	}

	back, _ := store.GetByID(ctx, old.ID)
	replacement, _ := store.GetByID(ctx, fresh.ID)
	if !back.Pinned {
		t.Error("restored memory lost its pin")
	}
	if replacement.Pinned {
		t.Error("replacement kept the inherited pin after the undo")
	}
	pinned, _ := store.PinnedMemories(ctx)
	if len(pinned) != 1 || pinned[0].ID != old.ID {
		t.Fatalf("working set = %+v, want just the restored memory", pinned)
	}
}

// countingBackend records how many rows each backend call had to materialise,
// so a hot-path lookup can be shown to push its filter into the store rather
// than loading the collection into Go.
type countingBackend struct {
	vector.Backend
	allRows      int
	allCalls     int
	metadataHits int
}

func (b *countingBackend) All(ctx context.Context) ([]vector.Item, error) {
	items, err := b.Backend.All(ctx)
	b.allCalls++
	b.allRows += len(items)
	return items, err
}

func (b *countingBackend) SearchMetadata(ctx context.Context, where map[string]string,
	limit int) ([]vector.Item, error) {
	items, err := b.Backend.SearchMetadata(ctx, where, limit)
	b.metadataHits++
	return items, err
}

// Python pushes the pinned lookup into Chroma as a `where` filter because it
// sits on the recall_pack / auto_context hot path; this port kept scanning the
// whole collection on every pack call, so the cheapest part of the packer was
// the most expensive. chromem-go's Where is exact-match on string metadata,
// which is exactly what `pinned` is stored as.
func TestPinnedLookupDoesNotScanTheCollection(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	inner, err := vector.NewChromem(filepath.Join(dir, "s"), "test", 16)
	if err != nil {
		t.Fatalf("NewChromem: %v", err)
	}
	t.Cleanup(func() { _ = inner.Close() })
	counting := &countingBackend{Backend: inner}
	store := NewMemoryStore(counting, &stubEmbedder{dim: 16},
		DefaultStoreConfig(dir, "test"))

	for i := 0; i < 30; i++ {
		if _, _, _, err := store.Remember(ctx,
			fmt.Sprintf("filler row %d", i), RememberOpts{}); err != nil {
			t.Fatalf("Remember: %v", err)
		}
	}
	want, _, _, err := store.Remember(ctx, "the standing instruction",
		RememberOpts{Pinned: true})
	if err != nil {
		t.Fatalf("Remember: %v", err)
	}

	counting.allRows, counting.allCalls = 0, 0
	pinned, err := store.PinnedMemories(ctx)
	if err != nil {
		t.Fatalf("PinnedMemories: %v", err)
	}
	if len(pinned) != 1 || pinned[0].ID != want.ID {
		t.Fatalf("pinned = %+v, want just the pinned row", pinned)
	}
	if counting.allCalls != 0 {
		t.Errorf("loaded %d rows via %d All() calls — the filter must run in the "+
			"store", counting.allRows, counting.allCalls)
	}
	if counting.metadataHits == 0 {
		t.Error("no metadata-filtered query was issued")
	}
}

// Superseded and expired rows are not part of the working set even when the
// metadata filter matches them.
func TestPinnedLookupSkipsDeadRows(t *testing.T) {
	ctx := context.Background()
	store := newTestStore(t)

	live, _, _, _ := store.Remember(ctx, "live instruction", RememberOpts{Pinned: true})
	gone, _, _, _ := store.Remember(ctx, "retired instruction", RememberOpts{Pinned: true})
	replacement, _, _, _ := store.Remember(ctx, "its replacement", RememberOpts{})
	if err := store.Supersede(ctx, gone.ID, replacement.ID); err != nil {
		t.Fatalf("Supersede: %v", err)
	}
	expired, _, _, _ := store.Remember(ctx, "expired instruction",
		RememberOpts{Pinned: true, ExpiresAt: f64ptr(1)})

	pinned, err := store.PinnedMemories(ctx)
	if err != nil {
		t.Fatalf("PinnedMemories: %v", err)
	}
	got := map[string]bool{}
	for _, m := range pinned {
		got[m.ID] = true
	}
	if !got[live.ID] {
		t.Error("live pinned memory missing from the working set")
	}
	if got[gone.ID] {
		t.Error("superseded memory is still in the working set")
	}
	if got[expired.ID] {
		t.Error("expired memory is still in the working set")
	}
	// The replacement inherited the pin, so it belongs there.
	if !got[replacement.ID] {
		t.Error("the replacement should hold the inherited pin")
	}
}
