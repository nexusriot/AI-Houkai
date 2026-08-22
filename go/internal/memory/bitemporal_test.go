package memory

import (
	"context"
	"testing"
)

// Bi-temporal validity: when a memory was true, as distinct from when we learned
// it. The journal already records TRANSACTION time — StateAt replays it to
// answer "as of when we knew". These cover VALID time: the half-open interval
// [ValidFrom, ValidUntil) and RecallOpts.AsOf asking what held at a moment.
//
// Mirrors tests/test_bitemporal.py in the Python port.

const (
	jan = 1_700_000_000.0
	feb = jan + 30*86_400
	mar = jan + 60*86_400
	apr = jan + 90*86_400
)

func f64(v float64) *float64 { return &v }

// recalledIDs runs a recall and returns the set of ids it produced.
func recalledIDs(t *testing.T, s *MemoryStore, q string, opts RecallOpts) map[string]bool {
	t.Helper()
	res, err := s.Recall(context.Background(), q, 10, opts)
	if err != nil {
		t.Fatalf("Recall(%q): %v", q, err)
	}
	out := map[string]bool{}
	for _, r := range res {
		out[r.ID] = true
	}
	return out
}

func TestValidityDefaultsAreUnbounded(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	m, _, _, err := store.Remember(ctx, "no validity recorded", RememberOpts{})
	if err != nil {
		t.Fatal(err)
	}
	if m.ValidFrom != 0 || m.ValidUntil != 0 {
		t.Errorf("defaults = %v..%v, want 0..0", m.ValidFrom, m.ValidUntil)
	}
	// Unbounded means valid at any instant asked about.
	for _, ts := range []float64{1, jan, apr} {
		if !recalledIDs(t, store, "validity recorded", RecallOpts{AsOf: ts})[m.ID] {
			t.Errorf("unbounded memory missing at as_of=%v", ts)
		}
	}
}

func TestValidityRoundTripsThroughMetadata(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	m, _, _, err := store.Remember(ctx, "the office is in Berlin",
		RememberOpts{ValidFrom: f64(jan), ValidUntil: f64(mar)})
	if err != nil {
		t.Fatal(err)
	}
	got, err := store.GetByID(ctx, m.ID)
	if err != nil {
		t.Fatal(err)
	}
	if got.ValidFrom != jan || got.ValidUntil != mar {
		t.Errorf("read back %v..%v, want %v..%v",
			got.ValidFrom, got.ValidUntil, jan, mar)
	}
}

func TestValidityRoundTripsThroughDict(t *testing.T) {
	// ToDict/MemoryFromDict is what the journal and export/import use.
	m := Memory{ID: "m1", Text: "t", Type: Semantic, ValidFrom: jan, ValidUntil: mar}
	back := MemoryFromDict(m.ToDict())
	if back.ValidFrom != jan || back.ValidUntil != mar {
		t.Errorf("dict round trip lost validity: %v..%v", back.ValidFrom, back.ValidUntil)
	}
}

func TestValidityRejectsAnInvertedInterval(t *testing.T) {
	store := newTestStore(t)
	if _, _, _, err := store.Remember(context.Background(), "backwards",
		RememberOpts{ValidFrom: f64(mar), ValidUntil: f64(jan)}); err == nil {
		t.Error("valid_until <= valid_from must be rejected")
	}
}

func TestValidityRejectsAZeroLengthInterval(t *testing.T) {
	// Half-open means [T, T) contains nothing — a memory true at no instant is
	// a caller bug, not a memory worth storing.
	store := newTestStore(t)
	if _, _, _, err := store.Remember(context.Background(), "instantaneous",
		RememberOpts{ValidFrom: f64(jan), ValidUntil: f64(jan)}); err == nil {
		t.Error("a zero-length interval must be rejected")
	}
}

func TestValidityRejectsNegativeBounds(t *testing.T) {
	store := newTestStore(t)
	if _, _, _, err := store.Remember(context.Background(), "negative",
		RememberOpts{ValidFrom: f64(-1)}); err == nil {
		t.Error("a negative bound must be rejected")
	}
}

func TestRecallRejectsNegativeAsOf(t *testing.T) {
	store := newTestStore(t)
	if _, err := store.Recall(context.Background(), "anything", 5,
		RecallOpts{AsOf: -1}); err == nil {
		t.Error("as_of < 0 must be rejected")
	}
}

// [from, until) — so two facts that succeed one another can share a boundary
// instant without both being true at it.
func TestValidityIntervalIsHalfOpen(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	first, _, _, _ := store.Remember(ctx, "the office is in Berlin",
		RememberOpts{ValidFrom: f64(jan), ValidUntil: f64(mar)})
	second, _, _, _ := store.Remember(ctx, "the office is in Munich",
		RememberOpts{ValidFrom: f64(mar)})

	atBoundary := recalledIDs(t, store, "where is the office", RecallOpts{AsOf: mar})
	if !atBoundary[second.ID] {
		t.Error("valid_from is inclusive: the successor must be current at the boundary")
	}
	if atBoundary[first.ID] {
		t.Error("valid_until is exclusive: the predecessor must not survive the boundary")
	}
}

func TestAsOfSelectsTheFactThatHeldThen(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	old, _, _, _ := store.Remember(ctx, "the office is in Berlin",
		RememberOpts{ValidFrom: f64(jan), ValidUntil: f64(mar)})
	fresh, _, _, _ := store.Remember(ctx, "the office is in Munich",
		RememberOpts{ValidFrom: f64(mar)})

	february := recalledIDs(t, store, "where is the office", RecallOpts{AsOf: feb})
	if !february[old.ID] || february[fresh.ID] {
		t.Error("as_of=February must return the Berlin fact and only it")
	}
	april := recalledIDs(t, store, "where is the office", RecallOpts{AsOf: apr})
	if !april[fresh.ID] || april[old.ID] {
		t.Error("as_of=April must return the Munich fact and only it")
	}
}

// Closing valid_until retires a fact without deleting it — that is the
// difference from TTL, which reclaims the row.
func TestARetiredFactDropsOutOfOrdinaryRecall(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	retired, _, _, _ := store.Remember(ctx, "the office is in Berlin",
		RememberOpts{ValidFrom: f64(jan), ValidUntil: f64(mar)})

	if recalledIDs(t, store, "where is the office", RecallOpts{})[retired.ID] {
		t.Error("a fact past its valid_until must not appear in a default recall")
	}
	if _, err := store.GetByID(ctx, retired.ID); err != nil {
		t.Error("retiring must not delete the row")
	}
	if !recalledIDs(t, store, "where is the office", RecallOpts{AsOf: feb})[retired.ID] {
		t.Error("a retired fact must still be reachable by asking about the past")
	}
}

func TestAFutureFactIsNotYetCurrent(t *testing.T) {
	store := newTestStore(t)
	future, _, _, _ := store.Remember(context.Background(),
		"the office moves to Munich next year", RememberOpts{ValidFrom: f64(apr * 10)})
	if recalledIDs(t, store, "office munich", RecallOpts{})[future.ID] {
		t.Error("a memory whose valid_from is in the future must not be current")
	}
}

// The subtle half, and the reason AsOf is not just a metadata filter.
func TestAsOfReturnsASupersededMemoryValidAtThatTime(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	old, _, _, _ := store.Remember(ctx, "the office is in Berlin",
		RememberOpts{ValidFrom: f64(jan), ValidUntil: f64(mar)})
	fresh, _, _, _ := store.Remember(ctx, "the office is in Munich",
		RememberOpts{ValidFrom: f64(mar)})
	if err := store.Supersede(ctx, old.ID, fresh.ID); err != nil {
		t.Fatal(err)
	}

	if recalledIDs(t, store, "where is the office", RecallOpts{})[old.ID] {
		t.Error("ordinary recall must still hide a superseded memory")
	}
	if !recalledIDs(t, store, "where is the office", RecallOpts{AsOf: feb})[old.ID] {
		t.Error("as_of must return a superseded memory that WAS true then")
	}
}

func TestAsOfStillHidesASupersededMemoryNotValidThen(t *testing.T) {
	// Admitting superseded rows must not become "return everything".
	store := newTestStore(t)
	ctx := context.Background()
	old, _, _, _ := store.Remember(ctx, "the office is in Berlin",
		RememberOpts{ValidFrom: f64(jan), ValidUntil: f64(feb)})
	fresh, _, _, _ := store.Remember(ctx, "the office is in Munich",
		RememberOpts{ValidFrom: f64(feb)})
	if err := store.Supersede(ctx, old.ID, fresh.ID); err != nil {
		t.Fatal(err)
	}

	march := recalledIDs(t, store, "where is the office", RecallOpts{AsOf: mar})
	if march[old.ID] {
		t.Error("a superseded memory outside the asked interval must stay hidden")
	}
	if !march[fresh.ID] {
		t.Error("the memory valid at that time must be returned")
	}
}

func TestWithoutAsOfSupersedeHidingIsUnchanged(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	old, _, _, _ := store.Remember(ctx, "superseded and unbounded", RememberOpts{})
	fresh, _, _, _ := store.Remember(ctx, "the replacement", RememberOpts{})
	if err := store.Supersede(ctx, old.ID, fresh.ID); err != nil {
		t.Fatal(err)
	}
	if recalledIDs(t, store, "superseded", RecallOpts{})[old.ID] {
		t.Error("without as_of, a superseded memory must stay hidden")
	}
}

func TestEditClosingAnIntervalRetiresTheFact(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	m, _, _, _ := store.Remember(ctx, "the office is in Berlin",
		RememberOpts{ValidFrom: f64(jan)})
	if !recalledIDs(t, store, "where is the office", RecallOpts{})[m.ID] {
		t.Fatal("precondition: an open-ended fact is current")
	}

	if _, err := store.Edit(ctx, m.ID, EditOpts{ValidUntil: f64(mar)}); err != nil {
		t.Fatal(err)
	}
	if recalledIDs(t, store, "where is the office", RecallOpts{})[m.ID] {
		t.Error("closing valid_until must retire the fact from default recall")
	}
	if !recalledIDs(t, store, "where is the office", RecallOpts{AsOf: feb})[m.ID] {
		t.Error("the retired fact must stay reachable via as_of")
	}
}

func TestEditReopeningAnEndMakesItCurrentAgain(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	m, _, _, _ := store.Remember(ctx, "temporarily retired",
		RememberOpts{ValidFrom: f64(jan), ValidUntil: f64(mar)})
	if recalledIDs(t, store, "temporarily retired", RecallOpts{})[m.ID] {
		t.Fatal("precondition: a closed interval is not current")
	}
	if _, err := store.Edit(ctx, m.ID, EditOpts{ValidUntil: f64(0)}); err != nil {
		t.Fatal(err)
	}
	if !recalledIDs(t, store, "temporarily retired", RecallOpts{})[m.ID] {
		t.Error("reopening valid_until must make the fact current again")
	}
}

func TestEditValidatesOneEndAgainstTheOther(t *testing.T) {
	// Closing an end must not silently produce until <= from.
	store := newTestStore(t)
	ctx := context.Background()
	m, _, _, _ := store.Remember(ctx, "bounded below", RememberOpts{ValidFrom: f64(mar)})
	if _, err := store.Edit(ctx, m.ID, EditOpts{ValidUntil: f64(jan)}); err == nil {
		t.Error("an edit producing until <= from must be rejected")
	}
}

func TestRememberManyCarriesPerItemValidity(t *testing.T) {
	store := newTestStore(t)
	mems, err := store.RememberMany(context.Background(), []RememberItem{
		{Text: "berlin era", RememberOpts: RememberOpts{ValidFrom: f64(jan), ValidUntil: f64(mar)}},
		{Text: "munich era", RememberOpts: RememberOpts{ValidFrom: f64(mar)}},
	}, 128, "", false)
	if err != nil {
		t.Fatal(err)
	}
	if len(mems) != 2 {
		t.Fatalf("stored %d, want 2", len(mems))
	}
	if mems[0].ValidFrom != jan || mems[0].ValidUntil != mar {
		t.Errorf("item 0 = %v..%v", mems[0].ValidFrom, mems[0].ValidUntil)
	}
	if mems[1].ValidFrom != mar || mems[1].ValidUntil != 0 {
		t.Errorf("item 1 = %v..%v", mems[1].ValidFrom, mems[1].ValidUntil)
	}
}

func TestRecallFastPathValidityShortfall(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	from, until := 1.0, 2.0
	retired, _, _, err := store.Remember(ctx, "quarterly revenue target details",
		RememberOpts{ValidFrom: &from, ValidUntil: &until})
	if err != nil {
		t.Fatal(err)
	}
	live, _, _, _ := store.Remember(ctx, "quarterly revenue target", RememberOpts{})

	// The retired row ranks first (exact text match); the fast path fetches
	// exactly k=1, the validity filter drops it, and pre-fix the result was
	// empty while the live row was never fetched.
	hits, err := store.Recall(ctx, "quarterly revenue target details", 1, RecallOpts{
		Mode: ModeSemantic, IncludeSuperseded: true, IncludeExpired: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 1 || hits[0].Memory.ID != live.ID {
		t.Fatalf("hits = %+v, want exactly the live memory", hits)
	}
	_ = retired

	// With every row retired the result is legitimately empty.
	if _, err := store.Forget(ctx, live.ID); err != nil {
		t.Fatal(err)
	}
	hits, err = store.Recall(ctx, "quarterly revenue target details", 1, RecallOpts{
		Mode: ModeSemantic, IncludeSuperseded: true, IncludeExpired: true,
	})
	if err != nil || len(hits) != 0 {
		t.Fatalf("hits = %+v err=%v, want empty", hits, err)
	}
}
