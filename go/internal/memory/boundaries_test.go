package memory

import (
	"context"
	"strings"
	"testing"
)

// A numeric knob supplied by a caller means what it says. Go's zero value is
// indistinguishable from "unset" inside a struct, and that convention had
// leaked out to the HTTP and MCP surfaces, where a client CAN send an explicit
// 0: `?limit=0` returned the whole store, `?depth=0` walked a hop anyway, and
// `token_budget: 0` packed a full context block. The Python port took all three
// literally, so the two ports answered the same request differently.
//
// The zero these tests defend is the one a caller computes — `limit =
// min(remaining, page)`, a depth budget walked down to nothing — where being
// handed everything instead of nothing is the worst possible answer.

func seedN(t *testing.T, store *MemoryStore, n int) []Memory {
	t.Helper()
	ctx := context.Background()
	var out []Memory
	for i := 0; i < n; i++ {
		m, _, _, err := store.Remember(ctx,
			"seeded subject number "+string(rune('a'+i)), RememberOpts{})
		if err != nil {
			t.Fatalf("Remember: %v", err)
		}
		out = append(out, m)
	}
	return out
}

func TestListRecentTakesLimitLiterally(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	seedN(t, store, 3)

	got, err := store.ListRecent(ctx, 0, true, true)
	if err != nil {
		t.Fatalf("ListRecent(0): %v", err)
	}
	if len(got) != 0 {
		t.Errorf("limit=0 returned %d memories, want 0 — a caller whose "+
			"remaining budget reached zero must not be handed the store",
			len(got))
	}

	got, err = store.ListRecent(ctx, 2, true, true)
	if err != nil {
		t.Fatalf("ListRecent(2): %v", err)
	}
	if len(got) != 2 {
		t.Errorf("limit=2 returned %d, want 2", len(got))
	}

	got, err = store.ListRecent(ctx, 99, true, true)
	if err != nil {
		t.Fatalf("ListRecent(99): %v", err)
	}
	if len(got) != 3 {
		t.Errorf("a limit past the end returned %d, want all 3", len(got))
	}
}

func TestListRecentRejectsNegativeLimit(t *testing.T) {
	store := newTestStore(t)
	seedN(t, store, 2)

	_, err := store.ListRecent(context.Background(), -1, true, true)
	if err == nil {
		t.Fatal("ListRecent(-1) should be an error, not the whole store")
	}
	if !IsValidationError(err) {
		t.Errorf("err = %T, want a ValidationError so transports answer 400", err)
	}
}

func TestListRecentPageKeepsZeroMeaningUnbounded(t *testing.T) {
	// The internal paging primitive keeps Go's zero-value convention: an
	// unset Limit field is "no bound". That is what every "give me every
	// row" caller inside the store relies on.
	store := newTestStore(t)
	seedN(t, store, 3)

	got, err := store.ListRecentPage(context.Background(), ListRecentOpts{
		IncludeSuperseded: true, IncludeExpired: true,
	})
	if err != nil {
		t.Fatalf("ListRecentPage: %v", err)
	}
	if len(got) != 3 {
		t.Errorf("unset Limit returned %d, want all 3", len(got))
	}
}

func TestNeighborsTakesDepthLiterally(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	mems := seedN(t, store, 2)
	if err := store.Link(ctx, mems[0].ID, mems[1].ID, "related"); err != nil {
		t.Fatalf("Link: %v", err)
	}

	for _, depth := range []int{0, -1} {
		got, err := store.Neighbors(ctx, mems[0].ID, "", "both", depth)
		if err != nil {
			t.Fatalf("Neighbors(depth=%d): %v", depth, err)
		}
		if len(got) != 0 {
			t.Errorf("depth=%d returned %d neighbours, want 0 — zero hops "+
				"reaches nothing", depth, len(got))
		}
	}

	got, err := store.Neighbors(ctx, mems[0].ID, "", "both", 1)
	if err != nil {
		t.Fatalf("Neighbors(depth=1): %v", err)
	}
	if len(got) != 1 {
		t.Errorf("depth=1 returned %d neighbours, want 1", len(got))
	}
}

func TestRecallPackTakesTokenBudgetLiterally(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	seedN(t, store, 3)

	pack, err := store.RecallPack(ctx, "seeded subject", PackOpts{TokenBudget: 0})
	if err != nil {
		t.Fatalf("RecallPack: %v", err)
	}
	if len(pack.Items) != 0 {
		t.Errorf("token_budget=0 packed %d memories, want none — a zero "+
			"budget is what a caller with no room left sends",
			len(pack.Items))
	}

	pack, err = store.RecallPack(ctx, "seeded subject",
		PackOpts{TokenBudget: DefaultTokenBudget})
	if err != nil {
		t.Fatalf("RecallPack: %v", err)
	}
	if len(pack.Items) == 0 {
		t.Error("the default budget packed nothing")
	}
}

func TestAutoContextPackTakesTokenBudgetLiterally(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	seedN(t, store, 3)

	pack, err := store.AutoContextPack(ctx, "seeded subject",
		AutoContextOpts{TokenBudget: 0})
	if err != nil {
		t.Fatalf("AutoContextPack: %v", err)
	}
	if len(pack.Items) != 0 {
		t.Errorf("token_budget=0 packed %d memories, want none",
			len(pack.Items))
	}
}

// Caller input must reach the transports as a ValidationError, or a 400 is
// reported to the client as a 500 and the message carries a Go type name.

func TestSelfMergeIsCallerError(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	m := seedN(t, store, 1)[0]

	_, err := store.Merge(ctx, m.ID, m.ID, "")
	if err == nil {
		t.Fatal("merging a memory into itself should fail")
	}
	if !IsValidationError(err) {
		t.Errorf("err = %T, want a ValidationError — a self-merge can never "+
			"succeed, so telling the client to retry (500) is wrong", err)
	}
}

func TestBadTagNameIsCallerError(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	seedN(t, store, 1)

	for _, tc := range []struct {
		name string
		call func() error
	}{
		{"rename to a comma'd tag", func() error {
			_, err := store.RenameTag(ctx, "x", "a,b")
			return err
		}},
		{"rename to an empty tag", func() error {
			_, err := store.RenameTag(ctx, "x", "")
			return err
		}},
		{"merge into a comma'd tag", func() error {
			_, err := store.MergeTags(ctx, []string{"x"}, "a,b")
			return err
		}},
	} {
		err := tc.call()
		if err == nil {
			t.Errorf("%s: should fail", tc.name)
			continue
		}
		if !IsValidationError(err) {
			t.Errorf("%s: err = %T, want a ValidationError", tc.name, err)
		}
		if strings.Contains(err.Error(), "errorString") {
			t.Errorf("%s: message leaks a Go type: %v", tc.name, err)
		}
	}
}

// A tag the write path accepts must be one the curation path can manage. The
// two validators disagreed: Remember took "" and "   " while RenameTag
// refused them, so a caller could create a tag that could never be renamed —
// and "" was swallowed by the comma-join and never seen again at all.
func TestWriteAndCurationAgreeOnLegalTags(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	for _, tag := range []string{"", "   ", "\t"} {
		_, _, _, err := store.Remember(ctx, "blank tag subject",
			RememberOpts{Tags: []string{tag}})
		if err == nil {
			t.Errorf("Remember accepted tag %q, which RenameTag refuses", tag)
			continue
		}
		if !IsValidationError(err) {
			t.Errorf("tag %q: err = %T, want a ValidationError", tag, err)
		}
	}

	// A tag that merely contains a space is ordinary.
	m, _, _, err := store.Remember(ctx, "spacey tag subject",
		RememberOpts{Tags: []string{"my tag"}})
	if err != nil {
		t.Fatalf(`Remember with tag "my tag": %v`, err)
	}
	got, err := store.GetByID(ctx, m.ID)
	if err != nil {
		t.Fatalf("GetByID: %v", err)
	}
	if len(got.Tags) != 1 || got.Tags[0] != "my tag" {
		t.Errorf("tags = %v, want [my tag]", got.Tags)
	}
}
