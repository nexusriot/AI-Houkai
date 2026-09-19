package cli

import (
	"strings"
	"testing"
)

// A command that slices a list by a user-supplied count has to take that count
// literally. `houkai list --limit 0` read the zero as "no bound" and printed
// the whole store — the worst possible answer for a caller whose page budget
// had run out — and a negative limit did the same instead of failing.
//
// The cobra command needs a store, so these drive the flag plumbing and the
// error text rather than the rows; TestListLimitZeroReturnsNothing in the
// httpserver package covers the answer itself end to end.

func TestListRejectsANegativeLimit(t *testing.T) {
	cmd := newListCmd()
	if err := cmd.Flags().Set("limit", "-1"); err != nil {
		t.Fatalf("set --limit: %v", err)
	}
	err := cmd.RunE(cmd, nil)
	if err == nil {
		t.Fatal("--limit -1 should fail, not list everything")
	}
	if !strings.Contains(err.Error(), ">= 0") {
		t.Errorf("err = %q, should say the limit must be >= 0", err)
	}
}

func TestListLimitFlagDefaults(t *testing.T) {
	// The default has to be a real bound, not the zero that now means "none".
	cmd := newListCmd()
	got, err := cmd.Flags().GetInt("limit")
	if err != nil {
		t.Fatalf("GetInt: %v", err)
	}
	if got <= 0 {
		t.Errorf("default --limit = %d, want a positive bound", got)
	}
}
