package memory

import "testing"

// trust is validated on write, so an unrecognised level can only arrive from a
// hand-edited store or from a build that knows a level this one does not. Both
// are named as expected cases, so reading one must fail safe rather than read
// as trusted — that would be a laundering path. An *absent* level is different:
// it means the row predates the feature, and it deserialises as trusted so an
// old store does not change behaviour just by being opened.
func TestTrustRankUnknownIsLeastTrusted(t *testing.T) {
	if got := TrustRank("from-the-future"); got != len(TrustLevels)-1 {
		t.Errorf("TrustRank(unknown) = %d, want %d (worst)", got, len(TrustLevels)-1)
	}
}

func TestTrustRankAbsentIsTrusted(t *testing.T) {
	if got := TrustRank(""); got != 0 {
		t.Errorf("TrustRank(\"\") = %d, want 0 so old rows keep working", got)
	}
}

func TestTrustRankOrdersKnownLevelsBestFirst(t *testing.T) {
	for i, level := range TrustLevels {
		if got := TrustRank(TrustLevel(level)); got != i {
			t.Errorf("TrustRank(%q) = %d, want %d", level, got, i)
		}
	}
}

// The known-level combinations live in TestWorstTrust (curation_test.go); this
// covers only the case that used to launder — an unrecognised level folding in
// as if it were trusted.
func TestWorstTrustTreatsAnUnknownLevelAsTheWorstCase(t *testing.T) {
	if got := WorstTrust("trusted", "from-the-future"); got != "untrusted" {
		t.Errorf("WorstTrust(trusted, unknown) = %q, want %q", got, "untrusted")
	}
}
