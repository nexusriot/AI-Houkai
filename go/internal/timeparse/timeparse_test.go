package timeparse

import (
	"testing"
	"time"
)

func TestParseEmptyIsUnbounded(t *testing.T) {
	ts, ok, err := Parse("")
	if err != nil || ok || ts != 0 {
		t.Fatalf("empty: ts=%v ok=%v err=%v", ts, ok, err)
	}
	if ts, ok, err := Parse("   "); err != nil || ok || ts != 0 {
		t.Fatalf("blank: ts=%v ok=%v err=%v", ts, ok, err)
	}
}

func TestParseEpoch(t *testing.T) {
	ts, ok, err := Parse("1700000000")
	if err != nil || !ok || ts != 1700000000 {
		t.Fatalf("epoch: ts=%v ok=%v err=%v", ts, ok, err)
	}
}

func TestParseRelativeSpan(t *testing.T) {
	now := time.Date(2026, 6, 17, 12, 0, 0, 0, time.UTC)
	cases := map[string]float64{
		"7d":   7 * 86400,
		"24h":  24 * 3600,
		"30m":  30 * 60,
		"2w":   2 * 604800,
		" 1d ": 86400,
	}
	for in, span := range cases {
		ts, ok, err := ParseAt(in, now)
		if err != nil || !ok {
			t.Fatalf("%q: ok=%v err=%v", in, ok, err)
		}
		want := float64(now.Unix()) - span
		if ts != want {
			t.Errorf("%q: got %v want %v", in, ts, want)
		}
	}
}

func TestParseISODate(t *testing.T) {
	ts, ok, err := Parse("2026-01-01")
	if err != nil || !ok {
		t.Fatalf("iso date: ok=%v err=%v", ok, err)
	}
	want := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC).Unix()
	if int64(ts) != want {
		t.Errorf("iso date: got %v want %v", int64(ts), want)
	}
}

func TestParseISODatetimeNaiveIsUTC(t *testing.T) {
	ts, _, err := Parse("2026-06-14T10:30:00")
	if err != nil {
		t.Fatalf("iso datetime: %v", err)
	}
	want := time.Date(2026, 6, 14, 10, 30, 0, 0, time.UTC).Unix()
	if int64(ts) != want {
		t.Errorf("naive datetime should read as UTC: got %v want %v", int64(ts), want)
	}
}

func TestParseISODatetimeZ(t *testing.T) {
	ts, _, err := Parse("2026-06-14T10:30:00Z")
	if err != nil {
		t.Fatalf("iso Z: %v", err)
	}
	want := time.Date(2026, 6, 14, 10, 30, 0, 0, time.UTC).Unix()
	if int64(ts) != want {
		t.Errorf("Z datetime: got %v want %v", int64(ts), want)
	}
}

func TestParseInvalid(t *testing.T) {
	if _, _, err := Parse("not-a-time"); err == nil {
		t.Error("expected error for garbage input")
	}
}
