package httpserver

import "testing"

func TestRound4(t *testing.T) {
	cases := []struct {
		in   float32
		want float64
	}{
		{0.12344, 0.1234},
		{0.12346, 0.1235},
		{1, 1},
		{0, 0},
		// Negative values must round to the NEAREST step, not truncate toward
		// zero (the old int64(x+0.5) trick returned -1.2345 here).
		{-1.23456, -1.2346},
		{-0.00004, 0},
	}
	for _, tc := range cases {
		if got := round4(tc.in); got != tc.want {
			t.Errorf("round4(%v) = %v, want %v", tc.in, got, tc.want)
		}
	}
}
