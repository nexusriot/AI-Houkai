// Package timeparse provides lenient timestamp parsing shared by the CLI, MCP
// server and HTTP API.
//
// Recall's since/until filters take Unix timestamps (seconds). Humans and other
// tools prefer ISO dates or relative spans, so the user-facing layers funnel
// their inputs through Parse to get a value the store accepts.
package timeparse

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"time"
)

var relRE = regexp.MustCompile(`(?i)^\s*(\d+(?:\.\d+)?)\s*([smhdw])\s*$`)

var relSeconds = map[string]float64{
	"s": 1,
	"m": 60,
	"h": 3600,
	"d": 86400,
	"w": 604800,
}

// isoLayouts are tried in order for non-relative, non-numeric input. Naive
// values (no zone) are interpreted as UTC, mirroring the Python implementation.
var isoLayouts = []string{
	time.RFC3339Nano,
	time.RFC3339,
	"2006-01-02T15:04:05",
	"2006-01-02T15:04",
	"2006-01-02 15:04:05",
	"2006-01-02",
}

// Parse coerces value into a Unix timestamp (seconds since epoch).
//
// It accepts, in order:
//   - an empty string → (0, false, nil): no bound
//   - a relative span like "7d", "24h", "30m" → now minus that span
//   - a bare numeric string → parsed as an epoch timestamp verbatim
//   - an ISO-8601 date/datetime (e.g. "2026-06-14" or "2026-06-14T10:30:00");
//     a trailing "Z" is honoured and naive values are interpreted as UTC
//
// The ok result reports whether a bound was set (false for empty input).
// A non-nil error is returned for anything that cannot be parsed, so callers
// can surface a clear message instead of silently dropping the filter.
func Parse(value string) (ts float64, ok bool, err error) {
	return ParseAt(value, time.Now())
}

// ParseAt is Parse with an explicit "now" for relative spans (testable).
func ParseAt(value string, now time.Time) (float64, bool, error) {
	text := strings.TrimSpace(value)
	if text == "" {
		return 0, false, nil
	}

	if m := relRE.FindStringSubmatch(text); m != nil {
		amount, _ := strconv.ParseFloat(m[1], 64)
		unit := strings.ToLower(m[2])
		return float64(now.Unix()) - amount*relSeconds[unit], true, nil
	}

	if n, perr := strconv.ParseFloat(text, 64); perr == nil {
		return n, true, nil
	}

	iso := text
	if strings.HasSuffix(iso, "Z") {
		iso = iso[:len(iso)-1] + "+00:00"
	}
	for _, layout := range isoLayouts {
		if t, perr := time.Parse(layout, iso); perr == nil {
			return float64(t.Unix()), true, nil
		}
		// Naive layouts (no zone) parse in UTC via ParseInLocation.
		if t, perr := time.ParseInLocation(layout, text, time.UTC); perr == nil {
			return float64(t.Unix()), true, nil
		}
	}

	return 0, false, fmt.Errorf(
		"invalid timestamp %q: expected epoch seconds, an ISO-8601 "+
			"date/datetime, or a relative span like '7d'", text)
}
