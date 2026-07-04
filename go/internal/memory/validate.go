// Validation layer for the closed enum vocabularies shared by every surface
// (CLI, HTTP, MCP, TUI). Validating in the store means a typo like
// mode="hybird" is rejected in ONE place instead of silently degrading.
package memory

import (
	"errors"
	"fmt"
	"strings"
)

// Canonical enum vocabularies (mirrors Python store.py).
var (
	MemoryTypes      = []string{"episodic", "semantic", "procedural", "feedback"}
	LinkRels         = []string{"related", "refines", "derived_from", "example_of", "contradicts", "supersedes"}
	RecallModes      = []string{"semantic", "hybrid"}
	Fusions          = []string{"weighted", "rrf"}
	ConflictPolicies = []string{"ignore", "warn", "supersede", "raise"}
	ImportPolicies   = []string{"skip", "overwrite", "rename", "error"}
	Directions       = []string{"out", "in", "both"}
)

// ValidationError marks an error caused by invalid caller input (a bad enum
// value, comma in a tag, self-link, …) so transports can map it to a 4xx
// response instead of a 500 — the Go analogue of Python's ValueError.
type ValidationError struct{ Msg string }

func (e *ValidationError) Error() string { return e.Msg }

// IsValidationError reports whether err (or anything it wraps) is a
// ValidationError.
func IsValidationError(err error) bool {
	var ve *ValidationError
	return errors.As(err, &ve)
}

func validationErrorf(format string, args ...any) error {
	return &ValidationError{Msg: fmt.Sprintf(format, args...)}
}

// validateChoice rejects values outside the allowed vocabulary, naming the
// parameter and the vocabulary in the error.
func validateChoice(value string, allowed []string, param string) error {
	for _, a := range allowed {
		if value == a {
			return nil
		}
	}
	return validationErrorf("%s must be one of: %s — got %q",
		param, strings.Join(allowed, ", "), value)
}

// validateTags rejects commas: tags are stored comma-joined in backend
// metadata, so a comma inside a tag would silently split it on the next read.
func validateTags(tags []string) error {
	for _, t := range tags {
		if strings.Contains(t, ",") {
			return validationErrorf(
				"tags must not contain commas — got %q (tags are stored as a comma-joined string)", t)
		}
	}
	return nil
}

func validatePolarity(p int) error {
	if p != -1 && p != 0 && p != 1 {
		return validationErrorf("polarity must be -1, 0, or +1 — got %d", p)
	}
	return nil
}

// clamp01 clamps v to [0, 1] (importance range).
func clamp01(v float32) float32 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

// Float32Ptr returns a pointer to v — convenience for optional fields like
// RememberOpts.Importance where nil means "unset".
func Float32Ptr(v float32) *float32 { return &v }
