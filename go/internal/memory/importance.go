package memory

import (
	"math"
	"regexp"
	"strings"
)

// Heuristic importance auto-assignment.
//
// ScoreImportance rates a memory 0..1 without an LLM, following the tiers
// sketched in docs/DESIGN.md §15:
//
//	0.90+  explicit standing instructions, corrections, user preferences
//	0.75   decisions, conventions, policies
//	0.60   task completions, durable project facts
//	0.50   neutral default (nothing matched)
//	0.35   passing observations, hedged statements
//
// The score is the strongest matching tier plus small modifiers (memory
// type, questions, very short fragments), clamped to [0.05, 0.98] so an
// auto-scored memory never claims absolute certainty in either direction.
//
// Deterministic by design: same text in, same score out — recall ranking
// and decay both consume importance, so it must be stable across runs.

type importanceTier struct {
	score    float32
	patterns []*regexp.Regexp
}

func compileAll(patterns ...string) []*regexp.Regexp {
	out := make([]*regexp.Regexp, len(patterns))
	for i, p := range patterns {
		out[i] = regexp.MustCompile("(?i)" + p)
	}
	return out
}

// First (highest) matching tier wins; patterns are case-insensitive and
// word-bounded where it matters.
var importanceTiers = []importanceTier{
	{0.90, compileAll(
		`\b(always|never)\b`,
		`\bfrom now on\b`,
		`\b(must|must not|mustn't)\b`,
		`\bdo not ever\b`,
		`\bimportant\b`,
		`\bcritical\b`,
		`^correction\b`,
		`\bactually,`,
		`\bthat('s| is| was) (wrong|incorrect)\b`,
		`\bnot \w+([ ,]+\w+){0,3}[ ,]+but\b`,
		`\binstead of\b`,
		`\bprefers?\b`,
		`\bfavou?rite\b`,
		`\b(i|user) (like|love|hate|dislike)s?\b`,
		`\bdon'?t (like|want|use)\b`,
	)},
	{0.75, compileAll(
		`\bdecided\b`,
		`\bdecision\b`,
		`\bwe (use|chose|agreed|settled on)\b`,
		`\bconvention\b`,
		`\bpolicy\b`,
		`\bstandard\b`,
		`\brule\b`,
		`\bworkflow\b`,
		`\brequired\b`,
	)},
	{0.60, compileAll(
		`\b(fixed|solved|resolved)\b`,
		`\b(implemented|added|built|created)\b`,
		`\b(deployed|released|shipped|merged|published)\b`,
		`\b(configured|installed|migrated)\b`,
	)},
	{0.35, compileAll(
		`\b(noticed|observed|saw|spotted)\b`,
		`\b(seems|appears|looks like)\b`,
		`\b(maybe|perhaps|might|possibly)\b`,
		`\bnot sure\b`,
		`\bfor now\b`,
		`\btemporar(y|ily)\b`,
	)},
}

const (
	importanceDefault float32 = 0.50
	importanceFloor   float32 = 0.05
	importanceCeil    float32 = 0.98
)

// Memory types that are durable by nature get a nudge upward.
var importanceTypeBonus = map[MemoryType]float32{
	Procedural: 0.10,
	Feedback:   0.10,
}

// ScoreImportance rates a memory's importance 0..1 from its text, type, and
// tags. The strongest matching tier wins (an instruction beats a hedge),
// then modifiers apply: +0.10 for procedural/feedback types, −0.15 for
// questions, −0.10 for fragments under 20 characters.
func ScoreImportance(text string, memType MemoryType, _ []string) float32 {
	body := strings.TrimSpace(text)

	score := importanceDefault
	for _, tier := range importanceTiers {
		matched := false
		for _, p := range tier.patterns {
			if p.MatchString(body) {
				matched = true
				break
			}
		}
		if matched {
			score = tier.score
			break
		}
	}

	score += importanceTypeBonus[memType]
	if strings.HasSuffix(body, "?") {
		score -= 0.15
	}
	if len(body) < 20 {
		score -= 0.10
	}

	if score < importanceFloor {
		score = importanceFloor
	}
	if score > importanceCeil {
		score = importanceCeil
	}
	// Round to 3 decimals, matching the Python implementation.
	return float32(math.Round(float64(score)*1000) / 1000)
}
