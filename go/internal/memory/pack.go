package memory

import (
	"context"
	"fmt"
	"strings"
)

// PackedMemory is a memory selected for a context pack, with its token cost.
type PackedMemory struct {
	Memory Memory  `json:"memory"`
	Score  float32 `json:"score"`
	Tokens int     `json:"tokens"`
}

// PackResult is the result of RecallPack — memories that fit a token budget
// plus a ready-to-inject context block.
type PackResult struct {
	Items      []PackedMemory `json:"items"`
	Text       string         `json:"text"` // formatted block (empty when no items fit)
	UsedTokens int            `json:"used_tokens"`
	Budget     int            `json:"budget"`
	Truncated  bool           `json:"truncated"` // true if ranked candidates were dropped to fit
}

// IDs returns the packed memory ids in rank order.
func (p PackResult) IDs() []string {
	out := make([]string, len(p.Items))
	for i, it := range p.Items {
		out[i] = it.Memory.ID
	}
	return out
}

// EstimateTokens is a cheap, tokenizer-free token estimate (~4 chars/token).
// A soft heuristic so the token budget stays a dependency-free ceiling;
// callers wanting exact counts pass their own TokenCounter.
func EstimateTokens(text string) int {
	n := (len(text) + 2) / 4 // round(len/4)
	if n < 1 {
		return 1
	}
	return n
}

// PackOpts holds optional RecallPack parameters.
type PackOpts struct {
	TokenBudget       int // default 800
	Type              MemoryType
	Tag               string
	MinImportance     float32
	Mode              RecallMode // default hybrid
	Weights           HybridWeights
	Expand            *ExpandSpec
	IncludeSuperseded bool
	MaxItems          int              // ranked candidates to consider (default 50)
	TokenCounter      func(string) int // default EstimateTokens
	Header            *string          // nil → "## Relevant memory"; empty string → no header

	// Source/Since/Until filter candidates exactly as in Recall.
	Source string
	Since  float64
	Until  float64
}

const defaultPackHeader = "## Relevant memory"

// RecallPack assembles the highest-ranked memories that fit a token budget
// into a ready-to-inject context block.
//
// Ranks candidates with Recall (hybrid by default), then greedily packs them
// in rank order while the running token estimate stays within TokenBudget.
// The budget covers the rendered memory lines only — the header is not
// counted against it. The default ~4-chars/token estimate makes the budget a
// soft ceiling unless an exact TokenCounter is supplied.
func (s *MemoryStore) RecallPack(ctx context.Context, query string, opts PackOpts) (PackResult, error) {
	if opts.TokenBudget == 0 {
		opts.TokenBudget = 800
	}
	if opts.Mode == "" {
		opts.Mode = ModeHybrid
	}
	if opts.MaxItems == 0 {
		opts.MaxItems = 50
	}
	countFn := opts.TokenCounter
	if countFn == nil {
		countFn = EstimateTokens
	}
	header := defaultPackHeader
	if opts.Header != nil {
		header = *opts.Header
	}

	ranked, err := s.Recall(ctx, query, opts.MaxItems, RecallOpts{
		Type:              opts.Type,
		Tag:               opts.Tag,
		MinImportance:     opts.MinImportance,
		Mode:              opts.Mode,
		Overfetch:         3,
		Weights:           opts.Weights,
		IncludeSuperseded: opts.IncludeSuperseded,
		Expand:            opts.Expand,
		Source:            opts.Source,
		Since:             opts.Since,
		Until:             opts.Until,
	})
	if err != nil {
		return PackResult{}, err
	}

	var items []PackedMemory
	used := 0
	truncated := false
	for _, r := range ranked {
		line := packLine(r.Memory)
		cost := countFn(line)
		if used+cost > opts.TokenBudget {
			truncated = true
			continue // a smaller, lower-ranked item may still fit
		}
		items = append(items, PackedMemory{Memory: r.Memory, Score: r.Score, Tokens: cost})
		used += cost
	}

	text := ""
	if len(items) > 0 {
		lines := make([]string, len(items))
		for i, p := range items {
			lines[i] = packLine(p.Memory)
		}
		body := strings.Join(lines, "\n")
		if header != "" {
			text = header + "\n" + body
		} else {
			text = body
		}
	}

	return PackResult{
		Items:      items,
		Text:       text,
		UsedTokens: used,
		Budget:     opts.TokenBudget,
		Truncated:  truncated,
	}, nil
}

func packLine(m Memory) string {
	return fmt.Sprintf("- (%s) %s", m.Type, m.Text)
}
