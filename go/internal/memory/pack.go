package memory

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strings"
)

// PackedMemory is a memory selected for a context pack, with its token cost.
type PackedMemory struct {
	Memory Memory  `json:"memory"`
	Score  float32 `json:"score"`
	Tokens int     `json:"tokens"`
}

// CompressedGroup folds several similar, budget-dropped memories into one
// summary line. Memories is the source set; Text is the rendered line.
type CompressedGroup struct {
	Memories []Memory
	Text     string
	Tokens   int
}

// IDs returns the source memory ids.
func (g CompressedGroup) IDs() []string {
	ids := make([]string, len(g.Memories))
	for i, m := range g.Memories {
		ids[i] = m.ID
	}
	return ids
}

// MarshalJSON renders a compressed group as {ids, count, text, tokens} so the
// full source memories are not dumped into API payloads.
func (g CompressedGroup) MarshalJSON() ([]byte, error) {
	return json.Marshal(map[string]any{
		"ids": g.IDs(), "count": len(g.Memories), "text": g.Text, "tokens": g.Tokens,
	})
}

// PackResult is the result of RecallPack — memories that fit a token budget
// plus a ready-to-inject context block.
type PackResult struct {
	Items            []PackedMemory    `json:"items"`
	Text             string            `json:"text"` // formatted block (empty when no items fit)
	UsedTokens       int               `json:"used_tokens"`
	Budget           int               `json:"budget"`
	Truncated        bool              `json:"truncated"` // true if ranked candidates were dropped to fit
	CompressedGroups []CompressedGroup `json:"compressed_groups,omitempty"`
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
// callers wanting exact counts pass their own TokenCounter. Uses round-half-to-
// even (like Python's round()) so packing decisions match the reference port.
func EstimateTokens(text string) int {
	n := int(math.RoundToEven(float64(len(text)) / 4.0))
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

	// Advanced retrieval controls, forwarded to Recall.
	Fusion         FusionMode
	Diversity      *float32
	DedupThreshold *float32
	MinCosine      *float32
	NoTouch        bool

	// Query-time compression: candidates that could not be packed individually
	// are clustered by Jaccard similarity; clusters of ≥ CompressMinGroup
	// members are folded into one summary line packed into the remaining budget.
	Compress          bool
	CompressThreshold float32 // default 0.30 when Compress and zero
	CompressMinGroup  int     // default 2 when Compress and zero
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
		Overfetch:         4,
		Weights:           opts.Weights,
		IncludeSuperseded: opts.IncludeSuperseded,
		Expand:            opts.Expand,
		Source:            opts.Source,
		Since:             opts.Since,
		Until:             opts.Until,
		Fusion:            opts.Fusion,
		Diversity:         opts.Diversity,
		DedupThreshold:    opts.DedupThreshold,
		MinCosine:         opts.MinCosine,
		NoTouch:           opts.NoTouch,
	})
	if err != nil {
		return PackResult{}, err
	}

	return packRanked(ranked, opts.TokenBudget, countFn, header,
		opts.Compress, opts.CompressThreshold, opts.CompressMinGroup), nil
}

// packRanked greedily packs already-ranked (memory, score) results to a token
// budget, optionally compressing budget-dropped candidates into summary lines.
// Shared by RecallPack and AutoContextPack (mirrors Python's _pack_ranked).
func packRanked(ranked []MemoryWithScore, budget int, countFn func(string) int, header string,
	compress bool, compressThreshold float32, compressMinGroup int) PackResult {

	if compressThreshold == 0 {
		compressThreshold = 0.30
	}
	if compressMinGroup == 0 {
		compressMinGroup = 2
	}

	var items []PackedMemory
	var dropped []MemoryWithScore
	used := 0
	truncated := false
	for _, r := range ranked {
		line := packLine(r.Memory)
		cost := countFn(line)
		if used+cost > budget {
			truncated = true
			dropped = append(dropped, r) // a smaller, lower-ranked item may still fit
			continue
		}
		items = append(items, PackedMemory{Memory: r.Memory, Score: r.Score, Tokens: cost})
		used += cost
	}

	var compressedGroups []CompressedGroup
	if compress && len(dropped) > 0 {
		for _, group := range clusterByJaccard(dropped, compressThreshold, compressMinGroup) {
			line := "- (compressed) " + compressGroup(group)
			cost := countFn(line)
			if used+cost <= budget {
				compressedGroups = append(compressedGroups, CompressedGroup{
					Memories: group, Text: line, Tokens: cost,
				})
				used += cost
			}
		}
	}

	text := ""
	if len(items) > 0 || len(compressedGroups) > 0 {
		parts := make([]string, 0, len(items)+len(compressedGroups))
		for _, p := range items {
			parts = append(parts, packLine(p.Memory))
		}
		for _, cg := range compressedGroups {
			parts = append(parts, cg.Text)
		}
		body := strings.Join(parts, "\n")
		if header != "" {
			text = header + "\n" + body
		} else {
			text = body
		}
	}

	return PackResult{
		Items:            items,
		Text:             text,
		UsedTokens:       used,
		Budget:           budget,
		Truncated:        truncated,
		CompressedGroups: compressedGroups,
	}
}

func packLine(m Memory) string {
	return fmt.Sprintf("- (%s) %s", m.Type, m.Text)
}

// jaccardSim is the token-set Jaccard similarity of two texts.
func jaccardSim(a, b string) float32 {
	ta := map[string]bool{}
	for _, t := range tokenize(a) {
		ta[t] = true
	}
	tb := map[string]bool{}
	for _, t := range tokenize(b) {
		tb[t] = true
	}
	if len(ta) == 0 && len(tb) == 0 {
		return 0
	}
	inter := 0
	for t := range ta {
		if tb[t] {
			inter++
		}
	}
	union := len(ta) + len(tb) - inter
	if union == 0 {
		return 0
	}
	return float32(inter) / float32(union)
}

// clusterByJaccard greedily single-links truncated candidates by token-Jaccard
// against each cluster seed. Clusters below minSize are dropped.
func clusterByJaccard(candidates []MemoryWithScore, threshold float32, minSize int) [][]Memory {
	n := len(candidates)
	used := make([]bool, n)
	var clusters [][]Memory
	for i := 0; i < n; i++ {
		if used[i] {
			continue
		}
		seed := candidates[i].Memory
		cluster := []Memory{seed}
		used[i] = true
		for j := i + 1; j < n; j++ {
			if used[j] {
				continue
			}
			if jaccardSim(seed.Text, candidates[j].Text) >= threshold {
				cluster = append(cluster, candidates[j].Memory)
				used[j] = true
			}
		}
		if len(cluster) >= minSize {
			clusters = append(clusters, cluster)
		}
	}
	return clusters
}

// compressGroup renders a cluster as "[×N similar] s1 | s2 | s3", using the
// first sentence of each of the (up to three) most-important members.
func compressGroup(mems []Memory) string {
	ordered := make([]Memory, len(mems))
	copy(ordered, mems)
	sort.SliceStable(ordered, func(i, j int) bool { return ordered[i].Importance > ordered[j].Importance })
	var snippets []string
	for i, m := range ordered {
		if i >= 3 {
			break
		}
		snippet := strings.TrimSpace(strings.SplitN(m.Text, ".", 2)[0])
		if snippet != "" {
			snippets = append(snippets, snippet)
		}
	}
	return fmt.Sprintf("[×%d similar] %s", len(mems), strings.Join(snippets, " | "))
}
