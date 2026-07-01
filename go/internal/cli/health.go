package cli

import (
	"fmt"
	"math"
	"sort"
	"strings"
	"time"

	"github.com/nexusriot/ai-houkai/internal/memory"
)

var bucketLabels = []string{"0.0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"}

// healthDecayScore mirrors DecayEngine.Score exactly, including the optional
// recall-frequency reinforcement term, so the histogram and at-risk count match
// what `houkai prune` would actually remove.
func healthDecayScore(importance float32, lastAccessed float64, decayRate float32, accessCount int, freqWeight float32) float32 {
	days := (float64(time.Now().Unix()) - lastAccessed) / 86400.0
	if days < 0 {
		days = 0
	}
	base := float64(importance) * math.Exp(-float64(decayRate)*days)
	if freqWeight != 0 {
		if accessCount < 0 {
			accessCount = 0
		}
		base *= 1.0 + float64(freqWeight)*math.Log1p(float64(accessCount))
	}
	return float32(base)
}

func decayBucket(score float32) int {
	b := int(score * 5)
	if b > 4 {
		b = 4
	}
	if b < 0 {
		b = 0
	}
	return b
}

// computeHealth builds the detailed health report over active memories,
// mirroring Python's _compute_health.
func computeHealth(active []memory.Memory, staleDays int, decayRate, minScore float32, protectTypes []string, freqWeight float32) map[string]any {
	now := float64(time.Now().Unix())
	staleTS := now - float64(staleDays)*86400.0
	protect := map[string]bool{}
	for _, t := range protectTypes {
		protect[t] = true
	}

	hist := make([]int, 5)
	atRisk, neverRecalled, stale, episodic, totalLinks := 0, 0, 0, 0, 0
	var totalImp float32
	for _, m := range active {
		s := healthDecayScore(m.Importance, m.LastAccessed, decayRate, m.AccessCount, freqWeight)
		hist[decayBucket(s)]++
		if s < minScore && !protect[string(m.Type)] {
			atRisk++
		}
		if m.AccessCount == 0 {
			neverRecalled++
		}
		if m.LastAccessed < staleTS {
			stale++
		}
		if m.Type == memory.Episodic {
			episodic++
		}
		totalImp += m.Importance
		totalLinks += len(m.Links)
	}

	var linkDensity, avgImp float64
	if len(active) > 0 {
		linkDensity = float64(totalLinks) / float64(len(active))
		avgImp = float64(totalImp) / float64(len(active))
	}

	sorted := make([]memory.Memory, len(active))
	copy(sorted, active)
	sort.SliceStable(sorted, func(i, j int) bool { return sorted[i].AccessCount > sorted[j].AccessCount })
	topRecalled := []map[string]any{}
	for _, m := range sorted {
		if len(topRecalled) >= 5 {
			break
		}
		if m.AccessCount == 0 {
			continue
		}
		snippet := m.Text
		if r := []rune(snippet); len(r) > 60 {
			snippet = string(r[:60]) + "…"
		}
		id := m.ID
		if len(id) > 8 {
			id = id[:8]
		}
		topRecalled = append(topRecalled, map[string]any{
			"id": id, "access_count": m.AccessCount, "text_snippet": snippet,
		})
	}

	histMap := map[string]int{}
	for i, label := range bucketLabels {
		histMap[label] = hist[i]
	}

	return map[string]any{
		"decay_histogram":       histMap,
		"at_risk_count":         atRisk,
		"never_recalled_count":  neverRecalled,
		"stale_count":           stale,
		"stale_days":            staleDays,
		"episodic_active_count": episodic,
		"link_density":          math.Round(linkDensity*1000) / 1000,
		"total_links":           totalLinks,
		"avg_importance":        math.Round(avgImp*1000) / 1000,
		"top_recalled":          topRecalled,
	}
}

// renderStatsText prints the human-readable stats report (and health, if present).
func renderStatsText(data map[string]any) {
	fmt.Printf("Store       %v\n", data["store_path"])
	fmt.Printf("Collection  %v\n", data["collection"])
	fmt.Printf("Total       %v (%v active, %v superseded)\n", data["total"], data["active"], data["superseded"])
	if sz, ok := data["store_size_bytes"].(int64); ok {
		fmt.Printf("Size        %.1f KB\n", float64(sz)/1024)
	}
	if bt, ok := data["by_type"].(map[string]int); ok && len(bt) > 0 {
		type kv struct {
			k string
			v int
		}
		var rows []kv
		for k, v := range bt {
			rows = append(rows, kv{k, v})
		}
		sort.Slice(rows, func(i, j int) bool { return rows[i].v > rows[j].v })
		fmt.Println("\nBy type:")
		for _, r := range rows {
			fmt.Printf("  %-12s %d\n", r.k, r.v)
		}
	}
	if tags, ok := data["top_tags"].([]string); ok && len(tags) > 0 {
		fmt.Println("\nTop tags:")
		for _, t := range tags {
			fmt.Printf("  %s\n", t)
		}
	}
	if h, ok := data["health"].(map[string]any); ok {
		renderHealthText(h)
	}
}

func renderHealthText(h map[string]any) {
	fmt.Println("\n── Health Report ──")
	fmt.Printf("Avg importance  %.2f    Link density  %.2f links/memory\n",
		toFloat(h["avg_importance"]), toFloat(h["link_density"]))
	fmt.Printf("At-risk         %v (below prune threshold)    Stale  %v (idle >%vd)\n",
		h["at_risk_count"], h["stale_count"], h["stale_days"])
	fmt.Printf("Never recalled  %v    Episodic (ripe for reflection)  %v\n",
		h["never_recalled_count"], h["episodic_active_count"])

	if hist, ok := h["decay_histogram"].(map[string]int); ok {
		maxCnt := 1
		for _, c := range hist {
			if c > maxCnt {
				maxCnt = c
			}
		}
		fmt.Println("\nDecay score distribution:")
		for _, label := range bucketLabels {
			c := hist[label]
			barLen := int(math.Round(float64(c) / float64(maxCnt) * 20))
			bar := strings.Repeat("█", barLen) + strings.Repeat("░", 20-barLen)
			fmt.Printf("  %-9s %4d  %s\n", label, c, bar)
		}
	}
	if top, ok := h["top_recalled"].([]map[string]any); ok && len(top) > 0 {
		fmt.Println("\nTop recalled:")
		for _, r := range top {
			fmt.Printf("  %-8v %4v  %v\n", r["id"], r["access_count"], r["text_snippet"])
		}
	}
}

func toFloat(v any) float64 {
	switch x := v.(type) {
	case float64:
		return x
	case float32:
		return float64(x)
	}
	return 0
}
