package memory

import (
	"math"
	"sort"

	"github.com/nexusriot/ai-houkai/internal/vector"
)

// cand is a scoring candidate: a memory plus its raw cosine similarity and
// (optionally) its embedding, carried straight from the vector query.
type cand struct {
	mem    Memory
	cosine float32
	emb    []float32
}

// scoredCand is a candidate after blend scoring.
type scoredCand struct {
	mem   Memory
	score float32
	emb   []float32
}

func round4(f float32) float64 { return math.Round(float64(f)*1e4) / 1e4 }
func round6(f float32) float64 { return math.Round(float64(f)*1e6) / 1e6 }

// applyReranker rescores the first-stage candidate pool with reranker and
// re-sorts descending. The reranker's score replaces the blended first-stage
// score; in explain mode each memory's breakdown gains a "rerank" block
// recording the first-stage score/rank and the new score/rank.
func applyReranker(reranker Reranker, query string, scored []scoredCand, expl map[string]map[string]any) ([]scoredCand, error) {
	mems := make([]Memory, len(scored))
	for i, sc := range scored {
		mems[i] = sc.mem
	}
	rr := reranker(query, mems)
	if len(rr) != len(mems) {
		return nil, validationErrorf("reranker returned %d scores for %d candidate(s)", len(rr), len(mems))
	}
	if expl != nil {
		for i, sc := range scored {
			e := expl[sc.mem.ID]
			if e == nil {
				e = map[string]any{}
				expl[sc.mem.ID] = e
			}
			e["rerank"] = map[string]any{
				"first_stage_score": round4(sc.score),
				"first_stage_rank":  i,
			}
		}
	}
	reranked := make([]scoredCand, len(scored))
	for i, sc := range scored {
		reranked[i] = scoredCand{mem: sc.mem, score: rr[i], emb: sc.emb}
	}
	sort.SliceStable(reranked, func(i, j int) bool { return reranked[i].score > reranked[j].score })
	if expl != nil {
		for newRank, sc := range reranked {
			if rrm, ok := expl[sc.mem.ID]["rerank"].(map[string]any); ok {
				rrm["score"] = round4(sc.score)
				rrm["rank"] = newRank
			}
		}
	}
	return reranked, nil
}

// scorerFilter applies the in-loop filters shared by all scorers: tag,
// superseded, and the absolute cosine floor.
func scorerFilter(m Memory, cosine float32, tag string, inclSup bool, minCosine *float32) bool {
	if tag != "" && !containsTag(m, tag) {
		return false
	}
	if !inclSup && m.SupersededBy != "" {
		return false
	}
	if minCosine != nil && cosine < *minCosine {
		return false
	}
	return true
}

// scoreSemantic scores by pure cosine plus a polarity nudge; results stay in
// query (descending-cosine) order unless the polarity weight re-orders them.
func scoreSemantic(cands []cand, w HybridWeights, tag string, inclSup bool, minCosine *float32, expl map[string]map[string]any) []scoredCand {
	pw := w.PolarityWeight
	out := make([]scoredCand, 0, len(cands))
	for _, c := range cands {
		if !scorerFilter(c.mem, c.cosine, tag, inclSup, minCosine) {
			continue
		}
		score := c.cosine + pw*float32(c.mem.Polarity)
		if expl != nil {
			expl[c.mem.ID] = map[string]any{
				"mode": "semantic", "cosine": round4(c.cosine),
				"polarity": c.mem.Polarity, "score": round4(score),
			}
		}
		out = append(out, scoredCand{mem: c.mem, score: score, emb: c.emb})
	}
	if pw != 0 {
		sort.SliceStable(out, func(i, j int) bool { return out[i].score > out[j].score })
	}
	return out
}

// scoreWeighted is the default hybrid blend: cw·cosine + lw·bm25 + rw·recency +
// iw·importance + pw·polarity, with a zero-lexical renormalisation when the
// query produces no lexical signal at all. bm25 is aligned to cands by index.
func (s *MemoryStore) scoreWeighted(cands []cand, bm25 []float32, w HybridWeights, tag string, inclSup bool, minCosine *float32, expl map[string]map[string]any, now float64) []scoredCand {
	cw, lw, rw, iw := w.Cosine, w.Lexical, w.Recency, w.Importance
	anyLex := false
	for _, b := range bm25 {
		if b > 0 {
			anyLex = true
			break
		}
	}
	// When the query yields no lexical signal, drop the lexical weight and
	// rescale the remaining core weights so scores stay comparable.
	if lw > 0 && !anyLex {
		core := cw + lw + rw + iw
		if core > lw {
			scale := core / (core - lw)
			cw, rw, iw, lw = cw*scale, rw*scale, iw*scale, 0
		}
	}
	pw := w.PolarityWeight
	out := make([]scoredCand, 0, len(cands))
	for i, c := range cands {
		if !scorerFilter(c.mem, c.cosine, tag, inclSup, minCosine) {
			continue
		}
		var lex float32
		if i < len(bm25) {
			lex = bm25[i]
		}
		rec := recencyScore(c.mem, w, s.cfg.DecayRate, now)
		score := cw*c.cosine + lw*lex + rw*rec + iw*c.mem.Importance + pw*float32(c.mem.Polarity)
		if expl != nil {
			expl[c.mem.ID] = map[string]any{
				"mode": "hybrid", "fusion": "weighted",
				"cosine": round4(c.cosine), "lexical": round4(lex),
				"recency": round4(rec), "importance": round4(c.mem.Importance),
				"polarity": c.mem.Polarity,
				"weights": map[string]any{
					"cosine": round4(cw), "lexical": round4(lw),
					"recency": round4(rw), "importance": round4(iw),
					"polarity": pw,
				},
				"score": round4(score),
			}
		}
		out = append(out, scoredCand{mem: c.mem, score: score, emb: c.emb})
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].score > out[j].score })
	return out
}

// scoreRRF fuses the four hybrid signals by Reciprocal Rank Fusion: each signal
// ranks the pool independently and the fused score is Σ wt/(rrfK+rank). Scale-
// free — immune to the cosine-vs-BM25 magnitude mismatch of the weighted blend.
func (s *MemoryStore) scoreRRF(cands []cand, bm25 []float32, w HybridWeights, tag string, inclSup bool, minCosine *float32, expl map[string]map[string]any, now float64, rrfK int) []scoredCand {
	type rrfRow struct {
		mem                                  Memory
		emb                                  []float32
		cosine, lexical, recency, importance float32
	}
	var rows []rrfRow
	for i, c := range cands {
		if !scorerFilter(c.mem, c.cosine, tag, inclSup, minCosine) {
			continue
		}
		var lex float32
		if i < len(bm25) {
			lex = bm25[i]
		}
		rows = append(rows, rrfRow{
			mem: c.mem, emb: c.emb, cosine: c.cosine, lexical: lex,
			recency: recencyScore(c.mem, w, s.cfg.DecayRate, now), importance: c.mem.Importance,
		})
	}
	if len(rows) == 0 {
		return nil
	}
	n := len(rows)
	signals := []struct {
		name string
		wt   float32
		val  func(rrfRow) float32
	}{
		{"cosine", w.Cosine, func(r rrfRow) float32 { return r.cosine }},
		{"lexical", w.Lexical, func(r rrfRow) float32 { return r.lexical }},
		{"recency", w.Recency, func(r rrfRow) float32 { return r.recency }},
		{"importance", w.Importance, func(r rrfRow) float32 { return r.importance }},
	}
	ranks := map[string][]int{}
	for _, sig := range signals {
		if sig.wt <= 0 {
			continue
		}
		order := make([]int, n)
		for j := range order {
			order[j] = j
		}
		val := sig.val
		sort.SliceStable(order, func(a, b int) bool { return val(rows[order[a]]) > val(rows[order[b]]) })
		r := make([]int, n)
		for pos, j := range order {
			r[j] = pos
		}
		ranks[sig.name] = r
	}
	pw := w.PolarityWeight
	out := make([]scoredCand, 0, n)
	for i, row := range rows {
		var score float32
		contrib := map[string]any{}
		for _, sig := range signals {
			if sig.wt <= 0 {
				continue
			}
			part := sig.wt / float32(rrfK+ranks[sig.name][i])
			score += part
			contrib[sig.name] = map[string]any{"rank": ranks[sig.name][i], "contribution": round6(part)}
		}
		score += pw * float32(row.mem.Polarity) / float32(rrfK+1)
		if expl != nil {
			expl[row.mem.ID] = map[string]any{
				"mode": "hybrid", "fusion": "rrf", "rrf_k": rrfK,
				"polarity": row.mem.Polarity, "signals": contrib, "score": round6(score),
			}
		}
		out = append(out, scoredCand{mem: row.mem, score: score, emb: row.emb})
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].score > out[j].score })
	return out
}

// mmrSelect re-selects up to k candidates balancing relevance and novelty
// (Maximal Marginal Relevance) and/or hard-dropping near-duplicates. Relevance
// is min-max normalised to [0,1] so the MMR trade-off is on the same scale as
// the cosine novelty penalty (critical for RRF's ~1/rrfK scores).
func mmrSelect(scored []scoredCand, k int, diversity, dedup *float32) []scoredCand {
	var lo, hi float32
	if len(scored) > 0 {
		lo, hi = scored[0].score, scored[0].score
		for _, sc := range scored {
			if sc.score < lo {
				lo = sc.score
			}
			if sc.score > hi {
				hi = sc.score
			}
		}
	}
	span := hi - lo
	rel := func(s float32) float32 {
		if span > 1e-12 {
			return (s - lo) / span
		}
		return 1.0
	}
	var selected []scoredCand
	var selEmbs [][]float32
	pool := make([]scoredCand, len(scored))
	copy(pool, scored)
	for len(pool) > 0 && len(selected) < k {
		bestIdx := -1
		var bestVal float32
		haveBest := false
		for idx, sc := range pool {
			var maxSim float32
			first := true
			if sc.emb != nil {
				for _, e := range selEmbs {
					sim := vector.CosineSim(sc.emb, e)
					if first || sim > maxSim {
						maxSim = sim
						first = false
					}
				}
			}
			if dedup != nil && len(selEmbs) > 0 && sc.emb != nil && maxSim >= *dedup {
				continue
			}
			var val float32
			if diversity != nil {
				lam := *diversity
				val = lam*rel(sc.score) - (1.0-lam)*maxSim
			} else {
				val = sc.score
			}
			if !haveBest || val > bestVal {
				bestVal = val
				bestIdx = idx
				haveBest = true
			}
		}
		if bestIdx < 0 {
			break // every remaining candidate was a near-duplicate
		}
		sc := pool[bestIdx]
		pool = append(pool[:bestIdx], pool[bestIdx+1:]...)
		selected = append(selected, sc)
		if sc.emb != nil {
			selEmbs = append(selEmbs, sc.emb)
		}
	}
	return selected
}
