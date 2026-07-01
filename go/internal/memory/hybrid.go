package memory

import "math"

// recencyScore is the recency term of the hybrid blend: exp(-decayRate·ageDays),
// where ageDays measures from created_at (default) or last_accessed depending
// on w.RecencyBasis. now is Unix seconds (sub-second precision ok).
func recencyScore(m Memory, w HybridWeights, decayRate float32, now float64) float32 {
	basis := m.CreatedAt
	if w.RecencyBasis == "accessed" {
		basis = m.LastAccessed
	}
	ageDays := (now - basis) / 86400.0
	if ageDays < 0 {
		ageDays = 0
	}
	return float32(math.Exp(float64(-decayRate) * ageDays))
}
