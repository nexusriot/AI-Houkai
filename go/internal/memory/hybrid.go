package memory

import (
	"math"
	"time"
)

// hybridScore combines cosine, BM25, recency, and importance.
func hybridScore(cosine, bm25, importance float32, lastAccessed float64,
	w HybridWeights, decayRate float32) float32 {

	ageDays := float32(time.Since(time.Unix(int64(lastAccessed), 0)).Hours() / 24)
	recency := float32(math.Exp(float64(-decayRate * ageDays)))

	return w.Cosine*cosine + w.Lexical*bm25 + w.Recency*recency + w.Importance*importance
}
