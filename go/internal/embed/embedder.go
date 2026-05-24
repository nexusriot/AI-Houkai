package embed

import "context"

// Embedder is the pluggable embedding backend.
type Embedder interface {
	Dim() int
	Embed(ctx context.Context, texts []string) ([][]float32, error)
}

// Provider names for config.
const (
	ProviderOllama = "ollama"
	ProviderOpenAI = "openai"
)
