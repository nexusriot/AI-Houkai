package embed

import "context"

// Embedder is the pluggable embedding backend.
type Embedder interface {
	Dim() int
	Embed(ctx context.Context, texts []string) ([][]float32, error)
}

// Provider names for config. Ollama has no constant: it is the default
// branch wherever providers are switched on, never compared by name.
const (
	ProviderOpenAI       = "openai"
	ProviderDigitalOcean = "digitalocean"
)

// DigitalOceanBaseURL is the default endpoint for DO Serverless Inference.
const DigitalOceanBaseURL = "https://inference.do-ai.run"

// NewDigitalOcean builds an embedder against DigitalOcean's Serverless
// Inference API. It is wire-compatible with OpenAI's /v1/embeddings, so this
// is a thin wrapper over NewOpenAICompatible with the DO base URL.
func NewDigitalOcean(apiKey, model string) *OpenAIEmbedder {
	return NewOpenAICompatible(apiKey, model, DigitalOceanBaseURL)
}
