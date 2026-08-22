package embed

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
)

// OpenAIEmbedder calls the OpenAI embeddings API.
type OpenAIEmbedder struct {
	APIKey  string
	Model   string
	BaseURL string
	dim     int
	client  *http.Client
}

func NewOpenAI(apiKey, model string) *OpenAIEmbedder {
	return NewOpenAICompatible(apiKey, model, "https://api.openai.com")
}

// NewOpenAICompatible builds an embedder against any service that speaks the
// OpenAI /v1/embeddings protocol (e.g. DigitalOcean Serverless Inference,
// llama.cpp's openai-compat server, vLLM, Together, etc.). The baseURL should
// be the host root without the /v1 suffix — e.g. "https://inference.do-ai.run".
func NewOpenAICompatible(apiKey, model, baseURL string) *OpenAIEmbedder {
	return &OpenAIEmbedder{
		APIKey:  apiKey,
		Model:   model,
		BaseURL: baseURL,
		client:  &http.Client{Timeout: embedTimeout},
	}
}

func (o *OpenAIEmbedder) Dim() int { return o.dim }

func (o *OpenAIEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	type req struct {
		Input []string `json:"input"`
		Model string   `json:"model"`
	}
	type embObj struct {
		Index     int       `json:"index"`
		Embedding []float32 `json:"embedding"`
	}
	type resp struct {
		Data []embObj `json:"data"`
	}

	body, _ := json.Marshal(req{Input: texts, Model: o.Model})
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost,
		o.BaseURL+"/v1/embeddings", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Authorization", "Bearer "+o.APIKey)
	httpReq.Header.Set("Content-Type", "application/json")

	res, err := o.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("openai embed: %w", err)
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("openai embed: HTTP %d", res.StatusCode)
	}

	var r resp
	if err := json.NewDecoder(res.Body).Decode(&r); err != nil {
		return nil, fmt.Errorf("openai embed decode: %w", err)
	}
	// A short response with HTTP 200 would otherwise surface as an index
	// panic in the store's batch loops.
	if len(r.Data) != len(texts) {
		return nil, fmt.Errorf("openai embed: got %d embeddings for %d inputs",
			len(r.Data), len(texts))
	}
	// The API documents index-ordered results but does not guarantee the
	// ordering — an out-of-order response would silently store the wrong
	// vector under the wrong text.
	sort.SliceStable(r.Data, func(i, j int) bool {
		return r.Data[i].Index < r.Data[j].Index
	})
	out := make([][]float32, len(r.Data))
	for i, d := range r.Data {
		out[i] = d.Embedding
	}
	if len(out) > 0 && o.dim == 0 {
		o.dim = len(out[0])
	}
	return out, nil
}
