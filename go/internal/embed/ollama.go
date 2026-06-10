package embed

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
)

// OllamaEmbedder calls the local Ollama /api/embed endpoint.
type OllamaEmbedder struct {
	BaseURL string
	Model   string
	dim     int
	client  *http.Client
}

func NewOllama(baseURL, model string) *OllamaEmbedder {
	return &OllamaEmbedder{
		BaseURL: baseURL,
		Model:   model,
		client:  &http.Client{},
	}
}

func (o *OllamaEmbedder) Dim() int { return o.dim }

func (o *OllamaEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	type req struct {
		Model  string   `json:"model"`
		Input  []string `json:"input"`
	}
	type resp struct {
		Embeddings [][]float32 `json:"embeddings"`
	}

	body, _ := json.Marshal(req{Model: o.Model, Input: texts})
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost,
		o.BaseURL+"/api/embed", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")

	res, err := o.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("ollama embed: %w", err)
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ollama embed: HTTP %d", res.StatusCode)
	}

	var r resp
	if err := json.NewDecoder(res.Body).Decode(&r); err != nil {
		return nil, fmt.Errorf("ollama embed decode: %w", err)
	}
	if len(r.Embeddings) == 0 {
		return nil, fmt.Errorf("ollama returned empty embeddings")
	}
	if o.dim == 0 {
		o.dim = len(r.Embeddings[0])
	}
	return r.Embeddings, nil
}
