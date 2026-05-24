package embed

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
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
	return &OpenAIEmbedder{
		APIKey:  apiKey,
		Model:   model,
		BaseURL: "https://api.openai.com",
		client:  &http.Client{},
	}
}

func (o *OpenAIEmbedder) Dim() int { return o.dim }

func (o *OpenAIEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	type req struct {
		Input []string `json:"input"`
		Model string   `json:"model"`
	}
	type embObj struct {
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
	out := make([][]float32, len(r.Data))
	for i, d := range r.Data {
		out[i] = d.Embedding
	}
	if len(out) > 0 && o.dim == 0 {
		o.dim = len(out[0])
	}
	return out, nil
}
