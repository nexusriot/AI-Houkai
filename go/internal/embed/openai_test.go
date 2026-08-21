package embed

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// fakeOpenAIServer stands in for any OpenAI-compatible /v1/embeddings host.
func fakeOpenAIServer(t *testing.T, wantAuth, wantModel string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/embeddings" {
			http.Error(w, "wrong path: "+r.URL.Path, http.StatusNotFound)
			return
		}
		if got := r.Header.Get("Authorization"); got != wantAuth {
			http.Error(w, "auth: "+got, http.StatusUnauthorized)
			return
		}
		body, _ := io.ReadAll(r.Body)
		var req struct {
			Model string   `json:"model"`
			Input []string `json:"input"`
		}
		if err := json.Unmarshal(body, &req); err != nil {
			http.Error(w, "bad json", 400)
			return
		}
		if req.Model != wantModel {
			http.Error(w, "model: "+req.Model, 400)
			return
		}
		// Return a 4-dim vector per input.
		type d struct {
			Embedding []float32 `json:"embedding"`
			Index     int       `json:"index"`
			Object    string    `json:"object"`
		}
		out := make([]d, len(req.Input))
		for i := range req.Input {
			out[i] = d{Embedding: []float32{0.1, 0.2, 0.3, 0.4}, Index: i, Object: "embedding"}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"data": out, "model": req.Model})
	}))
}

func TestOpenAIEmbedderSendsAuthAndModel(t *testing.T) {
	srv := fakeOpenAIServer(t, "Bearer test-key", "text-embedding-3-small")
	defer srv.Close()

	e := NewOpenAICompatible("test-key", "text-embedding-3-small", srv.URL)
	vecs, err := e.Embed(context.Background(), []string{"hello", "world"})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(vecs) != 2 || len(vecs[0]) != 4 {
		t.Errorf("unexpected vectors: %v", vecs)
	}
	if e.Dim() != 4 {
		t.Errorf("Dim() = %d, want 4 (cached from response)", e.Dim())
	}
}

func TestDigitalOceanEmbedder(t *testing.T) {
	// Use a stand-in server but verify the constructor wires the right model
	// + auth shape that DO expects.
	srv := fakeOpenAIServer(t, "Bearer do-token", "qwen3-embedding-0.6b")
	defer srv.Close()

	e := NewDigitalOcean("do-token", "qwen3-embedding-0.6b")
	// Override the production DO URL with our test server.
	e.BaseURL = srv.URL

	vecs, err := e.Embed(context.Background(), []string{"hello world"})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(vecs) != 1 || len(vecs[0]) != 4 {
		t.Errorf("unexpected vectors: %v", vecs)
	}
}

func TestDigitalOceanBaseURLConstant(t *testing.T) {
	e := NewDigitalOcean("k", "m")
	if !strings.HasPrefix(e.BaseURL, "https://inference.do-ai.run") {
		t.Errorf("DigitalOcean baseURL = %q, want https://inference.do-ai.run", e.BaseURL)
	}
}

func TestOpenAIEmbedderPropagatesNon200(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "boom", http.StatusInternalServerError)
	}))
	defer srv.Close()

	e := NewOpenAICompatible("k", "m", srv.URL)
	_, err := e.Embed(context.Background(), []string{"x"})
	if err == nil {
		t.Fatal("expected error on HTTP 500")
	}
	if !strings.Contains(err.Error(), "500") {
		t.Errorf("error should mention status code: %v", err)
	}
}
