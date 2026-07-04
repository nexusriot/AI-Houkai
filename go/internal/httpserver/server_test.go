package httpserver_test

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"

	"github.com/nexusriot/ai-houkai/internal/httpserver"
	"github.com/nexusriot/ai-houkai/internal/memory"
	"github.com/nexusriot/ai-houkai/internal/vector"
)

// stubEmbedder mirrors the deterministic test embedder used in the memory pkg.
type stubEmbedder struct{ dim int }

func (e *stubEmbedder) Dim() int { return e.dim }

func (e *stubEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, t := range texts {
		v := make([]float32, e.dim)
		h := fnv.New64a()
		h.Write([]byte(t))
		seed := h.Sum64()
		for j := 0; j < e.dim; j++ {
			seed = seed*6364136223846793005 + 1442695040888963407
			v[j] = float32(int64(seed>>33)%1000) / 1000.0
		}
		var norm float64
		for _, x := range v {
			norm += float64(x) * float64(x)
		}
		norm = math.Sqrt(norm)
		if norm > 0 {
			for j := range v {
				v[j] = float32(float64(v[j]) / norm)
			}
		}
		out[i] = v
	}
	return out, nil
}

func newTestServer(t *testing.T, token string) (*httptest.Server, *memory.MemoryStore) {
	t.Helper()
	dir := t.TempDir()
	backend, err := vector.NewChromem(filepath.Join(dir, "s"), "test", 16)
	if err != nil {
		t.Fatalf("NewChromem: %v", err)
	}
	t.Cleanup(func() { _ = backend.Close() })

	store := memory.NewMemoryStore(backend, &stubEmbedder{dim: 16}, memory.DefaultStoreConfig(dir, "test"))
	srv := httpserver.New(store, dir, "test", token)
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)
	return ts, store
}

func decode(t *testing.T, resp *http.Response) map[string]any {
	t.Helper()
	defer resp.Body.Close()
	var m map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&m); err != nil {
		t.Fatalf("decode: %v", err)
	}
	return m
}

func TestHealthIsOpen(t *testing.T) {
	ts, _ := newTestServer(t, "")
	resp, err := http.Get(ts.URL + "/health")
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != 200 {
		t.Fatalf("health status = %d", resp.StatusCode)
	}
	body := decode(t, resp)
	if body["status"] != "ok" {
		t.Errorf("health body = %v", body)
	}
}

func TestRememberGetForgetRoundTrip(t *testing.T) {
	ts, _ := newTestServer(t, "")

	// POST /memories
	resp, err := http.Post(ts.URL+"/memories", "application/json",
		bytes.NewReader([]byte(`{"text":"remember this","type":"semantic","importance":0.7}`)))
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != 201 {
		t.Fatalf("remember status = %d", resp.StatusCode)
	}
	body := decode(t, resp)
	if body["stored"] != true {
		t.Fatalf("stored = %v", body["stored"])
	}
	id, _ := body["id"].(string)
	if id == "" {
		t.Fatal("no id returned")
	}

	// GET /memories/{id}
	resp, err = http.Get(ts.URL + "/memories/" + id)
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != 200 {
		t.Fatalf("get status = %d", resp.StatusCode)
	}
	got := decode(t, resp)
	if got["text"] != "remember this" {
		t.Errorf("get text = %v", got["text"])
	}

	// DELETE /memories/{id}
	req, _ := http.NewRequest(http.MethodDelete, ts.URL+"/memories/"+id, nil)
	resp, err = http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != 200 {
		t.Fatalf("delete status = %d", resp.StatusCode)
	}
	resp.Body.Close()

	// Now 404.
	resp, _ = http.Get(ts.URL + "/memories/" + id)
	if resp.StatusCode != 404 {
		t.Errorf("get-after-delete status = %d, want 404", resp.StatusCode)
	}
	resp.Body.Close()
}

func TestRecallFiltersBySource(t *testing.T) {
	ts, _ := newTestServer(t, "")
	post := func(payload string) {
		resp, err := http.Post(ts.URL+"/memories", "application/json", bytes.NewReader([]byte(payload)))
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
	}
	post(`{"text":"deploy via git","source":"git"}`)
	post(`{"text":"deploy via runbook","source":"runbook"}`)

	resp, err := http.Get(ts.URL + "/recall?query=deploy&k=5&source=git")
	if err != nil {
		t.Fatal(err)
	}
	body := decode(t, resp)
	results, _ := body["results"].([]any)
	if len(results) != 1 {
		t.Fatalf("source filter: want 1 result, got %d (%v)", len(results), results)
	}
	first, _ := results[0].(map[string]any)
	if first["source"] != "git" {
		t.Errorf("source filter leaked: %v", first["source"])
	}
}

func TestRecallBadSinceIs400(t *testing.T) {
	ts, _ := newTestServer(t, "")
	resp, err := http.Get(ts.URL + "/recall?query=x&since=not-a-time")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 400 {
		t.Errorf("bad since: status = %d, want 400", resp.StatusCode)
	}
}

func TestAuthGuardsRoutesButNotHealth(t *testing.T) {
	ts, _ := newTestServer(t, "secret")

	// /health is open.
	resp, _ := http.Get(ts.URL + "/health")
	if resp.StatusCode != 200 {
		t.Errorf("health should be open, got %d", resp.StatusCode)
	}
	resp.Body.Close()

	// /stats requires the token.
	resp, _ = http.Get(ts.URL + "/stats")
	if resp.StatusCode != 401 {
		t.Errorf("stats without token = %d, want 401", resp.StatusCode)
	}
	resp.Body.Close()

	// With the right bearer token it succeeds.
	req, _ := http.NewRequest(http.MethodGet, ts.URL+"/stats", nil)
	req.Header.Set("Authorization", "Bearer secret")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != 200 {
		t.Errorf("stats with token = %d, want 200", resp.StatusCode)
	}
	resp.Body.Close()
}

func TestMethodNotAllowedAnd404(t *testing.T) {
	ts, _ := newTestServer(t, "")

	// Wrong method on a known path → 405.
	req, _ := http.NewRequest(http.MethodPut, ts.URL+"/memories", nil)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusMethodNotAllowed {
		t.Errorf("PUT /memories = %d, want 405", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); ct != "application/json" {
		t.Errorf("405 Content-Type = %q, want application/json", ct)
	}
	if body := decode(t, resp); body["error"] == nil {
		t.Errorf("405 body should carry an error field: %v", body)
	}

	// Unknown path → 404, also JSON.
	resp, _ = http.Get(ts.URL + "/nope")
	if resp.StatusCode != 404 {
		t.Errorf("unknown path = %d, want 404", resp.StatusCode)
	}
	if body := decode(t, resp); body["error"] == nil {
		t.Errorf("404 body should carry an error field: %v", body)
	}
}

func TestInvalidJSONBodyIs400(t *testing.T) {
	ts, _ := newTestServer(t, "")
	resp, err := http.Post(ts.URL+"/memories", "application/json", bytes.NewReader([]byte(`{bad`)))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 400 {
		b, _ := io.ReadAll(resp.Body)
		t.Errorf("invalid JSON = %d, want 400 (%s)", resp.StatusCode, b)
	}
}

func TestRecallPackReturnsBlock(t *testing.T) {
	ts, _ := newTestServer(t, "")
	for i := 0; i < 3; i++ {
		payload := fmt.Sprintf(`{"text":"deploy step %d","type":"procedural"}`, i)
		resp, _ := http.Post(ts.URL+"/memories", "application/json", bytes.NewReader([]byte(payload)))
		resp.Body.Close()
	}
	resp, err := http.Post(ts.URL+"/recall_pack", "application/json",
		bytes.NewReader([]byte(`{"query":"deploy","token_budget":500}`)))
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != 200 {
		t.Fatalf("recall_pack status = %d", resp.StatusCode)
	}
	body := decode(t, resp)
	if _, ok := body["text"].(string); !ok {
		t.Errorf("recall_pack missing text block: %v", body)
	}
	if _, ok := body["items"].([]any); !ok {
		t.Errorf("recall_pack missing items: %v", body)
	}
}

func TestAutoContextEndpoint(t *testing.T) {
	ts, store := newTestServer(t, "")
	ctx := context.Background()
	store.Remember(ctx, "deployment pipeline runbook and rollback", memory.RememberOpts{Type: memory.Procedural, Importance: memory.Float32Ptr(0.9)})

	body := bytes.NewBufferString(`{"task":"the deployment pipeline failed","token_budget":800}`)
	resp, err := http.Post(ts.URL+"/auto_context", "application/json", body)
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != 200 {
		t.Fatalf("auto_context status = %d", resp.StatusCode)
	}
	m := decode(t, resp)
	if _, ok := m["queries"].([]any); !ok {
		t.Errorf("auto_context response missing queries: %v", m)
	}
	if _, ok := m["items"].([]any); !ok {
		t.Errorf("auto_context response missing items: %v", m)
	}
}

func TestHealthOmitsCollection(t *testing.T) {
	ts, _ := newTestServer(t, "")
	resp, _ := http.Get(ts.URL + "/health")
	m := decode(t, resp)
	if _, present := m["collection"]; present {
		t.Error("/health must not expose the collection name")
	}
}
