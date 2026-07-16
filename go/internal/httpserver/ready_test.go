package httpserver_test

import (
	"net/http"
	"testing"
)

func TestReadyReturns200(t *testing.T) {
	ts, _ := newTestServer(t, "")
	resp, err := http.Get(ts.URL + "/ready")
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != 200 {
		t.Fatalf("ready status = %d, want 200", resp.StatusCode)
	}
	body := decode(t, resp)
	if body["ready"] != true {
		t.Errorf("ready body = %v", body)
	}
}

func TestReadyIsAuthExempt(t *testing.T) {
	// /ready must be reachable without the bearer token (like /health) so
	// readiness probes work without the secret.
	ts, _ := newTestServer(t, "s3cret")
	resp, err := http.Get(ts.URL + "/ready")
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != 200 {
		t.Fatalf("ready without token = %d, want 200 (auth-exempt)", resp.StatusCode)
	}
	// A protected route still requires the token.
	resp2, err := http.Get(ts.URL + "/stats")
	if err != nil {
		t.Fatal(err)
	}
	if resp2.StatusCode != 401 {
		t.Fatalf("stats without token = %d, want 401", resp2.StatusCode)
	}
}

func TestReadyBodyIsSanitized(t *testing.T) {
	// /ready is auth-exempt, so its body must expose only per-check ok bools —
	// no provider dim/latency/error strings that could enumerate internals.
	ts, _ := newTestServer(t, "")
	resp, err := http.Get(ts.URL + "/ready")
	if err != nil {
		t.Fatal(err)
	}
	body := decode(t, resp)
	checks, ok := body["checks"].(map[string]any)
	if !ok {
		t.Fatalf("no checks in body: %v", body)
	}
	emb, ok := checks["embedder"].(map[string]any)
	if !ok {
		t.Fatalf("no embedder check: %v", checks)
	}
	if _, ok := emb["ok"]; !ok {
		t.Error("embedder check should expose ok")
	}
	if len(emb) != 1 {
		t.Errorf("embedder check must expose only {ok}, got %v", emb)
	}
}
