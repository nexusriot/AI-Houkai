package httpserver_test

// Regression tests for the Python-parity review fixes on the HTTP surface:
// the PATCH edit endpoint, pointer importance, header="" on recall_pack, and
// ValidationError → 400 mapping.

import (
	"bytes"
	"context"
	"fmt"
	"net/http"
	"strings"
	"testing"

	"github.com/nexusriot/ai-houkai/internal/memory"
)

// doJSON issues an arbitrary-method JSON request.
func doJSON(t *testing.T, method, url, path, payload string) *http.Response {
	t.Helper()
	req, err := http.NewRequest(method, url+path, bytes.NewReader([]byte(payload)))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	return resp
}

func TestRememberExplicitZeroImportance(t *testing.T) {
	ts, _ := newTestServer(t, "")
	resp := postJSON(t, ts.URL, "/memories", `{"text":"worthless but explicit","importance":0}`)
	if resp.StatusCode != 201 {
		t.Fatalf("remember status = %d, want 201", resp.StatusCode)
	}
	body := decode(t, resp)
	if imp, ok := body["importance"].(float64); !ok || imp != 0 {
		t.Errorf("importance = %v, want 0 (explicit zero must not fall back to 0.5)", body["importance"])
	}

	// Omitted importance → the store default.
	resp = postJSON(t, ts.URL, "/memories", `{"text":"defaulted importance"}`)
	body = decode(t, resp)
	if imp, ok := body["importance"].(float64); !ok || imp != 0.5 {
		t.Errorf("default importance = %v, want 0.5", body["importance"])
	}
}

func TestValidationErrorsAre400(t *testing.T) {
	ts, _ := newTestServer(t, "")
	a := rememberHTTP(t, ts.URL, `{"text":"validation source","type":"semantic"}`)
	b := rememberHTTP(t, ts.URL, `{"text":"validation target","type":"semantic"}`)

	cases := []struct {
		name, path, payload string
	}{
		{"bad recall mode", "/recall", `{"query":"x","mode":"hybird"}`},
		{"bad recall fusion", "/recall", `{"query":"x","mode":"hybrid","fusion":"borda"}`},
		{"bad remember type", "/memories", `{"text":"x","type":"opinions"}`},
		{"bad remember polarity", "/memories", `{"text":"x","polarity":5}`},
		{"comma tag", "/memories", `{"text":"x","tags":["a,b"]}`},
		{"bad link rel", "/links", fmt.Sprintf(`{"src_id":%q,"dst_id":%q,"rel":"friend_of"}`, a, b)},
		{"self link", "/links", fmt.Sprintf(`{"src_id":%q,"dst_id":%q}`, a, a)},
	}
	for _, tc := range cases {
		resp := postJSON(t, ts.URL, tc.path, tc.payload)
		body := decode(t, resp)
		if resp.StatusCode != 400 {
			t.Errorf("%s: status = %d (%v), want 400", tc.name, resp.StatusCode, body)
		}
	}
}

func TestRecallPackHeaderEmptyVsAbsent(t *testing.T) {
	ts, _ := newTestServer(t, "")
	for i := 0; i < 3; i++ {
		resp := postJSON(t, ts.URL, "/memories",
			fmt.Sprintf(`{"text":"release step %d of the pipeline","type":"procedural"}`, i))
		resp.Body.Close()
	}

	// Absent header → the default "## Relevant memory" line.
	resp := postJSON(t, ts.URL, "/recall_pack", `{"query":"release pipeline","token_budget":500}`)
	body := decode(t, resp)
	text, _ := body["text"].(string)
	if !strings.HasPrefix(text, "## Relevant memory") {
		t.Errorf("absent header: text = %q, want the default header line", text)
	}

	// Explicit "" → NO header line (previously inexpressible over HTTP).
	resp = postJSON(t, ts.URL, "/recall_pack", `{"query":"release pipeline","token_budget":500,"header":""}`)
	body = decode(t, resp)
	text, _ = body["text"].(string)
	if text == "" || strings.Contains(text, "## Relevant memory") {
		t.Errorf("header \"\": text = %q, want items with no header line", text)
	}

	// A custom header is passed through.
	resp = postJSON(t, ts.URL, "/recall_pack", `{"query":"release pipeline","token_budget":500,"header":"### Context"}`)
	body = decode(t, resp)
	text, _ = body["text"].(string)
	if !strings.HasPrefix(text, "### Context") {
		t.Errorf("custom header: text = %q, want ### Context prefix", text)
	}
}

func TestEditEndpoint(t *testing.T) {
	ts, store := newTestServer(t, "")
	id := rememberHTTP(t, ts.URL, `{"text":"HTTP edit target","type":"semantic","tags":["orig"]}`)

	resp := doJSON(t, "PATCH", ts.URL, "/memories/"+id,
		`{"text":"HTTP edit target, revised","importance":0,"tags":["patched"],"source":"review"}`)
	if resp.StatusCode != 200 {
		t.Fatalf("PATCH status = %d, want 200", resp.StatusCode)
	}
	body := decode(t, resp)
	if body["text"] != "HTTP edit target, revised" {
		t.Errorf("text = %v", body["text"])
	}
	if imp, _ := body["importance"].(float64); imp != 0 {
		t.Errorf("importance = %v, want 0", body["importance"])
	}
	if body["source"] != "review" {
		t.Errorf("source = %v, want review", body["source"])
	}

	// The edit landed in the store, id unchanged.
	m, err := store.GetByID(context.Background(), id)
	if err != nil {
		t.Fatal(err)
	}
	if m.Text != "HTTP edit target, revised" || len(m.Tags) != 1 || m.Tags[0] != "patched" {
		t.Errorf("stored memory = %+v", m)
	}

	// Journaled: the edit is visible in the audit journal (and hence undoable).
	entries, _ := store.Journal().Read(memory.ReadOpts{Op: "edit", MemoryID: m.ID})
	if len(entries) != 1 {
		t.Errorf("journal edit entries = %d, want 1", len(entries))
	}
}

func TestEditEndpointErrors(t *testing.T) {
	ts, _ := newTestServer(t, "")
	id := rememberHTTP(t, ts.URL, `{"text":"error cases","type":"semantic"}`)

	// Unknown id → 404.
	resp := doJSON(t, "PATCH", ts.URL, "/memories/no-such-id", `{"text":"x"}`)
	if resp.StatusCode != 404 {
		t.Errorf("PATCH unknown id = %d, want 404", resp.StatusCode)
	}
	resp.Body.Close()

	// Bad enum → 400 (ValidationError mapping).
	resp = doJSON(t, "PATCH", ts.URL, "/memories/"+id, `{"type":"opinions"}`)
	if resp.StatusCode != 400 {
		t.Errorf("PATCH bad type = %d, want 400", resp.StatusCode)
	}
	resp.Body.Close()

	// Empty body → 400.
	resp = doJSON(t, "PATCH", ts.URL, "/memories/"+id, `{}`)
	if resp.StatusCode != 400 {
		t.Errorf("PATCH empty body = %d, want 400", resp.StatusCode)
	}
	resp.Body.Close()

	// Whitespace-only text → 400.
	resp = doJSON(t, "PATCH", ts.URL, "/memories/"+id, `{"text":"   "}`)
	if resp.StatusCode != 400 {
		t.Errorf("PATCH blank text = %d, want 400", resp.StatusCode)
	}
	resp.Body.Close()
}
