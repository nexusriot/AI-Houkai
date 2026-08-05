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

// pinned and trust can be set over HTTP but were never serialised back. The MCP
// surface carries both; the REST payload did not, so a client — ai-houkai-service
// is the main one — could pin a memory or mark it untrusted and then had no way
// to see that state again.
func TestRESTPayloadCarriesPinnedAndTrust(t *testing.T) {
	ts, _, _ := newJournaledServer(t)

	st, created := post(t, ts.URL+"/memories", map[string]any{
		"text": "a standing rule", "pinned": true, "trust": "reported",
	})
	if st != 200 && st != 201 {
		t.Fatalf("POST /memories = %d %v", st, created)
	}
	if created["pinned"] != true {
		t.Errorf("POST response pinned = %v, want true", created["pinned"])
	}
	if created["trust"] != "reported" {
		t.Errorf("POST response trust = %v, want reported", created["trust"])
	}
	id, _ := created["id"].(string)

	resp, err := http.Get(ts.URL + "/memories/" + id)
	if err != nil {
		t.Fatalf("GET: %v", err)
	}
	got := decode(t, resp)
	if got["pinned"] != true || got["trust"] != "reported" {
		t.Errorf("GET payload pinned=%v trust=%v, want true/reported",
			got["pinned"], got["trust"])
	}
}

// Absent is not the same as false: a client must not have to guess a default.
func TestRESTPayloadStatesTheDefaultsExplicitly(t *testing.T) {
	ts, _, _ := newJournaledServer(t)

	_, created := post(t, ts.URL+"/memories", map[string]any{"text": "a plain fact"})
	id, _ := created["id"].(string)
	resp, err := http.Get(ts.URL + "/memories/" + id)
	if err != nil {
		t.Fatalf("GET: %v", err)
	}
	got := decode(t, resp)
	if v, ok := got["pinned"]; !ok || v != false {
		t.Errorf("pinned = %v (present=%v), want an explicit false", v, ok)
	}
	if v, ok := got["trust"]; !ok || v != "trusted" {
		t.Errorf("trust = %v (present=%v), want an explicit \"trusted\"", v, ok)
	}
}

func TestRecallHitsCarryPinnedAndTrust(t *testing.T) {
	ts, _, _ := newJournaledServer(t)

	if st, body := post(t, ts.URL+"/memories", map[string]any{
		"text": "recallable standing rule", "pinned": true, "trust": "reported",
	}); st != 200 && st != 201 {
		t.Fatalf("seed: %d %v", st, body)
	}
	_, body := post(t, ts.URL+"/recall", map[string]any{
		"query": "recallable standing rule", "k": 5,
	})
	results, _ := body["results"].([]any)
	if len(results) == 0 {
		t.Fatalf("no results: %v", body)
	}
	hit, _ := results[0].(map[string]any)
	if hit["pinned"] != true || hit["trust"] != "reported" {
		t.Errorf("hit pinned=%v trust=%v, want true/reported",
			hit["pinned"], hit["trust"])
	}
}

// An idempotent repeat is the feature working, not a failure: the store found
// the existing row, bumped its access count and returned it. This port mapped
// "nothing new was written" onto 409 with an EMPTY conflicts list, so a client
// replaying a batch every session got a hard error on every known fact — while
// the Python port answered 201. 200 + stored:false says what happened.
func TestIdempotentRepeatIsNotAConflict(t *testing.T) {
	ts, _, _ := newJournaledServer(t)

	st1, first := post(t, ts.URL+"/memories",
		map[string]any{"text": "repeat me", "idempotent": true})
	if st1 != 201 {
		t.Fatalf("first write = %d %v, want 201", st1, first)
	}
	if first["stored"] != true {
		t.Errorf("first write stored = %v, want true", first["stored"])
	}

	st2, second := post(t, ts.URL+"/memories",
		map[string]any{"text": "repeat me", "idempotent": true})
	if st2 != 200 {
		t.Fatalf("repeat = %d %v, want 200", st2, second)
	}
	if second["stored"] != false {
		t.Errorf("repeat stored = %v, want false — no new row was written",
			second["stored"])
	}
	if second["id"] != first["id"] {
		t.Errorf("repeat id = %v, want the existing %v", second["id"], first["id"])
	}
	if _, ok := second["conflicts"]; ok {
		t.Errorf("repeat reported conflicts: %v", second["conflicts"])
	}
}

// A real conflict rejection must still be a 409 with the conflicts attached.
func TestRealConflictStillReturns409(t *testing.T) {
	ts, _, _ := newJournaledServer(t)

	if st, b := post(t, ts.URL+"/memories", map[string]any{
		"text": "the sky is blue", "polarity": 1,
	}); st != 201 {
		t.Fatalf("seed = %d %v", st, b)
	}
	st, body := post(t, ts.URL+"/memories", map[string]any{
		"text": "the sky is blue", "polarity": -1, "on_conflict": "raise",
	})
	if st != 409 {
		t.Fatalf("contradiction = %d %v, want 409", st, body)
	}
	conflicts, _ := body["conflicts"].([]any)
	if len(conflicts) == 0 {
		t.Errorf("409 carried no conflicts: %v", body)
	}
}

// Same contract as the single write: `stored` counts rows CREATED, so a
// replayed idempotent batch reports 0 and answers 200 rather than claiming it
// wrote every item again.
func TestBatchStoredCountsOnlyNewRows(t *testing.T) {
	ts, _, _ := newJournaledServer(t)
	items := []any{
		map[string]any{"text": "batch fact one"},
		map[string]any{"text": "batch fact two"},
	}

	st, first := post(t, ts.URL+"/memories/batch",
		map[string]any{"items": items, "idempotent": true})
	if st != 201 {
		t.Fatalf("first batch = %d %v, want 201", st, first)
	}
	if first["stored"] != float64(2) {
		t.Fatalf("first batch stored = %v, want 2", first["stored"])
	}

	st, again := post(t, ts.URL+"/memories/batch",
		map[string]any{"items": items, "idempotent": true})
	if st != 200 {
		t.Fatalf("replay = %d %v, want 200", st, again)
	}
	if again["stored"] != float64(0) {
		t.Errorf("replay stored = %v, want 0", again["stored"])
	}
	if mems, _ := again["memories"].([]any); len(mems) != 2 {
		t.Errorf("replay returned %d memories, want the 2 existing rows", len(mems))
	}
}

// Intra-batch duplicates collapse to one row, so they count once.
func TestBatchIntraDuplicatesCountOnce(t *testing.T) {
	ts, _, _ := newJournaledServer(t)
	st, out := post(t, ts.URL+"/memories/batch", map[string]any{
		"items": []any{
			map[string]any{"text": "same text"},
			map[string]any{"text": "same text"},
		},
		"idempotent": true,
	})
	if st != 201 {
		t.Fatalf("batch = %d %v, want 201", st, out)
	}
	if out["stored"] != float64(1) {
		t.Errorf("stored = %v, want 1", out["stored"])
	}
}

func TestBatchWithoutTheFlagCountsEveryItem(t *testing.T) {
	ts, _, _ := newJournaledServer(t)
	_, out := post(t, ts.URL+"/memories/batch", map[string]any{
		"items": []any{
			map[string]any{"text": "dup text"},
			map[string]any{"text": "dup text"},
		},
	})
	if out["stored"] != float64(2) {
		t.Errorf("stored = %v, want 2 — idempotency stays opt-in", out["stored"])
	}
}
