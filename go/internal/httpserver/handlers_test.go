package httpserver_test

import (
	"bytes"
	"context"
	"fmt"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/nexusriot/ai-houkai/internal/memory"
)

// remember stores a memory through the HTTP API and returns its id.
func rememberHTTP(t *testing.T, url, payload string) string {
	t.Helper()
	resp, err := http.Post(url+"/memories", "application/json", bytes.NewReader([]byte(payload)))
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != 201 {
		t.Fatalf("remember status = %d for %s", resp.StatusCode, payload)
	}
	body := decode(t, resp)
	id, _ := body["id"].(string)
	if id == "" {
		t.Fatalf("no id returned for %s", payload)
	}
	return id
}

func postJSONAuthed(t *testing.T, url, path, payload, token string) *http.Response {
	t.Helper()
	req, err := http.NewRequest(http.MethodPost, url+path, strings.NewReader(payload))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+token)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	return resp
}

func postJSON(t *testing.T, url, path, payload string) *http.Response {
	t.Helper()
	resp, err := http.Post(url+path, "application/json", bytes.NewReader([]byte(payload)))
	if err != nil {
		t.Fatal(err)
	}
	return resp
}

func TestListMemories(t *testing.T) {
	ts, _ := newTestServer(t, "")
	rememberHTTP(t, ts.URL, `{"text":"alpha memory one","type":"semantic"}`)
	rememberHTTP(t, ts.URL, `{"text":"beta memory two","type":"episodic"}`)

	resp, err := http.Get(ts.URL + "/memories?limit=10")
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != 200 {
		t.Fatalf("list status = %d", resp.StatusCode)
	}
	body := decode(t, resp)
	mems, ok := body["memories"].([]any)
	if !ok {
		t.Fatalf("list response missing memories array: %v", body)
	}
	if len(mems) != 2 {
		t.Fatalf("list len = %d, want 2", len(mems))
	}
	// Each entry should carry the memDict fields.
	first, _ := mems[0].(map[string]any)
	for _, key := range []string{"id", "text", "type", "importance", "created_at"} {
		if _, present := first[key]; !present {
			t.Errorf("list entry missing %q: %v", key, first)
		}
	}
}

func TestNeighborsEndpoint(t *testing.T) {
	ts, _ := newTestServer(t, "")
	a := rememberHTTP(t, ts.URL, `{"text":"memory a about deploys","type":"semantic"}`)
	b := rememberHTTP(t, ts.URL, `{"text":"memory b about releases","type":"semantic"}`)

	// Link a -> b so neighbors returns b.
	resp := postJSON(t, ts.URL, "/links", fmt.Sprintf(`{"src_id":%q,"dst_id":%q,"rel":"related"}`, a, b))
	if resp.StatusCode != 200 {
		t.Fatalf("link status = %d", resp.StatusCode)
	}
	if body := decode(t, resp); body["ok"] != true {
		t.Errorf("link body = %v", body)
	}

	resp, err := http.Get(ts.URL + "/memories/" + a + "/neighbors?direction=out")
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != 200 {
		t.Fatalf("neighbors status = %d", resp.StatusCode)
	}
	body := decode(t, resp)
	nb, ok := body["neighbors"].([]any)
	if !ok {
		t.Fatalf("neighbors missing array: %v", body)
	}
	if len(nb) != 1 {
		t.Fatalf("neighbors len = %d, want 1", len(nb))
	}
	entry, _ := nb[0].(map[string]any)
	if entry["id"] != b {
		t.Errorf("neighbor id = %v, want %s", entry["id"], b)
	}
	if entry["rel"] != "related" {
		t.Errorf("neighbor rel = %v, want related", entry["rel"])
	}
}

func TestLinkThenUnlink(t *testing.T) {
	ts, _ := newTestServer(t, "")
	a := rememberHTTP(t, ts.URL, `{"text":"linkable memory a","type":"semantic"}`)
	b := rememberHTTP(t, ts.URL, `{"text":"linkable memory b","type":"semantic"}`)

	if resp := postJSON(t, ts.URL, "/links", fmt.Sprintf(`{"src_id":%q,"dst_id":%q}`, a, b)); resp.StatusCode != 200 {
		t.Fatalf("link status = %d", resp.StatusCode)
	} else {
		resp.Body.Close()
	}

	resp := postJSON(t, ts.URL, "/unlink", fmt.Sprintf(`{"src_id":%q,"dst_id":%q}`, a, b))
	if resp.StatusCode != 200 {
		t.Fatalf("unlink status = %d", resp.StatusCode)
	}
	body := decode(t, resp)
	// removed defaults to related-relation removal; at least one link removed.
	removed, ok := body["removed"].(float64)
	if !ok {
		t.Fatalf("unlink missing removed count: %v", body)
	}
	if removed < 1 {
		t.Errorf("unlink removed = %v, want >= 1", removed)
	}
}

func TestLinkMissingFieldIs400(t *testing.T) {
	ts, _ := newTestServer(t, "")
	resp := postJSON(t, ts.URL, "/links", `{"src_id":"only-src"}`)
	defer resp.Body.Close()
	if resp.StatusCode != 400 {
		t.Errorf("link without dst_id = %d, want 400", resp.StatusCode)
	}
	if body := decode(t, resp); body["error"] == nil {
		t.Errorf("400 body should carry error: %v", body)
	}
}

// TestLinkUnknownSrcIs404 exercises the 404 path in the link handler: only a
// missing *source* is rejected, because linkRaw validates the source only.
func TestLinkUnknownSrcIs404(t *testing.T) {
	ts, _ := newTestServer(t, "")
	dst := rememberHTTP(t, ts.URL, `{"text":"real destination memory","type":"semantic"}`)
	resp := postJSON(t, ts.URL, "/links", fmt.Sprintf(`{"src_id":"does-not-exist","dst_id":%q}`, dst))
	defer resp.Body.Close()
	if resp.StatusCode != 404 {
		t.Errorf("link from unknown src = %d, want 404", resp.StatusCode)
	}
}

// A link to a non-existent destination is rejected: graph walkers skip
// targets that don't resolve, so a dangling edge would be stored but
// unreachable (matches Python, which raises for an unknown dst).
func TestLinkUnknownDstRejected(t *testing.T) {
	ts, _ := newTestServer(t, "")
	a := rememberHTTP(t, ts.URL, `{"text":"real memory here","type":"semantic"}`)
	resp := postJSON(t, ts.URL, "/links", fmt.Sprintf(`{"src_id":%q,"dst_id":"does-not-exist"}`, a))
	if resp.StatusCode != 404 {
		t.Errorf("link to unknown dst = %d, want 404", resp.StatusCode)
	}
	if body := decode(t, resp); body["error"] == nil {
		t.Errorf("link-to-unknown-dst body = %v, want an error message", body)
	}
}

func TestSupersedeEndpoint(t *testing.T) {
	ts, store := newTestServer(t, "")
	oldID := rememberHTTP(t, ts.URL, `{"text":"the old deployment procedure","type":"procedural"}`)
	newID := rememberHTTP(t, ts.URL, `{"text":"the new deployment procedure","type":"procedural"}`)

	resp := postJSON(t, ts.URL, "/supersede", fmt.Sprintf(`{"old_id":%q,"new_id":%q}`, oldID, newID))
	if resp.StatusCode != 200 {
		t.Fatalf("supersede status = %d", resp.StatusCode)
	}
	if body := decode(t, resp); body["ok"] != true {
		t.Errorf("supersede body = %v", body)
	}

	// The old memory should now be marked superseded in the store.
	mem, err := store.GetByID(context.Background(), oldID)
	if err != nil {
		t.Fatal(err)
	}
	if mem.SupersededBy != newID {
		t.Errorf("old memory SupersededBy = %q, want %q", mem.SupersededBy, newID)
	}
}

func TestSupersedeUnknownIdIs404(t *testing.T) {
	ts, _ := newTestServer(t, "")
	newID := rememberHTTP(t, ts.URL, `{"text":"only the new one exists","type":"semantic"}`)
	resp := postJSON(t, ts.URL, "/supersede", fmt.Sprintf(`{"old_id":"nope","new_id":%q}`, newID))
	defer resp.Body.Close()
	if resp.StatusCode != 404 {
		t.Errorf("supersede unknown old = %d, want 404", resp.StatusCode)
	}
}

func TestConflictsEndpoint(t *testing.T) {
	ts, _ := newTestServer(t, "")
	// Two near-identical memories should be flagged as a duplicate/contradiction.
	rememberHTTP(t, ts.URL, `{"text":"the server listens on port 8080","type":"semantic"}`)
	rememberHTTP(t, ts.URL, `{"text":"the server listens on port 8080","type":"semantic","on_conflict":"ignore"}`)

	resp := postJSON(t, ts.URL, "/conflicts", `{"threshold":0.5}`)
	if resp.StatusCode != 200 {
		t.Fatalf("conflicts status = %d", resp.StatusCode)
	}
	body := decode(t, resp)
	conflicts, ok := body["conflicts"].([]any)
	if !ok {
		t.Fatalf("conflicts missing array: %v", body)
	}
	if len(conflicts) == 0 {
		t.Fatalf("expected at least one conflict for identical memories, got %v", body)
	}
	entry, _ := conflicts[0].(map[string]any)
	for _, key := range []string{"kind", "reason", "similarity", "a", "b"} {
		if _, present := entry[key]; !present {
			t.Errorf("conflict entry missing %q: %v", key, entry)
		}
	}
	// The nested a/b carry a clipped text + type.
	aSide, _ := entry["a"].(map[string]any)
	if _, present := aSide["text"]; !present {
		t.Errorf("conflict a side missing text: %v", aSide)
	}
}

func TestRecallPackCompressed(t *testing.T) {
	ts, _ := newTestServer(t, "")
	// Store several near-identical memories. Each item line is ~30 tokens, so a
	// tight budget admits only one; the other four are dropped, cluster by
	// Jaccard (they differ by a single trailing word), and — because each
	// summary snippet is the tiny first sentence "warm" — the compressed group
	// line is small enough to fit in the remaining budget.
	shared := "warm. the deployment cache pipeline warms and refreshes on every production release rollout event"
	for _, w := range []string{"alpha", "beta", "gamma", "delta", "epsilon"} {
		rememberHTTP(t, ts.URL, fmt.Sprintf(`{"text":%q,"type":"procedural"}`, shared+" "+w))
	}

	// Budget 45: one ~30-token item packs, four are dropped and collapse into a
	// single ~12-token compressed group line (30 + 12 <= 45).
	resp := postJSON(t, ts.URL, "/recall_pack",
		`{"query":"deployment cache pipeline","token_budget":45,"compress":true,"compress_threshold":0.3,"compress_min_group":2}`)
	if resp.StatusCode != 200 {
		t.Fatalf("recall_pack status = %d", resp.StatusCode)
	}
	body := decode(t, resp)
	if _, ok := body["items"].([]any); !ok {
		t.Fatalf("recall_pack missing items: %v", body)
	}
	if body["truncated"] != true {
		t.Fatalf("expected truncated=true with a tight budget, got %v", body["truncated"])
	}
	groups, ok := body["compressed_groups"].([]any)
	if !ok {
		t.Fatalf("recall_pack missing compressed_groups: %v", body)
	}
	if len(groups) == 0 {
		t.Fatalf("expected at least one compressed group, got %v", body)
	}
	g0, _ := groups[0].(map[string]any)
	for _, key := range []string{"ids", "count", "text", "tokens"} {
		if _, present := g0[key]; !present {
			t.Errorf("compressed group missing %q: %v", key, g0)
		}
	}
	// A compressed group must contain at least compress_min_group members.
	if cnt, _ := g0["count"].(float64); cnt < 2 {
		t.Errorf("compressed group count = %v, want >= 2", g0["count"])
	}
	ids, _ := g0["ids"].([]any)
	if len(ids) < 2 {
		t.Errorf("compressed group ids = %v, want >= 2", ids)
	}
}

// TestRecallPackNoCompressWithoutFlag confirms compressed_groups is absent when
// compression is not requested, even under a tight budget.
func TestRecallPackNoCompressWithoutFlag(t *testing.T) {
	ts, _ := newTestServer(t, "")
	for i := 0; i < 5; i++ {
		rememberHTTP(t, ts.URL, fmt.Sprintf(`{"text":"deploy the service step %d to prod cluster now","type":"procedural"}`, i))
	}
	resp := postJSON(t, ts.URL, "/recall_pack", `{"query":"deploy service","token_budget":30}`)
	if resp.StatusCode != 200 {
		t.Fatalf("recall_pack status = %d", resp.StatusCode)
	}
	body := decode(t, resp)
	if _, present := body["compressed_groups"]; present {
		t.Errorf("compressed_groups should be absent without compress=true: %v", body["compressed_groups"])
	}
}

func TestPatchPinnedOnlyIsAValidEdit(t *testing.T) {
	ts, store := newTestServer(t, "")
	id := rememberHTTP(t, ts.URL, `{"text":"pin me later"}`)

	resp := doJSON(t, "PATCH", ts.URL, "/memories/"+id, `{"pinned":true}`)
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("pinned-only PATCH = %d, want 200", resp.StatusCode)
	}
	m, err := store.GetByID(context.Background(), id)
	if err != nil || !m.Pinned {
		t.Fatalf("pinned = %v err=%v, want true", m.Pinned, err)
	}
}

func TestPatchValidUntilOnlyRetiresAFact(t *testing.T) {
	ts, store := newTestServer(t, "")
	id := rememberHTTP(t, ts.URL, `{"text":"retire me via validity"}`)

	resp := doJSON(t, "PATCH", ts.URL, "/memories/"+id, `{"valid_until": 2.0}`)
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("valid_until-only PATCH = %d, want 200", resp.StatusCode)
	}
	m, err := store.GetByID(context.Background(), id)
	if err != nil || m.ValidUntil != 2.0 {
		t.Fatalf("valid_until = %v err=%v, want 2.0", m.ValidUntil, err)
	}
}

func TestPatchTrustOnly(t *testing.T) {
	ts, store := newTestServer(t, "")
	id := rememberHTTP(t, ts.URL, `{"text":"downgrade my origin"}`)

	resp := doJSON(t, "PATCH", ts.URL, "/memories/"+id, `{"trust":"untrusted"}`)
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("trust-only PATCH = %d, want 200", resp.StatusCode)
	}
	m, _ := store.GetByID(context.Background(), id)
	if m.Trust != memory.TrustLevel("untrusted") {
		t.Fatalf("trust = %q, want untrusted", m.Trust)
	}
}

func TestRecallHonoursOverfetch(t *testing.T) {
	ts, _ := newTestServer(t, "")
	rememberHTTP(t, ts.URL, `{"text":"overfetch subject"}`)

	// Not a behavioral probe (the effect only shows on filtered stores) —
	// but the knob must at least be accepted on both methods without error.
	resp := postJSON(t, ts.URL, "/recall", `{"query":"overfetch subject","overfetch":12}`)
	if resp.StatusCode != 200 {
		t.Fatalf("POST overfetch = %d, want 200", resp.StatusCode)
	}
	resp = doJSON(t, "GET", ts.URL, "/recall?query=overfetch+subject&overfetch=12", "")
	if resp.StatusCode != 200 {
		t.Fatalf("GET overfetch = %d, want 200", resp.StatusCode)
	}
	resp = doJSON(t, "GET", ts.URL, "/recall?query=x&overfetch=banana", "")
	if resp.StatusCode != 400 {
		t.Fatalf("GET bad overfetch = %d, want 400", resp.StatusCode)
	}
}

func TestGetRecallTouchFalse(t *testing.T) {
	ts, store := newTestServer(t, "")
	id := rememberHTTP(t, ts.URL, `{"text":"get untouchable"}`)

	resp := doJSON(t, "GET", ts.URL, "/recall?query=get+untouchable&k=1&touch=false", "")
	if resp.StatusCode != 200 {
		t.Fatalf("GET touch=false = %d, want 200", resp.StatusCode)
	}
	m, _ := store.GetByID(context.Background(), id)
	if m.AccessCount != 0 {
		t.Fatalf("access_count = %d, want 0 (read-only recall)", m.AccessCount)
	}
}

func TestImportConflictIs409(t *testing.T) {
	// Tokened: the archive routes refuse tokenless servers outright.
	ts, store := newTestServer(t, "s3cret")
	ctx := context.Background()
	m, _, _, _ := store.Remember(ctx, "collision subject", memory.RememberOpts{})
	dir := t.TempDir()
	out := dir + "/dump.ahkai"
	if _, err := store.Export(ctx, out, memory.ExportOpts{}); err != nil {
		t.Fatal(err)
	}

	resp := postJSONAuthed(t, ts.URL, "/import",
		fmt.Sprintf(`{"path":%q,"on_conflict":"error"}`, out), "s3cret")
	if resp.StatusCode != 409 {
		t.Fatalf("conflicting import = %d, want 409", resp.StatusCode)
	}
	resp.Body.Close()
	_ = m

	resp = postJSONAuthed(t, ts.URL, "/import", `{"path":"/nonexistent/x.ahkai"}`, "s3cret")
	if resp.StatusCode != 404 {
		t.Fatalf("missing archive = %d, want 404", resp.StatusCode)
	}
	resp.Body.Close()
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

func TestMetricsEndpoint(t *testing.T) {
	ts, _ := newTestServer(t, "")
	postObj(t, ts.URL+"/memories", map[string]any{"text": "metric me"})
	getObj(t, ts.URL+"/recall?query=metric&k=3")
	status, body := getObj(t, ts.URL+"/metrics")
	if status != 200 {
		t.Fatalf("metrics status = %d", status)
	}
	calls := body["calls"].(map[string]any)
	if calls["remember"].(float64) != 1 || calls["recall"].(float64) != 1 {
		t.Errorf("metrics calls = %v", calls)
	}
	if _, ok := body["uptime_seconds"]; !ok {
		t.Error("metrics missing uptime_seconds")
	}
}

func TestRememberTTLAndGetExposesExpiresAt(t *testing.T) {
	ts, _ := newTestServer(t, "")
	status, body := postObj(t, ts.URL+"/memories", map[string]any{"text": "ephemeral", "ttl_seconds": 500})
	if status != 201 {
		t.Fatalf("remember status = %d", status)
	}
	if body["expires_at"] == nil {
		t.Error("remember response missing expires_at")
	}
	id := body["id"].(string)
	_, got := getObj(t, ts.URL+"/memories/"+id)
	if got["expires_at"] == nil {
		t.Error("get response missing expires_at")
	}
}

func TestRecallHidesExpiredHTTP(t *testing.T) {
	ts, _ := newTestServer(t, "")
	postObj(t, ts.URL+"/memories", map[string]any{"text": "soon gone deployment note", "expires_at": 1.0})
	_, hidden := getObj(t, ts.URL+"/recall?query=deployment%20note&k=5")
	for _, h := range hidden["results"].([]any) {
		if h.(map[string]any)["text"] == "soon gone deployment note" {
			t.Error("expired memory should be hidden from recall")
		}
	}
	_, shown := getObj(t, ts.URL+"/recall?query=deployment%20note&k=5&include_expired=true")
	found := false
	for _, h := range shown["results"].([]any) {
		if h.(map[string]any)["text"] == "soon gone deployment note" {
			found = true
		}
	}
	if !found {
		t.Error("include_expired should surface the expired memory")
	}
}

func TestPurgeExpiredHTTP(t *testing.T) {
	ts, _ := newTestServer(t, "")
	postObj(t, ts.URL+"/memories", map[string]any{"text": "expired doc", "expires_at": 1.0})
	status, dry := postObj(t, ts.URL+"/purge_expired", map[string]any{"dry_run": true})
	if status != 200 || dry["purged"].(float64) != 1 || dry["dry_run"] != true {
		t.Fatalf("dry-run purge = %v (status %d)", dry, status)
	}
	_, done := postObj(t, ts.URL+"/purge_expired", map[string]any{})
	if done["purged"].(float64) != 1 {
		t.Errorf("purge removed %v, want 1", done["purged"])
	}
}

func TestRecallExplainHTTP(t *testing.T) {
	ts, _ := newTestServer(t, "")
	postObj(t, ts.URL+"/memories", map[string]any{"text": "explainable memory"})
	_, body := getObj(t, ts.URL+"/recall?query=explainable&k=1&mode=hybrid&explain=true")
	results := body["results"].([]any)
	if len(results) == 0 {
		t.Fatal("no results")
	}
	expl, ok := results[0].(map[string]any)["explain"].(map[string]any)
	if !ok {
		t.Fatalf("hit missing explain: %v", results[0])
	}
	if expl["mode"] != "hybrid" {
		t.Errorf("explain mode = %v, want hybrid", expl["mode"])
	}
}

func TestHistoryEndpoint(t *testing.T) {
	ts, _ := newTestServer(t, "")
	_, mem := postObj(t, ts.URL+"/memories", map[string]any{"text": "v1"})
	id := mem["id"].(string)
	req, _ := http.NewRequest(http.MethodPatch, ts.URL+"/memories/"+id,
		bytes.NewReader([]byte(`{"text":"v2"}`)))
	http.DefaultClient.Do(req)

	status, body := getObj(t, ts.URL+"/memories/"+id+"/history")
	if status != 200 {
		t.Fatalf("history status = %d", status)
	}
	hist := body["history"].([]any)
	ops := make([]string, len(hist))
	for i, e := range hist {
		ops[i] = e.(map[string]any)["op"].(string)
	}
	if len(ops) != 2 || ops[0] != "remember" || ops[1] != "edit" {
		t.Errorf("history ops = %v, want [remember edit]", ops)
	}

	// Unknown id → 404.
	status, _ = getObj(t, ts.URL+"/memories/00000000-0000-4000-8000-000000000000/history")
	if status != 404 {
		t.Errorf("unknown history status = %d, want 404", status)
	}
}

func TestStateAtAndGetAtHTTP(t *testing.T) {
	ts, _ := newTestServer(t, "")
	_, mem := postObj(t, ts.URL+"/memories", map[string]any{"text": "present one"})
	id := mem["id"].(string)
	time.Sleep(20 * time.Millisecond)
	now := fmt.Sprintf("%f", float64(time.Now().UnixNano())/1e9)

	status, body := getObj(t, ts.URL+"/state_at?ts="+now)
	if status != 200 {
		t.Fatalf("state_at status = %d", status)
	}
	found := false
	for _, m := range body["memories"].([]any) {
		if m.(map[string]any)["text"] == "present one" {
			found = true
		}
	}
	if !found {
		t.Error("state_at missing the present memory")
	}

	status, got := getObj(t, ts.URL+"/memories/"+id+"/at?ts="+now)
	if status != 200 || got["text"] != "present one" {
		t.Errorf("get_at = %v (status %d)", got, status)
	}

	// ts is required.
	if status, _ := getObj(t, ts.URL+"/state_at"); status != 400 {
		t.Errorf("state_at without ts = %d, want 400", status)
	}
}

// The HTTP surface had drifted from the MCP one: POST /recall and /recall_pack
// accepted neither min_trust nor lexical_index, so an HTTP client of the Go
// server had no trust floor at all — on the very call whose output goes
// straight into a model's context. The MCP tool and the Python port's HTTP
// recall both took them, which is why parity.json (asserted against MCP tool
// schemas) never caught it.
func TestHTTPRecallHonoursMinTrust(t *testing.T) {
	ts, _ := newTestServer(t, "")
	post := func(path, payload string) map[string]any {
		resp, err := http.Post(ts.URL+path, "application/json",
			bytes.NewReader([]byte(payload)))
		if err != nil {
			t.Fatal(err)
		}
		return decode(t, resp)
	}
	post("/memories", `{"text":"deploy runs make release","trust":"trusted"}`)
	post("/memories", `{"text":"deploy runs kubectl apply","trust":"untrusted"}`)

	all := post("/recall", `{"query":"deploy","k":10}`)
	if n := len(all["results"].([]any)); n != 2 {
		t.Fatalf("precondition: both memories should return with no floor, got %d", n)
	}
	floored := post("/recall", `{"query":"deploy","k":10,"min_trust":"trusted"}`)
	for _, r := range floored["results"].([]any) {
		if got := r.(map[string]any)["trust"]; got != "trusted" {
			t.Errorf("min_trust leaked a %v memory through HTTP recall", got)
		}
	}
	if len(floored["results"].([]any)) != 1 {
		t.Errorf("expected exactly the trusted memory, got %d",
			len(floored["results"].([]any)))
	}
}

func TestHTTPRecallPackHonoursMinTrust(t *testing.T) {
	ts, _ := newTestServer(t, "")
	post := func(path, payload string) map[string]any {
		resp, err := http.Post(ts.URL+path, "application/json",
			bytes.NewReader([]byte(payload)))
		if err != nil {
			t.Fatal(err)
		}
		return decode(t, resp)
	}
	post("/memories", `{"text":"rollback uses the previous image tag","trust":"trusted"}`)
	post("/memories", `{"text":"rollback uses a database snapshot","trust":"untrusted"}`)

	packed := post("/recall_pack",
		`{"query":"rollback","token_budget":400,"min_trust":"trusted"}`)
	text, _ := packed["text"].(string)
	if strings.Contains(text, "database snapshot") {
		t.Error("min_trust did not reach the packed block over HTTP")
	}
	if !strings.Contains(text, "previous image tag") {
		t.Error("the trusted memory should survive the floor")
	}
}

// Bi-temporal validity over HTTP: write an interval, then ask what was true.
func TestHTTPValidityAndAsOf(t *testing.T) {
	ts, _ := newTestServer(t, "")
	post := func(path, payload string) map[string]any {
		resp, err := http.Post(ts.URL+path, "application/json",
			bytes.NewReader([]byte(payload)))
		if err != nil {
			t.Fatal(err)
		}
		return decode(t, resp)
	}
	const jan, feb, mar = 1_700_000_000.0, 1_702_592_000.0, 1_705_184_000.0

	old := post("/memories", fmt.Sprintf(
		`{"text":"the office is in Berlin","valid_from":%f,"valid_until":%f}`, jan, mar))
	fresh := post("/memories", fmt.Sprintf(
		`{"text":"the office is in Munich","valid_from":%f}`, mar))

	if got := old["valid_until"]; got != mar {
		t.Errorf("valid_until did not round-trip: %v", got)
	}

	// Default recall is "valid now", so the closed interval is gone.
	now := post("/recall", `{"query":"where is the office","k":10}`)
	for _, r := range now["results"].([]any) {
		if r.(map[string]any)["id"] == old["id"] {
			t.Error("a memory past its valid_until must not appear in a default recall")
		}
	}
	// Asking about February brings it back, and excludes its successor.
	past := post("/recall", fmt.Sprintf(
		`{"query":"where is the office","k":10,"as_of":%f}`, feb))
	ids := map[any]bool{}
	for _, r := range past["results"].([]any) {
		ids[r.(map[string]any)["id"]] = true
	}
	if !ids[old["id"]] {
		t.Error("as_of must return the fact that held then")
	}
	if ids[fresh["id"]] {
		t.Error("as_of must not return a fact that was not yet true")
	}
}
