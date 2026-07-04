package httpserver_test

import (
	"bytes"
	"context"
	"fmt"
	"net/http"
	"testing"
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
