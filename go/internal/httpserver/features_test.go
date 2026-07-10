package httpserver_test

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"testing"
	"time"
)

func postObj(t *testing.T, url string, body any) (int, map[string]any) {
	t.Helper()
	data, _ := json.Marshal(body)
	resp, err := http.Post(url, "application/json", bytes.NewReader(data))
	if err != nil {
		t.Fatalf("POST %s: %v", url, err)
	}
	defer resp.Body.Close()
	var m map[string]any
	json.NewDecoder(resp.Body).Decode(&m)
	return resp.StatusCode, m
}

func getObj(t *testing.T, url string) (int, map[string]any) {
	t.Helper()
	resp, err := http.Get(url)
	if err != nil {
		t.Fatalf("GET %s: %v", url, err)
	}
	defer resp.Body.Close()
	var m map[string]any
	json.NewDecoder(resp.Body).Decode(&m)
	return resp.StatusCode, m
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
