package httpserver_test

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
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
