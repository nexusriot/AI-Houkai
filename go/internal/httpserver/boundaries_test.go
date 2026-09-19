package httpserver_test

import (
	"net/http"
	"strings"
	"testing"
)

// The HTTP surface is where the two ports' numeric-knob conventions collided:
// a client CAN send an explicit 0, and Go read it as "unset" while Python read
// it as zero. `?limit=0` handed back the whole store, `?depth=0` walked a hop,
// `token_budget: 0` packed a full block, and `?limit=-1` — which Python
// answers 400 — also returned everything.

func seedThree(t *testing.T, url string) []string {
	t.Helper()
	ids := make([]string, 0, 3)
	for _, text := range []string{
		"the deploy script lives at ops/deploy.sh",
		"the api is versioned at /api/v1",
		"vlad prefers tabs over spaces",
	} {
		ids = append(ids, rememberHTTP(t, url, `{"text":"`+text+`"}`))
	}
	return ids
}

func TestListLimitZeroReturnsNothing(t *testing.T) {
	ts, _ := newTestServer(t, "")
	seedThree(t, ts.URL)

	resp, err := http.Get(ts.URL + "/memories?limit=0")
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != 200 {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	mems, _ := decode(t, resp)["memories"].([]any)
	if len(mems) != 0 {
		t.Errorf("?limit=0 returned %d memories, want 0 — a paging client "+
			"whose remaining count reached zero must not be handed the store",
			len(mems))
	}
}

func TestListNegativeLimitIsClientError(t *testing.T) {
	ts, _ := newTestServer(t, "")
	seedThree(t, ts.URL)

	resp, err := http.Get(ts.URL + "/memories?limit=-1")
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != 400 {
		t.Fatalf("status = %d, want 400 (matching the Python port)",
			resp.StatusCode)
	}
	if msg, _ := decode(t, resp)["error"].(string); !strings.Contains(msg, "limit") {
		t.Errorf("error = %q, should name the offending parameter", msg)
	}
}

func TestListLimitStillBounds(t *testing.T) {
	ts, _ := newTestServer(t, "")
	seedThree(t, ts.URL)

	for _, tc := range []struct{ limit, want int }{{1, 1}, {2, 2}, {99, 3}} {
		resp, err := http.Get(ts.URL + "/memories?limit=" +
			strings.TrimSpace(itoa(tc.limit)))
		if err != nil {
			t.Fatal(err)
		}
		mems, _ := decode(t, resp)["memories"].([]any)
		if len(mems) != tc.want {
			t.Errorf("?limit=%d returned %d, want %d", tc.limit, len(mems), tc.want)
		}
	}
}

func TestNeighborsDepthZeroReachesNothing(t *testing.T) {
	ts, _ := newTestServer(t, "")
	ids := seedThree(t, ts.URL)
	resp := postJSON(t, ts.URL, "/links",
		`{"src_id":"`+ids[0]+`","dst_id":"`+ids[1]+`","rel":"related"}`)
	if resp.StatusCode >= 300 {
		t.Fatalf("link status = %d", resp.StatusCode)
	}

	for _, depth := range []string{"0", "-1"} {
		resp, err := http.Get(ts.URL + "/memories/" + ids[0] + "/neighbors?depth=" + depth)
		if err != nil {
			t.Fatal(err)
		}
		if resp.StatusCode != 200 {
			t.Fatalf("depth=%s status = %d", depth, resp.StatusCode)
		}
		got, _ := decode(t, resp)["neighbors"].([]any)
		if len(got) != 0 {
			t.Errorf("?depth=%s returned %d neighbours, want 0", depth, len(got))
		}
	}

	resp2, err := http.Get(ts.URL + "/memories/" + ids[0] + "/neighbors?depth=1")
	if err != nil {
		t.Fatal(err)
	}
	got, _ := decode(t, resp2)["neighbors"].([]any)
	if len(got) != 1 {
		t.Errorf("?depth=1 returned %d neighbours, want 1", len(got))
	}
}

func TestPackZeroBudgetPacksNothing(t *testing.T) {
	ts, _ := newTestServer(t, "")
	seedThree(t, ts.URL)

	for _, route := range []struct{ path, payload string }{
		{"/recall_pack", `{"query":"deploy","token_budget":0}`},
		{"/auto_context", `{"task":"deploy","token_budget":0}`},
	} {
		resp := postJSON(t, ts.URL, route.path, route.payload)
		if resp.StatusCode != 200 {
			t.Fatalf("%s status = %d", route.path, resp.StatusCode)
		}
		body := decode(t, resp)
		items, _ := body["items"].([]any)
		if len(items) != 0 {
			t.Errorf("%s with token_budget=0 packed %d items, want none — a "+
				"caller with no room left sends exactly this", route.path,
				len(items))
		}
		if text, _ := body["text"].(string); strings.Contains(text, "deploy.sh") {
			t.Errorf("%s with token_budget=0 still rendered memory lines",
				route.path)
		}
	}
}

func TestPackDefaultBudgetStillPacks(t *testing.T) {
	// The other half of the contract: omitting the budget must not now mean
	// "pack nothing".
	ts, _ := newTestServer(t, "")
	seedThree(t, ts.URL)

	resp := postJSON(t, ts.URL, "/recall_pack", `{"query":"deploy"}`)
	if resp.StatusCode != 200 {
		t.Fatalf("status = %d", resp.StatusCode)
	}
	items, _ := decode(t, resp)["items"].([]any)
	if len(items) == 0 {
		t.Error("an omitted token_budget packed nothing")
	}
}

func TestCallerErrorsAreFourHundred(t *testing.T) {
	// A ValidationError from the store, not a plain error: these used to come
	// back as 500, telling the client to retry something that can never work,
	// with a Go type name in the message.
	ts, _ := newTestServer(t, "")
	ids := seedThree(t, ts.URL)

	for _, tc := range []struct{ name, path, payload string }{
		{"self-merge", "/merge",
			`{"target_id":"` + ids[0] + `","other_id":"` + ids[0] + `"}`},
		{"comma in a renamed tag", "/tags/rename", `{"old":"x","new":"a,b"}`},
		{"empty renamed tag", "/tags/rename", `{"old":"x","new":" "}`},
		{"comma in a merged tag", "/tags/merge",
			`{"sources":["x"],"into":"a,b"}`},
	} {
		resp := postJSON(t, ts.URL, tc.path, tc.payload)
		if resp.StatusCode != 400 {
			t.Errorf("%s: status = %d, want 400", tc.name, resp.StatusCode)
		}
		if msg, _ := decode(t, resp)["error"].(string); strings.Contains(msg, "errorString") {
			t.Errorf("%s: message leaks a Go type: %q", tc.name, msg)
		}
	}
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var b []byte
	for n > 0 {
		b = append([]byte{byte('0' + n%10)}, b...)
		n /= 10
	}
	if neg {
		b = append([]byte{'-'}, b...)
	}
	return string(b)
}

func TestUnknownIDIsNotFoundOnEverySubRoute(t *testing.T) {
	// /neighbors was the one /memories/{id}/… route that answered 200 for an
	// id that does not exist, so a client could not tell "this memory has no
	// links" from "there is no such memory".
	ts, _ := newTestServer(t, "")
	seedThree(t, ts.URL)

	for _, suffix := range []string{"", "/neighbors", "/history", "/versions", "/at?ts=1"} {
		resp, err := http.Get(ts.URL + "/memories/nosuchid" + suffix)
		if err != nil {
			t.Fatal(err)
		}
		if resp.StatusCode != 404 {
			t.Errorf("/memories/{unknown}%s = %d, want 404", suffix, resp.StatusCode)
		}
		resp.Body.Close()
	}
}

func TestNeighborsOfARealMemoryWithNoLinksIsEmpty(t *testing.T) {
	// The other half: a memory that exists but has no edges is 200 + [], not
	// a 404. The distinction is the whole point of the check above.
	ts, _ := newTestServer(t, "")
	ids := seedThree(t, ts.URL)

	resp, err := http.Get(ts.URL + "/memories/" + ids[0] + "/neighbors")
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != 200 {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	got, _ := decode(t, resp)["neighbors"].([]any)
	if len(got) != 0 {
		t.Errorf("neighbours = %d, want 0", len(got))
	}
}

func TestIntegerParamsRejectNonIntegers(t *testing.T) {
	// JSON has one number type, so an integer parameter arrives as a float
	// and int(...) would truncate. `{"k": 2.7}` silently became k=2, hiding
	// the caller's bug, where the Python port answered 400.
	ts, _ := newTestServer(t, "")
	seedThree(t, ts.URL)

	for _, tc := range []struct{ name, path, payload string }{
		{"recall k", "/recall", `{"query":"deploy","k":2.7}`},
		{"recall overfetch", "/recall", `{"query":"deploy","overfetch":1.5}`},
		{"pack token_budget", "/recall_pack", `{"query":"deploy","token_budget":10.5}`},
		{"auto_context max_phrases", "/auto_context", `{"task":"deploy","max_phrases":1.5}`},
		{"subgraph depth", "/subgraph", `{"memory_ids":[],"depth":1.5}`},
		{"remember polarity", "/memories", `{"text":"polarity probe","polarity":0.5}`},
		{"batch batch_size", "/memories/batch",
			`{"items":[{"text":"batch size probe"}],"batch_size":1.5}`},
		{"k as a bool", "/recall", `{"query":"deploy","k":true}`},
	} {
		resp := postJSON(t, ts.URL, tc.path, tc.payload)
		if resp.StatusCode != 400 {
			t.Errorf("%s: status = %d, want 400 for a non-integer",
				tc.name, resp.StatusCode)
		}
		resp.Body.Close()
	}
}

func TestIntegerParamsStillAcceptWholeNumbers(t *testing.T) {
	// A whole number arriving as a JSON float (2.0) is still an integer, and
	// a numeric string is still coerced, exactly as in the Python port.
	ts, _ := newTestServer(t, "")
	seedThree(t, ts.URL)

	for _, payload := range []string{
		`{"query":"deploy","k":2}`,
		`{"query":"deploy","k":2.0}`,
		`{"query":"deploy","k":"2"}`,
	} {
		resp := postJSON(t, ts.URL, "/recall", payload)
		if resp.StatusCode != 200 {
			t.Errorf("%s: status = %d, want 200", payload, resp.StatusCode)
			resp.Body.Close()
			continue
		}
		results, _ := decode(t, resp)["results"].([]any)
		if len(results) != 2 {
			t.Errorf("%s returned %d results, want 2", payload, len(results))
		}
	}
}

func TestEmptyBatchIsNotCreated(t *testing.T) {
	// 201 Created for zero creations is a lie, and it is the same rule the
	// non-empty path already applies by counting new ids.
	ts, _ := newTestServer(t, "")

	resp := postJSON(t, ts.URL, "/memories/batch", `{"items":[]}`)
	if resp.StatusCode != 200 {
		t.Errorf("empty batch status = %d, want 200", resp.StatusCode)
	}
	if stored, _ := decode(t, resp)["stored"].(float64); stored != 0 {
		t.Errorf("stored = %v, want 0", stored)
	}

	resp = postJSON(t, ts.URL, "/memories/batch",
		`{"items":[{"text":"a real batched memory"}]}`)
	if resp.StatusCode != 201 {
		t.Errorf("non-empty batch status = %d, want 201", resp.StatusCode)
	}
	resp.Body.Close()
}
