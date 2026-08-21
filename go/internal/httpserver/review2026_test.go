package httpserver_test

// Regression tests for the 2026-08 functional-bug review on the HTTP surface:
// pinned/trust/valid-time-only PATCH edits, the overfetch and touch recall
// knobs, strict body coercion (a mistyped dry_run ran a REAL purge), and the
// import conflict status mapping.

import (
	"context"
	"fmt"
	"testing"

	"github.com/nexusriot/ai-houkai/internal/memory"
)

func TestPatchPinnedOnlyIsAValidEdit(t *testing.T) {
	ts, store := newTestServer(t, "")
	id := rememberHTTP(t, ts.URL, `{"text":"pin me later"}`)

	resp := doJSON(t, "PATCH", ts.URL, "/memories/"+id, `{"pinned":true}`)
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

func TestMistypedDryRunMustNotPurge(t *testing.T) {
	ts, store := newTestServer(t, "")
	ctx := context.Background()
	exp := 1.0
	m, _, _, err := store.Remember(ctx, "expired but recoverable",
		memory.RememberOpts{ExpiresAt: &exp})
	if err != nil {
		t.Fatal(err)
	}

	// Pre-fix, bodyBool silently coerced the string to false → a REAL purge.
	resp := postJSON(t, ts.URL, "/purge_expired", `{"dry_run":"true"}`)
	if resp.StatusCode != 200 {
		t.Fatalf("purge_expired = %d, want 200", resp.StatusCode)
	}
	if _, err := store.GetByID(ctx, m.ID); err != nil {
		t.Fatal("dry_run:\"true\" performed a real purge — memory gone")
	}

	resp = postJSON(t, ts.URL, "/purge_expired", `{"dry_run":[1]}`)
	if resp.StatusCode != 400 {
		t.Fatalf("garbage dry_run = %d, want 400", resp.StatusCode)
	}
}

func TestMistypedNumbersAre400(t *testing.T) {
	ts, _ := newTestServer(t, "")
	cases := []struct{ name, path, payload string }{
		{"string k garbage", "/recall", `{"query":"x","k":"lots"}`},
		{"bool importance", "/memories", `{"text":"x","importance":true}`},
		{"garbage importance", "/memories", `{"text":"x","importance":"very"}`},
	}
	for _, c := range cases {
		resp := postJSON(t, ts.URL, c.path, c.payload)
		if resp.StatusCode != 400 {
			t.Errorf("%s: status = %d, want 400", c.name, resp.StatusCode)
		}
	}
	// Numeric strings coerce instead of silently dropping (mirrors Python).
	resp := postJSON(t, ts.URL, "/memories", `{"text":"coerced importance","importance":"0.9"}`)
	if resp.StatusCode != 201 {
		t.Fatalf("numeric-string importance = %d, want 201", resp.StatusCode)
	}
	if imp := decode(t, resp)["importance"].(float64); imp != 0.9 {
		t.Errorf("importance = %v, want 0.9 (coerced, not defaulted)", imp)
	}
}

func TestLoneStringTagsWrapsIntoList(t *testing.T) {
	ts, store := newTestServer(t, "")
	resp := postJSON(t, ts.URL, "/memories", `{"text":"tagged once","tags":"prod"}`)
	if resp.StatusCode != 201 {
		t.Fatalf("lone-string tags = %d, want 201", resp.StatusCode)
	}
	id := decode(t, resp)["id"].(string)
	m, _ := store.GetByID(context.Background(), id)
	if len(m.Tags) != 1 || m.Tags[0] != "prod" {
		t.Fatalf("tags = %v, want [prod]", m.Tags)
	}

	resp = postJSON(t, ts.URL, "/memories", `{"text":"bad tags","tags":{"a":1}}`)
	if resp.StatusCode != 400 {
		t.Fatalf("garbage tags = %d, want 400", resp.StatusCode)
	}
}

func TestImportConflictIs409(t *testing.T) {
	ts, store := newTestServer(t, "")
	ctx := context.Background()
	m, _, _, _ := store.Remember(ctx, "collision subject", memory.RememberOpts{})
	dir := t.TempDir()
	out := dir + "/dump.ahkai"
	if _, err := store.Export(ctx, out, memory.ExportOpts{}); err != nil {
		t.Fatal(err)
	}

	resp := postJSON(t, ts.URL, "/import",
		fmt.Sprintf(`{"path":%q,"on_conflict":"error"}`, out))
	if resp.StatusCode != 409 {
		t.Fatalf("conflicting import = %d, want 409", resp.StatusCode)
	}
	_ = m

	resp = postJSON(t, ts.URL, "/import", `{"path":"/nonexistent/x.ahkai"}`)
	if resp.StatusCode != 404 {
		t.Fatalf("missing archive = %d, want 404", resp.StatusCode)
	}
}
