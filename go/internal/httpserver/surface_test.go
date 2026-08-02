package httpserver_test

import (
	"bytes"
	"encoding/json"
	"net/http"
	"path/filepath"
	"testing"

	"github.com/nexusriot/ai-houkai/internal/httpserver"
	"github.com/nexusriot/ai-houkai/internal/memory"
	"github.com/nexusriot/ai-houkai/internal/vector"

	"net/http/httptest"
)

// Store capabilities that previously had no HTTP route (A3): restore,
// subgraph, undo, nuke, journal, export, import.

// newJournaledServer builds a server whose store has the journal ON, so the
// undo and journal routes have a log to work against.
func newJournaledServer(t *testing.T) (*httptest.Server, *memory.MemoryStore, string) {
	t.Helper()
	dir := t.TempDir()
	backend, err := vector.NewChromem(filepath.Join(dir, "s"), "test", 16)
	if err != nil {
		t.Fatalf("NewChromem: %v", err)
	}
	t.Cleanup(func() { _ = backend.Close() })
	cfg := memory.DefaultStoreConfig(dir, "test")
	cfg.JournalEnabled = true
	cfg.JournalPath = filepath.Join(dir, "journal.log")
	store := memory.NewMemoryStore(backend, &stubEmbedder{dim: 16}, cfg)
	ts := httptest.NewServer(httpserver.New(store, dir, "test", "").Handler())
	t.Cleanup(ts.Close)
	return ts, store, dir
}

// post sends a JSON body and returns (status, decoded body).
func post(t *testing.T, url string, body any) (int, map[string]any) {
	t.Helper()
	raw, _ := json.Marshal(body)
	resp, err := http.Post(url, "application/json", bytes.NewReader(raw))
	if err != nil {
		t.Fatalf("POST %s: %v", url, err)
	}
	status := resp.StatusCode
	return status, decode(t, resp)
}

// patch sends a JSON PATCH and discards the body.
func patch(t *testing.T, url string, body any) int {
	t.Helper()
	raw, _ := json.Marshal(body)
	req, _ := http.NewRequest(http.MethodPatch, url, bytes.NewReader(raw))
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("PATCH %s: %v", url, err)
	}
	defer resp.Body.Close()
	return resp.StatusCode
}

func remember(t *testing.T, base, text string) string {
	t.Helper()
	status, body := post(t, base+"/memories", map[string]any{"text": text})
	if status != 201 && status != 200 {
		t.Fatalf("remember %q status = %d (%v)", text, status, body)
	}
	return body["id"].(string)
}

func TestRestoreRoute(t *testing.T) {
	ts, _ := newTestServer(t, "")
	old := remember(t, ts.URL, "http old policy")
	fresh := remember(t, ts.URL, "http new policy")
	post(t, ts.URL+"/supersede", map[string]any{"old_id": old, "new_id": fresh})

	status, body := post(t, ts.URL+"/restore", map[string]any{"memory_id": old})
	if status != 200 || body["restored"] != true {
		t.Fatalf("restore = %d %v", status, body)
	}
}

func TestRestoreRouteUnknownIDIs404(t *testing.T) {
	ts, _ := newTestServer(t, "")
	if status, _ := post(t, ts.URL+"/restore", map[string]any{"memory_id": "nope"}); status != 404 {
		t.Errorf("status = %d, want 404", status)
	}
}

func TestSubgraphRoute(t *testing.T) {
	ts, _ := newTestServer(t, "")
	a := remember(t, ts.URL, "http graph a")
	b := remember(t, ts.URL, "http graph b")
	post(t, ts.URL+"/links", map[string]any{"src_id": a, "dst_id": b, "rel": "refines"})

	status, body := post(t, ts.URL+"/subgraph",
		map[string]any{"memory_ids": []string{a}, "depth": 1})
	if status != 200 {
		t.Fatalf("status = %d (%v)", status, body)
	}
	if n := len(body["nodes"].([]any)); n != 2 {
		t.Errorf("nodes = %d, want 2", n)
	}
	edges := body["edges"].([]any)
	if len(edges) != 1 {
		t.Fatalf("edges = %d, want 1", len(edges))
	}
	e := edges[0].(map[string]any)
	if e["src"] != a || e["dst"] != b || e["rel"] != "refines" {
		t.Errorf("edge = %v", e)
	}
}

func TestSubgraphRouteRequiresIDs(t *testing.T) {
	ts, _ := newTestServer(t, "")
	status, body := post(t, ts.URL+"/subgraph", map[string]any{})
	if status != 400 {
		t.Fatalf("status = %d, want 400 (%v)", status, body)
	}
}

func TestUndoRoute(t *testing.T) {
	ts, _, _ := newJournaledServer(t)
	id := remember(t, ts.URL, "http undo subject")
	if s := patch(t, ts.URL+"/memories/"+id, map[string]any{"text": "http changed"}); s != 200 {
		t.Fatalf("patch status = %d", s)
	}

	status, body := post(t, ts.URL+"/undo", map[string]any{})
	if status != 200 || body["ok"] != true || body["op"] != "edit" {
		t.Fatalf("undo = %d %v", status, body)
	}
	got := decode(t, mustGet(t, ts.URL+"/memories/"+id))
	if got["text"] != "http undo subject" {
		t.Errorf("text after undo = %v", got["text"])
	}
}

func TestUndoRouteNothingIs404(t *testing.T) {
	ts, _, _ := newJournaledServer(t)
	if status, _ := post(t, ts.URL+"/undo", map[string]any{}); status != 404 {
		t.Errorf("status = %d, want 404", status)
	}
}

func TestNukeRouteIsGuarded(t *testing.T) {
	ts, _ := newTestServer(t, "")
	remember(t, ts.URL, "http nuke subject")

	status, body := post(t, ts.URL+"/nuke", map[string]any{})
	if status != 400 {
		t.Fatalf("unconfirmed nuke = %d %v, want 400", status, body)
	}
	status, body = post(t, ts.URL+"/nuke", map[string]any{"confirm": "DELETE ALL"})
	if status != 200 || body["deleted"].(float64) != 1 {
		t.Fatalf("confirmed nuke = %d %v", status, body)
	}
}

func TestJournalRoute(t *testing.T) {
	ts, _, _ := newJournaledServer(t)
	remember(t, ts.URL, "http journal one")
	remember(t, ts.URL, "http journal two")

	body := decode(t, mustGet(t, ts.URL+"/journal?n=1"))
	if body["count"].(float64) != 1 {
		t.Fatalf("count = %v, want 1", body["count"])
	}
	entry := body["entries"].([]any)[0].(map[string]any)
	if entry["op"] != "remember" {
		t.Errorf("op = %v", entry["op"])
	}

	filtered := decode(t, mustGet(t, ts.URL+"/journal?op=forget"))
	if filtered["count"].(float64) != 0 {
		t.Errorf("op=forget count = %v, want 0", filtered["count"])
	}
}

func TestExportImportRoundtripRoutes(t *testing.T) {
	ts, store, dir := newJournaledServer(t)
	remember(t, ts.URL, "http archive subject")
	archive := filepath.Join(dir, "dump.ahkai")

	status, body := post(t, ts.URL+"/export", map[string]any{"path": archive})
	if status != 200 || body["count"].(float64) != 1 {
		t.Fatalf("export = %d %v", status, body)
	}
	post(t, ts.URL+"/nuke", map[string]any{"confirm": "DELETE ALL"})

	status, body = post(t, ts.URL+"/import", map[string]any{"path": archive})
	if status != 200 || body["imported"].(float64) != 1 {
		t.Fatalf("import = %d %v", status, body)
	}
	n, err := store.Count(t.Context())
	if err != nil || n != 1 {
		t.Errorf("count after import = %d (%v), want 1", n, err)
	}
}

func TestImportMissingArchiveIs404(t *testing.T) {
	ts, _, dir := newJournaledServer(t)
	status, _ := post(t, ts.URL+"/import",
		map[string]any{"path": filepath.Join(dir, "absent.ahkai")})
	if status != 404 {
		t.Errorf("status = %d, want 404", status)
	}
}
