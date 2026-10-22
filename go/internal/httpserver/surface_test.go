package httpserver_test

import (
	"bytes"
	"compress/gzip"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/nexusriot/ai-houkai/internal/httpserver"
	"github.com/nexusriot/ai-houkai/internal/memory"
	"github.com/nexusriot/ai-houkai/internal/vector"

	"net/http/httptest"
)

func TestGraphEligibilityThroughRecallAndPackRoutes(t *testing.T) {
	for _, endpoint := range []string{"/recall", "/recall_pack"} {
		for _, rerank := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/rerank=%v", endpoint, rerank), func(t *testing.T) {
				ts, _, _ := newJournaledServer(t)
				seed := remember(t, ts.URL, "trusted graph seed")
				for _, fields := range []map[string]any{
					{"text": "untrusted graph content", "trust": "untrusted"},
					{"text": "expired graph content", "expires_at": 1},
					{"text": "retired graph content", "valid_until": 1},
				} {
					status, child := post(t, ts.URL+"/memories", fields)
					if status != 201 {
						t.Fatalf("remember: %d %v", status, child)
					}
					status, result := post(t, ts.URL+"/links", map[string]any{"src_id": seed, "dst_id": child["id"], "rel": "related"})
					if status != 200 {
						t.Fatalf("link: %d %v", status, result)
					}
				}
				allowed := remember(t, ts.URL, "eligible graph content")
				post(t, ts.URL+"/links", map[string]any{"src_id": seed, "dst_id": allowed, "rel": "related"})
				status, result := post(t, ts.URL+endpoint, map[string]any{
					"query": "trusted graph seed", "k": 2, "max_items": 2,
					"mode": "semantic", "min_cosine": 0.999, "min_trust": "trusted", "token_budget": 1000,
					"expand": map[string]any{"rels": []string{"related"}, "cap": 1, "rerank": rerank},
				})
				if status != 200 {
					t.Fatalf("recall: %d %v", status, result)
				}
				key := "results"
				if endpoint == "/recall_pack" {
					key = "items"
				}
				rows := result[key].([]any)
				if len(rows) != 2 || rows[0].(map[string]any)["id"] != seed || rows[1].(map[string]any)["id"] != allowed {
					t.Fatalf("filters lost at HTTP boundary: %v", result)
				}
				if endpoint == "/recall_pack" {
					text := result["text"].(string)
					if strings.Contains(text, "untrusted graph") || strings.Contains(text, "expired graph") || strings.Contains(text, "retired graph") || !strings.Contains(text, "eligible graph content") {
						t.Fatalf("ineligible text entered packed context: %s", text)
					}
				}
			})
		}
	}
}

func TestJournalAndDefaultUndoRoutesStayInCollection(t *testing.T) {
	ts, store, dir := newJournaledServer(t)
	own := remember(t, ts.URL, "own original")
	if status := patch(t, ts.URL+"/memories/"+own, map[string]any{"text": "own edited"}); status != 200 {
		t.Fatal(status)
	}
	backend, err := vector.NewChromem(filepath.Join(dir, "s"), "foreign", 16)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { backend.Close() })
	cfg := memory.DefaultStoreConfig(dir, "foreign")
	cfg.JournalPath = store.Journal().Path
	foreign := memory.NewMemoryStore(backend, &stubEmbedder{dim: 16}, cfg)
	ctx := context.Background()
	other, _, _, err := foreign.Remember(ctx, "foreign private fact", memory.RememberOpts{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := foreign.Forget(ctx, other.ID); err != nil {
		t.Fatal(err)
	}
	status, journal := getObj(t, ts.URL+"/journal?n=10")
	if status != 200 || journal["count"] != float64(2) {
		t.Fatalf("journal: %d %v", status, journal)
	}
	for _, raw := range journal["entries"].([]any) {
		if raw.(map[string]any)["id"] != own {
			t.Fatal("journal leaked foreign event")
		}
	}
	status, state := getObj(t, fmt.Sprintf("%s/state_at?ts=%.6f", ts.URL, float64(time.Now().Unix())+1))
	if status != 200 || state["count"] != float64(1) || state["memories"].([]any)[0].(map[string]any)["id"] != own {
		t.Fatalf("state: %d %v", status, state)
	}
	if status, _ := getObj(t, ts.URL+"/memories/"+other.ID+"/history"); status != 404 {
		t.Fatal("foreign history was exposed")
	}
	if status, _ := post(t, ts.URL+"/undo", map[string]any{"memory_id": other.ID}); status != 404 {
		t.Fatal("foreign undo was selectable")
	}
	status, undone := post(t, ts.URL+"/undo", map[string]any{})
	if status != 200 || undone["ok"] != true || undone["id"] != own {
		t.Fatalf("undo: %d %v", status, undone)
	}
	if status, fetched := getObj(t, ts.URL+"/memories/"+own); status != 200 || fetched["text"] != "own original" {
		t.Fatalf("wrong undo result: %v", fetched)
	}
	if status, _ := getObj(t, ts.URL+"/memories/"+other.ID); status != 404 {
		t.Fatal("foreign row was resurrected")
	}
}

func writeRouteImport(t *testing.T, rows []map[string]any, dim int) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "route.ahkai")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	gz := gzip.NewWriter(f)
	enc := json.NewEncoder(gz)
	if err := enc.Encode(map[string]any{"format": "ai-houkai/export", "version": 1, "source": map[string]any{"embedding_dim": dim}}); err != nil {
		t.Fatal(err)
	}
	for _, row := range rows {
		if err := enc.Encode(row); err != nil {
			t.Fatal(err)
		}
	}
	if err := gz.Close(); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestOverwriteImportRoutePreservesMemoryOnFailure(t *testing.T) {
	ts, _, _ := newTokenedServer(t)
	status, original := postAuthed(t, ts.URL+"/memories", map[string]any{"text": "preserve this row", "tags": []string{"keep"}})
	if status != 201 {
		t.Fatal(original)
	}
	path := writeRouteImport(t, []map[string]any{{"id": original["id"], "text": "invalid replacement", "vector": []float32{1, 0}}}, 16)
	status, result := postAuthed(t, ts.URL+"/import", map[string]any{"path": path, "on_conflict": "overwrite"})
	if status != 200 || result["overwritten"] != float64(0) {
		t.Fatalf("import: %d %v", status, result)
	}
	errors := result["errors"].([]any)
	if len(errors) != 1 || errors[0].(map[string]any)["id"] != original["id"] {
		t.Fatalf("missing row error: %v", result)
	}
	status, recalled := postAuthed(t, ts.URL+"/recall", map[string]any{"query": "preserve this row", "k": 1})
	if status != 200 {
		t.Fatal(recalled)
	}
	rows := recalled["results"].([]any)
	if len(rows) != 1 || rows[0].(map[string]any)["id"] != original["id"] || rows[0].(map[string]any)["text"] != "preserve this row" {
		t.Fatalf("failed import lost original: %v", recalled)
	}
}

func TestImportConflictRouteLeavesNoPartialWrites(t *testing.T) {
	ts, store, _ := newTokenedServer(t)
	status, original := postAuthed(t, ts.URL+"/memories", map[string]any{"text": "existing fact"})
	if status != 201 {
		t.Fatal(original)
	}
	path := writeRouteImport(t, []map[string]any{
		{"id": "new-nonconflicting-id", "text": "must not be inserted"},
		{"id": original["id"], "text": "existing fact"},
	}, 0)
	before, err := store.Journal().Read(memory.ReadOpts{})
	if err != nil {
		t.Fatal(err)
	}
	status, result := postAuthed(t, ts.URL+"/import", map[string]any{"path": path, "on_conflict": "error"})
	if status != 409 {
		t.Fatalf("expected collision: %d %v", status, result)
	}
	if count, _ := store.Count(context.Background()); count != 1 {
		t.Fatalf("rejected import left %d rows", count)
	}
	after, err := store.Journal().Read(memory.ReadOpts{})
	if err != nil || !reflect.DeepEqual(before, after) {
		t.Fatalf("rejected import changed journal: %v", err)
	}
}

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
// newTokenedServer is newJournaledServer with a bearer token — the archive
// routes require one.
func newTokenedServer(t *testing.T) (*httptest.Server, *memory.MemoryStore, string) {
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
	ts := httptest.NewServer(httpserver.New(store, dir, "test", "s3cret").Handler())
	t.Cleanup(ts.Close)
	return ts, store, dir
}

// postAuthed is post with the bearer token attached.
func postAuthed(t *testing.T, url string, body any) (int, map[string]any) {
	t.Helper()
	raw, _ := json.Marshal(body)
	req, _ := http.NewRequest(http.MethodPost, url, bytes.NewReader(raw))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer s3cret")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("POST %s: %v", url, err)
	}
	return resp.StatusCode, decode(t, resp)
}

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
	// Archive routes require a configured token (see
	// TestArchiveRoutesRequireAToken for the refusal side).
	ts, store, dir := newTokenedServer(t)
	ctx := context.Background()
	if _, _, _, err := store.Remember(ctx, "http archive subject", memory.RememberOpts{}); err != nil {
		t.Fatal(err)
	}
	archive := filepath.Join(dir, "dump.ahkai")

	status, body := postAuthed(t, ts.URL+"/export", map[string]any{"path": archive})
	if status != 200 || body["count"].(float64) != 1 {
		t.Fatalf("export = %d %v", status, body)
	}
	postAuthed(t, ts.URL+"/nuke", map[string]any{"confirm": "DELETE ALL"})

	status, body = postAuthed(t, ts.URL+"/import", map[string]any{"path": archive})
	if status != 200 || body["imported"].(float64) != 1 {
		t.Fatalf("import = %d %v", status, body)
	}
	n, err := store.Count(t.Context())
	if err != nil || n != 1 {
		t.Errorf("count after import = %d (%v), want 1", n, err)
	}
}

func TestImportMissingArchiveIs404(t *testing.T) {
	ts, _, dir := newTokenedServer(t)
	status, _ := postAuthed(t, ts.URL+"/import",
		map[string]any{"path": filepath.Join(dir, "absent.ahkai")})
	if status != 404 {
		t.Errorf("status = %d, want 404", status)
	}
}
