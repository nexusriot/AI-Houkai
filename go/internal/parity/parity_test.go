// Package parity asserts the Go remote surface against the repo-root
// parity.json, the single source of truth shared with the Python port
// (tests/test_parity.py checks the same file).
//
// Neither CI job needs the other toolchain: each port validates itself against
// the manifest, so a tool or route added to only one side fails that side's
// build. This is the guard for the class of drift that let the Python MCP
// recall fall five ranking knobs behind the Go one unnoticed.
//
// It lives in its own package because it imports both the MCP and HTTP servers
// and belongs to neither.
package parity

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/nexusriot/ai-houkai/internal/httpserver"
	"github.com/nexusriot/ai-houkai/internal/mcpserver"
	"github.com/nexusriot/ai-houkai/internal/memory"
	"github.com/nexusriot/ai-houkai/internal/vector"
)

type manifest struct {
	MCPTools          []string `json:"mcp_tools"`
	HTTPRoutes        []string `json:"http_routes"`
	RecallKnobs       []string `json:"recall_knobs"`
	RecallExpandKnobs []string `json:"recall_expand_knobs"`
}

func loadManifest(t *testing.T) manifest {
	t.Helper()
	// go/internal/parity → repo root is three levels up.
	path := filepath.Join("..", "..", "..", "parity.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	var m manifest
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatalf("parse parity.json: %v", err)
	}
	return m
}

// zeroEmbedder is the cheapest possible embedder: the parity checks only need
// a server to introspect, never a meaningful vector.
type zeroEmbedder struct{}

func (zeroEmbedder) Dim() int { return 8 }

func (zeroEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i := range texts {
		v := make([]float32, 8)
		v[0] = 1
		out[i] = v
	}
	return out, nil
}

func newStore(t *testing.T) *memory.MemoryStore {
	t.Helper()
	dir := t.TempDir()
	backend, err := vector.NewChromem(filepath.Join(dir, "s"), "parity", 8)
	if err != nil {
		t.Fatalf("NewChromem: %v", err)
	}
	t.Cleanup(func() { _ = backend.Close() })
	cfg := memory.DefaultStoreConfig(dir, "parity")
	cfg.JournalEnabled = false
	return memory.NewMemoryStore(backend, zeroEmbedder{}, cfg)
}

func goMCPTools(t *testing.T) []string {
	t.Helper()
	s := mcpserver.New(newStore(t), "/p", "parity")
	req, _ := json.Marshal(map[string]any{
		"jsonrpc": "2.0", "id": 1, "method": "tools/list",
	})
	rb, _ := json.Marshal(s.HandleMessage(context.Background(), req))
	var env struct {
		Result struct {
			Tools []struct {
				Name        string         `json:"name"`
				InputSchema map[string]any `json:"inputSchema"`
			} `json:"tools"`
		} `json:"result"`
	}
	if err := json.Unmarshal(rb, &env); err != nil {
		t.Fatalf("parse tools/list: %v", err)
	}
	names := make([]string, 0, len(env.Result.Tools))
	for _, tool := range env.Result.Tools {
		names = append(names, tool.Name)
	}
	sort.Strings(names)
	return names
}

func goToolSchema(t *testing.T, want string) map[string]any {
	t.Helper()
	s := mcpserver.New(newStore(t), "/p", "parity")
	req, _ := json.Marshal(map[string]any{
		"jsonrpc": "2.0", "id": 1, "method": "tools/list",
	})
	rb, _ := json.Marshal(s.HandleMessage(context.Background(), req))
	var env struct {
		Result struct {
			Tools []struct {
				Name        string         `json:"name"`
				InputSchema map[string]any `json:"inputSchema"`
			} `json:"tools"`
		} `json:"result"`
	}
	if err := json.Unmarshal(rb, &env); err != nil {
		t.Fatalf("parse tools/list: %v", err)
	}
	for _, tool := range env.Result.Tools {
		if tool.Name == want {
			props, _ := tool.InputSchema["properties"].(map[string]any)
			return props
		}
	}
	t.Fatalf("tool %q not registered", want)
	return nil
}

func TestMCPToolsMatchManifest(t *testing.T) {
	m := loadManifest(t)
	want := append([]string(nil), m.MCPTools...)
	sort.Strings(want)
	got := goMCPTools(t)

	if len(got) != len(want) {
		t.Fatalf("tool count = %d, manifest has %d\n got: %v\nwant: %v",
			len(got), len(want), got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("tool[%d] = %q, want %q", i, got[i], want[i])
		}
	}
}

func TestRecallExposesEveryManifestKnob(t *testing.T) {
	m := loadManifest(t)
	props := goToolSchema(t, "recall")
	for _, knob := range append(append([]string(nil), m.RecallKnobs...),
		m.RecallExpandKnobs...) {
		if _, ok := props[knob]; !ok {
			t.Errorf("recall is missing knob %q", knob)
		}
	}
}

func TestRecallPackExposesRankingKnobs(t *testing.T) {
	m := loadManifest(t)
	props := goToolSchema(t, "recall_pack")
	// recall_pack packs rather than paginates, so k/overfetch/explain and the
	// expiry toggle do not apply; everything that shapes ranking must be there.
	skip := map[string]bool{
		"k": true, "overfetch": true, "explain": true, "include_expired": true,
	}
	for _, knob := range append(append([]string(nil), m.RecallKnobs...),
		m.RecallExpandKnobs...) {
		if skip[knob] {
			continue
		}
		if _, ok := props[knob]; !ok {
			t.Errorf("recall_pack is missing knob %q", knob)
		}
	}
}

func TestHTTPRoutesMatchManifest(t *testing.T) {
	m := loadManifest(t)
	handler := httpserver.New(newStore(t), "/p", "parity", "").Handler()
	ts := httptest.NewServer(handler)
	defer ts.Close()

	// Go's ServeMux exposes no route listing, so probe each manifest route.
	// An unregistered path falls through to the router's own 404, whose body
	// the middleware wraps verbatim as {"error": "404 page not found"} — a
	// string no handler of ours ever produces. A registered route answers
	// anything else (commonly 400 for the deliberately empty body).
	for _, entry := range m.HTTPRoutes {
		method, path := splitRoute(entry)
		probe := ts.URL + injectID(path)
		req, err := http.NewRequest(method, probe, http.NoBody)
		if err != nil {
			t.Fatalf("build %s: %v", entry, err)
		}
		req.Header.Set("Content-Type", "application/json")
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatalf("%s: %v", entry, err)
		}
		var body map[string]any
		_ = json.NewDecoder(resp.Body).Decode(&body)
		resp.Body.Close()

		msg, _ := body["error"].(string)
		if resp.StatusCode == 404 && strings.Contains(msg, "page not found") {
			t.Errorf("route %q is in parity.json but not registered", entry)
		}
		if resp.StatusCode == 405 {
			t.Errorf("route %q registered under a different method", entry)
		}
	}
}

func splitRoute(entry string) (method, path string) {
	for i := 0; i < len(entry); i++ {
		if entry[i] == ' ' {
			return entry[:i], entry[i+1:]
		}
	}
	return entry, ""
}

// injectID substitutes a concrete id for the {id} wildcard so the request
// actually reaches the handler rather than the router's pattern matcher.
func injectID(path string) string {
	out := ""
	for i := 0; i < len(path); i++ {
		if path[i] == '{' {
			j := i
			for j < len(path) && path[j] != '}' {
				j++
			}
			out += "parity-probe-id"
			i = j
			continue
		}
		out += string(path[i])
	}
	return out
}
