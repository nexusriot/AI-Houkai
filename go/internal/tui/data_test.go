package tui

import (
	"context"
	"hash/fnv"
	"math"
	"path/filepath"
	"strings"
	"testing"

	"github.com/nexusriot/ai-houkai/internal/memory"
	"github.com/nexusriot/ai-houkai/internal/vector"
)

// stubEmbedder deterministically hashes a text into a unit vector.
type stubEmbedder struct{ dim int }

func (e *stubEmbedder) Dim() int { return e.dim }

func (e *stubEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, t := range texts {
		v := make([]float32, e.dim)
		h := fnv.New64a()
		h.Write([]byte(t))
		seed := h.Sum64()
		for j := 0; j < e.dim; j++ {
			seed = seed*6364136223846793005 + 1442695040888963407
			v[j] = float32(int64(seed>>33)%1000) / 1000.0
		}
		var norm float64
		for _, x := range v {
			norm += float64(x) * float64(x)
		}
		norm = math.Sqrt(norm)
		if norm > 0 {
			for j := range v {
				v[j] = float32(float64(v[j]) / norm)
			}
		}
		out[i] = v
	}
	return out, nil
}

func newTestStore(t *testing.T) *memory.MemoryStore {
	t.Helper()
	dir := t.TempDir()
	backend, err := vector.NewChromem(filepath.Join(dir, "s"), "test", 16)
	if err != nil {
		t.Fatalf("NewChromem: %v", err)
	}
	t.Cleanup(func() { _ = backend.Close() })
	cfg := memory.DefaultStoreConfig(dir, "test")
	return memory.NewMemoryStore(backend, &stubEmbedder{dim: 16}, cfg)
}

func TestSnippet(t *testing.T) {
	if got := snippet("short text", 60); got != "short text" {
		t.Errorf("got %q", got)
	}
	long := strings.Repeat("word ", 30)
	got := snippet(long, 60)
	if len(got) > 62 { // 59 bytes + multi-byte ellipsis
		t.Errorf("snippet too long: %d", len(got))
	}
	if !strings.HasSuffix(got, "…") {
		t.Errorf("missing ellipsis: %q", got)
	}
	// Newlines flattened.
	if got := snippet("line one\nline two", 60); strings.Contains(got, "\n") {
		t.Errorf("newline survived: %q", got)
	}
}

func TestRecentView(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	for _, text := range []string{"first memory text here", "second memory text here"} {
		if _, _, _, err := store.Remember(ctx, text, memory.RememberOpts{}); err != nil {
			t.Fatal(err)
		}
	}
	v, err := recentView(ctx, store, 200)
	if err != nil {
		t.Fatal(err)
	}
	if v.Kind != "recent" || len(v.Rows) != 2 || len(v.Memories) != 2 {
		t.Errorf("view = %+v", v)
	}
	if !strings.HasPrefix(v.Title, "Recent (2)") {
		t.Errorf("title = %q", v.Title)
	}
}

func TestSearchView(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	if _, _, _, err := store.Remember(ctx, "the cat sat on the mat", memory.RememberOpts{}); err != nil {
		t.Fatal(err)
	}
	v, err := searchView(ctx, store, "the cat sat on the mat", 50)
	if err != nil {
		t.Fatal(err)
	}
	if v.Kind != "search" || len(v.Rows) == 0 {
		t.Errorf("view = %+v", v)
	}
	if v.Rows[0].Extra == "" {
		t.Error("search rows should carry a score")
	}
}

func TestNeighborsViewAndNavigator(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	a, _, _, _ := store.Remember(ctx, "memory a is about deployments", memory.RememberOpts{})
	b, _, _, _ := store.Remember(ctx, "memory b is about releases", memory.RememberOpts{})
	if err := store.Link(ctx, a.ID, b.ID, "related"); err != nil {
		t.Fatal(err)
	}

	nav := NewNavigator(store)
	if _, err := nav.OpenRecent(ctx, 200); err != nil {
		t.Fatal(err)
	}
	if nav.Breadcrumb() != nav.Current().Title {
		t.Errorf("breadcrumb = %q", nav.Breadcrumb())
	}

	v, err := nav.OpenNeighbors(ctx, a)
	if err != nil {
		t.Fatal(err)
	}
	if v.Kind != "neighbors" || len(v.Rows) != 1 {
		t.Errorf("view = %+v", v)
	}
	if v.Rows[0].Extra != "related" {
		t.Errorf("rel = %q", v.Rows[0].Extra)
	}
	if !strings.Contains(nav.Breadcrumb(), " > ") {
		t.Errorf("breadcrumb = %q", nav.Breadcrumb())
	}

	back := nav.Back()
	if back.Kind != "recent" {
		t.Errorf("back view kind = %q", back.Kind)
	}
	// Back at the root stays at the root.
	if nav.Back().Kind != "recent" {
		t.Error("back at root should stay at root")
	}
}

func TestDetailText(t *testing.T) {
	m := memory.Memory{
		ID:         "0123456789abcdef",
		Text:       "The full memory body.",
		Type:       memory.Procedural,
		Tags:       []string{"ops", "deploy"},
		Importance: 0.8,
		Source:     "cli",
		Links:      []memory.Link{{To: "fedcba9876543210", Rel: "related"}},
	}
	d := DetailText(m)
	for _, want := range []string{"01234567", "procedural", "#ops", "#deploy", "source: cli",
		"The full memory body.", "--related--> fedcba98"} {
		if !strings.Contains(d, want) {
			t.Errorf("detail missing %q in:\n%s", want, d)
		}
	}
	m.SupersededBy = "ffff0000ffff0000"
	if !strings.Contains(DetailText(m), "superseded by ffff0000") {
		t.Error("detail missing superseded marker")
	}
}
