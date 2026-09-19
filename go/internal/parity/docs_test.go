package parity

import (
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

// TestDesignTestInventoryIsCurrent holds go/DESIGN.md's "N test files / M test
// functions across P packages" line to what is actually on disk.
//
// It is here rather than in a package of its own because this is the same job
// parity_test.go already does for the remote surface: a number written by hand
// in a document, checked against the thing it describes. That line had drifted
// badly — it claimed ~20 files and 100+ functions across 9 packages when the
// real figures were roughly triple in every column — precisely because nothing
// re-derived it.
func TestDesignTestInventoryIsCurrent(t *testing.T) {
	root := filepath.Join("..", "..")
	design, err := os.ReadFile(filepath.Join(root, "DESIGN.md"))
	if err != nil {
		t.Fatalf("read DESIGN.md: %v", err)
	}

	claim := regexp.MustCompile(
		`(\d+) test files / (\d+) test functions across (\d+) packages`)
	m := claim.FindStringSubmatch(string(design))
	if m == nil {
		t.Fatal("go/DESIGN.md no longer states its test inventory " +
			`("N test files / M test functions across P packages")`)
	}

	files, err := filepath.Glob(filepath.Join(root, "internal", "*", "*_test.go"))
	if err != nil {
		t.Fatalf("glob: %v", err)
	}
	pkgs := map[string]bool{}
	funcs := 0
	for _, f := range files {
		pkgs[filepath.Base(filepath.Dir(f))] = true
		src, err := os.ReadFile(f)
		if err != nil {
			t.Fatalf("read %s: %v", f, err)
		}
		for _, line := range strings.Split(string(src), "\n") {
			if strings.HasPrefix(line, "func Test") {
				funcs++
			}
		}
	}

	for _, want := range []struct {
		label  string
		stated string
		actual int
	}{
		{"test files", m[1], len(files)},
		{"test functions", m[2], funcs},
		{"packages", m[3], len(pkgs)},
	} {
		if n, _ := strconv.Atoi(want.stated); n != want.actual {
			t.Errorf("go/DESIGN.md claims %s %s, actual %d — update the "+
				"\"Testing strategy\" line", want.stated, want.label, want.actual)
		}
	}
}

// TestDocumentedSurfaceSizesMatchParity holds the Go docs' prose counts —
// "41 MCP tools", "41 HTTP routes" — to the lists in parity.json.
//
// parity.json already guards the surface itself, but nothing guarded the
// numbers *written about* it, and go/README.md had said 22 tools in three
// places through the growth to 41. A reader following the README to decide
// whether a tool exists was being told something false.
//
// It lives on this side of the repo because that is the rule parity.json
// states for itself: each port asserts against the manifest in its OWN suite,
// "so neither CI job needs the other toolchain". The Python suite checking
// go/*.md broke that — and broke concretely, since the hermetic Python image
// excludes the whole Go tree.
func TestDocumentedSurfaceSizesMatchParity(t *testing.T) {
	m := loadManifest(t)
	// Any "<N> tools" / "<N> MCP tools" claim, however it is punctuated:
	// `**41 MCP tools**`, `(41 tools)`, `— 41 tools`.
	toolClaim := regexp.MustCompile(`(\d+)\s*(?:MCP\s+)?tools?\b`)
	routeClaim := regexp.MustCompile(
		`(?:(\d+)\s*HTTP\s+routes?\b|HTTP\s+routes?\s*\((\d+)\))`)

	for _, name := range []string{"README.md", "DESIGN.md"} {
		path := filepath.Join("..", "..", name)
		raw, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("read go/%s: %v", name, err)
		}
		text := string(raw)

		want := strconv.Itoa(len(m.MCPTools))
		for _, hit := range toolClaim.FindAllStringSubmatch(text, -1) {
			if hit[1] != want {
				t.Errorf("go/%s quotes %q tools; parity.json lists %s",
					name, hit[1], want)
			}
		}

		want = strconv.Itoa(len(m.HTTPRoutes))
		for _, hit := range routeClaim.FindAllStringSubmatch(text, -1) {
			got := hit[1]
			if got == "" {
				got = hit[2]
			}
			if got != want {
				t.Errorf("go/%s quotes %q HTTP routes; parity.json lists %s",
					name, got, want)
			}
		}
	}
}
