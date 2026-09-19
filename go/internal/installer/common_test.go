package installer

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The three installers used to carry their own copy of the block merge, the
// registration check and the env build. These cover the shared versions
// directly, so a regression names the helper rather than one client.

func TestMergeServerBlockPreservesUnrelatedConfig(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cfg.json")
	seed, _ := json.Marshal(map[string]any{
		"theme":      "dark",
		"mcpServers": map[string]any{"other": map[string]any{"command": "x"}},
	})
	if err := os.WriteFile(path, seed, 0o600); err != nil {
		t.Fatalf("seed: %v", err)
	}

	if _, err := mergeServerBlock(path, "mcpServers", "ai-houkai",
		map[string]any{"command": "m"}, nil); err != nil {
		t.Fatalf("mergeServerBlock: %v", err)
	}

	data, _ := os.ReadFile(path)
	var got map[string]any
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if got["theme"] != "dark" {
		t.Error("unrelated top-level keys must survive the merge")
	}
	servers := got["mcpServers"].(map[string]any)
	if _, ok := servers["other"]; !ok {
		t.Error("another client's MCP server was wiped")
	}
	if _, ok := servers["ai-houkai"]; !ok {
		t.Error("our entry was not added")
	}
}

func TestMergeServerBlockSeedsDefaultsWithoutClobbering(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cfg.json")
	defaults := map[string]any{"$schema": "https://example.test/schema.json"}

	if _, err := mergeServerBlock(path, "mcp", "ai-houkai",
		map[string]any{"enabled": true}, defaults); err != nil {
		t.Fatalf("mergeServerBlock: %v", err)
	}
	data, _ := os.ReadFile(path)
	var got map[string]any
	_ = json.Unmarshal(data, &got)
	if got["$schema"] != defaults["$schema"] {
		t.Errorf("$schema = %v, want seeded default", got["$schema"])
	}

	// A user who pinned their own schema keeps it: defaults are defaults.
	mine, _ := json.Marshal(map[string]any{"$schema": "https://mine.test/x.json"})
	_ = os.WriteFile(path, mine, 0o600)
	if _, err := mergeServerBlock(path, "mcp", "ai-houkai",
		map[string]any{"enabled": true}, defaults); err != nil {
		t.Fatalf("mergeServerBlock: %v", err)
	}
	data, _ = os.ReadFile(path)
	_ = json.Unmarshal(data, &got)
	if got["$schema"] != "https://mine.test/x.json" {
		t.Errorf("$schema = %v, want the user's own value", got["$schema"])
	}
}

func TestHasServerOnlyMatchesItsOwnKey(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cfg.json")
	if _, err := mergeServerBlock(path, "mcp", "ai-houkai",
		map[string]any{"enabled": true}, nil); err != nil {
		t.Fatalf("mergeServerBlock: %v", err)
	}
	if !hasServer(path, "mcp", "ai-houkai") {
		t.Error("registered server not found under its own key")
	}
	// OpenCode's `mcp` and Cursor's `mcpServers` are different schemas: a
	// registration under one must not read as a registration under the other.
	if hasServer(path, "mcpServers", "ai-houkai") {
		t.Error("a `mcp` entry must not satisfy a `mcpServers` check")
	}
	if hasServer(filepath.Join(t.TempDir(), "missing.json"), "mcp", "ai-houkai") {
		t.Error("a missing config must not read as registered")
	}
}

func TestServerEnvExtraOverridesDefaults(t *testing.T) {
	env := serverEnv("/mem", "col", map[string]string{
		"AI_HOUKAI_COLLECTION": "override",
		"EXTRA":                "1",
	})
	if env["AI_HOUKAI_PATH"] != "/mem" {
		t.Errorf("AI_HOUKAI_PATH = %v", env["AI_HOUKAI_PATH"])
	}
	if env["AI_HOUKAI_COLLECTION"] != "override" {
		t.Errorf("ExtraEnv must win over the default: %v", env["AI_HOUKAI_COLLECTION"])
	}
	if env["EXTRA"] != "1" {
		t.Errorf("EXTRA = %v", env["EXTRA"])
	}
}

func TestMCPCommandPrefersExplicitBinary(t *testing.T) {
	if got := mcpCommand("/opt/ai-houkai-mcp"); got != "/opt/ai-houkai-mcp" {
		t.Errorf("mcpCommand = %q, want the explicit path", got)
	}
	if got := mcpCommand(""); got == "" {
		t.Error("an unset BinaryPath must fall back to ResolveMCPCommand, not empty")
	}
}

// TestSnippetsShareOneGuide pins the fix for a real drift: each installer used
// to carry its own copy of the memory instructions, and the Claude Code one had
// fallen behind — it listed four tools and omitted `edit` entirely.
func TestSnippetsShareOneGuide(t *testing.T) {
	for name, snippet := range map[string]string{
		"claude-code": ClaudeMDSnippet(),
		"cursor":      CursorRuleSnippet,
		"opencode":    OpenCodeAgentsSnippet,
	} {
		if !strings.Contains(snippet, MemoryGuide) {
			t.Errorf("%s snippet does not wrap MemoryGuide", name)
		}
		if !strings.Contains(snippet, "edit(memory_id") {
			t.Errorf("%s snippet is missing the edit tool", name)
		}
	}
}
