package installer

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestOpenCodeInstallCreatesNewConfig(t *testing.T) {
	dir := t.TempDir()
	settings := filepath.Join(dir, ".config", "opencode", "opencode.json")

	inst := OpenCodeInstaller{
		MemoryPath:   "/tmp/mem",
		Collection:   "opencode",
		SettingsPath: settings,
		ServerName:   "ai-houkai",
		BinaryPath:   "ai-houkai-mcp",
	}
	out, err := inst.Install()
	if err != nil {
		t.Fatalf("Install: %v", err)
	}
	if out != settings {
		t.Errorf("Install returned %q, want %q", out, settings)
	}

	var got map[string]any
	data, err := os.ReadFile(settings)
	if err != nil {
		t.Fatalf("read config: %v", err)
	}
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if got["$schema"] != OpenCodeConfigSchemaURL {
		t.Errorf("$schema = %v", got["$schema"])
	}
	servers := got["mcp"].(map[string]any)
	block := servers["ai-houkai"].(map[string]any)
	if block["type"] != "local" {
		t.Errorf("type = %v, want local", block["type"])
	}
	if block["enabled"] != true {
		t.Errorf("enabled = %v, want true", block["enabled"])
	}
	cmd, ok := block["command"].([]any)
	if !ok || len(cmd) != 1 || cmd[0] != "ai-houkai-mcp" {
		t.Errorf("command = %v, want [ai-houkai-mcp]", block["command"])
	}
	env := block["environment"].(map[string]any)
	if env["AI_HOUKAI_PATH"] != "/tmp/mem" || env["AI_HOUKAI_COLLECTION"] != "opencode" {
		t.Errorf("environment = %v", env)
	}
}

func TestOpenCodeInstallPreservesExistingKeys(t *testing.T) {
	dir := t.TempDir()
	settings := filepath.Join(dir, "opencode.json")
	existing := `{"$schema": "custom", "theme": "dark", "mcp": {"other": {"type": "local"}}}`
	if err := os.WriteFile(settings, []byte(existing), 0o600); err != nil {
		t.Fatal(err)
	}

	inst := OpenCodeInstaller{
		MemoryPath:   "/tmp/mem",
		Collection:   "opencode",
		SettingsPath: settings,
		ServerName:   "ai-houkai",
		BinaryPath:   "ai-houkai-mcp",
	}
	if _, err := inst.Install(); err != nil {
		t.Fatalf("Install: %v", err)
	}

	var got map[string]any
	data, _ := os.ReadFile(settings)
	_ = json.Unmarshal(data, &got)
	if got["theme"] != "dark" {
		t.Error("existing top-level key was dropped")
	}
	if got["$schema"] != "custom" {
		t.Errorf("existing $schema overwritten: %v", got["$schema"])
	}
	servers := got["mcp"].(map[string]any)
	if _, ok := servers["other"]; !ok {
		t.Error("existing mcp entry was dropped")
	}
	if _, ok := servers["ai-houkai"]; !ok {
		t.Error("ai-houkai entry missing")
	}
}

func TestOpenCodeVerify(t *testing.T) {
	dir := t.TempDir()
	settings := filepath.Join(dir, "opencode.json")
	inst := OpenCodeInstaller{
		MemoryPath:   "/tmp/mem",
		Collection:   "opencode",
		SettingsPath: settings,
		ServerName:   "ai-houkai",
		BinaryPath:   "ai-houkai-mcp",
	}
	if inst.Verify() {
		t.Error("Verify true before install")
	}
	if _, err := inst.Install(); err != nil {
		t.Fatal(err)
	}
	if !inst.Verify() {
		t.Error("Verify false after install")
	}
}

func TestOpenCodeAgentsSnippet(t *testing.T) {
	if !strings.HasPrefix(OpenCodeAgentsSnippet, "## Memory (AI-Houkai MCP)") {
		t.Error("agents snippet missing heading")
	}
	if !strings.Contains(OpenCodeAgentsSnippet, "recall(query, k)") {
		t.Error("agents snippet missing memory guide")
	}
}
