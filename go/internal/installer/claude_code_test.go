package installer

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestInstallCreatesNewSettings(t *testing.T) {
	dir := t.TempDir()
	settings := filepath.Join(dir, ".claude", "settings.json")

	inst := ClaudeCodeInstaller{
		MemoryPath:   "/tmp/mem",
		Collection:   "ai_houkai",
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

	data, err := os.ReadFile(settings)
	if err != nil {
		t.Fatalf("read settings: %v", err)
	}
	var got map[string]any
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	servers, ok := got["mcpServers"].(map[string]any)
	if !ok {
		t.Fatalf("mcpServers missing or wrong type: %v", got["mcpServers"])
	}
	block, ok := servers["ai-houkai"].(map[string]any)
	if !ok {
		t.Fatalf("ai-houkai entry missing: %v", servers)
	}
	if block["command"] != "ai-houkai-mcp" {
		t.Errorf("command = %v, want ai-houkai-mcp", block["command"])
	}
	env := block["env"].(map[string]any)
	if env["AI_HOUKAI_PATH"] != "/tmp/mem" {
		t.Errorf("env AI_HOUKAI_PATH = %v, want /tmp/mem", env["AI_HOUKAI_PATH"])
	}
	if env["AI_HOUKAI_COLLECTION"] != "ai_houkai" {
		t.Errorf("env AI_HOUKAI_COLLECTION = %v, want ai_houkai", env["AI_HOUKAI_COLLECTION"])
	}
}

func TestInstallPreservesExistingKeys(t *testing.T) {
	dir := t.TempDir()
	settings := filepath.Join(dir, "settings.json")

	// Pre-existing settings with an unrelated MCP server and a top-level key.
	pre := map[string]any{
		"theme": "dark",
		"mcpServers": map[string]any{
			"other-server": map[string]any{"command": "other"},
		},
	}
	preBytes, _ := json.Marshal(pre)
	if err := os.WriteFile(settings, preBytes, 0o600); err != nil {
		t.Fatalf("seed settings: %v", err)
	}

	inst := ClaudeCodeInstaller{
		MemoryPath:   "/m",
		Collection:   "c",
		SettingsPath: settings,
		ServerName:   "ai-houkai",
		BinaryPath:   "ai-houkai-mcp",
	}
	if _, err := inst.Install(); err != nil {
		t.Fatalf("Install: %v", err)
	}

	data, _ := os.ReadFile(settings)
	var got map[string]any
	_ = json.Unmarshal(data, &got)

	if got["theme"] != "dark" {
		t.Error("top-level keys should survive merge")
	}
	servers := got["mcpServers"].(map[string]any)
	if _, ok := servers["other-server"]; !ok {
		t.Error("existing mcp server entry was wiped")
	}
	if _, ok := servers["ai-houkai"]; !ok {
		t.Error("our entry was not added")
	}
}

func TestVerify(t *testing.T) {
	dir := t.TempDir()
	settings := filepath.Join(dir, "settings.json")
	inst := ClaudeCodeInstaller{
		MemoryPath:   "/m",
		Collection:   "c",
		SettingsPath: settings,
		ServerName:   "ai-houkai",
		BinaryPath:   "ai-houkai-mcp",
	}
	if inst.Verify() {
		t.Error("Verify should be false before install")
	}
	if _, err := inst.Install(); err != nil {
		t.Fatalf("Install: %v", err)
	}
	if !inst.Verify() {
		t.Error("Verify should be true after install")
	}
}
