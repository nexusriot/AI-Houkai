package installer

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestCursorInstallCreatesNewConfig(t *testing.T) {
	dir := t.TempDir()
	settings := filepath.Join(dir, ".cursor", "mcp.json")

	inst := CursorInstaller{
		MemoryPath:   "/tmp/mem",
		Collection:   "cursor",
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
		t.Fatalf("read config: %v", err)
	}
	var got map[string]any
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	servers := got["mcpServers"].(map[string]any)
	block := servers["ai-houkai"].(map[string]any)
	if block["command"] != "ai-houkai-mcp" {
		t.Errorf("command = %v", block["command"])
	}
	env := block["env"].(map[string]any)
	if env["AI_HOUKAI_PATH"] != "/tmp/mem" || env["AI_HOUKAI_COLLECTION"] != "cursor" {
		t.Errorf("env = %v", env)
	}
}

func TestCursorInstallPreservesExistingServers(t *testing.T) {
	dir := t.TempDir()
	settings := filepath.Join(dir, "mcp.json")
	existing := `{"mcpServers": {"other": {"command": "other-mcp"}}}`
	if err := os.WriteFile(settings, []byte(existing), 0o600); err != nil {
		t.Fatal(err)
	}

	inst := CursorInstaller{
		MemoryPath:   "/tmp/mem",
		Collection:   "cursor",
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
	servers := got["mcpServers"].(map[string]any)
	if _, ok := servers["other"]; !ok {
		t.Error("existing server entry was dropped")
	}
	if _, ok := servers["ai-houkai"]; !ok {
		t.Error("ai-houkai entry missing")
	}
}

func TestCursorVerify(t *testing.T) {
	dir := t.TempDir()
	settings := filepath.Join(dir, "mcp.json")
	inst := CursorInstaller{
		MemoryPath:   "/tmp/mem",
		Collection:   "cursor",
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

func TestCursorRuleSnippet(t *testing.T) {
	if !strings.HasPrefix(CursorRuleSnippet, "---\n") {
		t.Error("rule snippet missing frontmatter")
	}
	if !strings.Contains(CursorRuleSnippet, "alwaysApply: true") {
		t.Error("rule snippet missing alwaysApply")
	}
	if !strings.Contains(CursorRuleSnippet, "remember(text, type, tags, importance)") {
		t.Error("rule snippet missing memory guide")
	}
}

func TestCursorInstallOverwritesUnparseable(t *testing.T) {
	dir := t.TempDir()
	settings := filepath.Join(dir, "mcp.json")
	if err := os.WriteFile(settings, []byte("{not json"), 0o600); err != nil {
		t.Fatal(err)
	}
	inst := CursorInstaller{
		MemoryPath:   "/tmp/mem",
		Collection:   "cursor",
		SettingsPath: settings,
		ServerName:   "ai-houkai",
		BinaryPath:   "ai-houkai-mcp",
	}
	if _, err := inst.Install(); err != nil {
		t.Fatalf("Install: %v", err)
	}
	if !inst.Verify() {
		t.Error("Verify false after overwrite-install")
	}
}
