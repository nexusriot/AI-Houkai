package installer

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// ClaudeCodeInstaller patches ~/.claude/settings.json to register the MCP server.
type ClaudeCodeInstaller struct {
	MemoryPath   string
	Collection   string
	SettingsPath string
	ServerName   string
	ExtraEnv     map[string]string
	BinaryPath   string // path to ai-houkai-mcp binary; defaults to "ai-houkai-mcp"
}

func DefaultInstaller() ClaudeCodeInstaller {
	home, _ := os.UserHomeDir()
	return ClaudeCodeInstaller{
		MemoryPath:   filepath.Join(home, ".ai_houkai"),
		Collection:   "ai_houkai",
		SettingsPath: filepath.Join(home, ".claude", "settings.json"),
		ServerName:   "ai-houkai",
		BinaryPath:   "ai-houkai-mcp",
	}
}

func (i ClaudeCodeInstaller) buildMCPBlock() map[string]any {
	env := map[string]any{
		"AI_HOUKAI_PATH":       i.MemoryPath,
		"AI_HOUKAI_COLLECTION": i.Collection,
	}
	for k, v := range i.ExtraEnv {
		env[k] = v
	}
	return map[string]any{
		"command": i.BinaryPath,
		"args":    []string{},
		"env":     env,
	}
}

// Install writes / merges the MCP server entry into settings.json.
func (i ClaudeCodeInstaller) Install() (string, error) {
	path := expandHome(i.SettingsPath)

	var settings map[string]any
	data, err := os.ReadFile(path)
	if err != nil && !os.IsNotExist(err) {
		return "", fmt.Errorf("read settings: %w", err)
	}
	if len(data) > 0 {
		if err := json.Unmarshal(data, &settings); err != nil {
			// Overwrite unparseable file.
			settings = map[string]any{}
		}
	}
	if settings == nil {
		settings = map[string]any{}
	}

	servers, _ := settings["mcpServers"].(map[string]any)
	if servers == nil {
		servers = map[string]any{}
	}
	servers[i.ServerName] = i.buildMCPBlock()
	settings["mcpServers"] = servers

	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return "", err
	}
	out, _ := json.MarshalIndent(settings, "", "  ")
	return path, os.WriteFile(path, out, 0o600)
}

// Verify returns true if the server entry exists in settings.json.
func (i ClaudeCodeInstaller) Verify() bool {
	path := expandHome(i.SettingsPath)
	data, err := os.ReadFile(path)
	if err != nil {
		return false
	}
	var settings map[string]any
	if err := json.Unmarshal(data, &settings); err != nil {
		return false
	}
	servers, _ := settings["mcpServers"].(map[string]any)
	_, ok := servers[i.ServerName]
	return ok
}

// PrintConfig prints the MCP block for manual inspection.
func (i ClaudeCodeInstaller) PrintConfig() {
	b, _ := json.MarshalIndent(map[string]any{
		"mcpServers": map[string]any{i.ServerName: i.buildMCPBlock()},
	}, "", "  ")
	fmt.Println(string(b))
}

// ClaudeMDSnippet returns a CLAUDE.md snippet for memory usage instructions.
func ClaudeMDSnippet() string {
	return `## Memory (AI-Houkai MCP)
- recall() before starting any task to surface relevant context
- remember() conventions, decisions, corrections, and user preferences
- forget() outdated or superseded facts
- maintenance_tick() occasionally to prune stale memories
`
}

func expandHome(path string) string {
	if len(path) >= 2 && path[:2] == "~/" {
		home, _ := os.UserHomeDir()
		return filepath.Join(home, path[2:])
	}
	return path
}
