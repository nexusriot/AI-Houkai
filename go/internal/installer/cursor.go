package installer

// Cursor installer for the AI-Houkai MCP server.
//
// Registers `ai-houkai-mcp` in Cursor's MCP config so the editor's agent
// gains a persistent memory. Cursor uses the same `mcpServers` schema as
// Claude Desktop / Claude Code, but reads it from a different file:
//
//	~/.cursor/mcp.json            global (all projects)
//	<project>/.cursor/mcp.json    project-scoped

import (
	"fmt"
	"os"
	"path/filepath"
)

// CursorProjectConfigPath is the project-scoped Cursor MCP config location.
const CursorProjectConfigPath = ".cursor/mcp.json"

// CursorRuleSnippet is the .cursor/rules/*.mdc content — Markdown with a
// small YAML frontmatter. `alwaysApply: true` keeps the rule in context
// every request.
const CursorRuleSnippet = `---
description: AI-Houkai persistent memory — when and how to use the MCP tools
alwaysApply: true
---

# Memory (AI-Houkai MCP)

` + MemoryGuide

// CursorInstaller registers the AI-Houkai MCP server with Cursor.
type CursorInstaller struct {
	MemoryPath   string
	Collection   string
	SettingsPath string
	ServerName   string
	BinaryPath   string // path to ai-houkai-mcp; defaults to ResolveMCPCommand()
	ExtraEnv     map[string]string
}

// DefaultCursorInstaller targets the global ~/.cursor/mcp.json.
func DefaultCursorInstaller() CursorInstaller {
	home, _ := os.UserHomeDir()
	return CursorInstaller{
		MemoryPath:   filepath.Join(home, ".ai_houkai"),
		Collection:   "cursor",
		SettingsPath: filepath.Join(home, ".cursor", "mcp.json"),
		ServerName:   "ai-houkai",
	}
}

func (i CursorInstaller) mcpCommand() string {
	if i.BinaryPath != "" {
		return i.BinaryPath
	}
	return ResolveMCPCommand()
}

func (i CursorInstaller) buildMCPBlock() map[string]any {
	env := map[string]any{
		"AI_HOUKAI_PATH":       i.MemoryPath,
		"AI_HOUKAI_COLLECTION": i.Collection,
	}
	for k, v := range i.ExtraEnv {
		env[k] = v
	}
	return map[string]any{"command": i.mcpCommand(), "env": env}
}

// Install patches Cursor's mcp.json with the MCP server block and returns
// the written path.
func (i CursorInstaller) Install() (string, error) {
	path := expandHome(i.SettingsPath)
	config := loadJSONFile(path)
	servers, _ := config["mcpServers"].(map[string]any)
	if servers == nil {
		servers = map[string]any{}
	}
	servers[i.ServerName] = i.buildMCPBlock()
	config["mcpServers"] = servers
	return path, writeJSONFile(path, config)
}

// Verify returns true if the server entry exists in Cursor's mcp.json.
func (i CursorInstaller) Verify() bool {
	config := loadJSONFile(expandHome(i.SettingsPath))
	servers, _ := config["mcpServers"].(map[string]any)
	_, ok := servers[i.ServerName]
	return ok
}

// PrintConfig prints the MCP block for manual pasting.
func (i CursorInstaller) PrintConfig() {
	block := map[string]any{
		"mcpServers": map[string]any{i.ServerName: i.buildMCPBlock()},
	}
	fmt.Printf("\nPaste this into %s:\n\n%s\n", i.SettingsPath, printJSONBlock(block))
	fmt.Printf("\nThen reload Cursor and open Settings → MCP to confirm %q is listed.\n\n", i.ServerName)
}
