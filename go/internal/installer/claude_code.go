package installer

// Claude Code installer for the AI-Houkai MCP server.
//
// Registers `ai-houkai-mcp` in Claude Code's settings.json under the same
// `mcpServers` schema Cursor uses.

import (
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
	BinaryPath   string // path to ai-houkai-mcp binary; defaults to ResolveMCPCommand()
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
	return map[string]any{
		"command": mcpCommand(i.BinaryPath),
		"args":    []string{},
		"env":     serverEnv(i.MemoryPath, i.Collection, i.ExtraEnv),
	}
}

// Install writes / merges the MCP server entry into settings.json.
func (i ClaudeCodeInstaller) Install() (string, error) {
	return mergeServerBlock(i.SettingsPath, "mcpServers", i.ServerName,
		i.buildMCPBlock(), nil)
}

// Verify returns true if the server entry exists in settings.json.
func (i ClaudeCodeInstaller) Verify() bool {
	return hasServer(i.SettingsPath, "mcpServers", i.ServerName)
}

// PrintConfig prints the MCP block for manual inspection.
func (i ClaudeCodeInstaller) PrintConfig() {
	fmt.Println(printJSONBlock(map[string]any{
		"mcpServers": map[string]any{i.ServerName: i.buildMCPBlock()},
	}))
}

// ClaudeMDSnippet returns a CLAUDE.md snippet for memory usage instructions.
// It wraps the same client-agnostic MemoryGuide the Cursor and OpenCode
// snippets do — the hand-written variant it replaced listed four tools out of
// forty-one and had drifted from every other copy.
func ClaudeMDSnippet() string {
	return "## Memory (AI-Houkai MCP)\n\n" + MemoryGuide
}
