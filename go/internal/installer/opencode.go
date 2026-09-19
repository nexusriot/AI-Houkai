package installer

// OpenCode installer for the AI-Houkai MCP server.
//
// Registers `ai-houkai-mcp` in OpenCode's config so the agent gains a
// persistent memory. OpenCode (sst/opencode) uses its own `mcp` schema —
// distinct from the `mcpServers` block used by Claude/Cursor: each server is
// an entry under `mcp` with `type: "local"`, a `command` *array*, an
// `environment` object, and an `enabled` flag.
//
//	~/.config/opencode/opencode.json   global (all projects)
//	<project>/opencode.json            project-scoped

import (
	"fmt"
	"os"
	"path/filepath"
)

// OpenCodeProjectConfigPath is the project-scoped OpenCode config location.
const OpenCodeProjectConfigPath = "opencode.json"

// OpenCodeConfigSchemaURL is the $schema value OpenCode configs carry.
const OpenCodeConfigSchemaURL = "https://opencode.ai/config.json"

// OpenCodeAgentsSnippet is the AGENTS.md content — OpenCode reads
// project/global instructions from AGENTS.md.
const OpenCodeAgentsSnippet = "## Memory (AI-Houkai MCP)\n\n" + MemoryGuide

// OpenCodeInstaller registers the AI-Houkai MCP server with OpenCode.
type OpenCodeInstaller struct {
	MemoryPath   string
	Collection   string
	SettingsPath string
	ServerName   string
	BinaryPath   string // path to ai-houkai-mcp; defaults to ResolveMCPCommand()
	ExtraEnv     map[string]string
}

// DefaultOpenCodeInstaller targets the global ~/.config/opencode/opencode.json.
func DefaultOpenCodeInstaller() OpenCodeInstaller {
	home, _ := os.UserHomeDir()
	return OpenCodeInstaller{
		MemoryPath:   filepath.Join(home, ".ai_houkai"),
		Collection:   "opencode",
		SettingsPath: filepath.Join(home, ".config", "opencode", "opencode.json"),
		ServerName:   "ai-houkai",
	}
}

func (i OpenCodeInstaller) buildMCPBlock() map[string]any {
	return map[string]any{
		"type":        "local",
		"command":     []string{mcpCommand(i.BinaryPath)},
		"enabled":     true,
		"environment": serverEnv(i.MemoryPath, i.Collection, i.ExtraEnv),
	}
}

// Install patches opencode.json with the MCP server block and returns the
// written path.
func (i OpenCodeInstaller) Install() (string, error) {
	return mergeServerBlock(i.SettingsPath, "mcp", i.ServerName,
		i.buildMCPBlock(),
		map[string]any{"$schema": OpenCodeConfigSchemaURL})
}

// Verify returns true if the server entry exists in opencode.json.
func (i OpenCodeInstaller) Verify() bool {
	return hasServer(i.SettingsPath, "mcp", i.ServerName)
}

// PrintConfig prints the MCP block for manual pasting.
func (i OpenCodeInstaller) PrintConfig() {
	printPasteBlock(i.SettingsPath,
		map[string]any{
			"$schema": OpenCodeConfigSchemaURL,
			"mcp":     map[string]any{i.ServerName: i.buildMCPBlock()},
		},
		fmt.Sprintf("Then restart OpenCode — %q tools become available to the agent.", i.ServerName))
}
