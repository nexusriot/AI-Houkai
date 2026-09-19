package installer

// Shared helpers for the AI-Houkai client installers.
//
// Each installer (Claude Code, Cursor, OpenCode, …) registers the same stdio
// MCP server — `ai-houkai-mcp` — into a client-specific config file. The only
// differences are the file location and the JSON schema the client expects.

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
)

// MCPBinary is the name of the MCP server binary registered by all installers.
const MCPBinary = "ai-houkai-mcp"

// ResolveMCPCommand returns the absolute path to the ai-houkai-mcp binary if
// it sits next to the running executable, otherwise the bare name (resolved
// via PATH at runtime).
func ResolveMCPCommand() string {
	if exe, err := os.Executable(); err == nil {
		candidate := filepath.Join(filepath.Dir(exe), MCPBinary)
		if fi, err := os.Stat(candidate); err == nil && !fi.IsDir() {
			return candidate
		}
	}
	return MCPBinary
}

// loadJSONFile loads a JSON config file, returning an empty map if the file
// is missing or unparseable (the installer will overwrite it).
func loadJSONFile(path string) map[string]any {
	data, err := os.ReadFile(path)
	if err != nil {
		return map[string]any{}
	}
	var out map[string]any
	if err := json.Unmarshal(data, &out); err != nil || out == nil {
		return map[string]any{}
	}
	return out
}

// writeJSONFile writes config to path (creating parent dirs).
func writeJSONFile(path string, config map[string]any) error {
	if parent := filepath.Dir(path); parent != "" && parent != "." {
		if err := os.MkdirAll(parent, 0o700); err != nil {
			return err
		}
	}
	out, err := json.MarshalIndent(config, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(out, '\n'), 0o600)
}

// VerifyBinary checks that the ai-houkai-mcp binary is reachable. It returns
// the resolved path and true on success.
func VerifyBinary() (string, bool) {
	cmd := ResolveMCPCommand()
	if filepath.IsAbs(cmd) {
		if fi, err := os.Stat(cmd); err == nil && !fi.IsDir() {
			return cmd, true
		}
		return cmd, false
	}
	if found, err := exec.LookPath(cmd); err == nil {
		return found, true
	}
	return cmd, false
}

// MemoryGuide is a client-agnostic description of when/how to use the memory
// tools. Each installer wraps it in its own instruction-file format
// (CLAUDE.md, AGENTS.md, .cursor/rules/*.mdc, …).
const MemoryGuide = `You have access to a persistent memory store via AI-Houkai MCP tools:

- **remember(text, type, tags, importance)** — store a fact, decision, or preference
- **recall(query, k)** — semantic search across stored memories
- **edit(memory_id, …)** — update a memory in place (keeps id, links, history)
- **forget(memory_id)** — remove a specific memory
- **list_recent()** — see the most recently created memories

### When to use memory

| Situation | Action |
|---|---|
| User states a preference or coding convention | ` + "`remember` with `type=\"feedback\"` or `\"procedural\"`" + ` |
| You learn something about the codebase | ` + "`remember` with `type=\"semantic\"`" + ` |
| Starting a new task | ` + "`recall` relevant context first" + ` |
| A stored fact is outdated or has a typo | ` + "`edit` it in place — don't forget+remember" + ` |
| User corrects you | ` + "`remember` the correction, `forget` the wrong fact" + ` |

### Memory types
- ` + "`episodic`" + ` — time-stamped events ("Fixed auth bug in PR #441")
- ` + "`semantic`" + ` — distilled facts ("API versioned at /api/v1/")
- ` + "`procedural`" + ` — how-to rules ("Always use tmp_path in tests")
- ` + "`feedback`" + ` — user preferences ("Prefers concise answers")`

// printJSONBlock pretty-prints a settings block for manual pasting.
func printJSONBlock(block map[string]any) string {
	b, _ := json.MarshalIndent(block, "", "  ")
	return string(b)
}

// expandHome resolves a leading `~/` against the user's home directory.
func expandHome(path string) string {
	if len(path) >= 2 && path[:2] == "~/" {
		home, _ := os.UserHomeDir()
		return filepath.Join(home, path[2:])
	}
	return path
}

// mcpCommand is the binary an installer registers: an explicitly configured
// BinaryPath wins, otherwise fall back to the usual resolution.
func mcpCommand(binaryPath string) string {
	if binaryPath != "" {
		return binaryPath
	}
	return ResolveMCPCommand()
}

// serverEnv is the environment block every client passes to the MCP server.
// ExtraEnv wins over the two defaults.
func serverEnv(memoryPath, collection string, extra map[string]string) map[string]any {
	env := map[string]any{
		"AI_HOUKAI_PATH":       memoryPath,
		"AI_HOUKAI_COLLECTION": collection,
	}
	for k, v := range extra {
		env[k] = v
	}
	return env
}

// mergeServerBlock merges one server block into a client's JSON config file
// and returns the written path.
//
// Read-modify-write rather than overwrite: every client config also holds
// settings that are none of our business, and other MCP servers besides ours.
// defaults seeds top-level keys only when absent (OpenCode's `$schema`).
func mergeServerBlock(
	path, configKey, serverName string,
	block map[string]any,
	defaults map[string]any,
) (string, error) {
	path = expandHome(path)
	config := loadJSONFile(path)
	for k, v := range defaults {
		if _, ok := config[k]; !ok {
			config[k] = v
		}
	}
	servers, _ := config[configKey].(map[string]any)
	if servers == nil {
		servers = map[string]any{}
	}
	servers[serverName] = block
	config[configKey] = servers
	return path, writeJSONFile(path, config)
}

// hasServer reports whether serverName is registered under configKey in the
// config file at path.
func hasServer(path, configKey, serverName string) bool {
	config := loadJSONFile(expandHome(path))
	servers, _ := config[configKey].(map[string]any)
	_, ok := servers[serverName]
	return ok
}

// printPasteBlock prints a settings block for manual pasting, followed by the
// client's own "now reload me" hint.
func printPasteBlock(settingsPath string, block map[string]any, hint string) {
	fmt.Printf("\nPaste this into %s:\n\n%s\n", settingsPath, printJSONBlock(block))
	fmt.Printf("\n%s\n\n", hint)
}
