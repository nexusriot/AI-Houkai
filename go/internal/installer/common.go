package installer

// Shared helpers for the AI-Houkai client installers.
//
// Each installer (Claude Code, Cursor, OpenCode, …) registers the same stdio
// MCP server — `ai-houkai-mcp` — into a client-specific config file. The only
// differences are the file location and the JSON schema the client expects.

import (
	"encoding/json"
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
- **forget(memory_id)** — remove a specific memory
- **list_recent()** — see the most recently created memories

### When to use memory

| Situation | Action |
|---|---|
| User states a preference or coding convention | ` + "`remember` with `type=\"feedback\"` or `\"procedural\"`" + ` |
| You learn something about the codebase | ` + "`remember` with `type=\"semantic\"`" + ` |
| Starting a new task | ` + "`recall` relevant context first" + ` |
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
