package cli

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The three `houkai install <client>` subcommands were one 41-line RunE copied
// twice. They are now one builder parameterised by an installClient, so these
// drive each client through the shared body and check that the per-client
// differences — config key, block schema, project path, snippet flag — still
// land where they belong.

func runInstall(t *testing.T, args ...string) string {
	t.Helper()
	cmd := newInstallCmd()
	cmd.SetArgs(args)
	cmd.SetOut(os.Stdout)
	if err := cmd.Execute(); err != nil {
		t.Fatalf("houkai install %v: %v", args, err)
	}
	return ""
}

func TestInstallWritesEachClientsSchema(t *testing.T) {
	for _, tc := range []struct {
		client    string
		configKey string
		// The block key that holds the environment, which differs by client.
		envKey string
	}{
		{"claude-code", "mcpServers", "env"},
		{"cursor", "mcpServers", "env"},
		{"opencode", "mcp", "environment"},
	} {
		t.Run(tc.client, func(t *testing.T) {
			cfg := filepath.Join(t.TempDir(), "client.json")
			runInstall(t, tc.client, "--settings", cfg,
				"--memory-path", "/mem", "--collection", "col",
				"--binary", "/opt/ai-houkai-mcp")

			data, err := os.ReadFile(cfg)
			if err != nil {
				t.Fatalf("read config: %v", err)
			}
			var got map[string]any
			if err := json.Unmarshal(data, &got); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			servers, ok := got[tc.configKey].(map[string]any)
			if !ok {
				t.Fatalf("no %q object in %v", tc.configKey, got)
			}
			block, ok := servers["ai-houkai"].(map[string]any)
			if !ok {
				t.Fatalf("no ai-houkai entry under %q: %v", tc.configKey, servers)
			}
			env, ok := block[tc.envKey].(map[string]any)
			if !ok {
				t.Fatalf("no %q object in the block: %v", tc.envKey, block)
			}
			// --memory-path / --collection must reach the installer: the flags
			// are applied through pointers AFTER it is built, so a value copy
			// anywhere in that path would silently write the defaults.
			if env["AI_HOUKAI_PATH"] != "/mem" {
				t.Errorf("AI_HOUKAI_PATH = %v, want /mem", env["AI_HOUKAI_PATH"])
			}
			if env["AI_HOUKAI_COLLECTION"] != "col" {
				t.Errorf("AI_HOUKAI_COLLECTION = %v, want col", env["AI_HOUKAI_COLLECTION"])
			}
		})
	}
}

func TestInstallSettingsOverridesProject(t *testing.T) {
	cfg := filepath.Join(t.TempDir(), "explicit.json")
	runInstall(t, "cursor", "--project", "--settings", cfg)
	if _, err := os.Stat(cfg); err != nil {
		t.Fatalf("--settings must win over --project: %v", err)
	}
}

func TestInstallIsIdempotent(t *testing.T) {
	cfg := filepath.Join(t.TempDir(), "client.json")
	runInstall(t, "opencode", "--settings", cfg, "--binary", "m")
	first, _ := os.ReadFile(cfg)
	runInstall(t, "opencode", "--settings", cfg, "--binary", "m")
	second, _ := os.ReadFile(cfg)
	if string(first) != string(second) {
		t.Error("re-running install must not change the config")
	}
}

func TestInstallSnippetFlagsAreClientSpecific(t *testing.T) {
	// Only Cursor has --rule and only OpenCode has --agents; Claude Code has
	// neither (its snippet prints after a successful install).
	for _, tc := range []struct{ client, flag string }{
		{"cursor", "rule"},
		{"opencode", "agents"},
	} {
		if newInstallCmd().Flags().Lookup(tc.flag) != nil {
			t.Errorf("--%s must not be a flag of the bare `install` command", tc.flag)
		}
		sub, _, err := newInstallCmd().Find([]string{tc.client})
		if err != nil {
			t.Fatalf("find %s: %v", tc.client, err)
		}
		if sub.Flags().Lookup(tc.flag) == nil {
			t.Errorf("`install %s` is missing its --%s flag", tc.client, tc.flag)
		}
		other := map[string]string{"rule": "agents", "agents": "rule"}[tc.flag]
		if sub.Flags().Lookup(other) != nil {
			t.Errorf("`install %s` must not carry --%s", tc.client, other)
		}
	}
	claude, _, _ := newInstallCmd().Find([]string{"claude-code"})
	for _, flag := range []string{"rule", "agents"} {
		if claude.Flags().Lookup(flag) != nil {
			t.Errorf("`install claude-code` must not carry --%s", flag)
		}
	}
}

func TestInstallSnippetPrintsWithoutWriting(t *testing.T) {
	cfg := filepath.Join(t.TempDir(), "client.json")
	runInstall(t, "cursor", "--rule", "--settings", cfg)
	if _, err := os.Stat(cfg); !os.IsNotExist(err) {
		t.Error("--rule must print the snippet, not install")
	}
}

func TestBareInstallStillMeansClaudeCode(t *testing.T) {
	cmd := newInstallCmd()
	if cmd.RunE == nil {
		t.Fatal("bare `houkai install` lost its Claude Code fallback")
	}
	for _, flag := range []string{"settings", "memory-path", "collection", "binary",
		"project", "verify", "print"} {
		if cmd.Flags().Lookup(flag) == nil {
			t.Errorf("bare `install` is missing the --%s flag", flag)
		}
	}
	if !strings.Contains(cmd.Short, "claude-code") {
		t.Errorf("install Short should still name the clients: %q", cmd.Short)
	}
}
