package cli

import (
	"os"
	"path/filepath"
	"testing"
)

// withEnv sets env vars for the test and restores them afterwards.
func withEnv(t *testing.T, kv map[string]string) {
	t.Helper()
	prev := map[string]string{}
	for k := range kv {
		prev[k] = os.Getenv(k)
	}
	for k, v := range kv {
		if v == "" {
			os.Unsetenv(k)
		} else {
			os.Setenv(k, v)
		}
	}
	t.Cleanup(func() {
		for k, v := range prev {
			if v == "" {
				os.Unsetenv(k)
			} else {
				os.Setenv(k, v)
			}
		}
	})
}

func TestResolveConfigDefaults(t *testing.T) {
	// Isolate from real env + real ~/.config.
	tmpHome := t.TempDir()
	withEnv(t, map[string]string{
		"HOME":                     tmpHome,
		"AI_HOUKAI_PATH":           "",
		"AI_HOUKAI_COLLECTION":     "",
		"AI_HOUKAI_EMBED_PROVIDER": "",
		"AI_HOUKAI_CONFIG":         "",
		"OPENAI_API_KEY":           "",
	})
	cfg := ResolveConfig("", "")
	if cfg.Collection != "ai_houkai" {
		t.Errorf("default collection = %q, want ai_houkai", cfg.Collection)
	}
	if cfg.EmbedProvider != "ollama" {
		t.Errorf("default provider = %q, want ollama", cfg.EmbedProvider)
	}
	if cfg.EmbedDim != 384 {
		t.Errorf("default embed_dim = %d, want 384", cfg.EmbedDim)
	}
}

func TestResolveConfigFlagOverridesEnv(t *testing.T) {
	tmpHome := t.TempDir()
	withEnv(t, map[string]string{
		"HOME":                 tmpHome,
		"AI_HOUKAI_PATH":       "/env/path",
		"AI_HOUKAI_COLLECTION": "env_coll",
		"AI_HOUKAI_CONFIG":     "",
	})
	cfg := ResolveConfig("/flag/path", "flag_coll")
	if cfg.StorePath != "/flag/path" {
		t.Errorf("flag should win for store: got %q", cfg.StorePath)
	}
	if cfg.Collection != "flag_coll" {
		t.Errorf("flag should win for collection: got %q", cfg.Collection)
	}
}

func TestResolveConfigEnvOverridesFileFromExplicitPath(t *testing.T) {
	tmpHome := t.TempDir()
	cfgFile := filepath.Join(tmpHome, "ai_houkai.toml")
	if err := os.WriteFile(cfgFile, []byte(`collection = "from_file"`+"\n"), 0o600); err != nil {
		t.Fatalf("write cfg: %v", err)
	}
	withEnv(t, map[string]string{
		"HOME":                 tmpHome,
		"AI_HOUKAI_CONFIG":     cfgFile,
		"AI_HOUKAI_COLLECTION": "from_env",
		"AI_HOUKAI_PATH":       "",
	})
	cfg := ResolveConfig("", "")
	if cfg.Collection != "from_env" {
		t.Errorf("env should win over file: got %q", cfg.Collection)
	}
}

func TestResolveConfigReadsFile(t *testing.T) {
	tmpHome := t.TempDir()
	cfgFile := filepath.Join(tmpHome, "ai_houkai.toml")
	body := `
collection      = "file_coll"
embed_provider  = "openai"
openai_model    = "text-embedding-3-large"
embed_dim       = 1024
default_importance = 0.7
`
	if err := os.WriteFile(cfgFile, []byte(body), 0o600); err != nil {
		t.Fatalf("write cfg: %v", err)
	}
	withEnv(t, map[string]string{
		"HOME":                     tmpHome,
		"AI_HOUKAI_CONFIG":         cfgFile,
		"AI_HOUKAI_COLLECTION":     "",
		"AI_HOUKAI_EMBED_PROVIDER": "",
		"AI_HOUKAI_PATH":           "",
	})
	cfg := ResolveConfig("", "")
	if cfg.Collection != "file_coll" {
		t.Errorf("collection from file: got %q", cfg.Collection)
	}
	if cfg.EmbedProvider != "openai" {
		t.Errorf("provider from file: got %q", cfg.EmbedProvider)
	}
	if cfg.EmbedDim != 1024 {
		t.Errorf("embed_dim from file: got %d", cfg.EmbedDim)
	}
	if cfg.DefaultImportance != 0.7 {
		t.Errorf("default_importance from file: got %v", cfg.DefaultImportance)
	}
}

func TestEditorFallback(t *testing.T) {
	withEnv(t, map[string]string{"EDITOR": ""})
	if got := editorCmd(Config{}); got != "nano" {
		t.Errorf("fallback editor = %q, want nano", got)
	}
	withEnv(t, map[string]string{"EDITOR": "vim"})
	if got := editorCmd(Config{}); got != "vim" {
		t.Errorf("$EDITOR-derived editor = %q, want vim", got)
	}
	if got := editorCmd(Config{Editor: "code"}); got != "code" {
		t.Errorf("explicit editor should win: got %q", got)
	}
}
