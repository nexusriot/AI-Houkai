package cli

import (
	"os"
	"path/filepath"

	"github.com/BurntSushi/toml"
)

// Config holds resolved CLI configuration.
type Config struct {
	StorePath         string  `toml:"store_path"`
	Collection        string  `toml:"collection"`
	DefaultType       string  `toml:"default_type"`
	DefaultImportance float32 `toml:"default_importance"`
	Editor            string  `toml:"editor"`
	EmbedProvider     string  `toml:"embed_provider"`
	OllamaURL         string  `toml:"ollama_url"`
	OllamaModel       string  `toml:"ollama_model"`
	OpenAIKey         string  `toml:"openai_api_key"`
	OpenAIModel       string  `toml:"openai_model"`
	EmbedDim          int     `toml:"embed_dim"`
}

func defaultConfig() Config {
	home, _ := os.UserHomeDir()
	return Config{
		StorePath:         filepath.Join(home, ".ai_houkai", ".chroma"),
		Collection:        "ai_houkai",
		DefaultType:       "episodic",
		DefaultImportance: 0.5,
		Editor:            "",
		EmbedProvider:     "ollama",
		OllamaURL:         "http://localhost:11434",
		OllamaModel:       "all-minilm",
		OpenAIModel:       "text-embedding-3-small",
		EmbedDim:          384,
	}
}

// configSearchPaths returns the config file locations in resolution order
// (lowest → highest priority). Later files override earlier ones.
func configSearchPaths() []string {
	var paths []string

	// 1. System-wide config (lowest priority).
	paths = append(paths, "/etc/ai-houkai/config.toml")

	// 2. User config (higher priority).
	home, _ := os.UserHomeDir()
	if home != "" {
		paths = append(paths, filepath.Join(home, ".config", "ai_houkai", "config.toml"))
	}

	// 3. Explicit override via env var (highest priority among files).
	if v := os.Getenv("AI_HOUKAI_CONFIG"); v != "" {
		paths = append(paths, v)
	}

	return paths
}

// ResolveConfig merges: built-in defaults → /etc config → user config →
// $AI_HOUKAI_CONFIG → env vars → CLI flags.
func ResolveConfig(storePath, collection string) Config {
	cfg := defaultConfig()

	for _, p := range configSearchPaths() {
		data, err := os.ReadFile(p)
		if err != nil {
			continue
		}
		_ = toml.Unmarshal(data, &cfg)
	}

	// Env vars.
	if v := os.Getenv("AI_HOUKAI_PATH"); v != "" {
		cfg.StorePath = v
	}
	if v := os.Getenv("AI_HOUKAI_COLLECTION"); v != "" {
		cfg.Collection = v
	}
	if v := os.Getenv("AI_HOUKAI_EMBED_PROVIDER"); v != "" {
		cfg.EmbedProvider = v
	}
	if v := os.Getenv("AI_HOUKAI_OLLAMA_URL"); v != "" {
		cfg.OllamaURL = v
	}
	if v := os.Getenv("AI_HOUKAI_OLLAMA_MODEL"); v != "" {
		cfg.OllamaModel = v
	}
	if v := os.Getenv("OPENAI_API_KEY"); v != "" && cfg.OpenAIKey == "" {
		cfg.OpenAIKey = v
	}

	// CLI flag overrides (empty string = not set).
	if storePath != "" {
		cfg.StorePath = storePath
	}
	if collection != "" {
		cfg.Collection = collection
	}

	return cfg
}

func editorCmd(cfg Config) string {
	if cfg.Editor != "" {
		return cfg.Editor
	}
	if e := os.Getenv("EDITOR"); e != "" {
		return e
	}
	return "nano"
}
