package cli

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/BurntSushi/toml"
)

// ImportanceDefault is a default_importance config value: either a float or
// the literal string "auto", which enables the heuristic importance scorer.
type ImportanceDefault struct {
	Value float32
	Auto  bool
}

// UnmarshalTOML accepts `default_importance = 0.7` and
// `default_importance = "auto"`.
func (d *ImportanceDefault) UnmarshalTOML(v any) error {
	switch x := v.(type) {
	case string:
		if x == "auto" {
			d.Auto = true
			d.Value = 0.5
			return nil
		}
		return fmt.Errorf("default_importance must be a float or \"auto\", got %q", x)
	case float64:
		d.Value = float32(x)
	case int64:
		d.Value = float32(x)
	default:
		return fmt.Errorf("default_importance must be a float or \"auto\", got %T", v)
	}
	return nil
}

// Config holds resolved CLI configuration.
type Config struct {
	StorePath         string            `toml:"store_path"`
	Collection        string            `toml:"collection"`
	DefaultType       string            `toml:"default_type"`
	DefaultImportance ImportanceDefault `toml:"default_importance"`
	Summarizer        string            `toml:"summarizer"` // reflection summarizer spec, e.g. "ollama:llama3.1"
	Editor            string            `toml:"editor"`
	EmbedProvider     string            `toml:"embed_provider"`
	OllamaURL         string            `toml:"ollama_url"`
	OllamaModel       string            `toml:"ollama_model"`
	OpenAIKey         string            `toml:"openai_api_key"`
	OpenAIModel       string            `toml:"openai_model"`
	DOKey             string            `toml:"do_api_key"`
	DOModel           string            `toml:"do_model"`
	EmbedDim          int               `toml:"embed_dim"`
	Maintenance       MaintenanceConfig `toml:"maintenance"`
}

// MaintenanceConfig holds the [maintenance] config block.
type MaintenanceConfig struct {
	Decay        MaintenanceDecayConfig `toml:"decay"`
	IntervalSecs int                    `toml:"interval_secs"` // tick cadence, default 3600
	Reflect      bool                   `toml:"reflect"`       // run reflection each tick
	Consolidate  bool                   `toml:"consolidate"`   // soft-consolidate on reflection
	StatePath    string                 `toml:"state_path"`    // default: <store-dir>/maintenance_state.json
	PidPath      string                 `toml:"pid_path"`      // default: <store-dir>/maintenance.pid
	LogPath      string                 `toml:"log_path"`      // default: <store-dir>/maintenance.log
}

// MaintPaths returns the resolved state/pid/log paths, defaulting to files
// alongside the store directory.
func (c Config) MaintPaths() (statePath, pidPath, logPath string) {
	dir := filepath.Dir(c.StorePath)
	statePath = c.Maintenance.StatePath
	if statePath == "" {
		statePath = filepath.Join(dir, "maintenance_state.json")
	}
	pidPath = c.Maintenance.PidPath
	if pidPath == "" {
		pidPath = filepath.Join(dir, "maintenance.pid")
	}
	logPath = c.Maintenance.LogPath
	if logPath == "" {
		logPath = filepath.Join(dir, "maintenance.log")
	}
	return statePath, pidPath, logPath
}

// MaintenanceDecayConfig holds [maintenance.decay]: the effective decay/prune
// parameters used by `houkai prune`, the maintenance daemon, and the
// `stats --health` report so all three agree on what would be pruned.
type MaintenanceDecayConfig struct {
	DecayRate       float32  `toml:"decay_rate"`
	MinScore        float32  `toml:"min_score"`
	ProtectTypes    []string `toml:"protect_types"`
	FrequencyWeight float32  `toml:"frequency_weight"`
}

func defaultConfig() Config {
	home, _ := os.UserHomeDir()
	return Config{
		StorePath:         filepath.Join(home, ".ai_houkai", ".chroma"),
		Collection:        "ai_houkai",
		DefaultType:       "episodic",
		DefaultImportance: ImportanceDefault{Value: 0.5},
		Editor:            "",
		EmbedProvider:     "ollama",
		OllamaURL:         "http://localhost:11434",
		OllamaModel:       "all-minilm",
		OpenAIModel:       "text-embedding-3-small",
		DOModel:           "qwen3-embedding-0.6b",
		EmbedDim:          384,
		Maintenance: MaintenanceConfig{
			Decay: MaintenanceDecayConfig{
				DecayRate:       0.1,
				MinScore:        0.05,
				ProtectTypes:    []string{"procedural"},
				FrequencyWeight: 0.0,
			},
			IntervalSecs: 3600,
		},
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
	if v := os.Getenv("DIGITALOCEAN_TOKEN"); v != "" && cfg.DOKey == "" {
		cfg.DOKey = v
	}
	if v := os.Getenv("AI_HOUKAI_DO_MODEL"); v != "" {
		cfg.DOModel = v
	}
	if v := os.Getenv("AI_HOUKAI_SUMMARIZER"); v != "" {
		cfg.Summarizer = v
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
