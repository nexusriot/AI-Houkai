package cli

import (
	"context"
	"fmt"
	"os"

	"github.com/nexusriot/ai-houkai/internal/embed"
	"github.com/nexusriot/ai-houkai/internal/memory"
	"github.com/nexusriot/ai-houkai/internal/vector"
	"github.com/nexusriot/ai-houkai/internal/version"
	"github.com/spf13/cobra"
)

type ctxKey string

const storeKey ctxKey = "store"
const cfgKey ctxKey = "cfg"
const fmtKey ctxKey = "fmt"
const backendKey ctxKey = "backend"

// storeFromCtx retrieves the MemoryStore from context.
func storeFromCtx(ctx context.Context) *memory.MemoryStore {
	return ctx.Value(storeKey).(*memory.MemoryStore)
}

func cfgFromCtx(ctx context.Context) Config {
	return ctx.Value(cfgKey).(Config)
}

func backendFromCtx(ctx context.Context) *vector.ChromemBackend {
	return ctx.Value(backendKey).(*vector.ChromemBackend)
}

func fmtFromCtx(ctx context.Context) OutputFormat {
	v, _ := ctx.Value(fmtKey).(OutputFormat)
	if v == "" {
		return FormatAuto
	}
	return v
}

// NewRootCmd builds and returns the root cobra command.
func NewRootCmd() *cobra.Command {
	var storePath, collection, format string

	root := &cobra.Command{
		Use:     "houkai",
		Short:   "AI-Houkai — persistent memory for AI agents",
		Long:    "houkai gives you direct terminal access to your agent's long-term memory store.",
		Version: version.Version + " (built " + version.BuildTime + ")",
		PersistentPreRunE: func(cmd *cobra.Command, _ []string) error {
			// We're past flag/arg parsing, so any error from here on (or from
			// the command's RunE) is a runtime failure, not misuse — don't dump
			// the usage text after it. Parse/arg errors happen before this hook
			// runs, so they still show usage.
			cmd.SilenceUsage = true

			cfg := ResolveConfig(storePath, collection)

			var embedder embed.Embedder
			switch cfg.EmbedProvider {
			case embed.ProviderOpenAI:
				if cfg.OpenAIKey == "" {
					return fmt.Errorf("OPENAI_API_KEY is required for openai embed provider")
				}
				embedder = embed.NewOpenAI(cfg.OpenAIKey, cfg.OpenAIModel)
			case embed.ProviderDigitalOcean:
				if cfg.DOKey == "" {
					return fmt.Errorf("DIGITALOCEAN_TOKEN (or do_api_key in config) is required for digitalocean embed provider")
				}
				embedder = embed.NewDigitalOcean(cfg.DOKey, cfg.DOModel)
			default: // ollama
				embedder = embed.NewOllama(cfg.OllamaURL, cfg.OllamaModel)
			}

			backend, err := vector.NewChromem(cfg.StorePath, cfg.Collection, cfg.EmbedDim)
			if err != nil {
				return fmt.Errorf("open store: %w", err)
			}

			storeCfg := memory.DefaultStoreConfig(cfg.StorePath, cfg.Collection)
			storeCfg.DefaultImportance = cfg.DefaultImportance.Value
			if cfg.DefaultImportance.Auto {
				storeCfg.ImportanceFn = memory.ScoreImportance
			}
			storeCfg.Actor = "cli"
			storeCfg.EmbeddingModel = embedModelName(cfg)
			store := memory.NewMemoryStore(backend, embedder, storeCfg)

			// Inject into context.
			ctx := context.WithValue(cmd.Context(), storeKey, store)
			ctx = context.WithValue(ctx, cfgKey, cfg)
			ctx = context.WithValue(ctx, fmtKey, OutputFormat(format))
			ctx = context.WithValue(ctx, backendKey, backend)
			cmd.SetContext(ctx)
			return nil
		},
	}

	root.PersistentFlags().StringVarP(&storePath, "store", "s", "", "Path to chromem-go store (overrides config/env)")
	root.PersistentFlags().StringVarP(&collection, "collection", "c", "", "Collection name (overrides config/env)")
	root.PersistentFlags().StringVar(&format, "format", "auto", "Output format: auto|json|tsv")

	root.AddCommand(
		newRememberCmd(),
		newRecallCmd(),
		newPackCmd(),
		newListCmd(),
		newShowCmd(),
		newForgetCmd(),
		newNukeCmd(),
		newEditCmd(),
		newTagCmd(),
		newBumpCmd(),
		newLinkCmd(),
		newUnlinkCmd(),
		newNeighborsCmd(),
		newGraphCmd(),
		newConflictsCmd(),
		newSupersedeCmd(),
		newRestoreCmd(),
		newPruneCmd(),
		newPurgeCmd(),
		newReflectCmd(),
		newMaintenanceCmd(),
		newExportCmd(),
		newImportCmd(),
		newInfoCmd(),
		newBackupCmd(),
		newStatsCmd(),
		newDoctorCmd(),
		newIngestCmd(),
		newServeCmd(),
		newCollectionsCmd(),
		newTuiCmd(),
		newJournalCmd(),
		newInstallCmd(),
		newConfigCmd(),
	)

	return root
}

// embedModelName returns the embedding model name for the active provider
// so the export header records a stable identifier.
func embedModelName(cfg Config) string {
	switch cfg.EmbedProvider {
	case "openai":
		return "openai:" + cfg.OpenAIModel
	case "digitalocean":
		return "digitalocean:" + cfg.DOModel
	default:
		return "ollama:" + cfg.OllamaModel
	}
}

// Execute runs the root command.
func Execute() {
	if err := NewRootCmd().Execute(); err != nil {
		os.Exit(1)
	}
}
