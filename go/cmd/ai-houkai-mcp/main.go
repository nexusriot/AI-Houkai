package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/mark3labs/mcp-go/server"
	"github.com/nexusriot/ai-houkai/internal/cli"
	"github.com/nexusriot/ai-houkai/internal/embed"
	"github.com/nexusriot/ai-houkai/internal/mcpserver"
	"github.com/nexusriot/ai-houkai/internal/memory"
	"github.com/nexusriot/ai-houkai/internal/vector"
)

func main() {
	// Use the same resolution chain as the CLI: defaults → /etc config →
	// user config → env vars. CLI flags don't apply here (MCP runs headless).
	cfg := cli.ResolveConfig("", "")

	var embedder embed.Embedder
	switch cfg.EmbedProvider {
	case embed.ProviderOpenAI:
		key := cfg.OpenAIKey
		if key == "" {
			key = os.Getenv("OPENAI_API_KEY")
		}
		if key == "" {
			log.Fatal("OPENAI_API_KEY required for openai embed provider")
		}
		embedder = embed.NewOpenAI(key, cfg.OpenAIModel)
	case embed.ProviderDigitalOcean:
		key := cfg.DOKey
		if key == "" {
			key = os.Getenv("DIGITALOCEAN_TOKEN")
		}
		if key == "" {
			log.Fatal("DIGITALOCEAN_TOKEN required for digitalocean embed provider")
		}
		embedder = embed.NewDigitalOcean(key, cfg.DOModel)
	default:
		embedder = embed.NewOllama(cfg.OllamaURL, cfg.OllamaModel)
	}

	backend, err := vector.NewChromem(cfg.StorePath, cfg.Collection, cfg.EmbedDim)
	if err != nil {
		log.Fatalf("open store: %v", err)
	}
	defer backend.Close()

	// Warm up embedder to set dimensions.
	if _, err := embedder.Embed(context.Background(), []string{"warmup"}); err != nil {
		fmt.Fprintf(os.Stderr, "warning: embed warmup failed: %v\n", err)
	}

	storeCfg := memory.DefaultStoreConfig(cfg.StorePath, cfg.Collection)
	storeCfg.DefaultImportance = cfg.DefaultImportance
	store := memory.NewMemoryStore(backend, embedder, storeCfg)

	s := mcpserver.New(store, cfg.StorePath, cfg.Collection)
	if err := server.ServeStdio(s); err != nil {
		log.Fatalf("MCP server: %v", err)
	}
}
