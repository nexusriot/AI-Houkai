// Command ai-houkai-serve starts the HTTP/REST API server, configured entirely
// through the environment so it needs no CLI flags — symmetric with
// ai-houkai-mcp.
//
//	AI_HOUKAI_PATH         chromem-go store path    (default from config)
//	AI_HOUKAI_COLLECTION   collection name          (default from config)
//	AI_HOUKAI_HTTP_HOST    bind address             (default 127.0.0.1)
//	AI_HOUKAI_HTTP_PORT    bind port                (default 8077)
//	AI_HOUKAI_HTTP_TOKEN   optional bearer token
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"
	"strconv"
	"syscall"

	"github.com/nexusriot/ai-houkai/internal/cli"
	"github.com/nexusriot/ai-houkai/internal/embed"
	"github.com/nexusriot/ai-houkai/internal/httpserver"
	"github.com/nexusriot/ai-houkai/internal/memory"
	"github.com/nexusriot/ai-houkai/internal/vector"
)

func main() {
	// Same resolution chain as the CLI/MCP: defaults → /etc → user → env.
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

	storeCfg := memory.DefaultStoreConfig(cfg.StorePath, cfg.Collection)
	storeCfg.DefaultImportance = cfg.DefaultImportance.Value
	if cfg.DefaultImportance.Auto {
		storeCfg.ImportanceFn = memory.ScoreImportance
	}
	storeCfg.Actor = "http"
	store := memory.NewMemoryStore(backend, embedder, storeCfg)

	host := envOr("AI_HOUKAI_HTTP_HOST", "127.0.0.1")
	port := 8077
	if v := os.Getenv("AI_HOUKAI_HTTP_PORT"); v != "" {
		if n, perr := strconv.Atoi(v); perr == nil {
			port = n
		}
	}
	token := os.Getenv("AI_HOUKAI_HTTP_TOKEN")

	auth := "(open — no auth)"
	if token != "" {
		auth = "(token required)"
	}
	fmt.Fprintf(os.Stderr, "AI-Houkai HTTP API → http://%s:%d  %s\nstore: %s  collection: %s\n",
		host, port, auth, cfg.StorePath, cfg.Collection)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	srv := httpserver.New(store, cfg.StorePath, cfg.Collection, token)
	if err := srv.ListenAndServe(ctx, host, port); err != nil {
		log.Fatalf("http server: %v", err)
	}
}

func envOr(name, def string) string {
	if v := os.Getenv(name); v != "" {
		return v
	}
	return def
}
