package cli

import (
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"github.com/nexusriot/ai-houkai/internal/httpserver"
	"github.com/spf13/cobra"
)

func newServeCmd() *cobra.Command {
	var host string
	var port int
	var token string

	cmd := &cobra.Command{
		Use:   "serve",
		Short: "Serve the memory store over a JSON HTTP/REST API",
		Long: `Serve the memory store over a JSON HTTP/REST API.

Reuses the store selected by --store/--collection. Examples:

  houkai serve --port 8077
  curl -s localhost:8077/health
  curl -s 'localhost:8077/recall?query=auth&k=3&since=7d'
  curl -s localhost:8077/memories -d '{"text":"remember this","type":"semantic"}'

Pass --token (or set AI_HOUKAI_HTTP_TOKEN) to require
'Authorization: Bearer <token>' on every request except /health. The server
binds 127.0.0.1 by default — set --host 0.0.0.0 only behind a trusted network
or reverse proxy.`,
		RunE: func(cmd *cobra.Command, _ []string) error {
			store := storeFromCtx(cmd.Context())
			cfg := cfgFromCtx(cmd.Context())
			// Attribute journal writes to the HTTP front-end while it owns the store.
			store.AsActor("http")

			if token == "" {
				token = os.Getenv("AI_HOUKAI_HTTP_TOKEN")
			}
			auth := "(open — no auth)"
			if token != "" {
				auth = "(token required)"
			}
			fmt.Fprintf(os.Stderr,
				"AI-Houkai HTTP API → http://%s:%d  %s\nstore: %s  collection: %s\nPress Ctrl-C to stop.\n",
				host, port, auth, cfg.StorePath, cfg.Collection)

			ctx, stop := signal.NotifyContext(cmd.Context(), os.Interrupt, syscall.SIGTERM)
			defer stop()

			srv := httpserver.New(store, cfg.StorePath, cfg.Collection, token)
			return srv.ListenAndServe(ctx, host, port)
		},
	}
	cmd.Flags().StringVarP(&host, "host", "H", "127.0.0.1", "Bind address")
	cmd.Flags().IntVarP(&port, "port", "p", 8077, "Bind port")
	cmd.Flags().StringVar(&token, "token", "",
		"Require 'Authorization: Bearer <token>' on every request except /health (env: AI_HOUKAI_HTTP_TOKEN)")
	return cmd
}
