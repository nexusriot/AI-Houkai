package cli

import (
	"github.com/nexusriot/ai-houkai/internal/tui"
	"github.com/spf13/cobra"
)

func newTuiCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "tui",
		Short: "Browse memories interactively (search, detail, link-graph walking)",
		RunE: func(cmd *cobra.Command, _ []string) error {
			store := storeFromCtx(cmd.Context())
			cfg := cfgFromCtx(cmd.Context())
			return tui.New(store, cfg.Collection).Run()
		},
	}
}
