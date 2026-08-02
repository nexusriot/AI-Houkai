package cli

import (
	"fmt"

	"github.com/spf13/cobra"
)

// newReindexCmd rebuilds the derived SQLite sidecar index.
//
// Needed when enabling `index = "sqlite"` on an existing store (nothing has
// been indexed yet), and the only way back after the index has been disabled —
// which happens whenever it disagrees with the backend, so that a stale index
// makes reads slower rather than wrong.
func newReindexCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "reindex",
		Short: "Rebuild the sidecar index from the vector store",
		RunE: func(cmd *cobra.Command, _ []string) error {
			store := storeFromCtx(cmd.Context())
			if store.Index() == nil {
				// The root command builds the store from config, so an
				// unconfigured index means there is nothing to rebuild — but
				// the user clearly wants one, so build it here rather than
				// making them edit config first.
				if err := store.EnableIndex(cmd.Context(), ""); err != nil {
					return err
				}
			}
			res, err := store.Reindex(cmd.Context())
			if err != nil {
				return err
			}
			if fmtFromCtx(cmd.Context()) == FormatJSON {
				return printJSON(res)
			}
			if !res.Enabled {
				return fmt.Errorf("%s", res.Error)
			}
			fmt.Printf("Indexed %d memories → %s\n", res.Indexed, res.Path)
			fts := "no (SQLite built without FTS5)"
			if res.FTS {
				fts = "yes"
			}
			fmt.Printf("  full-text search: %s\n", fts)
			if !res.Healthy {
				return fmt.Errorf("index still unhealthy — %s", res.Error)
			}
			return nil
		},
	}
}
