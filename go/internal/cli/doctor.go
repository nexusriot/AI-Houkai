package cli

import (
	"encoding/json"
	"errors"
	"fmt"
	"sort"

	"github.com/spf13/cobra"
)

func newDoctorCmd() *cobra.Command {
	var asJSON bool
	cmd := &cobra.Command{
		Use:   "doctor",
		Short: "Diagnose store, embedding backend, and configuration health",
		Long: "Actively probes the embedding backend (otherwise only contacted lazily on\n" +
			"first use), checks the store is reachable, guards against an embedding-\n" +
			"dimension mismatch, and reports the resolved configuration. Exits non-zero\n" +
			"if any check fails, so it doubles as a scriptable readiness gate.",
		RunE: func(cmd *cobra.Command, _ []string) error {
			ctx := cmd.Context()
			store := storeFromCtx(ctx)
			cfg := cfgFromCtx(ctx)

			// Each check is a flat map {"name","ok", ...detail} — the same shape
			// the Python port's `doctor --json` emits, so a monitoring script
			// works against either port unchanged.
			var checks []map[string]any
			add := func(name string, ok bool, detail map[string]any) {
				c := map[string]any{"name": name, "ok": ok}
				for k, v := range detail {
					if k != "name" && k != "ok" && v != nil {
						c[k] = v
					}
				}
				checks = append(checks, c)
			}

			// 1. Resolved configuration.
			add("config", true, map[string]any{
				"store_path":     cfg.StorePath,
				"collection":     cfg.Collection,
				"embed_provider": cfg.EmbedProvider,
				"embed_model":    embedModelName(cfg),
				"embed_dim":      cfg.EmbedDim,
			})

			// 2. Store reachable.
			if n, err := store.Count(ctx); err != nil {
				add("store", false, map[string]any{"error": err.Error()})
			} else {
				add("store", true, map[string]any{"count": n})
			}

			// 3. Embedding backend reachability + dimension + latency. Strip the
			//    probe's own "ok" so it is not duplicated at both levels.
			probe := store.ProbeEmbedding(ctx)
			probeOK, _ := probe["ok"].(bool)
			detail := map[string]any{"model": embedModelName(cfg)}
			for k, v := range probe {
				if k != "ok" {
					detail[k] = v
				}
			}
			add("embedder", probeOK, detail)

			// 4. Embed-dim guardrail: the probe's dimension must match the
			//    configured dimension the backend was created with.
			if probeOK {
				if pd, ok := probe["dim"].(int); ok {
					add("embed_dim", pd == cfg.EmbedDim, map[string]any{
						"probe_dim": pd, "configured_dim": cfg.EmbedDim,
					})
				}
			}

			// 5. Audit journal reachable.
			add("journal", true, map[string]any{"enabled": store.Journal() != nil})

			ok := true
			for _, c := range checks {
				if b, _ := c["ok"].(bool); !b {
					ok = false
				}
			}

			if asJSON || fmtFromCtx(ctx) == FormatJSON {
				b, _ := json.MarshalIndent(map[string]any{"ok": ok, "checks": checks}, "", "  ")
				fmt.Println(string(b))
			} else {
				for _, c := range checks {
					mark := "[ok]"
					if b, _ := c["ok"].(bool); !b {
						mark = "[!!]"
					}
					// Sorted keys so the report is byte-stable run-to-run.
					keys := make([]string, 0, len(c))
					for k := range c {
						if k != "name" && k != "ok" {
							keys = append(keys, k)
						}
					}
					sort.Strings(keys)
					fmt.Printf("%s %-11s", mark, c["name"])
					for _, k := range keys {
						fmt.Printf(" %s=%v", k, c[k])
					}
					fmt.Println()
				}
				fmt.Println()
				if ok {
					fmt.Println("OK — ready")
				} else {
					fmt.Println("NOT READY — see failures above")
				}
			}

			if !ok {
				return errors.New("diagnostics failed")
			}
			return nil
		},
	}
	cmd.Flags().BoolVar(&asJSON, "json", false, "Emit the report as JSON")
	return cmd
}
