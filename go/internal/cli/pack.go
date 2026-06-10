package cli

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/nexusriot/ai-houkai/internal/memory"
	"github.com/spf13/cobra"
)

func newPackCmd() *cobra.Command {
	var budget, maxItems int
	var memType, tag, mode, header, format string
	var minImp float32
	var inclSup bool

	cmd := &cobra.Command{
		Use:   "pack <query>",
		Short: "Assemble the most relevant memories into a token-budgeted context block",
		Long: `Assemble the most relevant memories into a token-budgeted context block.

Ranks with hybrid scoring by default, then greedily packs results until the
token budget is reached. The block is printed to stdout (pipe it into a
prompt); a one-line summary goes to stderr.`,
		Args: cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			store := storeFromCtx(cmd.Context())
			result, err := store.RecallPack(cmd.Context(), args[0], memory.PackOpts{
				TokenBudget:       budget,
				Type:              memory.MemoryType(memType),
				Tag:               tag,
				MinImportance:     minImp,
				Mode:              memory.RecallMode(mode),
				MaxItems:          maxItems,
				IncludeSuperseded: inclSup,
				Header:            &header,
			})
			if err != nil {
				return err
			}

			if format == "json" {
				type item struct {
					ID         string   `json:"id"`
					Text       string   `json:"text"`
					Type       string   `json:"type"`
					Tags       []string `json:"tags"`
					Importance float32  `json:"importance"`
					Score      float32  `json:"score"`
					Tokens     int      `json:"tokens"`
				}
				items := make([]item, len(result.Items))
				for i, p := range result.Items {
					items[i] = item{
						ID:         p.Memory.ID,
						Text:       p.Memory.Text,
						Type:       string(p.Memory.Type),
						Tags:       p.Memory.Tags,
						Importance: p.Memory.Importance,
						Score:      p.Score,
						Tokens:     p.Tokens,
					}
				}
				b, _ := json.MarshalIndent(map[string]any{
					"text":        result.Text,
					"used_tokens": result.UsedTokens,
					"budget":      result.Budget,
					"truncated":   result.Truncated,
					"items":       items,
				}, "", "  ")
				fmt.Println(string(b))
				return nil
			}

			if len(result.Items) == 0 {
				fmt.Fprintln(os.Stderr, "No memories found.")
				return nil
			}

			fmt.Println(result.Text)
			truncNote := ""
			if result.Truncated {
				truncNote = " · truncated"
			}
			fmt.Fprintf(os.Stderr, "[%d memories · %d/%d tokens%s]\n",
				len(result.Items), result.UsedTokens, result.Budget, truncNote)
			return nil
		},
	}
	cmd.Flags().IntVarP(&budget, "budget", "b", 800, "Token budget for the packed block")
	cmd.Flags().StringVarP(&memType, "type", "t", "", "Filter by memory type")
	cmd.Flags().StringVarP(&tag, "tag", "g", "", "Filter by tag")
	cmd.Flags().Float32Var(&minImp, "min-importance", 0, "Minimum importance")
	cmd.Flags().StringVar(&mode, "mode", "hybrid", "semantic|hybrid")
	cmd.Flags().IntVar(&maxItems, "max-items", 50, "Ranked candidates to consider")
	cmd.Flags().BoolVar(&inclSup, "include-superseded", false, "Include superseded memories")
	cmd.Flags().StringVar(&header, "header", "## Relevant memory", "Block header (empty string to omit)")
	cmd.Flags().StringVarP(&format, "output", "f", "text", "Output format: text|json")
	return cmd
}
