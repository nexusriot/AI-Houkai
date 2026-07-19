package cli

// Ingest command — chunk files (or stdin) into memories.

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/nexusriot/ai-houkai/internal/ingest"
	"github.com/nexusriot/ai-houkai/internal/memory"
	"github.com/spf13/cobra"
)

func newIngestCmd() *cobra.Command {
	var memType, source string
	var tags []string
	var importance float32
	var autoImportance, dryRun, yes bool
	var maxChars, minChars int

	cmd := &cobra.Command{
		Use:   "ingest [files...]",
		Short: "Split files into chunks and store each chunk as a memory",
		Long: `Split files into chunks and store each chunk as a memory.

Splits on blank lines, keeps markdown headings attached to their
paragraph, re-packs long paragraphs on sentence boundaries. Omit files
(or pass '-') to read stdin.`,
		RunE: func(cmd *cobra.Command, args []string) error {
			store := storeFromCtx(cmd.Context())
			cfg := cfgFromCtx(cmd.Context())

			// (label, text) per input.
			type input struct{ label, body string }
			var inputs []input
			if len(args) == 0 || (len(args) == 1 && args[0] == "-") {
				body, err := io.ReadAll(os.Stdin)
				if err != nil {
					return err
				}
				if strings.TrimSpace(string(body)) == "" {
					return fmt.Errorf("no input provided")
				}
				inputs = append(inputs, input{"stdin", string(body)})
			} else {
				for _, f := range args {
					data, err := os.ReadFile(f)
					if err != nil {
						return fmt.Errorf("%s not found", f)
					}
					inputs = append(inputs, input{filepath.Base(f), string(data)})
				}
			}

			explicit := cmd.Flags().Changed("importance")
			auto := autoImportance || (!explicit && cfg.DefaultImportance.Auto)

			type planned struct {
				label, chunk string
				imp          float32
			}
			var plan []planned
			for _, in := range inputs {
				for _, chunk := range ingest.ChunkText(in.body, maxChars, minChars) {
					var imp float32
					switch {
					case explicit:
						imp = importance
					case auto:
						imp = memory.ScoreImportance(chunk, memory.MemoryType(memType), tags)
					default:
						imp = cfg.DefaultImportance.Value
					}
					plan = append(plan, planned{in.label, chunk, imp})
				}
			}

			if len(plan) == 0 {
				fmt.Println("Nothing to ingest (all chunks below --min-chars?).")
				return nil
			}

			for i, p := range plan {
				firstLine := strings.SplitN(p.chunk, "\n", 2)[0]
				if len(firstLine) > 70 {
					firstLine = firstLine[:70]
				}
				fmt.Printf("  [%3d] %.2f  %s: %s\n", i+1, p.imp, p.label, firstLine)
			}
			fmt.Printf("\n%d chunk(s) from %d input(s).\n", len(plan), len(inputs))

			if dryRun {
				fmt.Println("Dry-run — nothing written.")
				return nil
			}

			if !yes && !Confirm(fmt.Sprintf("Store %d memories?", len(plan))) {
				fmt.Println("Aborted.")
				return nil
			}

			defer store.AsActor("import")()
			// One batched write — collapses N per-chunk encodes into
			// ceil(N/batch) (see MemoryStore.RememberMany). Ingesting raw
			// document chunks shouldn't trigger conflict management, so ignore.
			batch := make([]memory.RememberItem, len(plan))
			for i, p := range plan {
				src := source
				if src == "" {
					src = "ingest:" + p.label
				}
				batch[i] = memory.RememberItem{
					Text: p.chunk,
					RememberOpts: memory.RememberOpts{
						Type:       memory.MemoryType(memType),
						Tags:       tags,
						Importance: memory.Float32Ptr(p.imp),
						Source:     src,
					},
				}
			}
			if _, err := store.RememberMany(cmd.Context(), batch, 128, memory.PolicyIgnore); err != nil {
				return err
			}
			fmt.Printf("Stored %d memories.\n", len(plan))
			return nil
		},
	}
	cmd.Flags().StringVarP(&memType, "type", "t", "episodic", "episodic|semantic|procedural|feedback")
	cmd.Flags().StringSliceVarP(&tags, "tag", "g", nil, "Tag (repeatable)")
	// No -s shorthand: it would collide with the persistent --store flag.
	cmd.Flags().StringVar(&source, "source", "", "Source label; default ingest:<filename> (or ingest:stdin)")
	cmd.Flags().Float32VarP(&importance, "importance", "i", 0, "Importance 0.0–1.0")
	cmd.Flags().BoolVar(&autoImportance, "auto-importance", false, "Score each chunk heuristically")
	cmd.Flags().IntVar(&maxChars, "max-chars", 500, "Max chunk size")
	cmd.Flags().IntVar(&minChars, "min-chars", 30, "Drop chunks shorter than this")
	cmd.Flags().BoolVar(&dryRun, "dry-run", false, "Show chunks without writing")
	cmd.Flags().BoolVarP(&yes, "yes", "y", false, "Skip confirmation")
	return cmd
}
