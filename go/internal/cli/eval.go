package cli

import (
	"fmt"

	"github.com/nexusriot/ai-houkai/internal/eval"
	"github.com/spf13/cobra"
)

// newEvalCmd scores retrieval quality against a JSONL gold set.
//
// The eval harness existed in Python but was reachable from no surface, and
// had no Go port at all — so the graph damping, the lexical weight and the MMR
// defaults were all tuned by intuition. This is the ruler, in both ports,
// reading the same gold-set format.
func newEvalCmd() *cobra.Command {
	var k int
	var mode, fusion string
	var graph, diversity, dedup, minCosine float32
	var expandRerank, perCase bool

	cmd := &cobra.Command{
		Use:   "eval <goldset.jsonl>",
		Short: "Score retrieval quality against a gold set",
		Long: "Reports recall@k, precision@k, MRR, MAP and nDCG@k over a JSONL gold set:\n\n" +
			`  {"query": "how do we deploy", "relevant_ids": ["<uuid>"]}` + "\n" +
			`  {"query": "test isolation", "relevant_ids": ["<id>", "<id>"], "k": 3}` + "\n\n" +
			"Recall runs read-only, so evaluating never perturbs access-count or recency.\n" +
			"Pass ranking flags to A/B a config: --mode hybrid --fusion rrf --graph 0.15",
		Args: cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			cases, err := eval.LoadGoldset(args[0])
			if err != nil {
				return err
			}
			opts := eval.Options{
				DefaultK:    k,
				DefaultMode: mode,
				Recall:      eval.RecallOpts{Fusion: fusion, ExpandRerank: expandRerank},
			}
			// Only forward a knob the user actually set: a zero value is a
			// meaningful setting for some of these, absence is not.
			if cmd.Flags().Changed("graph") {
				opts.Recall.Graph = &graph
			}
			if cmd.Flags().Changed("diversity") {
				opts.Recall.Diversity = &diversity
			}
			if cmd.Flags().Changed("dedup") {
				opts.Recall.DedupThreshold = &dedup
			}
			if cmd.Flags().Changed("min-cosine") {
				opts.Recall.MinCosine = &minCosine
			}

			store := storeFromCtx(cmd.Context())
			res, err := eval.Evaluate(cmd.Context(), eval.StoreAdapter{Store: store},
				cases, opts)
			if err != nil {
				return err
			}
			if fmtFromCtx(cmd.Context()) == FormatJSON {
				return printJSON(res)
			}

			ks := "mixed"
			if res.K >= 0 {
				ks = fmt.Sprintf("%d", res.K)
			}
			fmt.Printf("Retrieval eval — %s (%d cases)\n", args[0], res.N)
			fmt.Printf("  recall@%s     %.3f\n", ks, res.RecallAtK)
			fmt.Printf("  precision@%s  %.3f\n", ks, res.PrecisionAtK)
			fmt.Printf("  MRR           %.3f\n", res.MRR)
			fmt.Printf("  MAP           %.3f\n", res.MAP)
			fmt.Printf("  nDCG@%s       %.3f\n", ks, res.NDCGAtK)
			if perCase {
				fmt.Println()
				for _, c := range res.PerCase {
					fmt.Printf("  %-50s k=%-3d recall=%.3f RR=%.3f nDCG=%.3f\n",
						truncate(c.Query, 50), c.K, c.RecallAtK, c.RR, c.NDCGAtK)
				}
			}
			return nil
		},
	}
	cmd.Flags().IntVarP(&k, "k", "k", 5, "Default top-k when a case omits it")
	cmd.Flags().StringVar(&mode, "mode", "hybrid", "semantic | hybrid")
	cmd.Flags().StringVar(&fusion, "fusion", "weighted", "weighted | rrf")
	cmd.Flags().Float32Var(&graph, "graph", 0, "Graph-proximity weight (hybrid only)")
	cmd.Flags().Float32Var(&diversity, "diversity", 0, "MMR λ in [0,1]")
	cmd.Flags().Float32Var(&dedup, "dedup", 0, "Drop near-duplicates above this cosine")
	cmd.Flags().Float32Var(&minCosine, "min-cosine", 0, "Absolute relevance floor")
	cmd.Flags().BoolVar(&expandRerank, "expand-rerank", false,
		"Merge graph-expanded nodes into the pool before top-k")
	cmd.Flags().BoolVar(&perCase, "per-case", false, "Show a row per query")
	return cmd
}
