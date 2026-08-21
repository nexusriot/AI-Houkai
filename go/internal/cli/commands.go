package cli

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/nexusriot/ai-houkai/internal/decay"
	"github.com/nexusriot/ai-houkai/internal/memory"
	reflectpkg "github.com/nexusriot/ai-houkai/internal/reflect"
	"github.com/nexusriot/ai-houkai/internal/timeparse"
	"github.com/spf13/cobra"
)

func newRememberCmd() *cobra.Command {
	var tags []string
	var importance float32
	var memType, source, onConflict string
	var polarity int
	var autoImportance, stdin bool
	var ttlSeconds float64
	var pinned, idempotent bool
	var trust string

	cmd := &cobra.Command{
		Use:   "remember <text>",
		Short: "Store a new memory",
		Args:  cobra.MaximumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			var text string
			if len(args) > 0 && !stdin {
				text = args[0]
			} else {
				// Read from stdin (no positional arg, or --stdin given).
				scanner := bufio.NewScanner(os.Stdin)
				var lines []string
				for scanner.Scan() {
					lines = append(lines, scanner.Text())
				}
				text = strings.Join(lines, "\n")
			}
			text = strings.TrimRight(text, "\n")
			if text == "" {
				return fmt.Errorf("text is required")
			}
			if polarity < -1 || polarity > 1 {
				return fmt.Errorf("polarity must be -1, 0, or 1")
			}
			store := storeFromCtx(cmd.Context())
			cfg := cfgFromCtx(cmd.Context())

			// Honor default_type from config when --type is not given.
			if !cmd.Flags().Changed("type") && cfg.DefaultType != "" {
				memType = cfg.DefaultType
			}

			// Explicit -i wins (including 0); --auto-importance (or
			// default_importance = "auto" in config) scores heuristically;
			// else nil = unset → the store default from config applies.
			var imp *float32
			if cmd.Flags().Changed("importance") {
				imp = memory.Float32Ptr(importance)
			} else if autoImportance || cfg.DefaultImportance.Auto {
				imp = memory.Float32Ptr(memory.ScoreImportance(text, memory.MemoryType(memType), tags))
			}
			opts := memory.RememberOpts{
				Type:       memory.MemoryType(memType),
				Tags:       tags,
				Importance: imp,
				Source:     source,
				Polarity:   polarity,
				OnConflict: memory.ConflictPolicy(onConflict),
				Pinned:     pinned,
				Trust:      memory.TrustLevel(trust),
				Idempotent: idempotent,
			}
			if cmd.Flags().Changed("ttl") {
				opts.TTLSeconds = &ttlSeconds
			}
			m, stored, conflicts, err := store.Remember(cmd.Context(), text, opts)
			if err != nil {
				if ce, ok := err.(*memory.ConflictError); ok {
					fmt.Printf("not stored: %d conflict(s) detected (use --on-conflict to override)\n", len(ce.Conflicts))
					return nil
				}
				return err
			}
			if !stored {
				fmt.Printf("not stored: %d conflict(s)\n", len(conflicts))
				return nil
			}
			fmt.Printf("stored %s (importance %.2f)\n", m.ID, m.Importance)
			return nil
		},
	}
	// Shorthands match Python (and the export command): -t = type, -g = tag.
	cmd.Flags().StringSliceVarP(&tags, "tag", "g", nil, "Tags (comma-separated or repeated)")
	cmd.Flags().Float32VarP(&importance, "importance", "i", 0.5, "Importance 0.0–1.0")
	cmd.Flags().StringVarP(&memType, "type", "t", "", "Memory type: episodic|semantic|procedural|feedback (default from config default_type, \"semantic\")")
	cmd.Flags().StringVar(&source, "source", "", "Provenance label")
	cmd.Flags().IntVar(&polarity, "polarity", 0, "Polarity: -1, 0, or 1")
	cmd.Flags().StringVar(&onConflict, "on-conflict", "", "Conflict policy for this write: ignore|warn|supersede|raise")
	cmd.Flags().BoolVar(&stdin, "stdin", false, "Read the memory text from stdin")
	cmd.Flags().BoolVar(&autoImportance, "auto-importance", false,
		"Score importance heuristically from the text (also: default_importance = \"auto\" in config.toml)")
	cmd.Flags().Float64Var(&ttlSeconds, "ttl", 0, "Time-to-live in seconds; the memory expires (and is hidden from recall) after this")
	cmd.Flags().BoolVar(&pinned, "pin", false,
		"Standing instruction: always offered to `pack --include-pinned`, never pruned by decay")
	cmd.Flags().StringVar(&trust, "trust", "",
		"trusted|reported|untrusted — how much the memory's ORIGIN is trusted. Use untrusted for anything read from content you did not author")
	cmd.Flags().BoolVar(&idempotent, "idempotent", false,
		"No-op if a live memory already has the same normalised text")
	return cmd
}

func newRecallCmd() *cobra.Command {
	var k int
	var tag, memType, mode, source, since, until, minTrust string
	var minImp float32
	var inclSup, inclExp bool

	cmd := &cobra.Command{
		Use:   "recall <query>",
		Short: "Search memories",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			store := storeFromCtx(cmd.Context())
			sinceTS, _, err := timeparse.Parse(since)
			if err != nil {
				return err
			}
			untilTS, _, err := timeparse.Parse(until)
			if err != nil {
				return err
			}
			opts := memory.RecallOpts{
				Type:              memory.MemoryType(memType),
				Tag:               tag,
				MinImportance:     minImp,
				Mode:              memory.RecallMode(mode),
				Overfetch:         4,
				IncludeSuperseded: inclSup,
				IncludeExpired:    inclExp,
				Source:            source,
				Since:             sinceTS,
				Until:             untilTS,
				MinTrust:          memory.TrustLevel(minTrust),
			}
			results, err := store.Recall(cmd.Context(), args[0], k, opts)
			if err != nil {
				return err
			}
			rows := make([]MemRow, len(results))
			for i, r := range results {
				rows[i] = MemRow{
					ID:           r.ID,
					Text:         r.Text,
					Type:         string(r.Type),
					Tags:         r.Tags,
					Importance:   r.Importance,
					Score:        r.Score,
					CreatedAt:    r.CreatedAt,
					SupersededBy: r.SupersededBy,
				}
			}
			PrintRows(os.Stdout, rows, fmtFromCtx(cmd.Context()))
			return nil
		},
	}
	cmd.Flags().IntVarP(&k, "limit", "k", 5, "Max results")
	cmd.Flags().StringVar(&tag, "tag", "", "Filter by tag")
	cmd.Flags().StringVar(&memType, "type", "", "Filter by memory type")
	cmd.Flags().Float32Var(&minImp, "min-importance", 0, "Minimum importance")
	cmd.Flags().StringVar(&mode, "mode", "semantic", "Scoring mode: semantic|hybrid")
	cmd.Flags().BoolVar(&inclSup, "include-superseded", false, "Include superseded memories")
	cmd.Flags().BoolVar(&inclExp, "include-expired", false, "Include memories whose TTL has passed")
	cmd.Flags().StringVar(&minTrust, "min-trust", "",
		"trusted|reported|untrusted — keep only memories at least this trusted")
	cmd.Flags().StringVar(&source, "source", "", "Filter by exact provenance string")
	cmd.Flags().StringVar(&since, "since", "", "Only memories created at/after (ISO date, epoch, or '7d')")
	cmd.Flags().StringVar(&until, "until", "", "Only memories created at/before (ISO date, epoch, or '7d')")
	return cmd
}

func newListCmd() *cobra.Command {
	var limit int
	var inclSup, inclExp bool
	var tag, memType, sortBy, since string

	cmd := &cobra.Command{
		Use:   "list",
		Short: "List recent memories",
		RunE: func(cmd *cobra.Command, _ []string) error {
			if sortBy != "created" && sortBy != "importance" {
				return fmt.Errorf("--sort must be 'created' or 'importance', got %q", sortBy)
			}
			sinceTS, _, err := timeparse.Parse(since)
			if err != nil {
				return err
			}
			store := storeFromCtx(cmd.Context())
			// Fetch unbounded, then filter/sort/limit client-side (matching Python).
			mems, err := store.ListRecent(cmd.Context(), 0, inclSup, inclExp)
			if err != nil {
				return err
			}
			var filtered []memory.Memory
			for _, m := range mems {
				if tag != "" && !containsTag(m.Tags, tag) {
					continue
				}
				if memType != "" && string(m.Type) != memType {
					continue
				}
				if sinceTS > 0 && m.CreatedAt < sinceTS {
					continue
				}
				filtered = append(filtered, m)
			}
			if sortBy == "importance" {
				sortByImportance(filtered)
			}
			if limit > 0 && len(filtered) > limit {
				filtered = filtered[:limit]
			}
			rows := make([]MemRow, len(filtered))
			for i, m := range filtered {
				rows[i] = MemRow{
					ID:           m.ID,
					Text:         m.Text,
					Type:         string(m.Type),
					Tags:         m.Tags,
					Importance:   m.Importance,
					CreatedAt:    m.CreatedAt,
					SupersededBy: m.SupersededBy,
				}
			}
			PrintRows(os.Stdout, rows, fmtFromCtx(cmd.Context()))
			return nil
		},
	}
	cmd.Flags().IntVar(&limit, "limit", 20, "Max results")
	cmd.Flags().BoolVar(&inclSup, "include-superseded", false, "Include superseded")
	cmd.Flags().BoolVar(&inclExp, "include-expired", false, "Include memories whose TTL has passed")
	cmd.Flags().StringVar(&tag, "tag", "", "Filter by tag")
	cmd.Flags().StringVar(&memType, "type", "", "Filter by type")
	cmd.Flags().StringVar(&sortBy, "sort", "created", "Sort order: created|importance")
	cmd.Flags().StringVar(&since, "since", "", "Only memories created at/after (ISO date, epoch, or '7d')")
	return cmd
}

// sortByImportance sorts descending by importance, then created_at desc.
func sortByImportance(mems []memory.Memory) {
	sort.SliceStable(mems, func(i, j int) bool {
		if mems[i].Importance != mems[j].Importance {
			return mems[i].Importance > mems[j].Importance
		}
		return mems[i].CreatedAt > mems[j].CreatedAt
	})
}

func newShowCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "show <id>",
		Short: "Show full details of a memory",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			store := storeFromCtx(cmd.Context())
			m, err := store.GetByID(cmd.Context(), args[0])
			if err != nil {
				return err
			}
			b, _ := json.MarshalIndent(m, "", "  ")
			fmt.Println(string(b))
			return nil
		},
	}
}

func newForgetCmd() *cobra.Command {
	var yes bool
	cmd := &cobra.Command{
		Use:   "forget <id> [id...]",
		Short: "Delete one or more memories",
		Args:  cobra.MinimumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			prompt := "Delete memory " + args[0] + "?"
			if len(args) > 1 {
				prompt = fmt.Sprintf("Delete %d memories?", len(args))
			}
			if !yes && !Confirm(prompt) {
				fmt.Println("aborted")
				return nil
			}
			store := storeFromCtx(cmd.Context())
			deleted := 0
			for _, id := range args {
				ok, err := store.Forget(cmd.Context(), id)
				if err != nil {
					return err
				}
				if ok {
					deleted++
				}
			}
			fmt.Printf("Deleted %d/%d\n", deleted, len(args))
			return nil
		},
	}
	cmd.Flags().BoolVarP(&yes, "yes", "y", false, "Skip confirmation")
	return cmd
}

func newNukeCmd() *cobra.Command {
	var yes bool
	cmd := &cobra.Command{
		Use:   "nuke",
		Short: "Delete EVERY memory in the current collection (irreversible)",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			store := storeFromCtx(cmd.Context())
			count, err := store.Count(cmd.Context())
			if err != nil {
				return err
			}
			if count == 0 {
				fmt.Println("Collection is already empty.")
				return nil
			}
			collection := cfgFromCtx(cmd.Context()).Collection
			if !yes && !Confirm(fmt.Sprintf("Destroy all %d memories in %q?", count, collection)) {
				fmt.Println("Aborted.")
				return nil
			}
			deleted, err := store.Nuke(cmd.Context())
			if err != nil {
				return err
			}
			fmt.Printf("Nuked %d memories.\n", deleted)
			return nil
		},
	}
	cmd.Flags().BoolVarP(&yes, "yes", "y", false, "Skip confirmation")
	return cmd
}

func newEditCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "edit <id>",
		Short: "Edit a memory's fields (type/importance/tags/source/polarity + text) in $EDITOR",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			store := storeFromCtx(cmd.Context())
			cfg := cfgFromCtx(cmd.Context())
			m, err := store.GetByID(cmd.Context(), args[0])
			if err != nil {
				return err
			}

			// Present a YAML-ish front-matter block: editable fields above a
			// "---" separator, the memory text below it.
			front := fmt.Sprintf(
				"# Edit memory — save and close to apply. Lines starting with # are ignored.\n"+
					"id: %s\ntype: %s\nimportance: %g\ntags: %s\nsource: %s\npolarity: %d\n---\n%s\n",
				m.ID, m.Type, m.Importance, strings.Join(m.Tags, ", "), m.Source, m.Polarity, m.Text)

			tmp, err := os.CreateTemp("", "houkai_edit-*.md")
			if err != nil {
				return err
			}
			_ = os.WriteFile(tmp.Name(), []byte(front), 0o600)
			tmp.Close()
			defer os.Remove(tmp.Name())

			c := exec.Command(editorCmd(cfg), tmp.Name())
			c.Stdin, c.Stdout, c.Stderr = os.Stdin, os.Stdout, os.Stderr
			if err := c.Run(); err != nil {
				return err
			}

			raw, err := os.ReadFile(tmp.Name())
			if err != nil {
				return err
			}
			if !strings.Contains(string(raw), "---") {
				return fmt.Errorf("missing '---' separator in edited file")
			}
			frontPart, bodyPart, _ := strings.Cut(string(raw), "---")
			newText := strings.TrimSpace(bodyPart)
			textChanged := newText != m.Text

			// Parse front-matter into EditOpts, skipping comment lines. The
			// body may contain markdown headings, so only the front matter
			// drops '#' lines. store.Edit keeps the id, re-embeds when the
			// text changed, preserves links / superseded_by / access
			// tracking, and journals the change so `journal undo` works.
			opts := memory.EditOpts{Text: &newText}
			for _, line := range strings.Split(frontPart, "\n") {
				if strings.HasPrefix(line, "#") || !strings.Contains(line, ":") {
					continue
				}
				k, v, _ := strings.Cut(line, ":")
				k, v = strings.TrimSpace(k), strings.TrimSpace(v)
				switch k {
				case "type":
					if v != "" {
						mt := memory.MemoryType(v)
						opts.Type = &mt
					}
				case "importance":
					if f, err := strconv.ParseFloat(v, 32); err == nil {
						opts.Importance = memory.Float32Ptr(float32(f))
					}
				case "tags":
					tags := []string{}
					for _, t := range strings.Split(v, ",") {
						if t = strings.TrimSpace(t); t != "" {
							tags = append(tags, t)
						}
					}
					opts.Tags = tags
				case "source":
					src := v
					opts.Source = &src
				case "polarity":
					if n, err := strconv.Atoi(v); err == nil {
						opts.Polarity = &n
					}
				}
			}

			if _, err := store.Edit(cmd.Context(), m.ID, opts); err != nil {
				return err
			}
			if textChanged {
				fmt.Printf("Updated (re-embedded) → %s\n", fmtID(m.ID))
			} else {
				fmt.Printf("Updated metadata for %s\n", fmtID(m.ID))
			}
			return nil
		},
	}
	return cmd
}

func newTagCmd() *cobra.Command {
	var add, remove []string
	cmd := &cobra.Command{
		Use:   "tag <id>",
		Short: "Add or remove tags without re-embedding",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			store := storeFromCtx(cmd.Context())
			m, err := store.GetByID(cmd.Context(), args[0])
			if err != nil {
				return err
			}
			tagSet := make(map[string]bool)
			for _, t := range m.Tags {
				tagSet[t] = true
			}
			for _, t := range add {
				tagSet[t] = true
			}
			for _, t := range remove {
				delete(tagSet, t)
			}
			tags := make([]string, 0, len(tagSet))
			for t := range tagSet {
				tags = append(tags, t)
			}
			sort.Strings(tags) // deterministic order, matching Python
			// store.Edit journals the change so `journal undo` can reverse it.
			m, err = store.Edit(cmd.Context(), m.ID, memory.EditOpts{Tags: tags})
			if err != nil {
				return err
			}
			label := strings.Join(m.Tags, ", ")
			if label == "" {
				label = "(none)"
			}
			fmt.Printf("%s tags: %s\n", fmtID(m.ID), label)
			return nil
		},
	}
	cmd.Flags().StringSliceVar(&add, "add", nil, "Tags to add")
	cmd.Flags().StringSliceVar(&remove, "remove", nil, "Tags to remove")
	return cmd
}

func newBumpCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "bump <id> <delta>",
		Short: "Adjust a memory's importance without re-embedding (=0.9 absolute, +0.2 / -0.1 relative)",
		Args:  cobra.ExactArgs(2),
		RunE: func(cmd *cobra.Command, args []string) error {
			store := storeFromCtx(cmd.Context())
			m, err := store.GetByID(cmd.Context(), args[0])
			if err != nil {
				return err
			}
			delta := args[1]
			old := m.Importance
			var newVal float64
			switch {
			case strings.HasPrefix(delta, "="):
				newVal, err = strconv.ParseFloat(delta[1:], 32)
			case strings.HasPrefix(delta, "+") || strings.HasPrefix(delta, "-"):
				var d float64
				d, err = strconv.ParseFloat(delta, 32)
				newVal = float64(old) + d
			default:
				return fmt.Errorf("delta must start with =, +, or - (e.g. =0.9, +0.2, -0.1)")
			}
			if err != nil {
				return fmt.Errorf("invalid importance delta %q: %w", delta, err)
			}
			// store.Edit clamps to [0, 1] and journals the change so
			// `journal undo` can reverse it.
			m, err = store.Edit(cmd.Context(), m.ID,
				memory.EditOpts{Importance: memory.Float32Ptr(float32(newVal))})
			if err != nil {
				return err
			}
			fmt.Printf("%s importance: %.2f → %.2f\n", fmtID(m.ID), old, m.Importance)
			return nil
		},
	}
	// Let a leading-dash delta (e.g. `bump abc -0.1`) be read as a positional
	// value rather than parsed as a flag.
	cmd.Flags().SetInterspersed(false)
	return cmd
}

func newLinkCmd() *cobra.Command {
	var rel string
	cmd := &cobra.Command{
		Use:   "link <src-id> <dst-id>",
		Short: "Create a directed link between two memories",
		Args:  cobra.ExactArgs(2),
		RunE: func(cmd *cobra.Command, args []string) error {
			store := storeFromCtx(cmd.Context())
			return store.Link(cmd.Context(), args[0], args[1], rel)
		},
	}
	cmd.Flags().StringVar(&rel, "rel", "related", "Relation type")
	return cmd
}

func newUnlinkCmd() *cobra.Command {
	var rel string
	cmd := &cobra.Command{
		Use:   "unlink <src-id> <dst-id>",
		Short: "Remove link(s) between two memories",
		Args:  cobra.ExactArgs(2),
		RunE: func(cmd *cobra.Command, args []string) error {
			store := storeFromCtx(cmd.Context())
			n, err := store.Unlink(cmd.Context(), args[0], args[1], rel)
			if err != nil {
				return err
			}
			fmt.Printf("removed %d link(s)\n", n)
			return nil
		},
	}
	cmd.Flags().StringVar(&rel, "rel", "", "Relation to remove (empty = all)")
	return cmd
}

func newNeighborsCmd() *cobra.Command {
	var rel, direction string
	var depth int
	cmd := &cobra.Command{
		Use:   "neighbors <id>",
		Short: "List linked memories",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			store := storeFromCtx(cmd.Context())
			results, err := store.Neighbors(cmd.Context(), args[0], rel, direction, depth)
			if err != nil {
				return err
			}
			rows := make([]MemRow, len(results))
			for i, r := range results {
				rows[i] = MemRow{
					ID:         r.ID,
					Text:       r.Text,
					Type:       string(r.Type),
					Tags:       r.Tags,
					Importance: r.Importance,
					CreatedAt:  r.CreatedAt,
				}
			}
			PrintRows(os.Stdout, rows, fmtFromCtx(cmd.Context()))
			return nil
		},
	}
	cmd.Flags().StringVar(&rel, "rel", "", "Filter by relation")
	cmd.Flags().StringVar(&direction, "direction", "both", "out|in|both")
	cmd.Flags().IntVar(&depth, "depth", 1, "BFS depth")
	return cmd
}

func newGraphCmd() *cobra.Command {
	var depth int
	var format string
	cmd := &cobra.Command{
		Use:   "graph <id> [id...]",
		Short: "Print the subgraph around given memory IDs (ascii|dot|json)",
		Args:  cobra.MinimumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			store := storeFromCtx(cmd.Context())
			g, err := store.Subgraph(cmd.Context(), args, depth)
			if err != nil {
				return err
			}
			switch format {
			case "json":
				b, _ := json.MarshalIndent(g, "", "  ")
				fmt.Println(string(b))
			case "dot":
				fmt.Print(graphToDOT(g))
			default: // ascii
				fmt.Print(graphToASCII(g))
			}
			return nil
		},
	}
	cmd.Flags().IntVar(&depth, "depth", 1, "BFS expansion depth")
	cmd.Flags().StringVar(&format, "format", "ascii", "Output format: ascii|dot|json")
	return cmd
}

// graphToASCII renders a subgraph as an indented node/edge listing.
func graphToASCII(g memory.Graph) string {
	var b strings.Builder
	for _, n := range g.Nodes {
		snippet := n.Text
		if r := []rune(snippet); len(r) > 50 {
			snippet = string(r[:50]) + "…"
		}
		fmt.Fprintf(&b, "%s (%s) %s\n", fmtID(n.ID), n.Type, snippet)
	}
	for _, e := range g.Edges {
		fmt.Fprintf(&b, "  %s --%s--> %s\n", fmtID(e.From), e.Rel, fmtID(e.To))
	}
	if len(g.Nodes) == 0 {
		b.WriteString("(empty)\n")
	}
	return b.String()
}

// graphToDOT renders a subgraph as Graphviz DOT.
func graphToDOT(g memory.Graph) string {
	var b strings.Builder
	b.WriteString("digraph memory {\n  rankdir=LR;\n  node [shape=box];\n")
	for _, n := range g.Nodes {
		snippet := strings.ReplaceAll(n.Text, "\"", "'")
		if r := []rune(snippet); len(r) > 40 {
			snippet = string(r[:40]) + "…"
		}
		fmt.Fprintf(&b, "  %q [label=%q];\n", fmtID(n.ID), fmtID(n.ID)+"\\n"+snippet)
	}
	for _, e := range g.Edges {
		fmt.Fprintf(&b, "  %q -> %q [label=%q];\n", fmtID(e.From), fmtID(e.To), e.Rel)
	}
	b.WriteString("}\n")
	return b.String()
}

func newConflictsCmd() *cobra.Command {
	var threshold float32
	var idFlag string
	cmd := &cobra.Command{
		Use:   "conflicts [id]",
		Short: "Detect contradictions or duplicates (whole store, or one memory via id/--id)",
		Args:  cobra.MaximumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			store := storeFromCtx(cmd.Context())
			// Accept the id either as a positional arg or via --id (Python uses --id).
			id := idFlag
			if len(args) > 0 {
				id = args[0]
			}
			conflicts, err := store.FindConflicts(cmd.Context(), id, threshold)
			if err != nil {
				return err
			}
			b, _ := json.MarshalIndent(conflicts, "", "  ")
			fmt.Println(string(b))
			return nil
		},
	}
	cmd.Flags().StringVar(&idFlag, "id", "", "Check a single memory (id or 8-char prefix) instead of the whole store")
	cmd.Flags().Float32VarP(&threshold, "threshold", "T", 0.80, "Similarity threshold")
	return cmd
}

func newSupersedeCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "supersede <old-id> <new-id>",
		Short: "Mark old-id as superseded by new-id",
		Args:  cobra.ExactArgs(2),
		RunE: func(cmd *cobra.Command, args []string) error {
			store := storeFromCtx(cmd.Context())
			return store.Supersede(cmd.Context(), args[0], args[1])
		},
	}
}

func newRestoreCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "restore <id>",
		Short: "Undo a supersede on a memory",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			store := storeFromCtx(cmd.Context())
			restored, err := store.Restore(cmd.Context(), args[0])
			if err != nil {
				return err
			}
			if restored {
				fmt.Printf("%s restored.\n", fmtID(args[0]))
			} else {
				fmt.Fprintf(os.Stderr, "%s was not superseded.\n", fmtID(args[0]))
			}
			return nil
		},
	}
}

func newPruneCmd() *cobra.Command {
	var apply, yes bool
	var decayRate, minScore, frequencyWeight float32
	var protectTypes []string
	cmd := &cobra.Command{
		Use:   "prune",
		Short: "Remove stale memories via decay scoring (dry-run by default)",
		Long: `Remove stale memories via decay scoring (dry-run by default).

With --frequency-weight > 0, frequently-recalled memories age out more slowly
than untouched ones of equal importance and age. Memories whose type is in
--protect-type are never pruned regardless of score.`,
		RunE: func(cmd *cobra.Command, _ []string) error {
			store := storeFromCtx(cmd.Context())
			// Default protected types from config when the flag is unset.
			protect := protectTypes
			if !cmd.Flags().Changed("protect-type") {
				protect = cfgFromCtx(cmd.Context()).Maintenance.Decay.ProtectTypes
			}
			protectMT := make([]memory.MemoryType, len(protect))
			for i, t := range protect {
				protectMT[i] = memory.MemoryType(t)
			}
			engine := decay.New(store, decayRate, minScore, protectMT, frequencyWeight)
			candidates, err := engine.Prune(cmd.Context(), true) // preview first
			if err != nil {
				return err
			}
			if len(candidates) == 0 {
				fmt.Println("no memories to prune")
				return nil
			}
			for _, m := range candidates {
				action := "[dry-run]"
				if apply {
					action = "[prune]"
				}
				fmt.Printf("%s %s  %.2f  %s\n", action, fmtID(m.ID), m.Importance, fmtAge(m.CreatedAt))
			}
			if !apply {
				fmt.Printf("\n%d memories would be pruned. Use --apply to delete.\n", len(candidates))
				return nil
			}
			if !yes && !Confirm(fmt.Sprintf("Delete %d memories?", len(candidates))) {
				fmt.Println("aborted")
				return nil
			}
			deleted, err := engine.Prune(cmd.Context(), false)
			if err != nil {
				return err
			}
			fmt.Printf("Pruned %d memories.\n", len(deleted))
			return nil
		},
	}
	cmd.Flags().BoolVar(&apply, "apply", false, "Actually delete (default: dry-run)")
	cmd.Flags().BoolVarP(&yes, "yes", "y", false, "Skip the confirmation prompt when applying")
	cmd.Flags().Float32Var(&decayRate, "decay-rate", 0.1, "Decay rate λ")
	cmd.Flags().Float32Var(&minScore, "min-score", 0.05, "Prune threshold")
	cmd.Flags().StringSliceVar(&protectTypes, "protect-type", []string{"procedural"},
		"Memory types never pruned regardless of score (repeatable; default from [maintenance.decay])")
	cmd.Flags().Float32Var(&frequencyWeight, "frequency-weight", 0.0,
		"Reinforcement: how strongly recall count resists decay (0 = off)")
	return cmd
}

func newPurgeCmd() *cobra.Command {
	var apply, yes bool
	cmd := &cobra.Command{
		Use:   "purge",
		Short: "Hard-delete memories whose TTL has passed (dry-run by default)",
		Long: `Hard-delete memories whose TTL (expires_at) has passed. Dry-run by default.

Expired memories are already hidden from recall/list; this reclaims their
storage. Unlike prune it ignores protect-types — an explicit TTL is a stronger
signal than the decay heuristic.`,
		RunE: func(cmd *cobra.Command, _ []string) error {
			store := storeFromCtx(cmd.Context())
			candidates, err := store.PurgeExpired(cmd.Context(), 0, true) // preview
			if err != nil {
				return err
			}
			if len(candidates) == 0 {
				fmt.Println("no expired memories to purge")
				return nil
			}
			for _, m := range candidates {
				action := "[dry-run]"
				if apply {
					action = "[purge]"
				}
				fmt.Printf("%s %s  %s\n", action, fmtID(m.ID), fmtAge(m.CreatedAt))
			}
			if !apply {
				fmt.Printf("\n%d expired memories would be purged. Use --apply to delete.\n", len(candidates))
				return nil
			}
			if !yes && !Confirm(fmt.Sprintf("Delete %d expired memories?", len(candidates))) {
				fmt.Println("aborted")
				return nil
			}
			deleted, err := store.PurgeExpired(cmd.Context(), 0, false)
			if err != nil {
				return err
			}
			fmt.Printf("Purged %d expired memories.\n", len(deleted))
			return nil
		},
	}
	cmd.Flags().BoolVar(&apply, "apply", false, "Actually delete (default: dry-run)")
	cmd.Flags().BoolVarP(&yes, "yes", "y", false, "Skip the confirmation prompt when applying")
	return cmd
}

func newReflectCmd() *cobra.Command {
	var apply, yes bool
	var consolidate string
	var threshold float32
	var minCluster int
	var summarizer, types string
	var maxLevel int
	cmd := &cobra.Command{
		Use:   "reflect",
		Short: "Cluster episodic memories into semantic reflections (dry-run by default)",
		RunE: func(cmd *cobra.Command, _ []string) error {
			store := storeFromCtx(cmd.Context())
			spec := summarizer
			if !cmd.Flags().Changed("summarizer") {
				spec = cfgFromCtx(cmd.Context()).Summarizer
			}
			summarize, err := reflectpkg.BuildSummarizer(spec, true)
			if err != nil {
				return err
			}
			if spec != "" {
				fmt.Printf("Summarizer: %s\n", spec)
			}
			mode := reflectpkg.ConsolidateModeFromString(consolidate)
			// Hard consolidation permanently deletes sources — confirm first.
			if apply && mode == reflectpkg.ConsolidateHard && !yes {
				if !Confirm("Hard-consolidate will permanently delete source episodics. Continue?") {
					fmt.Println("aborted")
					return nil
				}
			}
			engine := reflectpkg.New(store, threshold, minCluster, summarize)
			for _, t := range strings.Split(types, ",") {
				if t = strings.TrimSpace(t); t != "" {
					engine.Types = append(engine.Types, memory.MemoryType(t))
				}
			}
			engine.MaxLevel = maxLevel
			created, err := engine.Reflect(cmd.Context(), !apply, mode)
			if err != nil {
				return err
			}
			if len(created) == 0 {
				fmt.Println("no clusters found")
				return nil
			}
			for _, m := range created {
				prefix := "[dry-run]"
				if apply {
					prefix = "[created]"
				}
				fmt.Printf("%s %s  %.2f  %s\n", prefix, fmtID(m.ID), m.Importance, m.Text[:min(60, len(m.Text))])
			}
			if !apply {
				fmt.Printf("\n%d reflection(s) would be created. Use --apply to persist.\n", len(created))
			}
			return nil
		},
	}
	cmd.Flags().BoolVar(&apply, "apply", false, "Persist reflections")
	cmd.Flags().StringVar(&consolidate, "consolidate", "none",
		"After reflecting, consolidate sources: none|soft (supersede+link)|hard (delete)")
	cmd.Flags().BoolVarP(&yes, "yes", "y", false, "Skip the confirmation prompt for hard consolidation")
	cmd.Flags().Float32Var(&threshold, "threshold", 0.75, "Clustering similarity threshold")
	cmd.Flags().IntVar(&minCluster, "min-cluster-size", 3, "Minimum cluster size")
	cmd.Flags().IntVar(&minCluster, "min-cluster", 3, "Minimum cluster size (deprecated alias of --min-cluster-size)")
	_ = cmd.Flags().MarkHidden("min-cluster")
	cmd.Flags().StringVar(&summarizer, "summarizer", "",
		"provider:model (extractive|ollama:M|openai:M|anthropic:M); default from `summarizer` in config.toml. "+
			"LLM summarizers are also called for the dry-run preview.")
	cmd.Flags().StringVar(&types, "types", "episodic",
		"Comma-separated memory types to cluster. Historically episodic only, which meant summaries were never themselves consolidated")
	cmd.Flags().IntVar(&maxLevel, "max-level", 1,
		"Tiers of reflection-of-reflections to allow. Each summary is tagged level:N; 1 (default) never re-summarises")
	return cmd
}

func newExportCmd() *cobra.Command {
	var memType, tag string
	var includeSuperseded, noVectors bool
	cmd := &cobra.Command{
		Use:   "export <path>",
		Short: "Export memories to a portable .ahkai file (gzipped JSONL)",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			store := storeFromCtx(cmd.Context())
			opts := memory.ExportOpts{
				IncludeVectors:    !noVectors,
				IncludeSuperseded: includeSuperseded,
			}
			if memType != "" {
				opts.Types = []memory.MemoryType{memory.MemoryType(memType)}
			}
			if tag != "" {
				opts.Tags = []string{tag}
			}
			summary, err := store.Export(cmd.Context(), args[0], opts)
			if err != nil {
				return err
			}
			fmt.Printf("Exported %d memories → %s (%d bytes, %.2fs)\n",
				summary.Count, summary.Path, summary.Bytes, summary.Elapsed)
			return nil
		},
	}
	cmd.Flags().StringVarP(&memType, "type", "t", "", "Filter by type")
	cmd.Flags().StringVarP(&tag, "tag", "g", "", "Filter by tag")
	cmd.Flags().BoolVar(&includeSuperseded, "include-superseded", false, "Include superseded memories")
	cmd.Flags().BoolVar(&noVectors, "no-vectors", false, "Omit embeddings — smaller file")
	return cmd
}

func newImportCmd() *cobra.Command {
	var onConflict string
	var regenerateVectors, dryRun, yes bool
	cmd := &cobra.Command{
		Use:   "import <path>",
		Short: "Import memories from an .ahkai file",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			path := args[0]
			if !dryRun && !yes && !Confirm(fmt.Sprintf("Import from %s?", path)) {
				fmt.Println("aborted")
				return nil
			}
			store := storeFromCtx(cmd.Context())
			opts := memory.ImportOpts{
				OnConflict:        memory.ImportConflictPolicy(onConflict),
				RegenerateVectors: regenerateVectors,
				DryRun:            dryRun,
			}
			summary, err := store.Import(cmd.Context(), path, opts)
			if err != nil {
				return err
			}
			prefix := ""
			if dryRun {
				prefix = "[dry-run] "
			}
			fmt.Printf("%simported=%d skipped=%d overwritten=%d renamed=%d errors=%d\n",
				prefix, summary.Imported, summary.Skipped,
				summary.Overwritten, summary.Renamed, len(summary.Errors))
			if summary.VectorsRegenerated {
				fmt.Println("(embeddings were re-generated for the local model)")
			}
			for i, e := range summary.Errors {
				if i >= 5 {
					break
				}
				fmt.Fprintf(os.Stderr, "  ! %s: %s\n", e.ID, e.Msg)
			}
			return nil
		},
	}
	cmd.Flags().StringVar(&onConflict, "on-conflict", "skip", "skip | overwrite | rename | error")
	cmd.Flags().BoolVar(&regenerateVectors, "regenerate-vectors", false, "Re-embed text on import")
	cmd.Flags().BoolVar(&dryRun, "dry-run", false, "Preview without writing")
	cmd.Flags().BoolVarP(&yes, "yes", "y", false, "Skip confirmation")
	return cmd
}

func newInfoCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "info <path>",
		Short: "Inspect an .ahkai file header without touching the store",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			hdr, count, err := memory.PeekExportHeader(args[0])
			if err != nil {
				return err
			}
			b, _ := json.MarshalIndent(hdr, "", "  ")
			fmt.Println(string(b))
			fmt.Printf("\nmemories on disk: %d\n", count)
			return nil
		},
	}
}

func newJournalCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "journal",
		Short: "Inspect the audit journal",
	}
	cmd.AddCommand(newJournalTailCmd(), newJournalShowCmd(), newJournalUndoCmd(),
		newJournalUndoLastCmd())
	return cmd
}

func newJournalTailCmd() *cobra.Command {
	var n int
	var op, actor, memID string
	var includeArchives bool
	cmd := &cobra.Command{
		Use:   "tail",
		Short: "Show recent journal entries (newest first)",
		RunE: func(cmd *cobra.Command, _ []string) error {
			store := storeFromCtx(cmd.Context())
			j := store.Journal()
			if j == nil {
				fmt.Println("(journal disabled)")
				return nil
			}
			entries, err := j.Read(memory.ReadOpts{
				Op: op, Actor: actor, MemoryID: memID,
				IncludeArchives: includeArchives,
			})
			if err != nil {
				return err
			}
			// Tail-N then reverse. n <= 0 would slice out of bounds.
			if n <= 0 {
				entries = nil
			} else if len(entries) > n {
				entries = entries[len(entries)-n:]
			}
			if len(entries) == 0 {
				fmt.Println("(no journal entries)")
				return nil
			}
			for i := len(entries) - 1; i >= 0; i-- {
				e := entries[i]
				secs := int64(e.TS)
				ts := time.Unix(secs, 0).Format("2006-01-02 15:04:05")
				fmt.Printf("%s  %-10s  %-10s  %s  (ts=%.6f)\n",
					ts, e.Op, e.Actor, e.Summary(), e.TS)
			}
			return nil
		},
	}
	cmd.Flags().IntVarP(&n, "number", "n", 20, "Number of entries")
	cmd.Flags().StringVar(&op, "op", "", "Filter by operation")
	cmd.Flags().StringVar(&actor, "actor", "", "Filter by actor")
	cmd.Flags().StringVar(&memID, "id", "", "Filter by memory id")
	cmd.Flags().BoolVar(&includeArchives, "all", false, "Include rotated archives")
	return cmd
}

func newJournalShowCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "show <ts>",
		Short: "Pretty-print one journal entry by timestamp",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			ts, err := strconv.ParseFloat(args[0], 64)
			if err != nil {
				return fmt.Errorf("invalid ts: %w", err)
			}
			store := storeFromCtx(cmd.Context())
			j := store.Journal()
			if j == nil {
				return fmt.Errorf("journal disabled")
			}
			entry, err := j.FindByTS(ts, 0)
			if err != nil {
				return err
			}
			if entry == nil {
				fmt.Fprintf(os.Stderr, "No entry at ts=%v\n", ts)
				return fmt.Errorf("not found")
			}
			b, _ := json.MarshalIndent(entry, "", "  ")
			fmt.Println(string(b))
			return nil
		},
	}
}

func newJournalUndoCmd() *cobra.Command {
	var yes bool
	cmd := &cobra.Command{
		Use:   "undo <ts>",
		Short: "Reverse a single journal entry",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			ts, err := strconv.ParseFloat(args[0], 64)
			if err != nil {
				return fmt.Errorf("invalid ts: %w", err)
			}
			store := storeFromCtx(cmd.Context())
			j := store.Journal()
			if j == nil {
				return fmt.Errorf("journal disabled")
			}
			entry, err := j.FindByTS(ts, 0)
			if err != nil {
				return err
			}
			if entry == nil {
				return fmt.Errorf("no entry at ts=%v", ts)
			}
			short := entry.ID
			if len(short) > 8 {
				short = short[:8]
			}
			if !yes && !Confirm(fmt.Sprintf("Undo %s %s?", entry.Op, short)) {
				fmt.Println("aborted")
				return nil
			}
			ok, err := store.Undo(cmd.Context(), *entry)
			if err != nil {
				return err
			}
			if ok {
				fmt.Println("Undone.")
				return nil
			}
			fmt.Fprintln(os.Stderr, "Could not undo this entry.")
			return fmt.Errorf("undo failed")
		},
	}
	cmd.Flags().BoolVarP(&yes, "yes", "y", false, "Skip confirmation")
	return cmd
}

func newBackupCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "backup",
		Short: "Copy the store directory to a timestamped backup",
		RunE: func(cmd *cobra.Command, _ []string) error {
			cfg := cfgFromCtx(cmd.Context())
			ts := time.Now().Format("20060102T150405")
			backupsDir := filepath.Join(filepath.Dir(cfg.StorePath), "backups")
			dst := filepath.Join(backupsDir, ts)
			if err := os.MkdirAll(backupsDir, 0o700); err != nil {
				return err
			}
			if err := copyDir(cfg.StorePath, dst); err != nil {
				return err
			}
			fmt.Printf("backed up to %s\n", dst)
			return nil
		},
	}
}

func copyDir(src, dst string) error {
	return filepath.Walk(src, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		rel, _ := filepath.Rel(src, path)
		target := filepath.Join(dst, rel)
		if info.IsDir() {
			return os.MkdirAll(target, info.Mode())
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		return os.WriteFile(target, data, info.Mode())
	})
}

func newStatsCmd() *cobra.Command {
	var health bool
	var staleDays int
	var decayRateFlag, freqWeightFlag float32
	cmd := &cobra.Command{
		Use:   "stats",
		Short: "Show store statistics (add --health for a decay/link/recall report)",
		RunE: func(cmd *cobra.Command, _ []string) error {
			store := storeFromCtx(cmd.Context())
			cfg := cfgFromCtx(cmd.Context())
			data, err := store.Stats(cmd.Context())
			if err != nil {
				return err
			}

			if health {
				dec := cfg.Maintenance.Decay
				decayRate := dec.DecayRate
				if cmd.Flags().Changed("decay-rate") {
					decayRate = decayRateFlag
				}
				freqWeight := dec.FrequencyWeight
				if cmd.Flags().Changed("frequency-weight") {
					freqWeight = freqWeightFlag
				}
				active, err := store.ListRecent(cmd.Context(), 0, false, false)
				if err != nil {
					return err
				}
				data["health"] = computeHealth(active, staleDays, decayRate, dec.MinScore, dec.ProtectTypes, freqWeight)
			}

			if fmtFromCtx(cmd.Context()) == FormatJSON {
				b, _ := json.MarshalIndent(data, "", "  ")
				fmt.Println(string(b))
				return nil
			}
			renderStatsText(data)
			return nil
		},
	}
	cmd.Flags().BoolVarP(&health, "health", "H", false,
		"Show a detailed health report: decay histogram, stale/at-risk/never-recalled counts, link density, top recalled")
	cmd.Flags().IntVar(&staleDays, "stale-days", 30, "Days without access before a memory is stale (--health)")
	cmd.Flags().Float32Var(&decayRateFlag, "decay-rate", 0.1, "Decay λ for the health histogram (--health; default from [maintenance.decay])")
	cmd.Flags().Float32Var(&freqWeightFlag, "frequency-weight", 0.0, "Recall-reinforcement weight for health scores (--health; default from [maintenance.decay])")
	return cmd
}

func newConfigCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "config",
		Short: "Show resolved configuration and config-file search paths",
		RunE: func(cmd *cobra.Command, _ []string) error {
			cfg := cfgFromCtx(cmd.Context())

			fmt.Println("Resolved configuration:")
			b, _ := json.MarshalIndent(cfg, "  ", "  ")
			fmt.Println("  " + string(b))

			fmt.Println("\nConfig file search order (later wins):")
			for _, p := range configSearchPaths() {
				exists := "missing"
				if _, err := os.Stat(p); err == nil {
					exists = "found"
				}
				fmt.Printf("  [%s] %s\n", exists, p)
			}
			return nil
		},
	}
	return cmd
}

func containsTag(tags []string, tag string) bool {
	for _, t := range tags {
		if t == tag {
			return true
		}
	}
	return false
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
