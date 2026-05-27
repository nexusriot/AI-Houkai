package cli

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/nexusriot/ai-houkai/internal/decay"
	"github.com/nexusriot/ai-houkai/internal/installer"
	"github.com/nexusriot/ai-houkai/internal/memory"
	reflectpkg "github.com/nexusriot/ai-houkai/internal/reflect"
	"github.com/spf13/cobra"
)

func newRememberCmd() *cobra.Command {
	var tags []string
	var importance float32
	var memType, source string

	cmd := &cobra.Command{
		Use:   "remember <text>",
		Short: "Store a new memory",
		Args:  cobra.MaximumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			var text string
			if len(args) > 0 {
				text = args[0]
			} else {
				// Read from stdin.
				scanner := bufio.NewScanner(os.Stdin)
				var lines []string
				for scanner.Scan() {
					lines = append(lines, scanner.Text())
				}
				text = strings.Join(lines, "\n")
			}
			if text == "" {
				return fmt.Errorf("text is required")
			}
			store := storeFromCtx(cmd.Context())
			opts := memory.RememberOpts{
				Type:       memory.MemoryType(memType),
				Tags:       tags,
				Importance: importance,
				Source:     source,
			}
			m, _, _, err := store.Remember(cmd.Context(), text, opts)
			if err != nil {
				return err
			}
			fmt.Printf("stored %s\n", m.ID)
			return nil
		},
	}
	cmd.Flags().StringSliceVarP(&tags, "tag", "t", nil, "Tags (comma-separated or repeated)")
	cmd.Flags().Float32VarP(&importance, "importance", "i", 0.5, "Importance 0.0–1.0")
	cmd.Flags().StringVar(&memType, "type", "episodic", "Memory type")
	cmd.Flags().StringVar(&source, "source", "", "Provenance label")
	return cmd
}

func newRecallCmd() *cobra.Command {
	var k int
	var tag, memType, mode string
	var minImp float32
	var inclSup bool

	cmd := &cobra.Command{
		Use:   "recall <query>",
		Short: "Search memories",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			store := storeFromCtx(cmd.Context())
			opts := memory.RecallOpts{
				Type:              memory.MemoryType(memType),
				Tag:               tag,
				MinImportance:     minImp,
				Mode:              memory.RecallMode(mode),
				Overfetch:         3,
				IncludeSuperseded: inclSup,
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
	return cmd
}

func newListCmd() *cobra.Command {
	var limit int
	var inclSup bool
	var tag, memType string

	cmd := &cobra.Command{
		Use:   "list",
		Short: "List recent memories",
		RunE: func(cmd *cobra.Command, _ []string) error {
			store := storeFromCtx(cmd.Context())
			mems, err := store.ListRecent(cmd.Context(), limit, inclSup)
			if err != nil {
				return err
			}
			var rows []MemRow
			for _, m := range mems {
				if tag != "" && !containsTag(m.Tags, tag) {
					continue
				}
				if memType != "" && string(m.Type) != memType {
					continue
				}
				rows = append(rows, MemRow{
					ID:           m.ID,
					Text:         m.Text,
					Type:         string(m.Type),
					Tags:         m.Tags,
					Importance:   m.Importance,
					CreatedAt:    m.CreatedAt,
					SupersededBy: m.SupersededBy,
				})
			}
			PrintRows(os.Stdout, rows, fmtFromCtx(cmd.Context()))
			return nil
		},
	}
	cmd.Flags().IntVar(&limit, "limit", 20, "Max results")
	cmd.Flags().BoolVar(&inclSup, "include-superseded", false, "Include superseded")
	cmd.Flags().StringVar(&tag, "tag", "", "Filter by tag")
	cmd.Flags().StringVar(&memType, "type", "", "Filter by type")
	return cmd
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
		Use:   "forget <id>",
		Short: "Delete a memory",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			if !yes && !Confirm("Delete memory "+args[0]+"?") {
				fmt.Println("aborted")
				return nil
			}
			store := storeFromCtx(cmd.Context())
			deleted, err := store.Forget(cmd.Context(), args[0])
			if err != nil {
				return err
			}
			if deleted {
				fmt.Println("deleted")
			} else {
				fmt.Println("not found")
			}
			return nil
		},
	}
	cmd.Flags().BoolVarP(&yes, "yes", "y", false, "Skip confirmation")
	return cmd
}

func newEditCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "edit <id>",
		Short: "Open a memory in $EDITOR",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			store := storeFromCtx(cmd.Context())
			cfg := cfgFromCtx(cmd.Context())
			m, err := store.GetByID(cmd.Context(), args[0])
			if err != nil {
				return err
			}

			// Write text to temp file.
			tmp, err := os.CreateTemp("", "houkai-edit-*.txt")
			if err != nil {
				return err
			}
			_ = os.WriteFile(tmp.Name(), []byte(m.Text), 0o600)
			tmp.Close()
			defer os.Remove(tmp.Name())

			editor := editorCmd(cfg)
			c := exec.Command(editor, tmp.Name())
			c.Stdin, c.Stdout, c.Stderr = os.Stdin, os.Stdout, os.Stderr
			if err := c.Run(); err != nil {
				return err
			}

			newText, err := os.ReadFile(tmp.Name())
			if err != nil {
				return err
			}
			m.Text = strings.TrimSpace(string(newText))
			if err := store.UpdateMemory(cmd.Context(), m, true); err != nil {
				return err
			}
			fmt.Println("updated")
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
			m.Tags = tags
			return store.UpdateMemory(cmd.Context(), m, false)
		},
	}
	cmd.Flags().StringSliceVar(&add, "add", nil, "Tags to add")
	cmd.Flags().StringSliceVar(&remove, "remove", nil, "Tags to remove")
	return cmd
}

func newBumpCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "bump <id> <importance>",
		Short: "Update a memory's importance without re-embedding",
		Args:  cobra.ExactArgs(2),
		RunE: func(cmd *cobra.Command, args []string) error {
			store := storeFromCtx(cmd.Context())
			m, err := store.GetByID(cmd.Context(), args[0])
			if err != nil {
				return err
			}
			f, err := strconv.ParseFloat(args[1], 32)
			if err != nil {
				return fmt.Errorf("invalid importance: %w", err)
			}
			m.Importance = float32(f)
			return store.UpdateMemory(cmd.Context(), m, false)
		},
	}
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
	cmd := &cobra.Command{
		Use:   "graph <id> [id...]",
		Short: "Print the subgraph around given memory IDs as JSON",
		Args:  cobra.MinimumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			store := storeFromCtx(cmd.Context())
			g, err := store.Subgraph(cmd.Context(), args, depth)
			if err != nil {
				return err
			}
			b, _ := json.MarshalIndent(g, "", "  ")
			fmt.Println(string(b))
			return nil
		},
	}
	cmd.Flags().IntVar(&depth, "depth", 1, "BFS expansion depth")
	return cmd
}

func newConflictsCmd() *cobra.Command {
	var threshold float32
	cmd := &cobra.Command{
		Use:   "conflicts [id]",
		Short: "Detect contradictions or duplicates",
		Args:  cobra.MaximumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			store := storeFromCtx(cmd.Context())
			id := ""
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
	cmd.Flags().Float32Var(&threshold, "threshold", 0.80, "Similarity threshold")
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
			return store.Restore(cmd.Context(), args[0])
		},
	}
}

func newPruneCmd() *cobra.Command {
	var apply bool
	var decayRate, minScore float32
	cmd := &cobra.Command{
		Use:   "prune",
		Short: "Remove stale memories via decay scoring (dry-run by default)",
		RunE: func(cmd *cobra.Command, _ []string) error {
			store := storeFromCtx(cmd.Context())
			engine := decay.New(store, decayRate, minScore, nil)
			candidates, err := engine.Prune(cmd.Context(), !apply)
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
					action = "[pruned]"
				}
				fmt.Printf("%s %s  %.2f  %s\n", action, fmtID(m.ID), m.Importance, fmtAge(m.CreatedAt))
			}
			if !apply {
				fmt.Printf("\n%d memories would be pruned. Use --apply to delete.\n", len(candidates))
			}
			return nil
		},
	}
	cmd.Flags().BoolVar(&apply, "apply", false, "Actually delete (default: dry-run)")
	cmd.Flags().Float32Var(&decayRate, "decay-rate", 0.1, "Decay rate λ")
	cmd.Flags().Float32Var(&minScore, "min-score", 0.05, "Prune threshold")
	return cmd
}

func newReflectCmd() *cobra.Command {
	var apply, consolidate bool
	var threshold float32
	var minCluster int
	cmd := &cobra.Command{
		Use:   "reflect",
		Short: "Cluster episodic memories into semantic reflections (dry-run by default)",
		RunE: func(cmd *cobra.Command, _ []string) error {
			store := storeFromCtx(cmd.Context())
			engine := reflectpkg.New(store, threshold, minCluster, nil)
			created, err := engine.Reflect(cmd.Context(), !apply, consolidate && apply)
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
	cmd.Flags().BoolVar(&consolidate, "consolidate", false, "Delete source episodics after reflecting")
	cmd.Flags().Float32Var(&threshold, "threshold", 0.75, "Clustering similarity threshold")
	cmd.Flags().IntVar(&minCluster, "min-cluster", 2, "Minimum cluster size")
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
				fmt.Fprintf(os.Stderr, "Error: %v\n", err)
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
				fmt.Fprintf(os.Stderr, "Error: %v\n", err)
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
	cmd.AddCommand(newJournalTailCmd(), newJournalShowCmd(), newJournalUndoCmd())
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
			// Tail-N then reverse.
			if len(entries) > n {
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
	return &cobra.Command{
		Use:   "stats",
		Short: "Show store statistics",
		RunE: func(cmd *cobra.Command, _ []string) error {
			store := storeFromCtx(cmd.Context())
			stats, err := store.Stats(cmd.Context())
			if err != nil {
				return err
			}
			b, _ := json.MarshalIndent(stats, "", "  ")
			fmt.Println(string(b))
			return nil
		},
	}
}

func newInstallCmd() *cobra.Command {
	var settingsPath, memPath, collection, binaryPath string
	var project bool
	cmd := &cobra.Command{
		Use:   "install",
		Short: "Register ai-houkai-mcp in Claude Code settings.json",
		RunE: func(cmd *cobra.Command, _ []string) error {
			inst := installer.DefaultInstaller()
			if memPath != "" {
				inst.MemoryPath = memPath
			}
			if collection != "" {
				inst.Collection = collection
			}
			if binaryPath != "" {
				inst.BinaryPath = binaryPath
			}
			if project {
				inst.SettingsPath = ".claude/settings.json"
			}
			if settingsPath != "" {
				inst.SettingsPath = settingsPath
			}
			path, err := inst.Install()
			if err != nil {
				return err
			}
			fmt.Printf("installed to %s\n", path)
			fmt.Println(installer.ClaudeMDSnippet())
			return nil
		},
	}
	cmd.Flags().StringVar(&settingsPath, "settings", "", "Custom settings.json path")
	cmd.Flags().StringVar(&memPath, "memory-path", "", "Memory store path")
	cmd.Flags().StringVar(&collection, "collection", "", "Collection name")
	cmd.Flags().StringVar(&binaryPath, "binary", "", "Path to ai-houkai-mcp binary")
	cmd.Flags().BoolVar(&project, "project", false, "Install project-scoped (.claude/settings.json)")
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

// wrapStore wraps *memory.MemoryStore to satisfy decay.Storable + reflect.Storable.
// Both interfaces share ListRecent+Forget; reflect.Storable also needs Remember+Link+AllRaw.
// *memory.MemoryStore satisfies both directly — this is just a type alias for clarity.

// decayStore adapts *memory.MemoryStore for decay.Storable.
type decayStore struct{ s *memory.MemoryStore }

func (d decayStore) ListRecent(ctx context.Context, limit int, incSup bool) ([]memory.Memory, error) {
	return d.s.ListRecent(ctx, limit, incSup)
}
func (d decayStore) Forget(ctx context.Context, id string) (bool, error) {
	return d.s.Forget(ctx, id)
}
