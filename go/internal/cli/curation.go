package cli

import (
	"fmt"
	"strings"
	"time"

	"github.com/nexusriot/ai-houkai/internal/memory"
	"github.com/spf13/cobra"
)

// Curation commands: merge, versions, tags, path, trash.

func newMergeCmd() *cobra.Command {
	var separator string
	var yes bool
	cmd := &cobra.Command{
		Use:   "merge <target> <other>",
		Short: "Fold one memory into another and delete the absorbed one",
		Long: "Transfers the absorbed memory's outgoing links and re-points every INCOMING\n" +
			"link at the target — a plain forget would strand those relationships.",
		Args: cobra.ExactArgs(2),
		RunE: func(cmd *cobra.Command, args []string) error {
			store := storeFromCtx(cmd.Context())
			if !yes && !Confirm(fmt.Sprintf("Merge %s into %s and delete it?",
				fmtID(args[1]), fmtID(args[0]))) {
				fmt.Println("Aborted.")
				return nil
			}
			mem, err := store.Merge(cmd.Context(), args[0], args[1], separator)
			if err != nil {
				return err
			}
			fmt.Printf("Merged. %s now has %d chars and %d links.\n",
				fmtID(mem.ID), len(mem.Text), len(mem.Links))
			return nil
		},
	}
	cmd.Flags().StringVar(&separator, "separator", "", "Text joined between the two")
	cmd.Flags().BoolVarP(&yes, "yes", "y", false, "Skip confirmation")
	return cmd
}

func newVersionsCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "versions <id>",
		Short: "Show past text states of a memory, oldest first",
		Long: "Each entry is the state BEFORE an edit; the live text is excluded —\n" +
			"use `houkai show` for that.",
		Args: cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			store := storeFromCtx(cmd.Context())
			mem, err := store.GetByID(cmd.Context(), args[0])
			if err != nil {
				return err
			}
			vs, err := store.Versions(mem.ID)
			if err != nil {
				return err
			}
			if fmtFromCtx(cmd.Context()) == FormatJSON {
				if vs == nil {
					vs = []memory.Version{}
				}
				return printJSON(vs)
			}
			if len(vs) == 0 {
				fmt.Println("(no earlier versions — this memory has never been edited)")
				return nil
			}
			for _, v := range vs {
				fmt.Printf("%s  imp=%.2f  %s\n",
					time.Unix(int64(v.TS), 0).Format("2006-01-02 15:04:05"),
					v.Importance, truncate(v.Text, 70))
			}
			return nil
		},
	}
}

func newTagsCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "tags",
		Short: "Curate tags across the collection",
	}
	cmd.AddCommand(newTagsListCmd(), newTagsRenameCmd(), newTagsMergeCmd(),
		newTagsDeleteCmd())
	return cmd
}

func newTagsListCmd() *cobra.Command {
	var all bool
	cmd := &cobra.Command{
		Use:   "list",
		Short: "List every tag with its usage count",
		RunE: func(cmd *cobra.Command, _ []string) error {
			tags, err := storeFromCtx(cmd.Context()).ListTags(cmd.Context(), all)
			if err != nil {
				return err
			}
			if fmtFromCtx(cmd.Context()) == FormatJSON {
				if tags == nil {
					tags = []memory.TagCount{}
				}
				return printJSON(tags)
			}
			if len(tags) == 0 {
				fmt.Println("(no tags)")
				return nil
			}
			for _, t := range tags {
				fmt.Printf("%-30s %d\n", t.Tag, t.Count)
			}
			return nil
		},
	}
	cmd.Flags().BoolVar(&all, "all", false, "Count superseded memories too")
	return cmd
}

func newTagsRenameCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "rename <old> <new>",
		Short: "Rename a tag across the collection (de-duplicating on collision)",
		Args:  cobra.ExactArgs(2),
		RunE: func(cmd *cobra.Command, args []string) error {
			n, err := storeFromCtx(cmd.Context()).RenameTag(cmd.Context(), args[0], args[1])
			if err != nil {
				return err
			}
			fmt.Printf("Renamed %q → %q on %d memories.\n", args[0], args[1], n)
			return nil
		},
	}
}

func newTagsMergeCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "merge <into> <source> [source...]",
		Short: "Fold several tags into one across the collection",
		Args:  cobra.MinimumNArgs(2),
		RunE: func(cmd *cobra.Command, args []string) error {
			n, err := storeFromCtx(cmd.Context()).MergeTags(
				cmd.Context(), args[1:], args[0])
			if err != nil {
				return err
			}
			fmt.Printf("Merged %s → %q on %d memories.\n",
				strings.Join(args[1:], ", "), args[0], n)
			return nil
		},
	}
}

func newTagsDeleteCmd() *cobra.Command {
	var yes bool
	cmd := &cobra.Command{
		Use:   "delete <tag>",
		Short: "Strip a tag from every memory that carries it",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			if !yes && !Confirm(fmt.Sprintf("Remove tag %q from every memory?", args[0])) {
				fmt.Println("Aborted.")
				return nil
			}
			n, err := storeFromCtx(cmd.Context()).DeleteTag(cmd.Context(), args[0])
			if err != nil {
				return err
			}
			fmt.Printf("Removed %q from %d memories.\n", args[0], n)
			return nil
		},
	}
	cmd.Flags().BoolVarP(&yes, "yes", "y", false, "Skip confirmation")
	return cmd
}

func newPathCmd() *cobra.Command {
	var maxDepth int
	cmd := &cobra.Command{
		Use:   "path <from> <to>",
		Short: "Find the shortest link path between two memories",
		Long: "Undirected: \"how are these related?\" does not care which way the author\n" +
			"happened to draw the arrow.",
		Args: cobra.ExactArgs(2),
		RunE: func(cmd *cobra.Command, args []string) error {
			store := storeFromCtx(cmd.Context())
			hops, err := store.FindPath(cmd.Context(), args[0], args[1], maxDepth)
			if err != nil {
				return err
			}
			if fmtFromCtx(cmd.Context()) == FormatJSON {
				length := 0
				if len(hops) > 0 {
					length = len(hops) - 1
				}
				if hops == nil {
					hops = []memory.PathHop{}
				}
				return printJSON(map[string]any{
					"found": len(hops) > 0, "length": length, "path": hops,
				})
			}
			if len(hops) == 0 {
				return fmt.Errorf("no path within %d hops", maxDepth)
			}
			for i, h := range hops {
				arrow := "   "
				if i > 0 {
					arrow = "─" + h.Rel + "→"
				}
				text := "(missing)"
				if mem, gerr := store.GetByID(cmd.Context(), h.ID); gerr == nil {
					text = truncate(mem.Text, 60)
				}
				fmt.Printf("%s %s  %s\n", arrow, fmtID(h.ID), text)
			}
			return nil
		},
	}
	cmd.Flags().IntVar(&maxDepth, "max-depth", 6, "Hop limit")
	return cmd
}

func newTrashCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "trash",
		Short: "Soft-deleted memories (recoverable)",
	}
	cmd.AddCommand(newTrashPutCmd(), newTrashListCmd(), newTrashRestoreCmd(),
		newTrashPurgeCmd())
	return cmd
}

func newTrashPutCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "put <id>",
		Short: "Soft-delete a memory — recoverable, unlike `forget`",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			store := storeFromCtx(cmd.Context())
			mem, err := store.GetByID(cmd.Context(), args[0])
			if err != nil {
				return err
			}
			ok, err := store.Trash(cmd.Context(), mem.ID)
			if err != nil {
				return err
			}
			if !ok {
				return fmt.Errorf("memory %q not found", args[0])
			}
			fmt.Printf("Trashed %s. Restore with: houkai trash restore %s\n",
				fmtID(mem.ID), mem.ID[:8])
			return nil
		},
	}
}

func newTrashListCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "list",
		Short: "List soft-deleted memories, oldest first",
		RunE: func(cmd *cobra.Command, _ []string) error {
			entries, err := storeFromCtx(cmd.Context()).TrashList()
			if err != nil {
				return err
			}
			if fmtFromCtx(cmd.Context()) == FormatJSON {
				if entries == nil {
					entries = []memory.TrashEntry{}
				}
				return printJSON(entries)
			}
			if len(entries) == 0 {
				fmt.Println("(trash is empty)")
				return nil
			}
			for _, e := range entries {
				text, _ := e.Memory["text"].(string)
				fmt.Printf("%s  %s  %-10s  %s\n", fmtID(e.MemoryID),
					time.Unix(int64(e.DeletedAt), 0).Format("2006-01-02 15:04:05"),
					e.Actor, truncate(text, 60))
			}
			return nil
		},
	}
}

func newTrashRestoreCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "restore <id>",
		Short: "Bring a trashed memory back with its id, tags and links intact",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			store := storeFromCtx(cmd.Context())
			// Resolve against the trash, not the store — it is no longer there.
			entries, err := store.TrashList()
			if err != nil {
				return err
			}
			var matches []string
			for _, e := range entries {
				if strings.HasPrefix(e.MemoryID, args[0]) {
					matches = append(matches, e.MemoryID)
				}
			}
			if len(matches) == 0 {
				return fmt.Errorf("%q is not in the trash", args[0])
			}
			if len(matches) > 1 {
				return fmt.Errorf("%q is ambiguous (%d matches)", args[0], len(matches))
			}
			mem, ok, err := store.TrashRestore(cmd.Context(), matches[0])
			if err != nil {
				return err
			}
			if !ok {
				return fmt.Errorf("%q is not in the trash", args[0])
			}
			fmt.Printf("Restored %s.\n", fmtID(mem.ID))
			return nil
		},
	}
}

func newTrashPurgeCmd() *cobra.Command {
	var yes bool
	cmd := &cobra.Command{
		Use:   "purge [id]",
		Short: "Permanently drop trashed memories. Irreversible",
		Args:  cobra.MaximumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			id := ""
			what := "the ENTIRE trash"
			if len(args) == 1 {
				id = args[0]
				what = "memory " + id
			}
			if !yes && !Confirm(fmt.Sprintf(
				"Permanently delete %s? This cannot be undone.", what)) {
				fmt.Println("Aborted.")
				return nil
			}
			n, err := storeFromCtx(cmd.Context()).TrashPurge(id)
			if err != nil {
				return err
			}
			fmt.Printf("Purged %d entries.\n", n)
			return nil
		},
	}
	cmd.Flags().BoolVarP(&yes, "yes", "y", false, "Skip confirmation")
	return cmd
}
