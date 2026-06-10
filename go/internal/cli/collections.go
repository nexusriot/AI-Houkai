package cli

// Collections sub-commands — manage namespaces inside one chromem store.
//
//	houkai collections list           List collections with memory counts.
//	houkai collections create NAME    Create an empty collection.
//	houkai collections delete NAME    Delete a collection and its memories.
//	houkai collections copy SRC DST   Copy memories (with embeddings) SRC → DST.

import (
	"encoding/json"
	"fmt"
	"sort"

	"github.com/spf13/cobra"
)

func newCollectionsCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "collections",
		Short: "Manage collections (namespaces) in the store",
	}
	cmd.AddCommand(
		newCollectionsListCmd(),
		newCollectionsCreateCmd(),
		newCollectionsDeleteCmd(),
		newCollectionsCopyCmd(),
	)
	return cmd
}

func newCollectionsListCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "list",
		Short: "List all collections in the store with their memory counts",
		RunE: func(cmd *cobra.Command, _ []string) error {
			backend := backendFromCtx(cmd.Context())
			active := cfgFromCtx(cmd.Context()).Collection

			counts := backend.ListCollections()
			names := make([]string, 0, len(counts))
			for n := range counts {
				names = append(names, n)
			}
			sort.Strings(names)

			if fmtFromCtx(cmd.Context()) == FormatJSON {
				type row struct {
					Name   string `json:"name"`
					Count  int    `json:"count"`
					Active bool   `json:"active"`
				}
				rows := make([]row, len(names))
				for i, n := range names {
					rows[i] = row{Name: n, Count: counts[n], Active: n == active}
				}
				b, _ := json.MarshalIndent(rows, "", "  ")
				fmt.Println(string(b))
				return nil
			}

			fmt.Printf("%-30s  %8s  %s\n", "COLLECTION", "MEMORIES", "")
			for _, n := range names {
				star := ""
				if n == active {
					star = "*"
				}
				fmt.Printf("%-30s  %8d  %s\n", n, counts[n], star)
			}
			return nil
		},
	}
}

func newCollectionsCreateCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "create <name>",
		Short: "Create an empty collection",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			backend := backendFromCtx(cmd.Context())
			name := args[0]
			if backend.HasCollection(name) {
				return fmt.Errorf("collection %q already exists", name)
			}
			if err := backend.CreateCollection(name); err != nil {
				return err
			}
			fmt.Printf("Created collection %q.\n", name)
			return nil
		},
	}
}

func newCollectionsDeleteCmd() *cobra.Command {
	var yes bool
	cmd := &cobra.Command{
		Use:   "delete <name>",
		Short: "Delete a collection and every memory in it (irreversible)",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			backend := backendFromCtx(cmd.Context())
			name := args[0]
			if !backend.HasCollection(name) {
				return fmt.Errorf("collection %q not found", name)
			}
			if name == cfgFromCtx(cmd.Context()).Collection {
				return fmt.Errorf("refusing to delete the active collection %q — switch with --collection first", name)
			}
			count := backend.ListCollections()[name]
			if !yes && !Confirm(fmt.Sprintf("Delete collection %q (%d memories)?", name, count)) {
				fmt.Println("Aborted.")
				return nil
			}
			if err := backend.DeleteCollection(name); err != nil {
				return err
			}
			fmt.Printf("Deleted collection %q (%d memories).\n", name, count)
			return nil
		},
	}
	cmd.Flags().BoolVarP(&yes, "yes", "y", false, "Skip confirmation")
	return cmd
}

func newCollectionsCopyCmd() *cobra.Command {
	var yes bool
	cmd := &cobra.Command{
		Use:   "copy <src> <dst>",
		Short: "Copy all memories (embeddings included — no re-embedding) SRC → DST",
		Long: `Copy all memories (embeddings included — no re-embedding) SRC → DST.

DST is created if missing; existing DST ids are overwritten.`,
		Args: cobra.ExactArgs(2),
		RunE: func(cmd *cobra.Command, args []string) error {
			backend := backendFromCtx(cmd.Context())
			src, dst := args[0], args[1]
			if !backend.HasCollection(src) {
				return fmt.Errorf("collection %q not found", src)
			}
			if src == dst {
				return fmt.Errorf("SRC and DST are the same collection")
			}
			total := backend.ListCollections()[src]
			if total == 0 {
				fmt.Printf("Collection %q is empty — nothing to copy.\n", src)
				return nil
			}
			if !yes && !Confirm(fmt.Sprintf("Copy %d memories %q → %q?", total, src, dst)) {
				fmt.Println("Aborted.")
				return nil
			}
			copied, err := backend.CopyCollection(cmd.Context(), src, dst)
			if err != nil {
				return err
			}
			fmt.Printf("Copied %d memories %q → %q.\n", copied, src, dst)
			return nil
		},
	}
	cmd.Flags().BoolVarP(&yes, "yes", "y", false, "Skip confirmation")
	return cmd
}
