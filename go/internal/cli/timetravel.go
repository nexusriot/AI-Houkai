package cli

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/nexusriot/ai-houkai/internal/memory"
	"github.com/nexusriot/ai-houkai/internal/timeparse"
	"github.com/spf13/cobra"
)

// Point-in-time and runtime-introspection commands. history / state-at /
// get-at replay the audit journal ("what did I know as of T?"); metrics
// reports the process-local op counters and recall latency. All four wrap
// store methods that previously had no CLI surface in either port.

// requireTS resolves a CLI timestamp argument. "now" is accepted as a
// convenience; anything else goes through timeparse (epoch / ISO-8601 /
// relative span like "7d").
func requireTS(raw string) (float64, error) {
	if strings.EqualFold(strings.TrimSpace(raw), "now") {
		return float64(time.Now().UnixNano()) / 1e9, nil
	}
	ts, ok, err := timeparse.Parse(raw)
	if err != nil {
		return 0, err
	}
	if !ok {
		return 0, fmt.Errorf("could not parse timestamp %q", raw)
	}
	return ts, nil
}

func printJSON(v any) error {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	fmt.Println(string(b))
	return nil
}

// journalIDMatches collects every full memory id among entries matching
// prefix — the primary id plus the link/supersede/restore counterparts
// recorded only in meta.
func journalIDMatches(entries []memory.JournalEntry, prefix string) map[string]bool {
	ids := map[string]bool{}
	for _, e := range entries {
		if strings.HasPrefix(e.ID, prefix) {
			ids[e.ID] = true
		}
		for _, key := range []string{"dst_id", "new_id", "superseder_id"} {
			if v, _ := e.Meta[key].(string); v != "" && strings.HasPrefix(v, prefix) {
				ids[v] = true
			}
		}
	}
	return ids
}

// resolveJournalID resolves a memory-id prefix for the time-travel commands.
//
// Resolved against the JOURNAL first (archives included): history and get-at
// exist precisely for memories that may no longer be live, so the live store
// cannot be the primary universe. The live resolver only fills in when the
// journal has no match (journaling disabled, or history pruned past
// keep_days).
func resolveJournalID(cmd *cobra.Command, prefix string) (string, error) {
	if len(prefix) == 36 {
		return prefix, nil
	}
	store := storeFromCtx(cmd.Context())
	if j := store.Journal(); j != nil {
		entries, err := j.Read(memory.ReadOpts{IncludeArchives: true})
		if err != nil {
			return "", err
		}
		ids := journalIDMatches(entries, prefix)
		if len(ids) > 1 {
			return "", fmt.Errorf("%q is ambiguous (%d matches)", prefix, len(ids))
		}
		for id := range ids {
			return id, nil
		}
	}
	mem, err := store.GetByID(cmd.Context(), prefix)
	if err != nil {
		return "", err
	}
	return mem.ID, nil
}

func newHistoryCmd() *cobra.Command {
	var includeArchives bool
	cmd := &cobra.Command{
		Use:   "history <id>",
		Short: "Show the full journaled timeline of one memory",
		Long: "Includes entries that only reference the memory indirectly — the link, " +
			"supersede and undo counterparts recorded against the other side.",
		Args: cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			store := storeFromCtx(cmd.Context())
			fullID, err := resolveJournalID(cmd, args[0])
			if err != nil {
				return err
			}
			entries, err := store.History(cmd.Context(), fullID, includeArchives)
			if err != nil {
				return err
			}
			if fmtFromCtx(cmd.Context()) == FormatJSON {
				rows := make([]map[string]any, len(entries))
				for i, e := range entries {
					rows[i] = map[string]any{
						"ts": e.TS, "op": string(e.Op), "actor": e.Actor, "id": e.ID,
						"before": e.Before, "after": e.After, "meta": e.Meta,
					}
				}
				return printJSON(rows)
			}
			if len(entries) == 0 {
				fmt.Println("(no journal history — journaling may have been disabled)")
				return nil
			}
			for _, e := range entries {
				ts := time.Unix(int64(e.TS), 0).Format("2006-01-02 15:04:05")
				fmt.Printf("%s  %-10s  %-10s  %s  (ts=%.6f)\n",
					ts, e.Op, e.Actor, e.Summary(), e.TS)
			}
			return nil
		},
	}
	cmd.Flags().BoolVar(&includeArchives, "archives", true,
		"Include rotated journal segments")
	return cmd
}

func newStateAtCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "state-at <ts>",
		Short: "Reconstruct every live memory as it stood at a past time",
		Long: "Best-effort journal replay: only mutations still present in the journal " +
			"(and its archives) can be reversed, and a nuke resets the reconstruction. " +
			"<ts> accepts 'now', epoch seconds, ISO-8601, or a relative span like '7d'.",
		Args: cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			store := storeFromCtx(cmd.Context())
			ts, err := requireTS(args[0])
			if err != nil {
				return err
			}
			mems, err := store.StateAt(cmd.Context(), ts)
			if err != nil {
				return err
			}
			if fmtFromCtx(cmd.Context()) == FormatJSON {
				return printJSON(map[string]any{
					"ts": ts, "count": len(mems), "memories": mems,
				})
			}
			stamp := time.Unix(int64(ts), 0).Format("2006-01-02 15:04:05")
			fmt.Printf("%d memories as of %s\n", len(mems), stamp)
			rows := make([]MemRow, len(mems))
			for i, m := range mems {
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
			PrintRows(cmd.OutOrStdout(), rows, fmtFromCtx(cmd.Context()))
			return nil
		},
	}
}

func newGetAtCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "get-at <id> <ts>",
		Short: "Reconstruct one memory as it was at a past time (see state-at)",
		Args:  cobra.ExactArgs(2),
		RunE: func(cmd *cobra.Command, args []string) error {
			store := storeFromCtx(cmd.Context())
			fullID, err := resolveJournalID(cmd, args[0])
			if err != nil {
				return err
			}
			ts, err := requireTS(args[1])
			if err != nil {
				return err
			}
			mem, err := store.GetAt(cmd.Context(), fullID, ts)
			if err != nil {
				return err
			}
			if mem == nil {
				return fmt.Errorf("memory did not exist at that time")
			}
			return printJSON(mem)
		},
	}
}

func newMetricsCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "metrics",
		Short: "Show runtime op counters and recall latency for this process",
		Long: "Metrics are per-process and in-memory: a fresh CLI invocation starts from " +
			"zero, so this is mostly useful against a long-lived `houkai serve` via " +
			"GET /metrics. Shown here for parity and for scripted one-shot checks.",
		RunE: func(cmd *cobra.Command, _ []string) error {
			m, err := storeFromCtx(cmd.Context()).Metrics(cmd.Context())
			if err != nil {
				return err
			}
			return printJSON(m)
		},
	}
}

// newJournalUndoLastCmd reverses the most recent journaled mutation.
// `journal undo <ts>` targets an exact entry; this is the "undo my last
// change" shortcut, which is what an operator reaches for after a mistake.
func newJournalUndoLastCmd() *cobra.Command {
	var memID string
	var yes bool
	cmd := &cobra.Command{
		Use:   "undo-last",
		Short: "Reverse the most recent journaled mutation",
		RunE: func(cmd *cobra.Command, _ []string) error {
			store := storeFromCtx(cmd.Context())
			j := store.Journal()
			if j == nil {
				return fmt.Errorf("journaling is disabled — nothing to undo")
			}
			entries, err := j.Read(memory.ReadOpts{})
			if err != nil {
				return err
			}
			// Resolve an --id prefix against the same (active) journal the
			// entry is picked from: EntryTouches compares exact ids, and the
			// memory being undone is often already forgotten, so the live
			// store cannot resolve it.
			if memID != "" && len(memID) != 36 {
				ids := journalIDMatches(entries, memID)
				if len(ids) == 0 {
					return fmt.Errorf("no journal entry for id prefix %q", memID)
				}
				if len(ids) > 1 {
					return fmt.Errorf("%q is ambiguous (%d matches)", memID, len(ids))
				}
				for id := range ids {
					memID = id
				}
			}
			var entry *memory.JournalEntry
			for i := len(entries) - 1; i >= 0; i-- {
				if memID == "" || memory.EntryTouches(entries[i], memID) {
					entry = &entries[i]
					break
				}
			}
			if entry == nil {
				return fmt.Errorf("no journal entry to undo")
			}
			if !yes && !Confirm(fmt.Sprintf("Undo %s %s?", entry.Op, fmtID(entry.ID))) {
				fmt.Println("Aborted.")
				return nil
			}
			ok, err := store.Undo(cmd.Context(), *entry)
			if err != nil {
				return err
			}
			if !ok {
				return fmt.Errorf("could not undo this entry")
			}
			fmt.Println("Undone.")
			return nil
		},
	}
	cmd.Flags().StringVar(&memID, "id", "", "Undo the newest entry touching this memory")
	cmd.Flags().BoolVarP(&yes, "yes", "y", false, "Skip confirmation")
	return cmd
}
