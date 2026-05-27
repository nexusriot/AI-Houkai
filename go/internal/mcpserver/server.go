package mcpserver

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
	"github.com/nexusriot/ai-houkai/internal/decay"
	"github.com/nexusriot/ai-houkai/internal/memory"
	reflectpkg "github.com/nexusriot/ai-houkai/internal/reflect"
	"github.com/nexusriot/ai-houkai/internal/version"
)

// New wires up the MCP server with all 14 tools.
func New(store *memory.MemoryStore, path, collection string) *server.MCPServer {
	s := server.NewMCPServer("ai-houkai", version.Version,
		server.WithToolCapabilities(false),
	)

	addRemember(s, store)
	addRecall(s, store)
	addForget(s, store)
	addListRecent(s, store)
	addStats(s, store, path, collection)
	addLink(s, store)
	addUnlink(s, store)
	addNeighbors(s, store)
	addFindConflicts(s, store)
	addSupersede(s, store)
	addMaintenanceTick(s, store)
	addJournalTail(s, store)
	addExport(s, store)
	addImport(s, store)

	return s
}

func jsonText(v any) *mcp.CallToolResult {
	b, _ := json.Marshal(v)
	return mcp.NewToolResultText(string(b))
}

func errResult(err error) *mcp.CallToolResult {
	b, _ := json.Marshal(map[string]string{"error": err.Error()})
	return mcp.NewToolResultText(string(b))
}

func addRemember(s *server.MCPServer, store *memory.MemoryStore) {
	tool := mcp.NewTool("remember",
		mcp.WithDescription("Store a new memory. Returns {id, stored} or {stored:false, conflicts:[…]}."),
		mcp.WithString("text", mcp.Required(), mcp.Description("Memory content")),
		mcp.WithString("type", mcp.Description("episodic|semantic|procedural|feedback (default: episodic)")),
		mcp.WithArray("tags", mcp.Description("Topic labels")),
		mcp.WithNumber("importance", mcp.Description("0.0–1.0 (default: 0.5)")),
		mcp.WithString("source", mcp.Description("Provenance label")),
		mcp.WithString("on_conflict", mcp.Description("ignore|warn|supersede|raise")),
		mcp.WithNumber("polarity", mcp.Description("-1/0/+1")),
	)
	s.AddTool(tool, func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		text, err := req.RequireString("text")
		if err != nil {
			return errResult(err), nil
		}
		opts := memory.RememberOpts{
			Type:       memory.MemoryType(req.GetString("type", string(memory.Episodic))),
			Tags:       req.GetStringSlice("tags", nil),
			Importance: float32(req.GetFloat("importance", 0.5)),
			Source:     req.GetString("source", ""),
			Polarity:   req.GetInt("polarity", 0),
		}
		m, stored, conflicts, err := store.Remember(ctx, text, opts)
		if err != nil {
			if ce, ok := err.(*memory.ConflictError); ok {
				return jsonText(map[string]any{"stored": false, "conflicts": ce.Conflicts}), nil
			}
			return errResult(err), nil
		}
		if !stored {
			return jsonText(map[string]any{"stored": false, "conflicts": conflicts}), nil
		}
		return jsonText(map[string]any{"id": m.ID, "stored": true}), nil
	})
}

func addRecall(s *server.MCPServer, store *memory.MemoryStore) {
	tool := mcp.NewTool("recall",
		mcp.WithDescription("Search memories semantically. Returns ranked list."),
		mcp.WithString("query", mcp.Required(), mcp.Description("Search query")),
		mcp.WithNumber("k", mcp.Description("Max results (default: 5)")),
		mcp.WithString("type", mcp.Description("Filter by memory type")),
		mcp.WithString("tag", mcp.Description("Filter by tag")),
		mcp.WithNumber("min_importance", mcp.Description("Minimum importance threshold")),
		mcp.WithString("mode", mcp.Description("semantic|hybrid (default: semantic)")),
		mcp.WithNumber("overfetch", mcp.Description("Overfetch multiplier for hybrid (default: 3)")),
		mcp.WithBoolean("include_superseded", mcp.Description("Include superseded memories")),
	)
	s.AddTool(tool, func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		query, err := req.RequireString("query")
		if err != nil {
			return errResult(err), nil
		}
		k := req.GetInt("k", 5)
		opts := memory.RecallOpts{
			Type:              memory.MemoryType(req.GetString("type", "")),
			Tag:               req.GetString("tag", ""),
			MinImportance:     float32(req.GetFloat("min_importance", 0)),
			Mode:              memory.RecallMode(req.GetString("mode", string(memory.ModeSemantic))),
			Overfetch:         req.GetInt("overfetch", 3),
			IncludeSuperseded: req.GetBool("include_superseded", false),
		}
		results, err := store.Recall(ctx, query, k, opts)
		if err != nil {
			return errResult(err), nil
		}
		type row struct {
			ID           string   `json:"id"`
			Text         string   `json:"text"`
			Type         string   `json:"type"`
			Tags         []string `json:"tags"`
			Importance   float32  `json:"importance"`
			Score        float32  `json:"score"`
			CreatedAt    float64  `json:"created_at"`
			SupersededBy string   `json:"superseded_by,omitempty"`
		}
		out := make([]row, len(results))
		for i, r := range results {
			out[i] = row{
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
		return jsonText(out), nil
	})
}

func addForget(s *server.MCPServer, store *memory.MemoryStore) {
	tool := mcp.NewTool("forget",
		mcp.WithDescription("Delete a memory by ID."),
		mcp.WithString("memory_id", mcp.Required(), mcp.Description("Memory UUID or 8-char prefix")),
	)
	s.AddTool(tool, func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		id, err := req.RequireString("memory_id")
		if err != nil {
			return errResult(err), nil
		}
		deleted, err := store.Forget(ctx, id)
		if err != nil {
			return errResult(err), nil
		}
		return jsonText(map[string]bool{"deleted": deleted}), nil
	})
}

func addListRecent(s *server.MCPServer, store *memory.MemoryStore) {
	tool := mcp.NewTool("list_recent",
		mcp.WithDescription("List recently created memories."),
		mcp.WithNumber("limit", mcp.Description("Max results (default: 20)")),
		mcp.WithBoolean("include_superseded", mcp.Description("Include superseded memories")),
	)
	s.AddTool(tool, func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		limit := req.GetInt("limit", 20)
		incSup := req.GetBool("include_superseded", false)
		mems, err := store.ListRecent(ctx, limit, incSup)
		if err != nil {
			return errResult(err), nil
		}
		type row struct {
			ID           string   `json:"id"`
			Text         string   `json:"text"`
			Type         string   `json:"type"`
			Tags         []string `json:"tags"`
			Importance   float32  `json:"importance"`
			CreatedAt    float64  `json:"created_at"`
			SupersededBy string   `json:"superseded_by,omitempty"`
		}
		out := make([]row, len(mems))
		for i, m := range mems {
			out[i] = row{
				ID:           m.ID,
				Text:         m.Text,
				Type:         string(m.Type),
				Tags:         m.Tags,
				Importance:   m.Importance,
				CreatedAt:    m.CreatedAt,
				SupersededBy: m.SupersededBy,
			}
		}
		return jsonText(out), nil
	})
}

func addStats(s *server.MCPServer, store *memory.MemoryStore, path, collection string) {
	tool := mcp.NewTool("stats",
		mcp.WithDescription("Return store statistics."),
	)
	s.AddTool(tool, func(ctx context.Context, _ mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		stats, err := store.Stats(ctx)
		if err != nil {
			return errResult(err), nil
		}
		stats["path"] = path
		stats["collection"] = collection
		return jsonText(stats), nil
	})
}

func addLink(s *server.MCPServer, store *memory.MemoryStore) {
	tool := mcp.NewTool("link",
		mcp.WithDescription("Create a directed link between two memories."),
		mcp.WithString("src_id", mcp.Required(), mcp.Description("Source memory ID")),
		mcp.WithString("dst_id", mcp.Required(), mcp.Description("Destination memory ID")),
		mcp.WithString("rel", mcp.Description("Relation (default: related)")),
	)
	s.AddTool(tool, func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		src, err := req.RequireString("src_id")
		if err != nil {
			return errResult(err), nil
		}
		dst, err := req.RequireString("dst_id")
		if err != nil {
			return errResult(err), nil
		}
		rel := req.GetString("rel", memory.RelRelated)
		if err := store.Link(ctx, src, dst, rel); err != nil {
			return errResult(err), nil
		}
		return jsonText(map[string]any{"ok": true, "src_id": src, "dst_id": dst, "rel": rel}), nil
	})
}

func addUnlink(s *server.MCPServer, store *memory.MemoryStore) {
	tool := mcp.NewTool("unlink",
		mcp.WithDescription("Remove link(s) between two memories."),
		mcp.WithString("src_id", mcp.Required(), mcp.Description("Source memory ID")),
		mcp.WithString("dst_id", mcp.Required(), mcp.Description("Destination memory ID")),
		mcp.WithString("rel", mcp.Description("Relation to remove (empty = all)")),
	)
	s.AddTool(tool, func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		src, err := req.RequireString("src_id")
		if err != nil {
			return errResult(err), nil
		}
		dst, err := req.RequireString("dst_id")
		if err != nil {
			return errResult(err), nil
		}
		rel := req.GetString("rel", "")
		removed, err := store.Unlink(ctx, src, dst, rel)
		if err != nil {
			return errResult(err), nil
		}
		return jsonText(map[string]int{"removed": removed}), nil
	})
}

func addNeighbors(s *server.MCPServer, store *memory.MemoryStore) {
	tool := mcp.NewTool("neighbors",
		mcp.WithDescription("Traverse the memory graph from a given node."),
		mcp.WithString("memory_id", mcp.Required(), mcp.Description("Starting memory ID")),
		mcp.WithString("rel", mcp.Description("Filter by relation type")),
		mcp.WithString("direction", mcp.Description("out|in|both (default: both)")),
		mcp.WithNumber("depth", mcp.Description("BFS depth (default: 1)")),
	)
	s.AddTool(tool, func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		id, err := req.RequireString("memory_id")
		if err != nil {
			return errResult(err), nil
		}
		rel := req.GetString("rel", "")
		direction := req.GetString("direction", "both")
		depth := req.GetInt("depth", 1)
		results, err := store.Neighbors(ctx, id, rel, direction, depth)
		if err != nil {
			return errResult(err), nil
		}
		return jsonText(results), nil
	})
}

func addFindConflicts(s *server.MCPServer, store *memory.MemoryStore) {
	tool := mcp.NewTool("find_conflicts",
		mcp.WithDescription("Detect contradictions or duplicates in the memory store."),
		mcp.WithString("memory_id", mcp.Description("Check a single memory (empty = global scan)")),
		mcp.WithNumber("threshold", mcp.Description("Similarity threshold (default: 0.80)")),
	)
	s.AddTool(tool, func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		id := req.GetString("memory_id", "")
		threshold := float32(req.GetFloat("threshold", 0))
		conflicts, err := store.FindConflicts(ctx, id, threshold)
		if err != nil {
			return errResult(err), nil
		}
		return jsonText(conflicts), nil
	})
}

func addSupersede(s *server.MCPServer, store *memory.MemoryStore) {
	tool := mcp.NewTool("supersede",
		mcp.WithDescription("Mark old_id as superseded by new_id (soft delete)."),
		mcp.WithString("old_id", mcp.Required(), mcp.Description("ID of the memory to supersede")),
		mcp.WithString("new_id", mcp.Required(), mcp.Description("ID of the replacement memory")),
	)
	s.AddTool(tool, func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		oldID, err := req.RequireString("old_id")
		if err != nil {
			return errResult(err), nil
		}
		newID, err := req.RequireString("new_id")
		if err != nil {
			return errResult(err), nil
		}
		if err := store.Supersede(ctx, oldID, newID); err != nil {
			return errResult(err), nil
		}
		return jsonText(map[string]any{"ok": true, "old_id": oldID, "new_id": newID}), nil
	})
}

func addMaintenanceTick(s *server.MCPServer, store *memory.MemoryStore) {
	tool := mcp.NewTool("maintenance_tick",
		mcp.WithDescription("Run a maintenance pass: prune decayed memories and optionally reflect."),
		mcp.WithNumber("decay_rate", mcp.Description("Decay rate λ (default: 0.1)")),
		mcp.WithNumber("min_score", mcp.Description("Prune threshold (default: 0.05)")),
		mcp.WithBoolean("reflect", mcp.Description("Also run reflection (default: false)")),
		mcp.WithBoolean("consolidate", mcp.Description("Consolidate episodic memories after reflection")),
	)
	s.AddTool(tool, func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		decayRate := float32(req.GetFloat("decay_rate", 0.1))
		minScore := float32(req.GetFloat("min_score", 0.05))
		doReflect := req.GetBool("reflect", false)
		consolidate := req.GetBool("consolidate", false)

		de := decay.New(store, decayRate, minScore, nil)
		pruned, err := de.Prune(ctx, false)
		if err != nil {
			return errResult(fmt.Errorf("prune: %w", err)), nil
		}

		result := map[string]any{"pruned": len(pruned)}

		if doReflect {
			re := reflectpkg.New(store, 0, 0, nil)
			created, err := re.Reflect(ctx, false, consolidate)
			if err != nil {
				return errResult(fmt.Errorf("reflect: %w", err)), nil
			}
			result["reflected"] = len(created)
		}

		return jsonText(result), nil
	})
}

func addJournalTail(s *server.MCPServer, store *memory.MemoryStore) {
	tool := mcp.NewTool("journal_tail",
		mcp.WithDescription("Return the most recent audit-journal entries (newest first)."),
		mcp.WithNumber("n", mcp.Description("Max entries (default: 20)")),
		mcp.WithString("op", mcp.Description("Filter by op: remember|forget|supersede|link|...")),
		mcp.WithNumber("since_seconds", mcp.Description("Only entries within last N seconds")),
	)
	s.AddTool(tool, func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		j := store.Journal()
		if j == nil {
			return jsonText([]any{}), nil
		}
		n := req.GetInt("n", 20)
		op := req.GetString("op", "")
		var since float64
		if v := req.GetFloat("since_seconds", 0); v > 0 {
			since = float64(time.Now().Unix()) - v
		}
		entries, err := j.Read(memory.ReadOpts{Op: op, Since: since})
		if err != nil {
			return errResult(err), nil
		}
		if len(entries) > n {
			entries = entries[len(entries)-n:]
		}
		// Reverse to newest-first.
		rev := make([]map[string]any, len(entries))
		for i, e := range entries {
			rev[len(entries)-1-i] = map[string]any{
				"ts": e.TS, "op": e.Op, "actor": e.Actor,
				"id": e.ID, "summary": e.Summary(), "meta": e.Meta,
			}
		}
		return jsonText(rev), nil
	})
}

func addExport(s *server.MCPServer, store *memory.MemoryStore) {
	tool := mcp.NewTool("export",
		mcp.WithDescription("Export memories to a portable .ahkai file (gzipped JSONL). Path is server-local."),
		mcp.WithString("path", mcp.Required(), mcp.Description("Output .ahkai file path")),
		mcp.WithBoolean("include_vectors", mcp.Description("Include embeddings (default: true)")),
		mcp.WithBoolean("include_superseded", mcp.Description("Include superseded memories")),
		mcp.WithString("type", mcp.Description("Filter by type")),
		mcp.WithString("tag", mcp.Description("Filter by tag")),
		mcp.WithNumber("since", mcp.Description("Only memories created after this unix timestamp")),
	)
	s.AddTool(tool, func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		path, err := req.RequireString("path")
		if err != nil {
			return errResult(err), nil
		}
		opts := memory.ExportOpts{
			IncludeVectors:    req.GetBool("include_vectors", true),
			IncludeSuperseded: req.GetBool("include_superseded", false),
			Since:             req.GetFloat("since", 0),
		}
		if t := req.GetString("type", ""); t != "" {
			opts.Types = []memory.MemoryType{memory.MemoryType(t)}
		}
		if t := req.GetString("tag", ""); t != "" {
			opts.Tags = []string{t}
		}
		summary, err := store.Export(ctx, path, opts)
		if err != nil {
			return errResult(err), nil
		}
		return jsonText(summary), nil
	})
}

func addImport(s *server.MCPServer, store *memory.MemoryStore) {
	tool := mcp.NewTool("import",
		mcp.WithDescription("Import memories from a portable .ahkai file."),
		mcp.WithString("path", mcp.Required(), mcp.Description(".ahkai file to import")),
		mcp.WithString("on_conflict", mcp.Description("skip | overwrite | rename | error (default: skip)")),
		mcp.WithBoolean("regenerate_vectors", mcp.Description("Re-embed text on import")),
		mcp.WithBoolean("dry_run", mcp.Description("Preview without writing")),
	)
	s.AddTool(tool, func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		path, err := req.RequireString("path")
		if err != nil {
			return errResult(err), nil
		}
		opts := memory.ImportOpts{
			OnConflict:        memory.ImportConflictPolicy(req.GetString("on_conflict", "skip")),
			RegenerateVectors: req.GetBool("regenerate_vectors", false),
			DryRun:            req.GetBool("dry_run", false),
		}
		summary, err := store.Import(ctx, path, opts)
		if err != nil {
			return jsonText(map[string]any{"ok": false, "error": err.Error()}), nil
		}
		_ = fmt.Sprintf // keep fmt referenced
		return jsonText(map[string]any{
			"ok":                    true,
			"imported":              summary.Imported,
			"skipped":               summary.Skipped,
			"overwritten":           summary.Overwritten,
			"renamed":               summary.Renamed,
			"errors":                summary.Errors,
			"vectors_regenerated":   summary.VectorsRegenerated,
		}), nil
	})
}
