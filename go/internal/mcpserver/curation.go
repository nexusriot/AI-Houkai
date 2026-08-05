package mcpserver

import (
	"context"
	"errors"
	"fmt"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
	"github.com/nexusriot/ai-houkai/internal/memory"
)

// Curation tools graduated from ai-houkai-service (D): merge, versions, tag
// management, find_path and the trash trio.

func addCurationTools(s *server.MCPServer, store *memory.MemoryStore) {
	addMerge(s, store)
	addVersions(s, store)
	addListTags(s, store)
	addRenameTag(s, store)
	addMergeTags(s, store)
	addDeleteTag(s, store)
	addFindPath(s, store)
	addTrash(s, store)
	addTrashList(s, store)
	addTrashRestore(s, store)
	addTrashPurge(s, store)
}

func addMerge(s *server.MCPServer, store *memory.MemoryStore) {
	tool := mcp.NewTool("merge",
		mcp.WithDescription("Fold one memory into another and delete the absorbed one. Combines the "+
			"text, transfers the absorbed memory's outgoing links, and re-points every INCOMING link "+
			"at the target — `forget` does not clean up incoming edges, so a plain delete would strand "+
			"every relationship pointing at the absorbed memory. Journaled on both sides."),
		mcp.WithString("target_id", mcp.Required(), mcp.Description("Memory to keep")),
		mcp.WithString("other_id", mcp.Required(), mcp.Description("Memory to fold in and delete")),
		mcp.WithString("separator", mcp.Description("Text joined between the two (default: two newlines)")),
	)
	s.AddTool(tool, func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		targetID, err := req.RequireString("target_id")
		if err != nil {
			return errResult(err), nil
		}
		otherID, err := req.RequireString("other_id")
		if err != nil {
			return errResult(err), nil
		}
		mem, err := store.Merge(ctx, targetID, otherID, req.GetString("separator", ""))
		if err != nil {
			return jsonText(map[string]any{
				"ok": false, "error": err.Error(),
				"not_found": errors.Is(err, memory.ErrNotFound),
			}), nil
		}
		out := memRecord(mem)
		out["ok"] = true
		return jsonText(out), nil
	})
}

func addVersions(s *server.MCPServer, store *memory.MemoryStore) {
	tool := mcp.NewTool("versions",
		mcp.WithDescription("Past text states of a memory, oldest first. Each entry is the state "+
			"BEFORE an edit; the current live state is excluded (use `get`). Reads rotated journal "+
			"segments, so history survives a rollover."),
		mcp.WithString("memory_id", mcp.Required(), mcp.Description("Memory UUID or 8-char prefix")),
	)
	s.AddTool(tool, func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		id, err := req.RequireString("memory_id")
		if err != nil {
			return errResult(err), nil
		}
		if mem, gerr := store.GetByID(ctx, id); gerr == nil {
			id = mem.ID
		}
		vs, err := store.Versions(id)
		if err != nil {
			return errResult(err), nil
		}
		if vs == nil {
			vs = []memory.Version{}
		}
		return jsonText(vs), nil
	})
}

func addListTags(s *server.MCPServer, store *memory.MemoryStore) {
	tool := mcp.NewTool("list_tags",
		mcp.WithDescription("Every tag with its usage count, most-used first."),
		mcp.WithBoolean("include_superseded", mcp.Description("Count superseded memories too")),
	)
	s.AddTool(tool, func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		tags, err := store.ListTags(ctx, req.GetBool("include_superseded", false))
		if err != nil {
			return errResult(err), nil
		}
		if tags == nil {
			tags = []memory.TagCount{}
		}
		return jsonText(tags), nil
	})
}

func addRenameTag(s *server.MCPServer, store *memory.MemoryStore) {
	tool := mcp.NewTool("rename_tag",
		mcp.WithDescription("Rename a tag across the collection, de-duplicating on collision."),
		mcp.WithString("old", mcp.Required(), mcp.Description("Existing tag")),
		mcp.WithString("new", mcp.Required(), mcp.Description("Replacement tag")),
	)
	s.AddTool(tool, func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		old, err := req.RequireString("old")
		if err != nil {
			return errResult(err), nil
		}
		updated, err := req.RequireString("new")
		if err != nil {
			return errResult(err), nil
		}
		n, err := store.RenameTag(ctx, old, updated)
		if err != nil {
			return jsonText(map[string]any{"ok": false, "error": err.Error()}), nil
		}
		return jsonText(map[string]any{"ok": true, "changed": n, "tag": updated}), nil
	})
}

func addMergeTags(s *server.MCPServer, store *memory.MemoryStore) {
	tool := mcp.NewTool("merge_tags",
		mcp.WithDescription("Fold several tags into one across the collection."),
		mcp.WithArray("sources", mcp.Required(), mcp.Description("Tags to fold in"),
			mcp.Items(map[string]any{"type": "string"})),
		mcp.WithString("into", mcp.Required(), mcp.Description("Tag to keep")),
	)
	s.AddTool(tool, func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		raw, _ := req.GetArguments()["sources"].([]any)
		if len(raw) == 0 {
			return errResult(fmt.Errorf("sources is required")), nil
		}
		sources := make([]string, 0, len(raw))
		for _, v := range raw {
			if str, ok := v.(string); ok {
				sources = append(sources, str)
			}
		}
		into, err := req.RequireString("into")
		if err != nil {
			return errResult(err), nil
		}
		n, err := store.MergeTags(ctx, sources, into)
		if err != nil {
			return jsonText(map[string]any{"ok": false, "error": err.Error()}), nil
		}
		return jsonText(map[string]any{"ok": true, "changed": n, "tag": into}), nil
	})
}

func addDeleteTag(s *server.MCPServer, store *memory.MemoryStore) {
	tool := mcp.NewTool("delete_tag",
		mcp.WithDescription("Strip a tag from every memory that carries it."),
		mcp.WithString("tag", mcp.Required(), mcp.Description("Tag to remove")),
	)
	s.AddTool(tool, func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		tag, err := req.RequireString("tag")
		if err != nil {
			return errResult(err), nil
		}
		n, err := store.DeleteTag(ctx, tag)
		if err != nil {
			return errResult(err), nil
		}
		return jsonText(map[string]any{"ok": true, "changed": n, "tag": tag}), nil
	})
}

func addFindPath(s *server.MCPServer, store *memory.MemoryStore) {
	tool := mcp.NewTool("find_path",
		mcp.WithDescription("Shortest undirected link path between two memories. Undirected because "+
			"\"how are these related?\" does not care which way the author drew the arrow. Returns "+
			"{found, length, path:[{id, rel, text}]}; an empty path means no route within max_depth."),
		mcp.WithString("from_id", mcp.Required(), mcp.Description("Start memory")),
		mcp.WithString("to_id", mcp.Required(), mcp.Description("End memory")),
		mcp.WithNumber("max_depth", mcp.Description("Hop limit (default: 6)")),
	)
	s.AddTool(tool, func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		fromID, err := req.RequireString("from_id")
		if err != nil {
			return errResult(err), nil
		}
		toID, err := req.RequireString("to_id")
		if err != nil {
			return errResult(err), nil
		}
		hops, err := store.FindPath(ctx, fromID, toID, req.GetInt("max_depth", 6))
		if err != nil {
			return errResult(err), nil
		}
		path := make([]map[string]any, 0, len(hops))
		for _, h := range hops {
			entry := map[string]any{"id": h.ID, "rel": h.Rel}
			if mem, gerr := store.GetByID(ctx, h.ID); gerr == nil {
				entry["text"] = clipText(mem.Text, 120)
			}
			path = append(path, entry)
		}
		length := 0
		if len(hops) > 0 {
			length = len(hops) - 1
		}
		return jsonText(map[string]any{
			"found": len(hops) > 0, "length": length, "path": path,
		}), nil
	})
}

func addTrash(s *server.MCPServer, store *memory.MemoryStore) {
	tool := mcp.NewTool("trash",
		mcp.WithDescription("Soft-delete a memory: recoverable, unlike `forget`. The missing middle "+
			"between `supersede` (which asserts \"replaced by X\") and `forget` (irreversible). Use "+
			"`trash_restore` to bring it back."),
		mcp.WithString("memory_id", mcp.Required(), mcp.Description("Memory UUID or 8-char prefix")),
	)
	s.AddTool(tool, func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		id, err := req.RequireString("memory_id")
		if err != nil {
			return errResult(err), nil
		}
		ok, err := store.Trash(ctx, id)
		if err != nil {
			return errResult(err), nil
		}
		return jsonText(map[string]any{"trashed": ok, "id": id}), nil
	})
}

func addTrashList(s *server.MCPServer, store *memory.MemoryStore) {
	tool := mcp.NewTool("trash_list",
		mcp.WithDescription("Everything currently in the trash, oldest first."),
	)
	s.AddTool(tool, func(_ context.Context, _ mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		entries, err := store.TrashList()
		if err != nil {
			return errResult(err), nil
		}
		out := make([]map[string]any, 0, len(entries))
		for _, e := range entries {
			text, _ := e.Memory["text"].(string)
			out = append(out, map[string]any{
				"memory_id": e.MemoryID, "deleted_at": e.DeletedAt,
				"actor": e.Actor, "text": clipText(text, 200),
			})
		}
		return jsonText(out), nil
	})
}

func addTrashRestore(s *server.MCPServer, store *memory.MemoryStore) {
	tool := mcp.NewTool("trash_restore",
		mcp.WithDescription("Bring a trashed memory back with its id, tags and links intact."),
		mcp.WithString("memory_id", mcp.Required(), mcp.Description("Memory id from trash_list")),
	)
	s.AddTool(tool, func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		id, err := req.RequireString("memory_id")
		if err != nil {
			return errResult(err), nil
		}
		mem, ok, err := store.TrashRestore(ctx, id)
		if err != nil {
			return errResult(err), nil
		}
		if !ok {
			return jsonText(map[string]any{
				"restored": false, "id": id, "error": "not in the trash",
			}), nil
		}
		out := memRecord(mem)
		out["restored"] = true
		return jsonText(out), nil
	})
}

func addTrashPurge(s *server.MCPServer, store *memory.MemoryStore) {
	tool := mcp.NewTool("trash_purge",
		mcp.WithDescription("Permanently drop trashed memories. Irreversible. Pass memory_id for one "+
			"entry, older_than_days to apply a retention cutoff, or neither to empty the whole trash. "+
			"The two are mutually exclusive."),
		mcp.WithString("memory_id", mcp.Description("Memory id; omit to empty the whole trash")),
		mcp.WithNumber("older_than_days", mcp.Description("Purge only entries trashed more than this many days ago")),
	)
	s.AddTool(tool, func(_ context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		id := req.GetString("memory_id", "")
		ttl := optFloat32(req, "older_than_days")
		if id != "" && ttl != nil {
			return jsonText(map[string]any{
				"purged": 0,
				"error":  "pass either memory_id or older_than_days, not both",
			}), nil
		}
		var n int
		var err error
		if ttl != nil {
			n, err = store.TrashPurgeExpired(float64(*ttl), 0)
		} else {
			n, err = store.TrashPurge(id)
		}
		if err != nil {
			return errResult(err), nil
		}
		return jsonText(map[string]any{"purged": n}), nil
	})
}

func clipText(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n])
}
