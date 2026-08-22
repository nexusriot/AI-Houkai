package mcpserver

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
	"github.com/nexusriot/ai-houkai/internal/eval"
	"github.com/nexusriot/ai-houkai/internal/maintenance"
	"github.com/nexusriot/ai-houkai/internal/memory"
	reflectpkg "github.com/nexusriot/ai-houkai/internal/reflect"
	"github.com/nexusriot/ai-houkai/internal/timeparse"
	"github.com/nexusriot/ai-houkai/internal/version"
)

// New wires up the MCP server with all 41 tools.
func New(store *memory.MemoryStore, path, collection string) *server.MCPServer {
	// mcp-go's stdio transport serves tools/call through a worker pool, so a
	// client that pipelines tool calls executes handlers concurrently — over a
	// MemoryStore with no store-level lock. Serialise every tool call, exactly
	// as the HTTP server's storeMu does for its handlers: Recall's
	// access-count bump, Link, and Supersede are read-modify-write, and
	// concurrent handlers would clobber each other's metadata writes.
	var storeMu sync.Mutex
	s := server.NewMCPServer("ai-houkai", version.Version,
		server.WithToolCapabilities(false),
		server.WithToolHandlerMiddleware(func(next server.ToolHandlerFunc) server.ToolHandlerFunc {
			return func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
				storeMu.Lock()
				defer storeMu.Unlock()
				return next(ctx, req)
			}
		}),
	)

	addRemember(s, store)
	addRememberMany(s, store)
	addRecall(s, store)
	addRecallPack(s, store)
	addAutoContext(s, store)
	addGet(s, store)
	addForget(s, store)
	addEdit(s, store)
	addListRecent(s, store)
	addPurgeExpired(s, store)
	addStats(s, store, path, collection)
	addMetrics(s, store)
	addHistory(s, store)
	addStateAt(s, store)
	addGetAt(s, store)
	addLink(s, store)
	addUnlink(s, store)
	addNeighbors(s, store)
	addSubgraph(s, store)
	addFindConflicts(s, store)
	addSupersede(s, store)
	addRestore(s, store)
	addUndo(s, store)
	addNuke(s, store)
	addEvalRecall(s, store)
	addReady(s, store)
	addMaintenanceTick(s, store)
	addJournalTail(s, store)
	addExport(s, store)
	addImport(s, store)
	addCurationTools(s, store)

	return s
}

// summarizerSpec is the reflection summarizer spec ("provider:model") used
// by maintenance_tick. Empty → built-in extractive summarizer.
var summarizerSpec string

// SetSummarizerSpec configures the reflection summarizer used by the
// maintenance_tick tool (e.g. "ollama:llama3.1"). Invalid specs fall back to
// the extractive summarizer with a logged warning.
func SetSummarizerSpec(spec string) { summarizerSpec = spec }

func buildSummarizer() reflectpkg.Summarizer {
	s, err := reflectpkg.BuildSummarizer(summarizerSpec, true)
	if err != nil {
		log.Printf("ai-houkai: bad summarizer spec %q (%v) — using extractive.", summarizerSpec, err)
		return nil // reflect.New falls back to the default summarizer
	}
	return s
}

// parseSinceUntil resolves the optional `since`/`until` recall filters, which
// accept epoch seconds, an ISO-8601 date/datetime, or a relative span ("7d").
// Returns 0 for an absent bound.
func parseSinceUntil(req mcp.CallToolRequest) (since, until float64, err error) {
	if since, _, err = timeparse.Parse(req.GetString("since", "")); err != nil {
		return 0, 0, err
	}
	if until, _, err = timeparse.Parse(req.GetString("until", "")); err != nil {
		return 0, 0, err
	}
	return since, until, nil
}

// optFloat32 returns a *float32 for an optional numeric arg (nil when absent),
// used for the recall knobs whose "unset" state disables a feature.
func optFloat32(req mcp.CallToolRequest, key string) *float32 {
	if v, ok := req.GetArguments()[key]; ok {
		if f, ok := v.(float64); ok {
			x := float32(f)
			return &x
		}
	}
	return nil
}

// weightsFromReq builds HybridWeights from a `graph` argument. It starts from
// DefaultWeights so exposing only `graph` doesn't zero the core weights (which
// Recall would reject) — graph is a pure add-on. Absent → zero value, which
// Recall replaces with DefaultWeights.
func weightsFromReq(req mcp.CallToolRequest) memory.HybridWeights {
	if p := optFloat32(req, "graph"); p != nil {
		w := memory.DefaultWeights()
		w.Graph = *p
		return w
	}
	return memory.HybridWeights{}
}

// expandFromReq builds an *ExpandSpec from flat `expand_*` arguments (nil for
// no expansion). MCP tool schemas are flat, so the HTTP body's nested `expand`
// object is spelled here as one argument per field; unspecified fields fall
// back to the same defaults as Python's ExpandSpec.
func expandFromReq(req mcp.CallToolRequest) *memory.ExpandSpec {
	args := req.GetArguments()
	keys := []string{"expand_rels", "expand_depth", "expand_cap",
		"expand_score", "expand_decay", "expand_rerank"}
	present := false
	for _, k := range keys {
		if _, ok := args[k]; ok {
			present = true
			break
		}
	}
	if !present {
		return nil
	}
	spec := &memory.ExpandSpec{
		Rels:  []string{"refines", "example_of"},
		Depth: 1, Cap: 5, Score: 0.70, Decay: 1.0,
	}
	if v, ok := args["expand_rels"].([]any); ok && len(v) > 0 {
		rels := make([]string, 0, len(v))
		for _, r := range v {
			if s, ok := r.(string); ok {
				rels = append(rels, s)
			}
		}
		if len(rels) > 0 {
			spec.Rels = rels
		}
	}
	if v, ok := args["expand_depth"].(float64); ok {
		spec.Depth = int(v)
	}
	if v, ok := args["expand_cap"].(float64); ok {
		spec.Cap = int(v)
	}
	if v, ok := args["expand_score"].(float64); ok {
		spec.Score = float32(v)
	}
	if v, ok := args["expand_decay"].(float64); ok {
		spec.Decay = float32(v)
	}
	if v, ok := args["expand_rerank"].(bool); ok {
		spec.Rerank = v
	}
	return spec
}

// withExpandArgs appends the flat expand_* argument declarations shared by the
// recall and recall_pack tools.
func withExpandArgs(opts ...mcp.ToolOption) []mcp.ToolOption {
	return append(opts,
		mcp.WithArray("expand_rels", mcp.Description("Graph-walk expansion: link rels to follow (default: refines, example_of)"),
			mcp.Items(map[string]any{"type": "string"})),
		mcp.WithNumber("expand_depth", mcp.Description("Graph-walk expansion: hop depth (default: 1)")),
		mcp.WithNumber("expand_cap", mcp.Description("Graph-walk expansion: max neighbours added (default: 5)")),
		mcp.WithNumber("expand_score", mcp.Description("Graph-walk expansion: score assigned to a hop-1 neighbour (default: 0.70)")),
		mcp.WithNumber("expand_decay", mcp.Description("Graph-walk expansion: per-hop score multiplier beyond hop 1 (default: 1.0)")),
		mcp.WithBoolean("expand_rerank", mcp.Description("Merge expanded neighbours into the pool BEFORE dedup/MMR/top-k instead of appending after")),
	)
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
		mcp.WithString("type", mcp.Description("episodic|semantic|procedural|feedback (default: semantic)")),
		mcp.WithArray("tags", mcp.Description("Topic labels")),
		mcp.WithNumber("importance", mcp.Description("0.0–1.0; omit for the default (0.5, or a heuristic score when the server runs with AI_HOUKAI_AUTO_IMPORTANCE=1)")),
		mcp.WithString("source", mcp.Description("Provenance label")),
		mcp.WithString("on_conflict", mcp.Description("ignore|warn|supersede|raise (default: store policy)")),
		mcp.WithNumber("polarity", mcp.Description("-1/0/+1")),
		mcp.WithNumber("expires_at", mcp.Description("Absolute TTL as a Unix timestamp (hidden from recall once passed)")),
		mcp.WithNumber("ttl_seconds", mcp.Description("Relative TTL in seconds from now. Pass at most one of expires_at/ttl_seconds")),
		mcp.WithBoolean("pinned", mcp.Description("Standing instruction: always offered to recall_pack(include_pinned), never pruned by decay")),
		mcp.WithString("trust", mcp.Description("trusted (default) | reported | untrusted — how much the memory's ORIGIN is trusted. Use untrusted for anything read from content the agent did not author")),
		mcp.WithBoolean("idempotent", mcp.Description("No-op if a live memory already has the same normalised text: bump its access count and return it unchanged")),
		mcp.WithNumber("valid_from", mcp.Description("When the fact became true in the world (Unix timestamp; omit for 'always')")),
		mcp.WithNumber("valid_until", mcp.Description("When the fact stops being true (Unix timestamp; omit for 'still true')")),
	)
	s.AddTool(tool, rememberHandler(store))
}

// rememberHandler is exposed for tests.
func rememberHandler(store *memory.MemoryStore) func(context.Context, mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	return func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		text, err := req.RequireString("text")
		if err != nil {
			return errResult(err), nil
		}
		opts := memory.RememberOpts{
			Pinned:     req.GetBool("pinned", false),
			Trust:      memory.TrustLevel(req.GetString("trust", "")),
			Idempotent: req.GetBool("idempotent", false),
			Type:       memory.MemoryType(req.GetString("type", string(memory.Semantic))),
			Tags:       req.GetStringSlice("tags", nil),
			Importance: optFloat32(req, "importance"), // nil = unset → store default / ImportanceFn
			Source:     req.GetString("source", ""),
			Polarity:   req.GetInt("polarity", 0),
			OnConflict: memory.ConflictPolicy(req.GetString("on_conflict", "")),
		}
		args := req.GetArguments()
		if v, ok := args["expires_at"].(float64); ok {
			opts.ExpiresAt = &v
		}
		if v, ok := args["ttl_seconds"].(float64); ok {
			opts.TTLSeconds = &v
		}
		if v, ok := args["valid_from"].(float64); ok {
			opts.ValidFrom = &v
		}
		if v, ok := args["valid_until"].(float64); ok {
			opts.ValidUntil = &v
		}
		m, stored, conflicts, err := store.Remember(ctx, text, opts)
		if err != nil {
			if ce, ok := err.(*memory.ConflictError); ok {
				return jsonText(map[string]any{"stored": false, "conflicts": ce.Conflicts}), nil
			}
			return errResult(err), nil
		}
		if !stored && len(conflicts) > 0 {
			return jsonText(map[string]any{"stored": false, "conflicts": conflicts}), nil
		}
		// stored=false with no conflicts is an idempotent repeat: the store
		// found the existing row and bumped it. Reporting a bare
		// {stored:false, conflicts:[]} read as a rejected write with no reason.
		out := map[string]any{"id": m.ID, "stored": stored, "importance": m.Importance}
		if m.ExpiresAt != 0 {
			out["expires_at"] = m.ExpiresAt
		}
		return jsonText(out), nil
	}
}

func addRememberMany(s *server.MCPServer, store *memory.MemoryStore) {
	tool := mcp.NewTool("remember_many",
		mcp.WithDescription("Store many memories in one batched, embedding-efficient call. Returns {stored, ids}."),
		mcp.WithArray("items", mcp.Required(), mcp.Description("List of objects, each with a required \"text\" plus optional type/tags/importance/source/polarity/expires_at/ttl_seconds (same fields as remember)")),
		mcp.WithNumber("batch_size", mcp.Description("Items per embed batch (default: 128)")),
		mcp.WithString("on_conflict", mcp.Description("ignore|warn|supersede (default: store policy); raise is not supported in bulk")),
		mcp.WithBoolean("idempotent", mcp.Description("Collapse re-assertions by normalised text, against stored rows and within this call, so a replayed batch does not accumulate near-duplicates")),
	)
	s.AddTool(tool, rememberManyHandler(store))
}

// rememberManyHandler is exposed for tests.
func rememberManyHandler(store *memory.MemoryStore) func(context.Context, mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	return func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		rawItems, ok := req.GetArguments()["items"].([]any)
		if !ok {
			return jsonText(map[string]any{"stored": 0, "error": "items must be an array"}), nil
		}
		items := make([]memory.RememberItem, 0, len(rawItems))
		for i, raw := range rawItems {
			it, ok := raw.(map[string]any)
			if !ok {
				return jsonText(map[string]any{"stored": 0, "error": fmt.Sprintf("items[%d] must be an object", i)}), nil
			}
			text, _ := it["text"].(string)
			if text == "" {
				return jsonText(map[string]any{"stored": 0, "error": fmt.Sprintf("items[%d] missing 'text'", i)}), nil
			}
			var tags []string
			if arr, ok := it["tags"].([]any); ok {
				for _, t := range arr {
					if sv, ok := t.(string); ok {
						tags = append(tags, sv)
					}
				}
			}
			typ := memory.Semantic
			if sv, ok := it["type"].(string); ok && sv != "" {
				typ = memory.MemoryType(sv)
			}
			src, _ := it["source"].(string)
			ri := memory.RememberItem{
				Text: text,
				RememberOpts: memory.RememberOpts{
					Type:   typ,
					Tags:   tags,
					Source: src,
				},
			}
			if v, ok := it["importance"].(float64); ok {
				f := float32(v)
				ri.Importance = &f
			}
			if v, ok := it["polarity"].(float64); ok {
				ri.Polarity = int(v)
			}
			if v, ok := it["expires_at"].(float64); ok {
				ri.ExpiresAt = &v
			}
			if v, ok := it["ttl_seconds"].(float64); ok {
				ri.TTLSeconds = &v
			}
			items = append(items, ri)
		}
		batchSize := 128
		if v, ok := req.GetArguments()["batch_size"].(float64); ok {
			batchSize = int(v)
		}
		onConflict := memory.ConflictPolicy(req.GetString("on_conflict", ""))
		idempotent := req.GetBool("idempotent", false)
		started := float64(time.Now().UnixNano()) / 1e9
		mems, err := store.RememberMany(ctx, items, batchSize, onConflict, idempotent)
		if err != nil {
			return jsonText(map[string]any{"stored": 0, "error": err.Error()}), nil
		}
		ids := make([]string, len(mems))
		// Rows created, not items submitted: an idempotent replay returns the
		// pre-existing rows, and reporting len(mems) told the agent it had
		// written N facts when it had written none. Distinct ids also collapse
		// intra-batch duplicates, which map to one row.
		created := map[string]bool{}
		for i, m := range mems {
			ids[i] = m.ID
			if m.CreatedAt >= started {
				created[m.ID] = true
			}
		}
		return jsonText(map[string]any{"stored": len(created), "ids": ids}), nil
	}
}

func addRecall(s *server.MCPServer, store *memory.MemoryStore) {
	recallOpts := []mcp.ToolOption{
		mcp.WithDescription("Search memories semantically. Returns ranked list."),
		mcp.WithString("query", mcp.Required(), mcp.Description("Search query")),
		mcp.WithNumber("k", mcp.Description("Max results (default: 5)")),
		mcp.WithString("type", mcp.Description("Filter by memory type")),
		mcp.WithString("tag", mcp.Description("Filter by tag")),
		mcp.WithNumber("min_importance", mcp.Description("Minimum importance threshold")),
		mcp.WithString("source", mcp.Description("Keep only memories with this exact provenance string")),
		mcp.WithString("since", mcp.Description("Bound created_at: epoch seconds, an ISO-8601 date/datetime, or a relative span like \"7d\" / \"24h\"")),
		mcp.WithString("until", mcp.Description("Bound created_at (upper): epoch seconds, ISO-8601, or a relative span")),
		mcp.WithString("mode", mcp.Description("semantic|hybrid (default: semantic)")),
		mcp.WithNumber("overfetch", mcp.Description("Overfetch multiplier (default: 4)")),
		mcp.WithBoolean("include_superseded", mcp.Description("Include superseded memories")),
		mcp.WithBoolean("include_expired", mcp.Description("Include memories whose TTL has passed")),
		mcp.WithString("fusion", mcp.Description("weighted (default) | rrf — Reciprocal Rank Fusion of the hybrid signals")),
		mcp.WithNumber("diversity", mcp.Description("MMR λ in [0,1]: higher favours relevance, lower novelty (omit to disable)")),
		mcp.WithNumber("dedup_threshold", mcp.Description("Drop a candidate whose cosine to an already-selected result exceeds this [0,1]")),
		mcp.WithNumber("min_cosine", mcp.Description("Absolute cosine relevance floor in [-1,1]; drops weak hits (omit to disable)")),
		mcp.WithBoolean("touch", mcp.Description("Bump access-count/last_accessed on the hits (default: true; false = read-only)")),
		mcp.WithBoolean("explain", mcp.Description("Include a per-signal score breakdown on each result")),
		mcp.WithNumber("graph", mcp.Description("Graph-proximity weight (hybrid mode only): lifts candidates linked to other strong hits. Omit/0 disables the channel")),
		mcp.WithString("lexical_index", mcp.Description("pool (default) scores BM25 only over the vector over-fetch pool; corpus also pulls candidates whose text contains the query's tokens into the pool (hybrid mode)")),
		mcp.WithString("min_trust", mcp.Description("trusted|reported|untrusted — keep only memories whose provenance is at least this trusted (omit for no filter)")),
		mcp.WithNumber("as_of", mcp.Description("Unix time: return what was TRUE at that moment, using each memory's valid_from/valid_until interval. Superseded memories that were valid then ARE included — that is the point. Omit for 'valid now'. Distinct from state_at, which replays the journal for 'as of when we knew'")),
	}
	tool := mcp.NewTool("recall", withExpandArgs(recallOpts...)...)
	s.AddTool(tool, func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		query, err := req.RequireString("query")
		if err != nil {
			return errResult(err), nil
		}
		since, until, err := parseSinceUntil(req)
		if err != nil {
			return errResult(err), nil
		}
		k := req.GetInt("k", 5)
		opts := memory.RecallOpts{
			Type:              memory.MemoryType(req.GetString("type", "")),
			Tag:               req.GetString("tag", ""),
			MinImportance:     float32(req.GetFloat("min_importance", 0)),
			Mode:              memory.RecallMode(req.GetString("mode", string(memory.ModeSemantic))),
			Overfetch:         req.GetInt("overfetch", 4),
			IncludeSuperseded: req.GetBool("include_superseded", false),
			IncludeExpired:    req.GetBool("include_expired", false),
			Source:            req.GetString("source", ""),
			Since:             since,
			Until:             until,
			Fusion:            memory.FusionMode(req.GetString("fusion", "")),
			Diversity:         optFloat32(req, "diversity"),
			DedupThreshold:    optFloat32(req, "dedup_threshold"),
			MinCosine:         optFloat32(req, "min_cosine"),
			NoTouch:           !req.GetBool("touch", true),
			Explain:           req.GetBool("explain", false),
			Weights:           weightsFromReq(req),
			Expand:            expandFromReq(req),
			LexicalIndex:      memory.LexicalIndexMode(req.GetString("lexical_index", "")),
			MinTrust:          memory.TrustLevel(req.GetString("min_trust", "")),
			AsOf:              req.GetFloat("as_of", 0),
		}
		results, err := store.Recall(ctx, query, k, opts)
		if err != nil {
			return errResult(err), nil
		}
		type row struct {
			ID           string         `json:"id"`
			Text         string         `json:"text"`
			Type         string         `json:"type"`
			Tags         []string       `json:"tags"`
			Importance   float32        `json:"importance"`
			Score        float32        `json:"score"`
			CreatedAt    float64        `json:"created_at"`
			SupersededBy string         `json:"superseded_by,omitempty"`
			ExpiresAt    float64        `json:"expires_at,omitempty"`
			Explain      map[string]any `json:"explain,omitempty"`
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
				ExpiresAt:    r.ExpiresAt,
				Explain:      r.Explain,
			}
		}
		return jsonText(out), nil
	})
}

func addRecallPack(s *server.MCPServer, store *memory.MemoryStore) {
	packOpts := []mcp.ToolOption{
		mcp.WithDescription("Assemble the most relevant memories into a token-budgeted context block. " +
			"Ranks with hybrid scoring (cosine + BM25 + recency + importance) by default, then greedily " +
			"packs results until token_budget is reached. Returns a ready-to-inject `text` block plus the " +
			"packed items. token_budget is a soft ceiling (estimated at ~4 chars/token) covering the " +
			"memory lines, not the header."),
		mcp.WithString("query", mcp.Required(), mcp.Description("Search query")),
		mcp.WithNumber("token_budget", mcp.Description("Token budget for the packed block (default: 800)")),
		mcp.WithString("type", mcp.Description("Filter by memory type")),
		mcp.WithString("tag", mcp.Description("Filter by tag")),
		mcp.WithNumber("min_importance", mcp.Description("Minimum importance threshold")),
		mcp.WithString("source", mcp.Description("Keep only memories with this exact provenance string")),
		mcp.WithString("since", mcp.Description("Bound created_at: epoch seconds, an ISO-8601 date/datetime, or a relative span like \"7d\" / \"24h\"")),
		mcp.WithString("until", mcp.Description("Bound created_at (upper): epoch seconds, ISO-8601, or a relative span")),
		mcp.WithString("mode", mcp.Description("semantic|hybrid (default: hybrid)")),
		mcp.WithNumber("max_items", mcp.Description("Ranked candidates to consider (default: 50)")),
		mcp.WithBoolean("include_superseded", mcp.Description("Include superseded memories")),
		mcp.WithString("fusion", mcp.Description("weighted (default) | rrf")),
		mcp.WithNumber("diversity", mcp.Description("MMR λ in [0,1] to reduce near-duplicate memories in the pack (omit to disable)")),
		mcp.WithNumber("dedup_threshold", mcp.Description("Hard-drop candidates whose cosine exceeds this [0,1]")),
		mcp.WithNumber("min_cosine", mcp.Description("Absolute cosine relevance floor in [-1,1] (omit to disable)")),
		mcp.WithBoolean("compress", mcp.Description("Fold budget-dropped, similar memories into compressed summary lines")),
		mcp.WithNumber("compress_threshold", mcp.Description("Jaccard similarity for compression clustering (default: 0.30)")),
		mcp.WithNumber("compress_min_group", mcp.Description("Minimum cluster size to compress (default: 2)")),
		mcp.WithNumber("graph", mcp.Description("Graph-proximity weight (hybrid mode only): lifts candidates linked to other strong hits. Omit/0 disables the channel")),
		mcp.WithBoolean("touch", mcp.Description("Bump access-count/last_accessed on the packed memories (default: true; false = read-only)")),
		mcp.WithString("header", mcp.Description("Heading prepended to the block, not counted against token_budget (default: \"## Relevant memory\"; \"\" for none)")),
		mcp.WithString("lexical_index", mcp.Description("pool (default) scores BM25 only over the vector over-fetch pool; corpus also pulls candidates whose text contains the query's tokens into the pool (hybrid mode)")),
		mcp.WithString("min_trust", mcp.Description("trusted|reported|untrusted — keep only memories whose provenance is at least this trusted")),
		mcp.WithNumber("as_of", mcp.Description("Unix time: pack what was TRUE at that moment, using each memory's valid_from/valid_until interval. Omit for 'valid now'")),
		mcp.WithBoolean("include_pinned", mcp.Description("Prepend every pinned memory ahead of the ranked hits, so a standing instruction is present whether or not it matches the query. They compete for the same budget")),
	}
	tool := mcp.NewTool("recall_pack", withExpandArgs(packOpts...)...)
	s.AddTool(tool, func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		query, err := req.RequireString("query")
		if err != nil {
			return errResult(err), nil
		}
		since, until, err := parseSinceUntil(req)
		if err != nil {
			return errResult(err), nil
		}
		// Header is tri-state: absent = default heading, present (incl. "") =
		// verbatim. A plain GetString would make "" indistinguishable from absent.
		var header *string
		if v, ok := req.GetArguments()["header"].(string); ok {
			header = &v
		}
		pack, err := store.RecallPack(ctx, query, memory.PackOpts{
			TokenBudget:       req.GetInt("token_budget", 800),
			Type:              memory.MemoryType(req.GetString("type", "")),
			Tag:               req.GetString("tag", ""),
			MinImportance:     float32(req.GetFloat("min_importance", 0)),
			Mode:              memory.RecallMode(req.GetString("mode", string(memory.ModeHybrid))),
			MaxItems:          req.GetInt("max_items", 50),
			IncludeSuperseded: req.GetBool("include_superseded", false),
			Source:            req.GetString("source", ""),
			Since:             since,
			Until:             until,
			Fusion:            memory.FusionMode(req.GetString("fusion", "")),
			Diversity:         optFloat32(req, "diversity"),
			DedupThreshold:    optFloat32(req, "dedup_threshold"),
			MinCosine:         optFloat32(req, "min_cosine"),
			Compress:          req.GetBool("compress", false),
			CompressThreshold: float32(req.GetFloat("compress_threshold", 0.30)),
			CompressMinGroup:  req.GetInt("compress_min_group", 2),
			Weights:           weightsFromReq(req),
			Expand:            expandFromReq(req),
			NoTouch:           !req.GetBool("touch", true),
			Header:            header,
			LexicalIndex:      memory.LexicalIndexMode(req.GetString("lexical_index", "")),
			MinTrust:          memory.TrustLevel(req.GetString("min_trust", "")),
			AsOf:              req.GetFloat("as_of", 0),
			IncludePinned:     req.GetBool("include_pinned", false),
		})
		if err != nil {
			return errResult(err), nil
		}
		return jsonText(packResultJSON(pack)), nil
	})
}

// packResultJSON renders a PackResult as the MCP/HTTP response payload,
// including compressed groups as {ids, count, text, tokens}.
func packResultJSON(pack memory.PackResult) map[string]any {
	items := make([]map[string]any, len(pack.Items))
	for i, p := range pack.Items {
		items[i] = map[string]any{
			"id":         p.Memory.ID,
			"text":       p.Memory.Text,
			"type":       string(p.Memory.Type),
			"tags":       p.Memory.Tags,
			"importance": p.Memory.Importance,
			"score":      p.Score,
			"tokens":     p.Tokens,
		}
	}
	out := map[string]any{
		"text":        pack.Text,
		"used_tokens": pack.UsedTokens,
		"budget":      pack.Budget,
		"truncated":   pack.Truncated,
		"items":       items,
	}
	if len(pack.CompressedGroups) > 0 {
		groups := make([]map[string]any, len(pack.CompressedGroups))
		for i, g := range pack.CompressedGroups {
			groups[i] = map[string]any{
				"ids": g.IDs(), "count": len(g.Memories), "text": g.Text, "tokens": g.Tokens,
			}
		}
		out["compressed_groups"] = groups
	}
	return out
}

func addAutoContext(s *server.MCPServer, store *memory.MemoryStore) {
	tool := mcp.NewTool("auto_context",
		mcp.WithDescription("Build a token-budgeted context block for a task by fanning out recall over the "+
			"task plus extracted key phrases, deduplicating by memory id (keeping the highest score), then "+
			"packing greedily. More thorough than a single recall_pack for open-ended tasks."),
		mcp.WithString("task", mcp.Required(), mcp.Description("Task or situation description to gather context for")),
		mcp.WithNumber("token_budget", mcp.Description("Token budget for the packed block (default: 800)")),
		mcp.WithNumber("max_phrases", mcp.Description("Max key phrases to fan out over (default: 3)")),
		mcp.WithString("mode", mcp.Description("semantic|hybrid (default: hybrid)")),
		mcp.WithNumber("min_cosine", mcp.Description("Absolute cosine relevance floor applied to every fan-out query [-1,1]")),
		mcp.WithBoolean("compress", mcp.Description("Fold budget-dropped, similar memories into compressed summary lines")),
		mcp.WithNumber("compress_threshold", mcp.Description("Jaccard similarity for compression clustering (default: 0.30)")),
		mcp.WithNumber("compress_min_group", mcp.Description("Minimum cluster size to compress (default: 2)")),
		mcp.WithBoolean("touch", mcp.Description("Bump access-count/last_accessed on every fan-out recall (default: true; false = read-only)")),
		mcp.WithString("header", mcp.Description("Heading prepended to the block, not counted against token_budget (default: \"## Relevant memory\"; \"\" for none)")),
		mcp.WithString("lexical_index", mcp.Description("pool (default) scores BM25 only over the vector over-fetch pool; corpus also pulls candidates whose text contains the query's tokens into the pool (hybrid mode). Applies to every fan-out query")),
		mcp.WithString("min_trust", mcp.Description("trusted|reported|untrusted — keep only memories whose provenance is at least this trusted. Worth setting here in particular: this is the tool you call without choosing a query, so it is the one most likely to pull scraped material into context unattended")),
	)
	s.AddTool(tool, func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		task, err := req.RequireString("task")
		if err != nil {
			return errResult(err), nil
		}
		maxPhrases := req.GetInt("max_phrases", 3)
		var header *string
		if v, ok := req.GetArguments()["header"].(string); ok {
			header = &v
		}
		pack, err := store.AutoContextPack(ctx, task, memory.AutoContextOpts{
			TokenBudget:       req.GetInt("token_budget", 800),
			MaxPhrases:        maxPhrases,
			Mode:              memory.RecallMode(req.GetString("mode", string(memory.ModeHybrid))),
			MinCosine:         optFloat32(req, "min_cosine"),
			Compress:          req.GetBool("compress", false),
			CompressThreshold: float32(req.GetFloat("compress_threshold", 0.30)),
			CompressMinGroup:  req.GetInt("compress_min_group", 2),
			NoTouch:           !req.GetBool("touch", true),
			Header:            header,
			LexicalIndex:      memory.LexicalIndexMode(req.GetString("lexical_index", "")),
			MinTrust:          memory.TrustLevel(req.GetString("min_trust", "")),
		})
		if err != nil {
			return errResult(err), nil
		}
		out := packResultJSON(pack)
		out["queries"] = append([]string{task}, memory.ExtractKeyPhrases(task, maxPhrases)...)
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

func addEdit(s *server.MCPServer, store *memory.MemoryStore) {
	tool := mcp.NewTool("edit",
		mcp.WithDescription("Update fields of an existing memory in place, keeping its id. "+
			"Omitted fields stay unchanged. Text changes are re-embedded; links, supersede state, "+
			"and access tracking are preserved (do NOT forget+remember to fix a typo — that loses "+
			"them). The change is journaled and undoable."),
		mcp.WithString("memory_id", mcp.Required(), mcp.Description("Memory UUID or 8-char prefix")),
		mcp.WithString("text", mcp.Description("New memory text (re-embedded)")),
		mcp.WithString("type", mcp.Description("episodic|semantic|procedural|feedback")),
		mcp.WithArray("tags", mcp.Description("Replacement tag list (empty list clears)")),
		mcp.WithNumber("importance", mcp.Description("0.0–1.0")),
		mcp.WithNumber("polarity", mcp.Description("-1/0/+1")),
		mcp.WithString("source", mcp.Description("Provenance label (empty string clears)")),
		mcp.WithNumber("expires_at", mcp.Description("Set the TTL to this Unix timestamp; pass 0 to clear it")),
		mcp.WithBoolean("pinned", mcp.Description("Pin (true) or unpin (false) the memory")),
		mcp.WithString("trust", mcp.Description("trusted | reported | untrusted — re-label the origin's trust")),
		mcp.WithNumber("valid_from", mcp.Description("Correct when the fact became true in the world (Unix timestamp; 0 reopens)")),
		mcp.WithNumber("valid_until", mcp.Description("Correct when the fact stopped being true (Unix timestamp; 0 reopens)")),
	)
	s.AddTool(tool, editHandler(store))
}

// editHandler is exposed for tests.
func editHandler(store *memory.MemoryStore) func(context.Context, mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	return func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		id, err := req.RequireString("memory_id")
		if err != nil {
			return errResult(err), nil
		}
		var opts memory.EditOpts
		args := req.GetArguments()
		if v, ok := args["text"].(string); ok {
			opts.Text = &v
		}
		if v, ok := args["type"].(string); ok {
			mt := memory.MemoryType(v)
			opts.Type = &mt
		}
		if _, ok := args["tags"]; ok {
			tags := req.GetStringSlice("tags", nil)
			if tags == nil {
				tags = []string{}
			}
			opts.Tags = tags
		}
		opts.Importance = optFloat32(req, "importance")
		if v, ok := args["polarity"].(float64); ok {
			p := int(v)
			opts.Polarity = &p
		}
		if v, ok := args["expires_at"].(float64); ok {
			opts.ExpiresAt = &v
		}
		if v, ok := args["source"].(string); ok {
			opts.Source = &v
		}
		// These four were write-only at creation over MCP: without them an
		// MCP client had no way to pin/unpin a memory, re-label its trust,
		// or correct its valid-time interval after the fact.
		if v, ok := args["pinned"].(bool); ok {
			opts.Pinned = &v
		}
		if v, ok := args["trust"].(string); ok && v != "" {
			t := memory.TrustLevel(v)
			opts.Trust = &t
		}
		if v, ok := args["valid_from"].(float64); ok {
			opts.ValidFrom = &v
		}
		if v, ok := args["valid_until"].(float64); ok {
			opts.ValidUntil = &v
		}
		m, err := store.Edit(ctx, id, opts)
		if err != nil {
			return jsonText(map[string]any{"ok": false, "error": err.Error()}), nil
		}
		out := map[string]any{
			"ok":           true,
			"id":           m.ID,
			"text":         m.Text,
			"type":         string(m.Type),
			"tags":         m.Tags,
			"importance":   m.Importance,
			"polarity":     m.Polarity,
			"source":       m.Source,
			"pinned":       m.Pinned,
			"trust":        string(m.Trust),
			"content_hash": m.ContentHash,
			"valid_from":   m.ValidFrom,
			"valid_until":  m.ValidUntil,
		}
		if m.ExpiresAt != 0 {
			out["expires_at"] = m.ExpiresAt
		}
		return jsonText(out), nil
	}
}

// memRecord renders a full memory record, shared by the get / restore /
// subgraph tools. Mirrors Python's _mem_dict.
func memRecord(m memory.Memory) map[string]any {
	links := make([]map[string]string, len(m.Links))
	for i, l := range m.Links {
		links[i] = map[string]string{"to": l.To, "rel": l.Rel}
	}
	return map[string]any{
		"id":            m.ID,
		"text":          m.Text,
		"type":          string(m.Type),
		"tags":          m.Tags,
		"importance":    m.Importance,
		"source":        m.Source,
		"polarity":      m.Polarity,
		"created_at":    m.CreatedAt,
		"last_accessed": m.LastAccessed,
		"access_count":  m.AccessCount,
		"superseded_by": m.SupersededBy,
		"superseded_at": m.SupersededAt,
		"expires_at":    m.ExpiresAt,
		// Carried explicitly — including their defaults — like the HTTP
		// serializer: an agent that pins, re-labels trust, or retires a fact
		// must be able to verify it afterwards.
		"pinned":       m.Pinned,
		"trust":        string(m.Trust),
		"content_hash": m.ContentHash,
		"valid_from":   m.ValidFrom,
		"valid_until":  m.ValidUntil,
		"links":        links,
	}
}

func addGet(s *server.MCPServer, store *memory.MemoryStore) {
	tool := mcp.NewTool("get",
		mcp.WithDescription("Fetch one memory by its exact id (or 8-char prefix). A plain read: no "+
			"access-count bump and no filtering — a superseded or expired memory is still returned. "+
			"Use `recall` for ranked search and `get_at` for a past point in time."),
		mcp.WithString("memory_id", mcp.Required(), mcp.Description("Memory UUID or 8-char prefix")),
	)
	s.AddTool(tool, func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		id, err := req.RequireString("memory_id")
		if err != nil {
			return errResult(err), nil
		}
		mem, err := store.GetByID(ctx, id)
		if err != nil {
			return jsonText(map[string]any{"found": false, "id": id}), nil
		}
		out := memRecord(mem)
		out["found"] = true
		return jsonText(out), nil
	})
}

func addListRecent(s *server.MCPServer, store *memory.MemoryStore) {
	tool := mcp.NewTool("list_recent",
		mcp.WithDescription("List recently created memories."),
		mcp.WithNumber("limit", mcp.Description("Max results (default: 20)")),
		mcp.WithBoolean("include_superseded", mcp.Description("Include superseded memories")),
		mcp.WithBoolean("include_expired", mcp.Description("Include memories whose TTL has passed")),
	)
	s.AddTool(tool, func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		limit := req.GetInt("limit", 20)
		incSup := req.GetBool("include_superseded", false)
		incExp := req.GetBool("include_expired", false)
		mems, err := store.ListRecent(ctx, limit, incSup, incExp)
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
			ExpiresAt    float64  `json:"expires_at,omitempty"`
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
				ExpiresAt:    m.ExpiresAt,
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

func addMetrics(s *server.MCPServer, store *memory.MemoryStore) {
	tool := mcp.NewTool("metrics",
		mcp.WithDescription("Runtime metrics: op counters + recall latency since server start (process-local, reset on restart)."),
	)
	s.AddTool(tool, func(ctx context.Context, _ mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		m, err := store.Metrics(ctx)
		if err != nil {
			return errResult(err), nil
		}
		return jsonText(m), nil
	})
}

func addPurgeExpired(s *server.MCPServer, store *memory.MemoryStore) {
	tool := mcp.NewTool("purge_expired",
		mcp.WithDescription("Hard-delete memories whose TTL has passed (reclaims storage). Expired memories are already hidden from recall."),
		mcp.WithBoolean("dry_run", mcp.Description("Report what would be purged without deleting")),
	)
	s.AddTool(tool, func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		dry := req.GetBool("dry_run", false)
		purged, err := store.PurgeExpired(ctx, 0, dry)
		if err != nil {
			return errResult(err), nil
		}
		ids := make([]string, len(purged))
		for i, p := range purged {
			ids[i] = p.ID
		}
		return jsonText(map[string]any{"purged": len(purged), "dry_run": dry, "ids": ids}), nil
	})
}

func addHistory(s *server.MCPServer, store *memory.MemoryStore) {
	tool := mcp.NewTool("history",
		mcp.WithDescription("Full journaled timeline of one memory, oldest first (creation, edits, supersede/restore, links pointing at it, forget)."),
		mcp.WithString("memory_id", mcp.Required(), mcp.Description("Memory UUID or 8-char prefix")),
	)
	s.AddTool(tool, func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		id, err := req.RequireString("memory_id")
		if err != nil {
			return errResult(err), nil
		}
		entries, err := store.History(ctx, id, true)
		if err != nil {
			return errResult(err), nil
		}
		out := make([]map[string]any, len(entries))
		for i, e := range entries {
			out[i] = map[string]any{
				"ts": e.TS, "op": string(e.Op), "actor": e.Actor, "id": e.ID,
				"before": e.Before, "after": e.After, "meta": e.Meta,
				"summary": e.Summary(),
			}
		}
		return jsonText(out), nil
	})
}

func addStateAt(s *server.MCPServer, store *memory.MemoryStore) {
	tool := mcp.NewTool("state_at",
		mcp.WithDescription("Reconstruct the store's live memories as of a past time (best-effort journal replay)."),
		mcp.WithString("ts", mcp.Required(), mcp.Description("Epoch seconds, an ISO-8601 date/datetime, or a relative span like \"7d\"")),
	)
	s.AddTool(tool, func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		raw, err := req.RequireString("ts")
		if err != nil {
			return errResult(err), nil
		}
		ts, _, perr := timeparse.Parse(raw)
		if perr != nil {
			return errResult(perr), nil
		}
		mems, err := store.StateAt(ctx, ts)
		if err != nil {
			return errResult(err), nil
		}
		items := make([]map[string]any, len(mems))
		for i, m := range mems {
			items[i] = map[string]any{
				"id": m.ID, "text": m.Text, "type": string(m.Type),
				"tags": m.Tags, "importance": m.Importance, "created_at": m.CreatedAt,
			}
		}
		return jsonText(map[string]any{"ts": ts, "count": len(mems), "memories": items}), nil
	})
}

func addGetAt(s *server.MCPServer, store *memory.MemoryStore) {
	tool := mcp.NewTool("get_at",
		mcp.WithDescription("Reconstruct a single memory as it was at a past time (see state_at)."),
		mcp.WithString("memory_id", mcp.Required(), mcp.Description("Memory UUID or 8-char prefix")),
		mcp.WithString("ts", mcp.Required(), mcp.Description("Epoch seconds, an ISO-8601 date/datetime, or a relative span like \"7d\"")),
	)
	s.AddTool(tool, func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		id, err := req.RequireString("memory_id")
		if err != nil {
			return errResult(err), nil
		}
		raw, err := req.RequireString("ts")
		if err != nil {
			return errResult(err), nil
		}
		ts, _, perr := timeparse.Parse(raw)
		if perr != nil {
			return errResult(perr), nil
		}
		mem, err := store.GetAt(ctx, id, ts)
		if err != nil {
			return errResult(err), nil
		}
		if mem == nil {
			return jsonText(map[string]any{"ok": false, "error": "memory did not exist at that time"}), nil
		}
		return jsonText(map[string]any{
			"ok": true, "ts": ts, "id": mem.ID, "text": mem.Text,
			"type": string(mem.Type), "tags": mem.Tags, "importance": mem.Importance,
			"created_at": mem.CreatedAt, "expires_at": mem.ExpiresAt,
		}), nil
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

func addRestore(s *server.MCPServer, store *memory.MemoryStore) {
	tool := mcp.NewTool("restore",
		mcp.WithDescription("Undo a supersede: clear the soft-delete so the memory is visible again. "+
			"Also removes the 'supersedes' link the superseder gained. Returns restored:false when the "+
			"memory does not exist or was not superseded."),
		mcp.WithString("memory_id", mcp.Required(), mcp.Description("Memory UUID or 8-char prefix")),
	)
	s.AddTool(tool, func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		id, err := req.RequireString("memory_id")
		if err != nil {
			return errResult(err), nil
		}
		var wasSupersededBy string
		if mem, err := store.GetByID(ctx, id); err == nil {
			wasSupersededBy = mem.SupersededBy
		}
		ok, err := store.Restore(ctx, id)
		if err != nil {
			return errResult(err), nil
		}
		return jsonText(map[string]any{
			"restored": ok, "id": id, "was_superseded_by": wasSupersededBy,
		}), nil
	})
}

func addSubgraph(s *server.MCPServer, store *memory.MemoryStore) {
	tool := mcp.NewTool("subgraph",
		mcp.WithDescription("Return the link graph reachable from the given memory ids within depth hops. "+
			"Follows OUTGOING links only (use `neighbors` with direction=\"in\" for the reverse). "+
			"Returns {nodes, edges:[{src,dst,rel}]}."),
		mcp.WithArray("memory_ids", mcp.Required(), mcp.Description("Seed memory ids"),
			mcp.Items(map[string]any{"type": "string"})),
		mcp.WithNumber("depth", mcp.Description("Hops to follow (default: 1)")),
	)
	s.AddTool(tool, func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		raw, ok := req.GetArguments()["memory_ids"].([]any)
		if !ok || len(raw) == 0 {
			return errResult(fmt.Errorf("memory_ids is required")), nil
		}
		ids := make([]string, 0, len(raw))
		for _, v := range raw {
			if s, ok := v.(string); ok {
				ids = append(ids, s)
			}
		}
		graph, err := store.Subgraph(ctx, ids, req.GetInt("depth", 1))
		if err != nil {
			return errResult(err), nil
		}
		nodes := make([]map[string]any, len(graph.Nodes))
		for i, n := range graph.Nodes {
			nodes[i] = memRecord(n)
		}
		edges := make([]map[string]string, len(graph.Edges))
		for i, e := range graph.Edges {
			edges[i] = map[string]string{"src": e.From, "dst": e.To, "rel": e.Rel}
		}
		return jsonText(map[string]any{"nodes": nodes, "edges": edges}), nil
	})
}

func addUndo(s *server.MCPServer, store *memory.MemoryStore) {
	tool := mcp.NewTool("undo",
		mcp.WithDescription("Reverse a journaled mutation — the newest one by default. Pass `ts` to undo "+
			"the entry with that exact journal timestamp (as reported by journal_tail), or `memory_id` to "+
			"undo the newest entry touching that memory. Undo refuses when the current state has diverged "+
			"from the entry's \"after\" snapshot. The undo itself is journaled."),
		mcp.WithNumber("ts", mcp.Description("Exact journal timestamp of the entry to reverse")),
		mcp.WithString("memory_id", mcp.Description("Undo the newest entry touching this memory")),
	)
	s.AddTool(tool, func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		j := store.Journal()
		if j == nil {
			return errResult(fmt.Errorf("journaling is disabled — nothing to undo")), nil
		}
		var entry *memory.JournalEntry
		if p := optFloat32(req, "ts"); p != nil {
			ts := req.GetFloat("ts", 0)
			found, err := j.FindByTS(ts, 1e-3)
			if err != nil || found == nil {
				return errResult(fmt.Errorf("no journal entry at ts=%v", ts)), nil
			}
			entry = found
		} else {
			entries, err := j.Read(memory.ReadOpts{})
			if err != nil {
				return errResult(err), nil
			}
			memID := req.GetString("memory_id", "")
			for i := len(entries) - 1; i >= 0; i-- {
				if memID == "" || memory.EntryTouches(entries[i], memID) {
					entry = &entries[i]
					break
				}
			}
			if entry == nil {
				return errResult(fmt.Errorf("no journal entry to undo")), nil
			}
		}
		ok, err := store.Undo(ctx, *entry)
		if err != nil {
			return errResult(err), nil
		}
		return jsonText(map[string]any{
			"ok": ok, "op": entry.Op, "id": entry.ID, "ts": entry.TS, "actor": entry.Actor,
		}), nil
	})
}

func addNuke(s *server.MCPServer, store *memory.MemoryStore) {
	tool := mcp.NewTool("nuke",
		mcp.WithDescription("Delete EVERY memory in the collection. Irreversible. Guarded: pass "+
			"confirm=\"DELETE ALL\" to proceed. The journal keeps a single 'nuke' entry with the count, "+
			"but the memories themselves are gone — undo cannot bring them back."),
		mcp.WithString("confirm", mcp.Required(), mcp.Description(`Must be exactly "DELETE ALL"`)),
	)
	s.AddTool(tool, func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		if req.GetString("confirm", "") != "DELETE ALL" {
			return jsonText(map[string]any{
				"ok": false, "deleted": 0,
				"error": `refusing to nuke: pass confirm="DELETE ALL"`,
			}), nil
		}
		n, err := store.Nuke(ctx)
		if err != nil {
			return errResult(err), nil
		}
		return jsonText(map[string]any{"ok": true, "deleted": n}), nil
	})
}

func addEvalRecall(s *server.MCPServer, store *memory.MemoryStore) {
	tool := mcp.NewTool("eval_recall",
		mcp.WithDescription("Score retrieval quality against a gold set. Read-only. Each case is "+
			`{"query": str, "relevant_ids": [id, ...], "k"?: int, "mode"?: str}. Returns recall@k, `+
			"precision@k, MRR, MAP and nDCG@k averaged over the cases, plus a per-case breakdown. "+
			"Recall runs with touch=false, so evaluating never perturbs access counts or recency. "+
			"Pass the ranking knobs to A/B a configuration."),
		mcp.WithArray("cases", mcp.Required(), mcp.Description("Gold-set cases"),
			mcp.Items(map[string]any{"type": "object"})),
		mcp.WithNumber("k", mcp.Description("Default top-k when a case omits it (default: 5)")),
		mcp.WithString("mode", mcp.Description("semantic|hybrid (default: hybrid)")),
		mcp.WithString("fusion", mcp.Description("weighted (default) | rrf")),
		mcp.WithNumber("graph", mcp.Description("Graph-proximity weight (hybrid only)")),
		mcp.WithNumber("diversity", mcp.Description("MMR λ in [0,1]")),
		mcp.WithNumber("dedup_threshold", mcp.Description("Drop near-duplicates above this cosine")),
		mcp.WithNumber("min_cosine", mcp.Description("Absolute cosine relevance floor")),
	)
	s.AddTool(tool, func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		raw, ok := req.GetArguments()["cases"].([]any)
		if !ok || len(raw) == 0 {
			return errResult(fmt.Errorf("no cases supplied")), nil
		}
		cases := make([]eval.Case, 0, len(raw))
		for i, item := range raw {
			obj, ok := item.(map[string]any)
			if !ok {
				return errResult(fmt.Errorf("case %d: expected an object", i)), nil
			}
			query, _ := obj["query"].(string)
			if query == "" {
				return errResult(fmt.Errorf("case %d: missing 'query'", i)), nil
			}
			idsRaw, _ := obj["relevant_ids"].([]any)
			if len(idsRaw) == 0 {
				return errResult(fmt.Errorf(
					"case %d: 'relevant_ids' must be a non-empty list", i)), nil
			}
			ids := make([]string, 0, len(idsRaw))
			for _, v := range idsRaw {
				if str, ok := v.(string); ok {
					ids = append(ids, str)
				}
			}
			c := eval.Case{Query: query, RelevantIDs: ids}
			if v, ok := obj["k"].(float64); ok {
				c.K = int(v)
			}
			if v, ok := obj["mode"].(string); ok {
				c.Mode = v
			}
			cases = append(cases, c)
		}

		opts := eval.Options{
			DefaultK:    req.GetInt("k", 5),
			DefaultMode: req.GetString("mode", "hybrid"),
			Recall: eval.RecallOpts{
				Fusion:         req.GetString("fusion", ""),
				Graph:          optFloat32(req, "graph"),
				Diversity:      optFloat32(req, "diversity"),
				DedupThreshold: optFloat32(req, "dedup_threshold"),
				MinCosine:      optFloat32(req, "min_cosine"),
			},
		}
		res, err := eval.Evaluate(ctx, eval.StoreAdapter{Store: store}, cases, opts)
		if err != nil {
			return errResult(err), nil
		}
		return jsonText(res), nil
	})
}

func addReady(s *server.MCPServer, store *memory.MemoryStore) {
	tool := mcp.NewTool("ready",
		mcp.WithDescription("Readiness probe: is the store reachable and the embedder working? Returns "+
			"{ready, checks:{store, embedder}} with the embedder check carrying its measured dimension "+
			"and latency. Unlike the HTTP /ready endpoint this is not sanitized — an MCP client is "+
			"already authenticated."),
	)
	s.AddTool(tool, func(ctx context.Context, _ mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		return jsonText(store.Readiness(ctx, 0)), nil
	})
}

// maintOpts holds the schedule configuration for the maintenance_tick tool,
// wired in by the host process via SetMaintenance. When unset, the tool runs
// ungated (no state file → every job is due, nothing is persisted).
var maintOpts struct {
	cfg       maintenance.Config
	statePath string
	set       bool
}

// SetMaintenance configures the schedule (decay_every/reflect_every gates)
// and state path used by the maintenance_tick tool, mirroring Python's
// config-driven MaintenanceScheduler.
func SetMaintenance(cfg maintenance.Config, statePath string) {
	maintOpts.cfg = cfg
	maintOpts.statePath = statePath
	maintOpts.set = true
}

func addMaintenanceTick(s *server.MCPServer, store *memory.MemoryStore) {
	tool := mcp.NewTool("maintenance_tick",
		mcp.WithDescription("Run one maintenance tick: prune stale memories via decay and optionally "+
			"consolidate episodic clusters via reflection. Jobs are gated on the configured schedule "+
			"(decay_every/reflect_every) — they only run when their interval has elapsed since the "+
			"last recorded run, so the tool is safe to call frequently."),
		mcp.WithNumber("decay_rate", mcp.Description("Decay rate λ (default: from config, 0.1)")),
		mcp.WithNumber("min_score", mcp.Description("Prune threshold (default: from config, 0.05)")),
		mcp.WithNumber("frequency_weight", mcp.Description("Reinforcement: how strongly recall count resists decay (default: from config, 0 = off)")),
		mcp.WithBoolean("reflect", mcp.Description("Also run reflection when due (default: from config, false)")),
		mcp.WithString("consolidate", mcp.Description("Consolidate sources after reflection: none|soft|hard (default: from config, none)")),
	)
	s.AddTool(tool, maintenanceTickHandler(store))
}

// maintenanceTickHandler is exposed for tests.
func maintenanceTickHandler(store *memory.MemoryStore) func(context.Context, mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	return func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		cfg := maintOpts.cfg
		if !maintOpts.set {
			cfg = maintenance.Config{DecayRate: 0.1, MinScore: 0.05}
		}
		if cfg.Summarizer == nil {
			cfg.Summarizer = buildSummarizer()
		}
		// Per-call overrides on top of the configured schedule.
		cfg.DecayRate = float32(req.GetFloat("decay_rate", float64(cfg.DecayRate)))
		cfg.MinScore = float32(req.GetFloat("min_score", float64(cfg.MinScore)))
		cfg.FrequencyWeight = float32(req.GetFloat("frequency_weight", float64(cfg.FrequencyWeight)))
		cfg.Reflect = req.GetBool("reflect", cfg.Reflect)
		if v := req.GetString("consolidate", ""); v != "" {
			cfg.Consolidate = reflectpkg.ConsolidateModeFromString(v)
		}

		res := maintenance.Tick(ctx, store, store, cfg, maintOpts.statePath, float64(time.Now().Unix()))
		if res.Err != nil {
			return errResult(res.Err), nil
		}
		return jsonText(map[string]any{
			"ran_decay":    res.RanDecay,
			"ran_reflect":  res.RanReflect,
			"ran_purge":    res.RanPurge,
			"pruned":       res.Pruned,
			"reflected":    res.Reflected,
			"purged":       res.Purged,
			"trash_purged": res.TrashPurged,
		}), nil
	}
}

func addJournalTail(s *server.MCPServer, store *memory.MemoryStore) {
	tool := mcp.NewTool("journal_tail",
		mcp.WithDescription("Return the most recent audit-journal entries (newest first)."),
		mcp.WithNumber("n", mcp.Description("Max entries (default: 20)")),
		mcp.WithString("op", mcp.Description("Filter by op: remember|forget|supersede|link|...")),
		mcp.WithNumber("since_seconds", mcp.Description("Only entries within last N seconds")),
	)
	s.AddTool(tool, journalTailHandler(store))
}

// journalTailHandler is exposed for tests.
func journalTailHandler(store *memory.MemoryStore) func(context.Context, mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	return func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		j := store.Journal()
		if j == nil {
			return jsonText([]any{}), nil
		}
		n := req.GetInt("n", 20)
		if n <= 0 {
			// A negative n would slice out of bounds; mirror Python's [] here.
			return jsonText([]any{}), nil
		}
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
	}
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
		return jsonText(map[string]any{
			"ok":                  true,
			"imported":            summary.Imported,
			"skipped":             summary.Skipped,
			"overwritten":         summary.Overwritten,
			"renamed":             summary.Renamed,
			"errors":              summary.Errors,
			"vectors_regenerated": summary.VectorsRegenerated,
		}), nil
	})
}
