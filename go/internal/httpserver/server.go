// Package httpserver exposes a MemoryStore over a small JSON HTTP/REST API,
// for clients that cannot speak MCP — web apps, shell scripts, automation
// tools, non-MCP agents.
//
// Routes (all JSON in / JSON out):
//
//	GET    /health                         liveness + memory count (skips auth)
//	GET    /ready                           readiness — backend + embedder probe, 200/503 (skips auth)
//	GET    /stats                          store statistics
//	GET    /metrics                        runtime counters + recall latency
//	GET    /memories?limit=&include_superseded=&include_expired=
//	                                       recent memories (list_recent)
//	POST   /memories                       store a memory (remember; ttl_seconds/expires_at)
//	GET    /memories/{id}                  fetch one memory
//	PATCH  /memories/{id}                  edit fields in place (journaled; expires_at)
//	DELETE /memories/{id}                  forget one memory
//	GET    /memories/{id}/neighbors?rel=&direction=&depth=
//	                                       linked memories
//	GET    /memories/{id}/history          journaled timeline of one memory
//	GET    /memories/{id}/at?ts=           reconstruct one memory as of a past time
//	GET    /state_at?ts=                   reconstruct all live memories as of a past time
//	POST   /purge_expired  {dry_run?}      hard-delete memories whose TTL passed
//	GET    /recall?query=&k=&type=&tag=&min_importance=&source=&since=&until=&mode=&include_expired=&explain=
//	POST   /recall                         same, via JSON body
//	POST   /recall_pack                    token-budgeted context block
//	POST   /auto_context                   fan-out context block for a task
//	POST   /links        {src_id,dst_id,rel?}      add a directed link
//	POST   /unlink       {src_id,dst_id,rel?}      remove link(s)
//	POST   /supersede    {old_id,new_id}           soft-delete + supersede link
//	POST   /restore      {memory_id}               clear a supersede (un-soft-delete)
//	POST   /conflicts    {memory_id?,threshold?}   duplicate / contradiction scan
//	POST   /subgraph     {memory_ids,depth?}       link graph reachable from ids
//	POST   /undo         {ts?,memory_id?}          reverse a journaled mutation
//	POST   /nuke         {confirm}                 delete every memory (guarded)
//	GET    /journal?n=&op=&since=                  audit-journal tail
//	POST   /export       {path,…}                  write a .ahkai archive (server-side path)
//	POST   /import       {path,on_conflict?,…}     read a .ahkai archive (server-side path)
//	POST   /merge        {target_id,other_id,…}    fold one memory into another
//	GET    /memories/{id}/versions                 past text states from the journal
//	GET    /tags?include_superseded=               tag usage counts
//	POST   /tags/rename  {old,new}                 rename a tag collection-wide
//	POST   /tags/merge   {sources,into}            fold several tags into one
//	DELETE /tags/{tag}                             strip a tag from every memory
//	POST   /find_path    {from_id,to_id,max_depth?} shortest undirected link path
//	POST   /trash        {memory_id}               soft-delete (recoverable)
//	GET    /trash                                  list soft-deleted memories
//	POST   /trash/restore {memory_id}              bring one back
//	POST   /trash/purge  {memory_id?, older_than_days?}  permanently drop (irreversible)
//
// Optional bearer-token auth: pass a token (or set AI_HOUKAI_HTTP_TOKEN) and
// every request must carry "Authorization: Bearer <token>". /health and /ready
// stay reachable so liveness/readiness probes work without the secret.
package httpserver

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"math"
	"net"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/nexusriot/ai-houkai/internal/memory"
	"github.com/nexusriot/ai-houkai/internal/timeparse"
)

const maxBody = 4 << 20 // 4 MiB cap on request bodies

// httpError short-circuits a handler with a specific status + message.
type httpError struct {
	status int
	msg    string
}

func (e *httpError) Error() string { return e.msg }

func errStatus(status int, format string, args ...any) *httpError {
	return &httpError{status: status, msg: fmt.Sprintf(format, args...)}
}

// Server wraps a MemoryStore with the HTTP routing table and optional auth.
type Server struct {
	store      *memory.MemoryStore
	path       string
	collection string
	token      string

	// storeMu serialises every store call. Handlers run on separate goroutines,
	// and store operations like Recall's access-count bump, Link, and Supersede
	// are read-modify-write against the backend — concurrent handlers would
	// clobber each other's metadata writes. Mirrors the Python HTTP server's
	// store_lock, held tightly around just the store call (not the response
	// write). Serve is the store's only in-process user, so this is sufficient.
	storeMu sync.Mutex
}

// New builds a Server. path/collection are reported by /health and /stats;
// token (when non-empty) gates every route except /health.
func New(store *memory.MemoryStore, path, collection, token string) *Server {
	return &Server{store: store, path: path, collection: collection, token: token}
}

// Handler returns the routed http.Handler with auth + panic recovery applied.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", s.wrap(s.health))
	mux.HandleFunc("GET /ready", s.wrap(s.ready))
	mux.HandleFunc("GET /stats", s.wrap(s.stats))
	mux.HandleFunc("GET /metrics", s.wrap(s.metrics))
	mux.HandleFunc("GET /memories", s.wrap(s.list))
	mux.HandleFunc("POST /memories", s.wrap(s.remember))
	mux.HandleFunc("POST /memories/batch", s.wrap(s.rememberMany))
	mux.HandleFunc("GET /memories/{id}", s.wrap(s.getOne))
	mux.HandleFunc("PATCH /memories/{id}", s.wrap(s.edit))
	mux.HandleFunc("DELETE /memories/{id}", s.wrap(s.forget))
	mux.HandleFunc("GET /memories/{id}/neighbors", s.wrap(s.neighbors))
	mux.HandleFunc("GET /memories/{id}/history", s.wrap(s.history))
	mux.HandleFunc("GET /memories/{id}/at", s.wrap(s.getAt))
	mux.HandleFunc("GET /state_at", s.wrap(s.stateAt))
	mux.HandleFunc("POST /purge_expired", s.wrap(s.purgeExpired))
	mux.HandleFunc("GET /recall", s.wrap(s.recall))
	mux.HandleFunc("POST /recall", s.wrap(s.recall))
	mux.HandleFunc("POST /recall_pack", s.wrap(s.recallPack))
	mux.HandleFunc("POST /auto_context", s.wrap(s.autoContext))
	mux.HandleFunc("POST /links", s.wrap(s.link))
	mux.HandleFunc("POST /unlink", s.wrap(s.unlink))
	mux.HandleFunc("POST /supersede", s.wrap(s.supersede))
	mux.HandleFunc("POST /restore", s.wrap(s.restore))
	mux.HandleFunc("POST /conflicts", s.wrap(s.conflicts))
	mux.HandleFunc("POST /subgraph", s.wrap(s.subgraph))
	mux.HandleFunc("POST /undo", s.wrap(s.undo))
	mux.HandleFunc("POST /nuke", s.wrap(s.nuke))
	mux.HandleFunc("GET /journal", s.wrap(s.journalTail))
	mux.HandleFunc("POST /export", s.wrap(s.export))
	mux.HandleFunc("POST /import", s.wrap(s.importArchive))
	mux.HandleFunc("POST /merge", s.wrap(s.merge))
	mux.HandleFunc("GET /memories/{id}/versions", s.wrap(s.versions))
	mux.HandleFunc("GET /tags", s.wrap(s.tags))
	mux.HandleFunc("POST /tags/rename", s.wrap(s.renameTag))
	mux.HandleFunc("POST /tags/merge", s.wrap(s.mergeTags))
	mux.HandleFunc("DELETE /tags/{tag}", s.wrap(s.deleteTag))
	mux.HandleFunc("POST /find_path", s.wrap(s.findPath))
	mux.HandleFunc("POST /trash", s.wrap(s.trash))
	mux.HandleFunc("GET /trash", s.wrap(s.trashList))
	mux.HandleFunc("POST /trash/restore", s.wrap(s.trashRestore))
	mux.HandleFunc("POST /trash/purge", s.wrap(s.trashPurge))
	return s.middleware(mux)
}

// ListenAndServe binds host:port and serves until ctx is cancelled.
func (s *Server) ListenAndServe(ctx context.Context, host string, port int) error {
	srv := &http.Server{
		Addr:              net.JoinHostPort(host, strconv.Itoa(port)),
		Handler:           s.Handler(),
		ReadHeaderTimeout: 10 * time.Second,
	}
	go func() {
		<-ctx.Done()
		shutCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = srv.Shutdown(shutCtx)
	}()
	if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		return err
	}
	return nil
}

// middleware applies optional bearer-token auth, turns any panic into a 500 so
// a single bad request never takes the server down, and normalises ServeMux's
// built-in plain-text 404/405 responses into the JSON error envelope.
func (s *Server) middleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if rec := recover(); rec != nil {
				log.Printf("ai-houkai http: panic: %v", rec)
				writeJSON(w, 500, map[string]any{"error": fmt.Sprintf("%v", rec)})
			}
		}()
		if !s.authorized(r) {
			writeJSON(w, 401, map[string]any{"error": "unauthorized"})
			return
		}
		cw := &captureWriter{ResponseWriter: w}
		next.ServeHTTP(cw, r)
		cw.finish()
	})
}

// captureWriter buffers the response so the router's default non-JSON error
// pages (404/405) can be re-rendered as JSON. Handlers that already emit JSON
// pass straight through.
type captureWriter struct {
	http.ResponseWriter
	status      int
	wroteHeader bool
	json        bool
	buf         []byte
}

func (c *captureWriter) WriteHeader(status int) {
	if c.wroteHeader {
		return
	}
	c.wroteHeader = true
	c.status = status
	c.json = strings.HasPrefix(c.Header().Get("Content-Type"), "application/json")
	if c.json {
		c.ResponseWriter.WriteHeader(status)
	}
}

func (c *captureWriter) Write(b []byte) (int, error) {
	if !c.wroteHeader {
		c.WriteHeader(http.StatusOK)
	}
	if c.json {
		return c.ResponseWriter.Write(b)
	}
	c.buf = append(c.buf, b...)
	return len(b), nil
}

// finish flushes a buffered non-JSON response (a ServeMux default error page)
// as a JSON envelope.
func (c *captureWriter) finish() {
	if c.json || !c.wroteHeader {
		return
	}
	msg := strings.TrimSpace(string(c.buf))
	if msg == "" {
		msg = http.StatusText(c.status)
	}
	writeJSON(c.ResponseWriter, c.status, map[string]any{"error": msg})
}

func (s *Server) authorized(r *http.Request) bool {
	if s.token == "" || r.URL.Path == "/health" || r.URL.Path == "/ready" {
		return true
	}
	return r.Header.Get("Authorization") == "Bearer "+s.token
}

// wrap adapts a (status, payload, error) handler to http.HandlerFunc, rendering
// *httpError with its status and any other error as a 500.
type apiFunc func(r *http.Request) (int, any, error)

func (s *Server) wrap(fn apiFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		status, payload, err := s.callLocked(fn, r)
		if err != nil {
			var he *httpError
			if errors.As(err, &he) {
				writeJSON(w, he.status, map[string]any{"error": he.msg})
				return
			}
			// Store-level validation (bad mode/type/rel/polarity …) is caller
			// error, not an internal fault (mirrors Python's ValueError → 400).
			if memory.IsValidationError(err) {
				writeJSON(w, 400, map[string]any{"error": err.Error()})
				return
			}
			writeJSON(w, 500, map[string]any{"error": fmt.Sprintf("%T: %v", err, err)})
			return
		}
		writeJSON(w, status, payload)
	}
}

// callLocked runs a handler while holding storeMu, releasing it (even on panic,
// via defer) before the caller writes the response — the response write must
// not hold the lock. Mirrors Python's `with store_lock: fn(...)`.
func (s *Server) callLocked(fn apiFunc, r *http.Request) (status int, payload any, err error) {
	s.storeMu.Lock()
	defer s.storeMu.Unlock()
	// The body/query coercers reject mistyped values by panicking with an
	// *httpError — Go handler signatures give them no error channel of their
	// own. Anything else keeps panicking.
	defer func() {
		if rec := recover(); rec != nil {
			he, ok := rec.(*httpError)
			if !ok {
				panic(rec)
			}
			status, payload, err = 0, nil, he
		}
	}()
	return fn(r)
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	body, err := json.Marshal(payload)
	if err != nil {
		body, status = []byte(`{"error":"failed to encode response"}`), 500
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_, _ = w.Write(body)
}

func (s *Server) health(r *http.Request) (int, any, error) {
	n, err := s.store.Count(r.Context())
	if err != nil {
		return 0, nil, err
	}
	// Liveness only — the collection name is intentionally omitted so an
	// unauthenticated probe cannot enumerate the store (matches Python).
	return 200, map[string]any{"status": "ok", "count": n}, nil
}

func (s *Server) ready(r *http.Request) (int, any, error) {
	// Readiness (distinct from the always-open liveness /health): exercises the
	// backend and an actual embed so orchestrators learn whether the store can
	// serve requests. 503 when any dependency check fails.
	//
	// Auth-exempt like /health, so the body is deliberately minimal: only the
	// overall flag and a per-check ok bool. Raw error strings, provider
	// latency/dim, and paths are withheld so an unauthenticated probe can't
	// enumerate backend internals — use `houkai doctor` for the detailed
	// report. The cacheTTL bounds embed calls under rapid polling.
	rep := s.store.Readiness(r.Context(), 5*time.Second)
	status := 200
	if b, _ := rep["ready"].(bool); !b {
		status = 503
	}
	safe := map[string]any{"ready": rep["ready"]}
	if checks, ok := rep["checks"].(map[string]any); ok {
		sc := map[string]any{}
		for name, c := range checks {
			okv := false
			if m, ok := c.(map[string]any); ok {
				okv, _ = m["ok"].(bool)
			}
			sc[name] = map[string]any{"ok": okv}
		}
		safe["checks"] = sc
	}
	return status, safe, nil
}

func (s *Server) stats(r *http.Request) (int, any, error) {
	st, err := s.store.Stats(r.Context())
	if err != nil {
		return 0, nil, err
	}
	st["path"] = s.path
	st["collection"] = s.collection
	return 200, st, nil
}

func (s *Server) metrics(r *http.Request) (int, any, error) {
	m, err := s.store.Metrics(r.Context())
	if err != nil {
		return 0, nil, err
	}
	return 200, m, nil
}

func (s *Server) purgeExpired(r *http.Request) (int, any, error) {
	b, err := readBody(r)
	if err != nil {
		return 0, nil, err
	}
	dry := bodyBool(b, "dry_run")
	purged, err := s.store.PurgeExpired(r.Context(), 0, dry)
	if err != nil {
		return 0, nil, err
	}
	ids := make([]string, len(purged))
	for i, p := range purged {
		ids[i] = p.ID
	}
	return 200, map[string]any{"purged": len(purged), "dry_run": dry, "ids": ids}, nil
}

func journalEntryDict(e memory.JournalEntry) map[string]any {
	return map[string]any{
		"ts": e.TS, "op": string(e.Op), "actor": e.Actor, "id": e.ID,
		"before": e.Before, "after": e.After, "meta": e.Meta,
		"summary": e.Summary(),
	}
}

func (s *Server) history(r *http.Request) (int, any, error) {
	id := r.PathValue("id")
	entries, err := s.store.History(r.Context(), id, true)
	if err != nil {
		return 0, nil, err
	}
	// Distinguish an unknown id from a live memory with no journal history.
	if len(entries) == 0 {
		if _, gerr := s.store.GetByID(r.Context(), id); errors.Is(gerr, memory.ErrNotFound) {
			return 0, nil, errStatus(404, "memory not found")
		}
	}
	out := make([]map[string]any, len(entries))
	for i, e := range entries {
		out[i] = journalEntryDict(e)
	}
	return 200, map[string]any{"id": id, "history": out}, nil
}

func (s *Server) stateAt(r *http.Request) (int, any, error) {
	raw := qsStr(r, "ts", "")
	if raw == "" {
		return 0, nil, errStatus(400, "missing required field: ts")
	}
	ts, err := parseTimeVal(raw)
	if err != nil {
		return 0, nil, err
	}
	mems, err := s.store.StateAt(r.Context(), ts)
	if err != nil {
		return 0, nil, err
	}
	out := make([]map[string]any, len(mems))
	for i, m := range mems {
		out[i] = memDict(m)
	}
	return 200, map[string]any{"ts": ts, "count": len(mems), "memories": out}, nil
}

func (s *Server) getAt(r *http.Request) (int, any, error) {
	raw := qsStr(r, "ts", "")
	if raw == "" {
		return 0, nil, errStatus(400, "missing required field: ts")
	}
	ts, err := parseTimeVal(raw)
	if err != nil {
		return 0, nil, err
	}
	mem, err := s.store.GetAt(r.Context(), r.PathValue("id"), ts)
	if err != nil {
		return 0, nil, err
	}
	if mem == nil {
		return 0, nil, errStatus(404, "memory did not exist at that time")
	}
	return 200, memDict(*mem), nil
}

func (s *Server) list(r *http.Request) (int, any, error) {
	limit := qsInt(r, "limit", 20)
	inc := qsBool(r, "include_superseded")
	incExp := qsBool(r, "include_expired")
	mems, err := s.store.ListRecent(r.Context(), limit, inc, incExp)
	if err != nil {
		return 0, nil, err
	}
	out := make([]map[string]any, len(mems))
	for i, m := range mems {
		out[i] = memDict(m)
	}
	return 200, map[string]any{"memories": out}, nil
}

func (s *Server) remember(r *http.Request) (int, any, error) {
	b, err := readBody(r)
	if err != nil {
		return 0, nil, err
	}
	text, err := requireStr(b, "text")
	if err != nil {
		return 0, nil, err
	}
	opts := memory.RememberOpts{
		Type:       memory.MemoryType(bodyStr(b, "type", string(memory.Semantic))),
		Tags:       bodyStrSlice(b, "tags"),
		Importance: bodyFloatPtr(b, "importance"), // nil = unset → store default
		Source:     bodyStr(b, "source", ""),
		Polarity:   int(bodyFloat(b, "polarity", 0)),
		OnConflict: memory.ConflictPolicy(bodyStr(b, "on_conflict", "")),
		Pinned:     bodyBool(b, "pinned"),
		Trust:      memory.TrustLevel(bodyStr(b, "trust", "")),
		Idempotent: bodyBool(b, "idempotent"),
	}
	// Valid time — when the memory was true, as opposed to when we learned it.
	// Probed as float64 like the TTL below: an epoch needs the precision.
	if v, ok := b["valid_from"].(float64); ok {
		opts.ValidFrom = &v
	}
	if v, ok := b["valid_until"].(float64); ok {
		opts.ValidUntil = &v
	}
	// TTL: probe as float64 (JSON numbers) — an epoch needs float64 precision,
	// which bodyFloatPtr (*float32) can't carry.
	if v, ok := b["expires_at"].(float64); ok {
		opts.ExpiresAt = &v
	}
	if v, ok := b["ttl_seconds"].(float64); ok {
		opts.TTLSeconds = &v
	}
	mem, stored, conflicts, err := s.store.Remember(r.Context(), text, opts)
	if err != nil {
		var ce *memory.ConflictError
		if errors.As(err, &ce) {
			return 409, conflictPayload(ce.Conflicts), nil
		}
		return 0, nil, err
	}
	if !stored {
		// Two very different reasons to write nothing. A conflict policy
		// rejected the write → 409 with the conflicts. An idempotent repeat
		// found the existing row and bumped it → the feature working, so 200
		// with that row and stored:false. Mapping both onto 409 made every
		// replayed batch fail against a store that already knew the fact.
		if len(conflicts) > 0 {
			return 409, conflictPayload(conflicts), nil
		}
		out := memDict(mem)
		out["stored"] = false
		return 200, out, nil
	}
	out := memDict(mem)
	out["stored"] = true
	return 201, out, nil
}

func (s *Server) rememberMany(r *http.Request) (int, any, error) {
	b, err := readBody(r)
	if err != nil {
		return 0, nil, err
	}
	rawItems, ok := b["items"].([]any)
	if !ok {
		return 0, nil, errStatus(400, "body must include an 'items' array")
	}
	items := make([]memory.RememberItem, 0, len(rawItems))
	for i, raw := range rawItems {
		it, ok := raw.(map[string]any)
		if !ok {
			return 0, nil, errStatus(400, "items[%d]: must be an object", i)
		}
		text, err := requireStr(it, "text")
		if err != nil {
			return 0, nil, errStatus(400, "items[%d]: missing 'text'", i)
		}
		ri := memory.RememberItem{
			Text: text,
			RememberOpts: memory.RememberOpts{
				Type:       memory.MemoryType(bodyStr(it, "type", string(memory.Semantic))),
				Tags:       bodyStrSlice(it, "tags"),
				Importance: bodyFloatPtr(it, "importance"),
				Source:     bodyStr(it, "source", ""),
				Polarity:   int(bodyFloat(it, "polarity", 0)),
			},
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
	if v, ok := b["batch_size"].(float64); ok {
		batchSize = int(v)
	}
	started := float64(time.Now().UnixNano()) / 1e9
	mems, err := s.store.RememberMany(
		r.Context(), items, batchSize,
		memory.ConflictPolicy(bodyStr(b, "on_conflict", "")),
		bodyBool(b, "idempotent"),
	)
	if err != nil {
		return 0, nil, err // validation (raise / bad item) → 400 via wrap
	}
	out := make([]map[string]any, len(mems))
	// `stored` is how many rows the batch CREATED. An idempotent replay returns
	// the pre-existing rows, and reporting len(mems) told the client it had
	// written N rows when it had written none. Counting distinct new ids also
	// collapses intra-batch duplicates, which map to one row.
	created := map[string]bool{}
	for i, m := range mems {
		out[i] = memDict(m)
		if m.CreatedAt >= started {
			created[m.ID] = true
		}
	}
	status := 201
	if len(created) == 0 {
		status = 200
	}
	return status, map[string]any{"stored": len(created), "memories": out}, nil
}

func (s *Server) getOne(r *http.Request) (int, any, error) {
	mem, err := s.store.GetByID(r.Context(), r.PathValue("id"))
	if errors.Is(err, memory.ErrNotFound) {
		return 0, nil, errStatus(404, "memory not found")
	}
	if err != nil {
		return 0, nil, err
	}
	return 200, memDict(mem), nil
}

// edit updates fields of a memory in place (PATCH /memories/{id}).
// Uniform null semantics mirror Python: null / omitted means "leave
// unchanged" for every field except `source`, where an explicit null clears.
// An explicit [] clears tags.
func (s *Server) edit(r *http.Request) (int, any, error) {
	b, err := readBody(r)
	if err != nil {
		return 0, nil, err
	}
	if len(b) == 0 {
		return 0, nil, errStatus(400, "empty edit: provide at least one field")
	}
	var opts memory.EditOpts
	// These four count as editable fields too: `{"pinned": true}` or a
	// validity correction alone is a legitimate edit, and dropping to the
	// "no editable fields" 400 silently made pin/trust/valid-time edits
	// impossible over HTTP.
	fields := 0
	if v, ok := b["pinned"].(bool); ok {
		opts.Pinned = &v
		fields++
	}
	if v, ok := b["trust"].(string); ok && v != "" {
		t := memory.TrustLevel(v)
		opts.Trust = &t
		fields++
	}
	// Valid time: correct the interval during which the memory was true. 0 on
	// an end reopens it.
	if v, ok := b["valid_from"].(float64); ok {
		opts.ValidFrom = &v
		fields++
	}
	if v, ok := b["valid_until"].(float64); ok {
		opts.ValidUntil = &v
		fields++
	}
	if v, ok := b["text"].(string); ok {
		opts.Text = &v
		fields++
	}
	if v, ok := b["type"].(string); ok {
		mt := memory.MemoryType(v)
		opts.Type = &mt
		fields++
	}
	if raw, present := b["tags"]; present && raw != nil {
		if arr, ok := raw.([]any); ok {
			tags := make([]string, 0, len(arr))
			for _, t := range arr {
				if ts, ok := t.(string); ok {
					tags = append(tags, ts)
				}
			}
			opts.Tags = tags
			fields++
		}
	}
	if p := bodyFloatPtr(b, "importance"); p != nil {
		opts.Importance = p
		fields++
	}
	if raw, present := b["polarity"]; present && raw != nil {
		n := int(bodyFloat(b, "polarity", 0))
		opts.Polarity = &n
		fields++
	}
	if raw, present := b["expires_at"]; present && raw != nil {
		if v, ok := raw.(float64); ok { // 0 clears the TTL
			opts.ExpiresAt = &v
			fields++
		}
	}
	if _, present := b["source"]; present {
		src, _ := b["source"].(string) // null → "" (clears)
		opts.Source = &src
		fields++
	}
	if fields == 0 {
		return 0, nil, errStatus(400, "no editable fields in body")
	}
	mem, err := s.store.Edit(r.Context(), r.PathValue("id"), opts)
	if err != nil {
		if errors.Is(err, memory.ErrNotFound) {
			return 0, nil, errStatus(404, "memory not found")
		}
		return 0, nil, err // validation → 400 via wrap
	}
	return 200, memDict(mem), nil
}

func (s *Server) forget(r *http.Request) (int, any, error) {
	id := r.PathValue("id")
	ok, err := s.store.Forget(r.Context(), id)
	if err != nil {
		return 0, nil, err
	}
	if !ok {
		return 0, nil, errStatus(404, "memory not found")
	}
	return 200, map[string]any{"forgotten": true, "id": id}, nil
}

func (s *Server) neighbors(r *http.Request) (int, any, error) {
	hits, err := s.store.Neighbors(
		r.Context(),
		r.PathValue("id"),
		qsStr(r, "rel", ""),
		qsStr(r, "direction", "both"),
		qsInt(r, "depth", 1),
	)
	if err != nil {
		return 0, nil, err
	}
	out := make([]map[string]any, len(hits))
	for i, h := range hits {
		d := memDict(h.Memory)
		d["rel"] = h.Rel
		out[i] = d
	}
	return 200, map[string]any{"neighbors": out}, nil
}

func (s *Server) recall(r *http.Request) (int, any, error) {
	query, opts, k, err := s.recallParams(r)
	if err != nil {
		return 0, nil, err
	}
	hits, err := s.store.Recall(r.Context(), query, k, opts)
	if err != nil {
		return 0, nil, err
	}
	out := make([]map[string]any, len(hits))
	for i, h := range hits {
		d := memDict(h.Memory)
		d["score"] = round4(h.Score)
		if h.Explain != nil {
			d["explain"] = h.Explain
		}
		out[i] = d
	}
	return 200, map[string]any{"results": out}, nil
}

func (s *Server) recallPack(r *http.Request) (int, any, error) {
	b, err := readBody(r)
	if err != nil {
		return 0, nil, err
	}
	query, err := requireStr(b, "query")
	if err != nil {
		return 0, nil, err
	}
	since, until, err := bodySinceUntil(b)
	if err != nil {
		return 0, nil, err
	}
	packOpts := memory.PackOpts{
		TokenBudget:       int(bodyFloat(b, "token_budget", 800)),
		Type:              memory.MemoryType(bodyStr(b, "type", "")),
		Tag:               bodyStr(b, "tag", ""),
		MinImportance:     float32(bodyFloat(b, "min_importance", 0)),
		Source:            bodyStr(b, "source", ""),
		Since:             since,
		Until:             until,
		Mode:              memory.RecallMode(bodyStr(b, "mode", string(memory.ModeHybrid))),
		MaxItems:          int(bodyFloat(b, "max_items", 50)),
		IncludeSuperseded: bodyBool(b, "include_superseded"),
		Fusion:            memory.FusionMode(bodyStr(b, "fusion", "")),
		Weights:           weightsFromBody(b),
		Diversity:         bodyFloatPtr(b, "diversity"),
		DedupThreshold:    bodyFloatPtr(b, "dedup_threshold"),
		MinCosine:         bodyFloatPtr(b, "min_cosine"),
		Expand:            expandFromBody(b),
		Compress:          bodyBool(b, "compress"),
		CompressThreshold: float32(bodyFloat(b, "compress_threshold", 0.30)),
		CompressMinGroup:  int(bodyFloat(b, "compress_min_group", 2)),
		// Same gap as recall had: these were reachable through the MCP tool but
		// not over HTTP, so an HTTP client could not set a trust floor on the
		// one call whose output goes straight into a model's context.
		MinTrust:      memory.TrustLevel(bodyStr(b, "min_trust", "")),
		LexicalIndex:  memory.LexicalIndexMode(bodyStr(b, "lexical_index", "")),
		IncludePinned: bodyBool(b, "include_pinned"),
		AsOf:          bodyFloat(b, "as_of", 0),
	}
	// header: absent → default; explicit "" → no header line (PackOpts.Header
	// contract). bodyStr can't express the empty string, so probe the map.
	if v, ok := b["header"].(string); ok {
		packOpts.Header = &v
	}
	res, err := s.store.RecallPack(r.Context(), query, packOpts)
	if err != nil {
		return 0, nil, err
	}
	return 200, packResponse(res), nil
}

// packResponse renders a PackResult as an HTTP JSON payload, including any
// compressed groups.
func packResponse(res memory.PackResult) map[string]any {
	items := make([]map[string]any, len(res.Items))
	for i, p := range res.Items {
		d := memDict(p.Memory)
		d["score"] = round4(p.Score)
		d["tokens"] = p.Tokens
		items[i] = d
	}
	out := map[string]any{
		"text":        res.Text,
		"used_tokens": res.UsedTokens,
		"budget":      res.Budget,
		"truncated":   res.Truncated,
		"items":       items,
	}
	if len(res.CompressedGroups) > 0 {
		groups := make([]map[string]any, len(res.CompressedGroups))
		for i, g := range res.CompressedGroups {
			groups[i] = map[string]any{
				"ids": g.IDs(), "count": len(g.Memories), "text": g.Text, "tokens": g.Tokens,
			}
		}
		out["compressed_groups"] = groups
	}
	return out
}

func (s *Server) autoContext(r *http.Request) (int, any, error) {
	b, err := readBody(r)
	if err != nil {
		return 0, nil, err
	}
	task, err := requireStr(b, "task")
	if err != nil {
		return 0, nil, err
	}
	maxPhrases := int(bodyFloat(b, "max_phrases", 3))
	res, err := s.store.AutoContextPack(r.Context(), task, memory.AutoContextOpts{
		TokenBudget:       int(bodyFloat(b, "token_budget", 800)),
		MaxPhrases:        maxPhrases,
		Mode:              memory.RecallMode(bodyStr(b, "mode", string(memory.ModeHybrid))),
		MinCosine:         bodyFloatPtr(b, "min_cosine"),
		Compress:          bodyBool(b, "compress"),
		CompressThreshold: float32(bodyFloat(b, "compress_threshold", 0.30)),
		CompressMinGroup:  int(bodyFloat(b, "compress_min_group", 2)),
		LexicalIndex:      memory.LexicalIndexMode(bodyStr(b, "lexical_index", "")),
		MinTrust:          memory.TrustLevel(bodyStr(b, "min_trust", "")),
	})
	if err != nil {
		return 0, nil, err
	}
	out := packResponse(res)
	out["queries"] = append([]string{task}, memory.ExtractKeyPhrases(task, maxPhrases)...)
	return 200, out, nil
}

func (s *Server) link(r *http.Request) (int, any, error) {
	b, err := readBody(r)
	if err != nil {
		return 0, nil, err
	}
	src, err := requireStr(b, "src_id")
	if err != nil {
		return 0, nil, err
	}
	dst, err := requireStr(b, "dst_id")
	if err != nil {
		return 0, nil, err
	}
	if err := s.store.Link(r.Context(), src, dst, bodyStr(b, "rel", memory.RelRelated)); err != nil {
		if errors.Is(err, memory.ErrNotFound) {
			return 0, nil, errStatus(404, "%s", err.Error())
		}
		return 0, nil, err // validation → 400 via wrap
	}
	return 200, map[string]any{"ok": true}, nil
}

func (s *Server) unlink(r *http.Request) (int, any, error) {
	b, err := readBody(r)
	if err != nil {
		return 0, nil, err
	}
	src, err := requireStr(b, "src_id")
	if err != nil {
		return 0, nil, err
	}
	dst, err := requireStr(b, "dst_id")
	if err != nil {
		return 0, nil, err
	}
	removed, err := s.store.Unlink(r.Context(), src, dst, bodyStr(b, "rel", ""))
	if err != nil {
		return 0, nil, err
	}
	return 200, map[string]any{"removed": removed}, nil
}

func (s *Server) supersede(r *http.Request) (int, any, error) {
	b, err := readBody(r)
	if err != nil {
		return 0, nil, err
	}
	oldID, err := requireStr(b, "old_id")
	if err != nil {
		return 0, nil, err
	}
	newID, err := requireStr(b, "new_id")
	if err != nil {
		return 0, nil, err
	}
	if err := s.store.Supersede(r.Context(), oldID, newID); err != nil {
		if errors.Is(err, memory.ErrNotFound) {
			return 0, nil, errStatus(404, "%s", err.Error())
		}
		return 0, nil, err // validation (self/cycle) → 400 via wrap
	}
	return 200, map[string]any{"ok": true}, nil
}

func (s *Server) conflicts(r *http.Request) (int, any, error) {
	b, err := readBody(r)
	if err != nil {
		return 0, nil, err
	}
	found, err := s.store.FindConflicts(r.Context(),
		bodyStr(b, "memory_id", ""), float32(bodyFloat(b, "threshold", 0)))
	if err != nil {
		return 0, nil, err
	}
	out := make([]map[string]any, len(found))
	for i, c := range found {
		out[i] = map[string]any{
			"kind": string(c.Kind), "reason": c.Reason, "similarity": c.Similarity,
			"a": map[string]any{"id": c.A.ID, "text": clip(c.A.Text, 120), "type": string(c.A.Type)},
			"b": map[string]any{"id": c.B.ID, "text": clip(c.B.Text, 120), "type": string(c.B.Type)},
		}
	}
	return 200, map[string]any{"conflicts": out}, nil
}

func (s *Server) restore(r *http.Request) (int, any, error) {
	b, err := readBody(r)
	if err != nil {
		return 0, nil, err
	}
	id, err := requireStr(b, "memory_id")
	if err != nil {
		return 0, nil, err
	}
	if _, err := s.store.GetByID(r.Context(), id); err != nil {
		return 0, nil, errStatus(404, "memory not found")
	}
	ok, err := s.store.Restore(r.Context(), id)
	if err != nil {
		return 0, nil, err
	}
	return 200, map[string]any{"restored": ok, "id": id}, nil
}

func (s *Server) subgraph(r *http.Request) (int, any, error) {
	b, err := readBody(r)
	if err != nil {
		return 0, nil, err
	}
	var ids []string
	switch v := b["memory_ids"].(type) {
	case string:
		ids = []string{v}
	case []any:
		for _, x := range v {
			if str, ok := x.(string); ok {
				ids = append(ids, str)
			}
		}
	}
	if len(ids) == 0 {
		return 0, nil, errStatus(400, "missing required field: memory_ids")
	}
	graph, err := s.store.Subgraph(r.Context(), ids, int(bodyFloat(b, "depth", 1)))
	if err != nil {
		return 0, nil, err
	}
	nodes := make([]map[string]any, len(graph.Nodes))
	for i, n := range graph.Nodes {
		nodes[i] = memDict(n)
	}
	edges := make([]map[string]any, len(graph.Edges))
	for i, e := range graph.Edges {
		edges[i] = map[string]any{"src": e.From, "dst": e.To, "rel": e.Rel}
	}
	return 200, map[string]any{"nodes": nodes, "edges": edges}, nil
}

// undo reverses a journaled mutation: the newest, one by exact ts, or the
// newest touching a given memory.
func (s *Server) undo(r *http.Request) (int, any, error) {
	b, err := readBody(r)
	if err != nil {
		return 0, nil, err
	}
	j := s.store.Journal()
	if j == nil {
		return 0, nil, errStatus(409, "journaling is disabled — nothing to undo")
	}
	var entry *memory.JournalEntry
	if p := bodyFloatPtr(b, "ts"); p != nil {
		ts := bodyFloat(b, "ts", 0)
		found, ferr := j.FindByTS(ts, 1e-3)
		if ferr != nil || found == nil {
			return 0, nil, errStatus(404, "no journal entry at ts=%v", ts)
		}
		entry = found
	} else {
		entries, rerr := j.Read(memory.ReadOpts{})
		if rerr != nil {
			return 0, nil, rerr
		}
		memID := bodyStr(b, "memory_id", "")
		for i := len(entries) - 1; i >= 0; i-- {
			if memID == "" || memory.EntryTouches(entries[i], memID) {
				entry = &entries[i]
				break
			}
		}
		if entry == nil {
			return 0, nil, errStatus(404, "no journal entry to undo")
		}
	}
	ok, err := s.store.Undo(r.Context(), *entry)
	if err != nil {
		return 0, nil, err
	}
	return 200, map[string]any{"ok": ok, "op": string(entry.Op), "id": entry.ID,
		"ts": entry.TS, "actor": entry.Actor}, nil
}

// nuke deletes every memory, guarded by an explicit confirm string so a stray
// request can't empty the store.
func (s *Server) nuke(r *http.Request) (int, any, error) {
	b, err := readBody(r)
	if err != nil {
		return 0, nil, err
	}
	if bodyStr(b, "confirm", "") != "DELETE ALL" {
		return 0, nil, errStatus(400, `refusing to nuke: pass {"confirm": "DELETE ALL"}`)
	}
	n, err := s.store.Nuke(r.Context())
	if err != nil {
		return 0, nil, err
	}
	return 200, map[string]any{"ok": true, "deleted": n}, nil
}

func (s *Server) journalTail(r *http.Request) (int, any, error) {
	j := s.store.Journal()
	if j == nil {
		return 200, map[string]any{"count": 0, "entries": []any{}}, nil
	}
	since, _, err := timeparse.Parse(r.URL.Query().Get("since"))
	if err != nil {
		return 0, nil, errStatus(400, "%s", err.Error())
	}
	entries, err := j.Read(memory.ReadOpts{
		Op:    r.URL.Query().Get("op"),
		Since: since,
	})
	if err != nil {
		return 0, nil, err
	}
	if n := qsInt(r, "n", 20); n > 0 && len(entries) > n {
		entries = entries[len(entries)-n:]
	}
	out := make([]map[string]any, len(entries))
	for i, e := range entries {
		out[i] = journalEntryDict(e)
	}
	return 200, map[string]any{"count": len(out), "entries": out}, nil
}

// export writes a .ahkai archive to a server-side path. The path is resolved on
// the server, so this route is only as safe as the token protecting it.
func (s *Server) export(r *http.Request) (int, any, error) {
	b, err := readBody(r)
	if err != nil {
		return 0, nil, err
	}
	path, err := requireStr(b, "path")
	if err != nil {
		return 0, nil, err
	}
	since, err := parseTimeVal(b["since"])
	if err != nil {
		return 0, nil, err
	}
	var types []memory.MemoryType
	for _, t := range bodyStrSlice(b, "types") {
		types = append(types, memory.MemoryType(t))
	}
	summary, err := s.store.Export(r.Context(), path, memory.ExportOpts{
		IncludeVectors:    bodyBoolDef(b, "include_vectors", true),
		IncludeSuperseded: bodyBool(b, "include_superseded"),
		Types:             types,
		Tags:              bodyStrSlice(b, "tags"),
		Since:             since,
	})
	if err != nil {
		return 0, nil, err
	}
	return 200, summary, nil
}

func (s *Server) importArchive(r *http.Request) (int, any, error) {
	b, err := readBody(r)
	if err != nil {
		return 0, nil, err
	}
	path, err := requireStr(b, "path")
	if err != nil {
		return 0, nil, err
	}
	summary, err := s.store.Import(r.Context(), path, memory.ImportOpts{
		OnConflict:        memory.ImportConflictPolicy(bodyStr(b, "on_conflict", "")),
		RegenerateVectors: bodyBool(b, "regenerate_vectors"),
		DryRun:            bodyBool(b, "dry_run"),
	})
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return 0, nil, errStatus(404, "archive not found: %s", path)
		}
		// Mirror Python's status mapping: id collisions are a conflict (409),
		// any other import failure (bad format, model mismatch) is caller
		// error (400) — neither is an internal fault.
		var conflict *memory.ImportConflictError
		if errors.As(err, &conflict) {
			return 0, nil, errStatus(409, "%s", err.Error())
		}
		return 0, nil, errStatus(400, "%s", err.Error())
	}
	return 200, summary, nil
}


// recallParams pulls recall arguments from a JSON body (POST) or query (GET).
func (s *Server) recallParams(r *http.Request) (string, memory.RecallOpts, int, error) {
	if r.Method == http.MethodPost {
		b, err := readBody(r)
		if err != nil {
			return "", memory.RecallOpts{}, 0, err
		}
		query, err := requireStr(b, "query")
		if err != nil {
			return "", memory.RecallOpts{}, 0, err
		}
		since, until, err := bodySinceUntil(b)
		if err != nil {
			return "", memory.RecallOpts{}, 0, err
		}
		return query, memory.RecallOpts{
			Type:              memory.MemoryType(bodyStr(b, "type", "")),
			Tag:               bodyStr(b, "tag", ""),
			MinImportance:     float32(bodyFloat(b, "min_importance", 0)),
			Source:            bodyStr(b, "source", ""),
			Since:             since,
			Until:             until,
			Mode:              memory.RecallMode(bodyStr(b, "mode", string(memory.ModeSemantic))),
			Overfetch:         int(bodyFloat(b, "overfetch", 4)),
			IncludeSuperseded: bodyBool(b, "include_superseded"),
			IncludeExpired:    bodyBool(b, "include_expired"),
			Fusion:            memory.FusionMode(bodyStr(b, "fusion", "")),
			Weights:           weightsFromBody(b),
			Diversity:         bodyFloatPtr(b, "diversity"),
			DedupThreshold:    bodyFloatPtr(b, "dedup_threshold"),
			MinCosine:         bodyFloatPtr(b, "min_cosine"),
			Expand:            expandFromBody(b),
			NoTouch:           !bodyBoolDef(b, "touch", true),
			Explain:           bodyBool(b, "explain"),
			// These three were reachable through the MCP tool and through the
			// Python port's HTTP recall, but not here — so an HTTP client of the
			// Go server had no trust floor at all.
			MinTrust:     memory.TrustLevel(bodyStr(b, "min_trust", "")),
			LexicalIndex: memory.LexicalIndexMode(bodyStr(b, "lexical_index", "")),
			AsOf:         bodyFloat(b, "as_of", 0),
		}, int(bodyFloat(b, "k", 5)), nil
	}

	query := qsStr(r, "query", "")
	if query == "" {
		return "", memory.RecallOpts{}, 0, errStatus(400, "missing required field: query")
	}
	since, until, err := qsSinceUntil(r)
	if err != nil {
		return "", memory.RecallOpts{}, 0, err
	}
	return query, memory.RecallOpts{
		Type:              memory.MemoryType(qsStr(r, "type", "")),
		Tag:               qsStr(r, "tag", ""),
		MinImportance:     float32(qsFloat(r, "min_importance", 0)),
		Source:            qsStr(r, "source", ""),
		Since:             since,
		Until:             until,
		Mode:              memory.RecallMode(qsStr(r, "mode", string(memory.ModeSemantic))),
		Overfetch:         qsInt(r, "overfetch", 4),
		IncludeSuperseded: qsBool(r, "include_superseded"),
		IncludeExpired:    qsBool(r, "include_expired"),
		Explain:           qsBool(r, "explain"),
		// Plain scalars, so they map onto a query string too. touch=false
		// lets eval/monitoring traffic recall without inflating access
		// counters, which feed decay reinforcement.
		NoTouch:      !qsBoolDef(r, "touch", true),
		MinTrust:     memory.TrustLevel(qsStr(r, "min_trust", "")),
		LexicalIndex: memory.LexicalIndexMode(qsStr(r, "lexical_index", "")),
		AsOf:         qsFloat(r, "as_of", 0),
	}, qsInt(r, "k", 5), nil
}

// weightsFromBody builds HybridWeights from a body `graph` field. It starts
// from DefaultWeights so exposing only `graph` doesn't zero the core weights
// (which Recall would reject) — graph is a pure add-on. Absent → zero value,
// which Recall replaces with DefaultWeights.
func weightsFromBody(b map[string]any) memory.HybridWeights {
	if p := bodyFloatPtr(b, "graph"); p != nil {
		w := memory.DefaultWeights()
		w.Graph = *p
		return w
	}
	return memory.HybridWeights{}
}

// expandFromBody builds an *ExpandSpec from a body `expand` object (nil for no
// expansion). Unspecified fields fall back to the same defaults as Python's
// ExpandSpec so the two ports behave identically.
func expandFromBody(b map[string]any) *memory.ExpandSpec {
	raw, ok := b["expand"].(map[string]any)
	if !ok {
		return nil
	}
	spec := &memory.ExpandSpec{
		Rels:  []string{"refines", "example_of"},
		Depth: 1, Cap: 5, Score: 0.70, Decay: 1.0,
	}
	if v, ok := raw["rels"].([]any); ok && len(v) > 0 {
		rels := make([]string, 0, len(v))
		for _, r := range v {
			if s, ok := r.(string); ok {
				rels = append(rels, s)
			}
		}
		spec.Rels = rels
	}
	if v, ok := raw["depth"].(float64); ok {
		spec.Depth = int(v)
	}
	if v, ok := raw["cap"].(float64); ok {
		spec.Cap = int(v)
	}
	if v, ok := raw["score"].(float64); ok {
		spec.Score = float32(v)
	}
	if v, ok := raw["decay"].(float64); ok {
		spec.Decay = float32(v)
	}
	if v, ok := raw["rerank"].(bool); ok {
		spec.Rerank = v
	}
	return spec
}

func memDict(m memory.Memory) map[string]any {
	links := make([]map[string]any, len(m.Links))
	for i, l := range m.Links {
		links[i] = map[string]any{"to": l.To, "rel": l.Rel}
	}
	// Match Python's _mem_dict, which emits None (JSON null) for these when
	// empty rather than "" / 0.
	var superseded any
	if m.SupersededBy != "" {
		superseded = m.SupersededBy
	}
	var supersededAt any
	if m.SupersededAt != 0 {
		supersededAt = m.SupersededAt
	}
	var source any
	if m.Source != "" {
		source = m.Source
	}
	var expiresAt any
	if m.ExpiresAt != 0 {
		expiresAt = m.ExpiresAt
	}
	return map[string]any{
		"id":            m.ID,
		"text":          m.Text,
		"type":          string(m.Type),
		"tags":          m.Tags,
		"importance":    m.Importance,
		"source":        source,
		"created_at":    m.CreatedAt,
		"last_accessed": m.LastAccessed,
		"access_count":  m.AccessCount,
		"polarity":      m.Polarity,
		"links":         links,
		"superseded_by": superseded,
		"superseded_at": supersededAt,
		"expires_at":    expiresAt,
		// Always present, never elided to null: a REST client can set these on
		// a write, so it has to be able to read them back — and "absent" would
		// be indistinguishable from "not pinned" / "unlabelled".
		"pinned": m.Pinned,
		"trust":  string(memory.TrustOrDefault(m.Trust)),
		// Valid time — when the memory was true, as opposed to when we learned
		// it. 0 on either end means unbounded.
		"valid_from":  m.ValidFrom,
		"valid_until": m.ValidUntil,
	}
}

func conflictPayload(cs []memory.Conflict) map[string]any {
	out := make([]map[string]any, len(cs))
	for i, c := range cs {
		out[i] = map[string]any{
			"kind": string(c.Kind), "similarity": c.Similarity,
			"other_id": c.B.ID, "other_text": clip(c.B.Text, 100),
		}
	}
	return map[string]any{"stored": false, "conflicts": out}
}

func round4(f float32) float64 {
	// math.Round, not int64(x+0.5) truncation — the latter rounds negative
	// scores (e.g. a min_cosine explain value) toward zero instead of to the
	// nearest step.
	return math.Round(float64(f)*1e4) / 1e4
}

func clip(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n]
}

func readBody(r *http.Request) (map[string]any, error) {
	if r.ContentLength > maxBody {
		return nil, errStatus(413, "request body too large")
	}
	raw, err := io.ReadAll(io.LimitReader(r.Body, maxBody+1))
	if err != nil {
		return nil, errStatus(400, "failed to read body")
	}
	if int64(len(raw)) > maxBody {
		return nil, errStatus(413, "request body too large")
	}
	if len(raw) == 0 {
		return map[string]any{}, nil
	}
	var data any
	if err := json.Unmarshal(raw, &data); err != nil {
		return nil, errStatus(400, "invalid JSON body: %v", err)
	}
	obj, ok := data.(map[string]any)
	if !ok {
		return nil, errStatus(400, "JSON body must be an object")
	}
	return obj, nil
}

func requireStr(b map[string]any, key string) (string, error) {
	v, ok := b[key]
	if !ok || v == nil {
		return "", errStatus(400, "missing required field: %s", key)
	}
	s, ok := v.(string)
	if !ok || s == "" {
		return "", errStatus(400, "missing required field: %s", key)
	}
	return s, nil
}

// The body coercers below reject mistyped present values by panicking with
// an *httpError, recovered in callLocked into a 400 (mirrors Python's
// coercers raising HttpError). Silent defaulting was destructive in the
// worst case: `{"dry_run": "true"}` coerced to false and ran a real purge.

func bodyStr(b map[string]any, key, def string) string {
	if v, ok := b[key].(string); ok && v != "" {
		return v
	}
	return def
}

func bodyFloat(b map[string]any, key string, def float64) float64 {
	switch v := b[key].(type) {
	case nil:
		return def
	case bool:
		// bool would coerce to 0/1 — reject explicitly, like Python.
	case float64:
		return v
	case json.Number:
		f, _ := v.Float64()
		return f
	case string:
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f
		}
	}
	panic(errStatus(400, "%s: not a valid number", key))
}

func bodyBool(b map[string]any, key string) bool {
	return bodyBoolDef(b, key, false)
}

// bodyBoolDef reads an optional bool defaulting to def when the key is absent.
func bodyBoolDef(b map[string]any, key string, def bool) bool {
	switch v := b[key].(type) {
	case nil:
		return def
	case bool:
		return v
	case string:
		// JSON string "false" must not be truthy (mirrors Python).
		switch strings.ToLower(v) {
		case "1", "true", "yes", "on":
			return true
		}
		return false
	case float64:
		return v != 0
	case json.Number:
		f, _ := v.Float64()
		return f != 0
	}
	panic(errStatus(400, "%s: not a valid boolean", key))
}

// bodyFloatPtr returns a *float32 for an optional numeric field (nil = absent).
func bodyFloatPtr(b map[string]any, key string) *float32 {
	switch v := b[key].(type) {
	case nil:
		return nil
	case bool:
	case float64:
		x := float32(v)
		return &x
	case json.Number:
		f, _ := v.Float64()
		x := float32(f)
		return &x
	case string:
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			x := float32(f)
			return &x
		}
	}
	panic(errStatus(400, "%s: not a valid number", key))
}

// bodyStrSlice coerces a string-list field. A lone string becomes a
// one-element list — passing it through raw would drop the value entirely.
// Anything other than a string / list of strings is a 400 (mirrors Python's
// _body_tags).
func bodyStrSlice(b map[string]any, key string) []string {
	switch v := b[key].(type) {
	case nil:
		return nil
	case string:
		return []string{v}
	case []any:
		out := make([]string, 0, len(v))
		for _, x := range v {
			s, ok := x.(string)
			if !ok {
				panic(errStatus(400, "%s: must be a string or a list of strings", key))
			}
			out = append(out, s)
		}
		return out
	}
	panic(errStatus(400, "%s: must be a string or a list of strings", key))
}

// bodySinceUntil resolves the since/until time filters from a JSON body. They
// may arrive as numbers (epoch) or strings (ISO / relative span like "7d").
func bodySinceUntil(b map[string]any) (since, until float64, err error) {
	if since, err = parseTimeVal(b["since"]); err != nil {
		return 0, 0, err
	}
	if until, err = parseTimeVal(b["until"]); err != nil {
		return 0, 0, err
	}
	return since, until, nil
}

func parseTimeVal(v any) (float64, error) {
	switch x := v.(type) {
	case nil:
		return 0, nil
	case float64:
		return x, nil
	case json.Number:
		f, _ := x.Float64()
		return f, nil
	case string:
		ts, _, perr := timeparse.Parse(x)
		if perr != nil {
			return 0, errStatus(400, "%s", perr.Error())
		}
		return ts, nil
	}
	return 0, errStatus(400, "invalid time value: %v", v)
}

func qsStr(r *http.Request, key, def string) string {
	if v := r.URL.Query().Get(key); v != "" {
		return v
	}
	return def
}

func qsInt(r *http.Request, key string, def int) int {
	v := r.URL.Query().Get(key)
	if v == "" {
		return def
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		panic(errStatus(400, "'%s' is not a valid integer", v))
	}
	return n
}

func qsFloat(r *http.Request, key string, def float64) float64 {
	v := r.URL.Query().Get(key)
	if v == "" {
		return def
	}
	f, err := strconv.ParseFloat(v, 64)
	if err != nil {
		panic(errStatus(400, "'%s' is not a valid number", v))
	}
	return f
}

func qsBool(r *http.Request, key string) bool {
	return qsBoolDef(r, key, false)
}

func qsBoolDef(r *http.Request, key string, def bool) bool {
	v := r.URL.Query().Get(key)
	if v == "" {
		return def
	}
	switch strings.ToLower(v) {
	case "1", "true", "yes", "on":
		return true
	}
	return false
}

func qsSinceUntil(r *http.Request) (since, until float64, err error) {
	if since, _, err = timeparse.Parse(r.URL.Query().Get("since")); err != nil {
		return 0, 0, errStatus(400, "%s", err.Error())
	}
	if until, _, err = timeparse.Parse(r.URL.Query().Get("until")); err != nil {
		return 0, 0, errStatus(400, "%s", err.Error())
	}
	return since, until, nil
}
