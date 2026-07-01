// Package httpserver exposes a MemoryStore over a small JSON HTTP/REST API,
// for clients that cannot speak MCP — web apps, shell scripts, automation
// tools, non-MCP agents.
//
// Routes (all JSON in / JSON out):
//
//	GET    /health                         liveness + memory count (skips auth)
//	GET    /stats                          store statistics
//	GET    /memories?limit=&include_superseded=
//	                                       recent memories (list_recent)
//	POST   /memories                       store a memory (remember)
//	GET    /memories/{id}                  fetch one memory
//	DELETE /memories/{id}                  forget one memory
//	GET    /memories/{id}/neighbors?rel=&direction=&depth=
//	                                       linked memories
//	GET    /recall?query=&k=&type=&tag=&min_importance=&source=&since=&until=&mode=
//	POST   /recall                         same, via JSON body
//	POST   /recall_pack                    token-budgeted context block
//	POST   /auto_context                   fan-out context block for a task
//	POST   /links        {src_id,dst_id,rel?}      add a directed link
//	POST   /unlink       {src_id,dst_id,rel?}      remove link(s)
//	POST   /supersede    {old_id,new_id}           soft-delete + supersede link
//	POST   /conflicts    {memory_id?,threshold?}   duplicate / contradiction scan
//
// Optional bearer-token auth: pass a token (or set AI_HOUKAI_HTTP_TOKEN) and
// every request must carry "Authorization: Bearer <token>". /health is always
// reachable so liveness probes work without the secret.
package httpserver

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"strconv"
	"strings"
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
	mux.HandleFunc("GET /stats", s.wrap(s.stats))
	mux.HandleFunc("GET /memories", s.wrap(s.list))
	mux.HandleFunc("POST /memories", s.wrap(s.remember))
	mux.HandleFunc("GET /memories/{id}", s.wrap(s.getOne))
	mux.HandleFunc("DELETE /memories/{id}", s.wrap(s.forget))
	mux.HandleFunc("GET /memories/{id}/neighbors", s.wrap(s.neighbors))
	mux.HandleFunc("GET /recall", s.wrap(s.recall))
	mux.HandleFunc("POST /recall", s.wrap(s.recall))
	mux.HandleFunc("POST /recall_pack", s.wrap(s.recallPack))
	mux.HandleFunc("POST /auto_context", s.wrap(s.autoContext))
	mux.HandleFunc("POST /links", s.wrap(s.link))
	mux.HandleFunc("POST /unlink", s.wrap(s.unlink))
	mux.HandleFunc("POST /supersede", s.wrap(s.supersede))
	mux.HandleFunc("POST /conflicts", s.wrap(s.conflicts))
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
	if s.token == "" || r.URL.Path == "/health" {
		return true
	}
	return r.Header.Get("Authorization") == "Bearer "+s.token
}

// wrap adapts a (status, payload, error) handler to http.HandlerFunc, rendering
// *httpError with its status and any other error as a 500.
type apiFunc func(r *http.Request) (int, any, error)

func (s *Server) wrap(fn apiFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		status, payload, err := fn(r)
		if err != nil {
			var he *httpError
			if errors.As(err, &he) {
				writeJSON(w, he.status, map[string]any{"error": he.msg})
				return
			}
			writeJSON(w, 500, map[string]any{"error": fmt.Sprintf("%T: %v", err, err)})
			return
		}
		writeJSON(w, status, payload)
	}
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

func (s *Server) stats(r *http.Request) (int, any, error) {
	st, err := s.store.Stats(r.Context())
	if err != nil {
		return 0, nil, err
	}
	st["path"] = s.path
	st["collection"] = s.collection
	return 200, st, nil
}

func (s *Server) list(r *http.Request) (int, any, error) {
	limit := qsInt(r, "limit", 20)
	inc := qsBool(r, "include_superseded")
	mems, err := s.store.ListRecent(r.Context(), limit, inc)
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
		Type:       memory.MemoryType(bodyStr(b, "type", string(memory.Episodic))),
		Tags:       bodyStrSlice(b, "tags"),
		Importance: float32(bodyFloat(b, "importance", 0)),
		Source:     bodyStr(b, "source", ""),
		Polarity:   int(bodyFloat(b, "polarity", 0)),
		OnConflict: memory.ConflictPolicy(bodyStr(b, "on_conflict", "")),
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
		return 409, conflictPayload(conflicts), nil
	}
	out := memDict(mem)
	out["stored"] = true
	return 201, out, nil
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
	header := bodyStr(b, "header", "## Relevant memory")
	res, err := s.store.RecallPack(r.Context(), query, memory.PackOpts{
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
		Header:            &header,
		Fusion:            memory.FusionMode(bodyStr(b, "fusion", "")),
		Diversity:         bodyFloatPtr(b, "diversity"),
		DedupThreshold:    bodyFloatPtr(b, "dedup_threshold"),
		MinCosine:         bodyFloatPtr(b, "min_cosine"),
		Compress:          bodyBool(b, "compress"),
		CompressThreshold: float32(bodyFloat(b, "compress_threshold", 0.30)),
		CompressMinGroup:  int(bodyFloat(b, "compress_min_group", 2)),
	})
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
		return 0, nil, errStatus(404, "%s", err.Error())
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
		return 0, nil, errStatus(404, "%s", err.Error())
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
			Overfetch:         4,
			IncludeSuperseded: bodyBool(b, "include_superseded"),
			Fusion:            memory.FusionMode(bodyStr(b, "fusion", "")),
			Diversity:         bodyFloatPtr(b, "diversity"),
			DedupThreshold:    bodyFloatPtr(b, "dedup_threshold"),
			MinCosine:         bodyFloatPtr(b, "min_cosine"),
			NoTouch:           !bodyBoolDef(b, "touch", true),
			Explain:           bodyBool(b, "explain"),
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
		Overfetch:         4,
		IncludeSuperseded: qsBool(r, "include_superseded"),
	}, qsInt(r, "k", 5), nil
}

func memDict(m memory.Memory) map[string]any {
	links := make([]map[string]any, len(m.Links))
	for i, l := range m.Links {
		links[i] = map[string]any{"to": l.To, "rel": l.Rel}
	}
	var superseded any
	if m.SupersededBy != "" {
		superseded = m.SupersededBy
	}
	return map[string]any{
		"id":            m.ID,
		"text":          m.Text,
		"type":          string(m.Type),
		"tags":          m.Tags,
		"importance":    m.Importance,
		"source":        m.Source,
		"created_at":    m.CreatedAt,
		"last_accessed": m.LastAccessed,
		"access_count":  m.AccessCount,
		"links":         links,
		"superseded_by": superseded,
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
	return float64(int64(f*10000+0.5)) / 10000
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

func bodyStr(b map[string]any, key, def string) string {
	if v, ok := b[key].(string); ok && v != "" {
		return v
	}
	return def
}

func bodyFloat(b map[string]any, key string, def float64) float64 {
	switch v := b[key].(type) {
	case float64:
		return v
	case json.Number:
		f, _ := v.Float64()
		return f
	}
	return def
}

func bodyBool(b map[string]any, key string) bool {
	v, _ := b[key].(bool)
	return v
}

// bodyBoolDef reads an optional bool defaulting to def when the key is absent.
func bodyBoolDef(b map[string]any, key string, def bool) bool {
	if v, ok := b[key].(bool); ok {
		return v
	}
	return def
}

// bodyFloatPtr returns a *float32 for an optional numeric field (nil = absent).
func bodyFloatPtr(b map[string]any, key string) *float32 {
	switch v := b[key].(type) {
	case float64:
		x := float32(v)
		return &x
	case json.Number:
		f, _ := v.Float64()
		x := float32(f)
		return &x
	}
	return nil
}

func bodyStrSlice(b map[string]any, key string) []string {
	raw, ok := b[key].([]any)
	if !ok {
		return nil
	}
	out := make([]string, 0, len(raw))
	for _, v := range raw {
		if s, ok := v.(string); ok {
			out = append(out, s)
		}
	}
	return out
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
		return def
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
		return def
	}
	return f
}

func qsBool(r *http.Request, key string) bool {
	switch r.URL.Query().Get(key) {
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
