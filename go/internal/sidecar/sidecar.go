// Package sidecar keeps a derived SQLite index beside the vector store.
//
// The Go port of ai_houkai/memory_system/sidecar.py, schema-for-schema, so a
// store written by either port carries the same index layout.
//
// The vector backend is authoritative for text, metadata and vectors. It is
// not a metadata query engine: it has no reverse-link lookup, no cursor
// pagination, no tag aggregation, and its lexical matching is whatever the
// caller does to the rows it already fetched. So the store ends up scanning
// the whole collection for work that is a one-line query in SQL.
//
// This index delivers full-corpus BM25, cursor pagination, O(1) reverse links,
// cheap tag/type counts and an indexed expiry sweep.
//
// It is a cache, never a source of truth. Every read has a scan fallback, and
// an index that disagrees with the backend's row count is disabled until
// `houkai reindex` rebuilds it — a stale index must degrade to "slower", never
// to "wrong".
package sidecar

import (
	"database/sql"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"

	_ "modernc.org/sqlite" // pure-Go driver: no cgo, so cross-compiles freely
)

const SchemaVersion = 1

const schemaSQL = `
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memories (
    id            TEXT PRIMARY KEY,
    text          TEXT NOT NULL,
    type          TEXT NOT NULL,
    tags          TEXT NOT NULL DEFAULT '',
    importance    REAL NOT NULL DEFAULT 0.5,
    created_at    REAL NOT NULL DEFAULT 0,
    last_accessed REAL NOT NULL DEFAULT 0,
    access_count  INTEGER NOT NULL DEFAULT 0,
    source        TEXT,
    superseded_by TEXT NOT NULL DEFAULT '',
    expires_at    REAL NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_mem_created  ON memories(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_mem_type     ON memories(type);
CREATE INDEX IF NOT EXISTS idx_mem_expires  ON memories(expires_at)
    WHERE expires_at > 0;
CREATE INDEX IF NOT EXISTS idx_mem_active   ON memories(superseded_by);

CREATE TABLE IF NOT EXISTS links (
    src TEXT NOT NULL,
    dst TEXT NOT NULL,
    rel TEXT NOT NULL,
    PRIMARY KEY (src, dst, rel)
);

-- The whole point of the links table: an index on dst turns "who points at
-- me?" from a full-collection scan into a lookup.
CREATE INDEX IF NOT EXISTS idx_links_dst ON links(dst);

CREATE TABLE IF NOT EXISTS tags (
    memory_id TEXT NOT NULL,
    tag       TEXT NOT NULL,
    PRIMARY KEY (memory_id, tag)
);

CREATE INDEX IF NOT EXISTS idx_tags_tag ON tags(tag);
`

// Deliberately a STANDALONE FTS table, not an external-content one, matching
// Python. External content halves the storage but makes correctness depend on
// hand-maintaining rowid deltas; get that wrong and SQLite reports "database
// disk image is malformed".
const ftsSQL = `
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    id UNINDEXED,
    text,
    tokenize='unicode61 remove_diacritics 2'
);
`

// Row is one memory as the index stores it. The store fills this in so the
// sidecar package never imports memory (which would be a cycle).
type Row struct {
	ID           string
	Text         string
	Type         string
	Tags         []string
	Importance   float32
	CreatedAt    float64
	LastAccessed float64
	AccessCount  int
	Source       string
	SupersededBy string
	ExpiresAt    float64
	Links        []Link
}

// Link is one outgoing edge.
type Link struct {
	To  string
	Rel string
}

// Index is a SQLite metadata + full-text index mirroring a collection.
type Index struct {
	Path       string
	Collection string
	FTS        bool

	mu             sync.Mutex
	db             *sql.DB
	healthy        bool
	disabledReason string
}

// Open creates or opens the index at path.
func Open(path, collection string) (*Index, error) {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, fmt.Errorf("sidecar: %w", err)
	}
	db, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, fmt.Errorf("sidecar: cannot open %s: %w", path, err)
	}
	// One connection: SQLite writes serialise anyway, and a pool would let
	// concurrent writers trip over each other for no gain on a cache.
	db.SetMaxOpenConns(1)
	// WAL keeps a reader from blocking the writer; NORMAL is the right
	// durability tradeoff for a rebuildable cache.
	if _, err := db.Exec("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL"); err != nil {
		db.Close()
		return nil, fmt.Errorf("sidecar: %w", err)
	}
	if _, err := db.Exec(schemaSQL); err != nil {
		db.Close()
		return nil, fmt.Errorf("sidecar: schema: %w", err)
	}
	idx := &Index{Path: path, Collection: collection, db: db, healthy: true}
	if _, err := db.Exec(ftsSQL); err == nil {
		idx.FTS = true
	}
	_, _ = db.Exec(
		"INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)",
		fmt.Sprint(SchemaVersion))
	return idx, nil
}

func (i *Index) Close() error {
	i.mu.Lock()
	defer i.mu.Unlock()
	return i.db.Close()
}

// Healthy reports whether reads may trust the index.
func (i *Index) Healthy() bool {
	i.mu.Lock()
	defer i.mu.Unlock()
	return i.healthy
}

// DisabledReason explains why the index was taken out of service, for
// `doctor` and `reindex` to report.
func (i *Index) DisabledReason() string {
	i.mu.Lock()
	defer i.mu.Unlock()
	return i.disabledReason
}

// Disable takes the index out of service so every read falls back to scanning.
func (i *Index) Disable(reason string) {
	i.mu.Lock()
	defer i.mu.Unlock()
	i.disable(reason)
}

func (i *Index) disable(reason string) {
	if i.healthy {
		log.Printf("ai-houkai: sidecar index disabled (%s) — run `houkai reindex` to rebuild", reason)
	}
	i.healthy = false
	i.disabledReason = reason
}

// Count returns the number of indexed memories.
func (i *Index) Count() (int, error) {
	i.mu.Lock()
	defer i.mu.Unlock()
	var n int
	err := i.db.QueryRow("SELECT COUNT(*) FROM memories").Scan(&n)
	return n, err
}

// Verify compares row counts with the backend and disables on a mismatch.
//
// Cheap and conservative: it cannot detect a same-count divergence, but it
// catches the common cases (index built for another collection, writes lost
// while the sidecar was missing, the backend edited directly).
func (i *Index) Verify(authoritative int) bool {
	if !i.Healthy() {
		return false
	}
	mine, err := i.Count()
	if err != nil {
		i.Disable(fmt.Sprintf("count failed: %v", err))
		return false
	}
	if mine != authoritative {
		i.Disable(fmt.Sprintf("row count %d != collection count %d", mine, authoritative))
		return false
	}
	return true
}

// Upsert inserts or replaces rows with their tags and outgoing links.
func (i *Index) Upsert(rows []Row) {
	if len(rows) == 0 || !i.Healthy() {
		return
	}
	i.mu.Lock()
	defer i.mu.Unlock()
	tx, err := i.db.Begin()
	if err != nil {
		i.disable(fmt.Sprintf("write failed: %v", err))
		return
	}
	for _, r := range rows {
		if err := upsertOne(tx, r, i.FTS); err != nil {
			_ = tx.Rollback()
			i.disable(fmt.Sprintf("write failed: %v", err))
			return
		}
	}
	if err := tx.Commit(); err != nil {
		i.disable(fmt.Sprintf("commit failed: %v", err))
	}
}

func upsertOne(tx *sql.Tx, r Row, fts bool) error {
	_, err := tx.Exec(
		`INSERT INTO memories
           (id, text, type, tags, importance, created_at, last_accessed,
            access_count, source, superseded_by, expires_at)
         VALUES (?,?,?,?,?,?,?,?,?,?,?)
         ON CONFLICT(id) DO UPDATE SET
           text=excluded.text, type=excluded.type, tags=excluded.tags,
           importance=excluded.importance, created_at=excluded.created_at,
           last_accessed=excluded.last_accessed,
           access_count=excluded.access_count, source=excluded.source,
           superseded_by=excluded.superseded_by,
           expires_at=excluded.expires_at`,
		r.ID, r.Text, r.Type, strings.Join(r.Tags, ","), r.Importance,
		r.CreatedAt, r.LastAccessed, r.AccessCount, r.Source, r.SupersededBy,
		r.ExpiresAt)
	if err != nil {
		return err
	}
	if _, err := tx.Exec("DELETE FROM tags WHERE memory_id = ?", r.ID); err != nil {
		return err
	}
	for _, t := range r.Tags {
		if t == "" {
			continue
		}
		if _, err := tx.Exec(
			"INSERT OR IGNORE INTO tags (memory_id, tag) VALUES (?, ?)",
			r.ID, t); err != nil {
			return err
		}
	}
	if _, err := tx.Exec("DELETE FROM links WHERE src = ?", r.ID); err != nil {
		return err
	}
	for _, l := range r.Links {
		if _, err := tx.Exec(
			"INSERT OR IGNORE INTO links (src, dst, rel) VALUES (?, ?, ?)",
			r.ID, l.To, l.Rel); err != nil {
			return err
		}
	}
	if fts {
		// Delete-then-insert: an edit must replace the indexed text, not add a
		// second copy that would return the same id twice.
		if _, err := tx.Exec("DELETE FROM memories_fts WHERE id = ?", r.ID); err != nil {
			return err
		}
		if _, err := tx.Exec(
			"INSERT INTO memories_fts (id, text) VALUES (?, ?)",
			r.ID, r.Text); err != nil {
			return err
		}
	}
	return nil
}

// Delete removes rows, their tags, and every edge touching them.
func (i *Index) Delete(ids []string) {
	if len(ids) == 0 || !i.Healthy() {
		return
	}
	i.mu.Lock()
	defer i.mu.Unlock()
	tx, err := i.db.Begin()
	if err != nil {
		i.disable(fmt.Sprintf("delete failed: %v", err))
		return
	}
	for _, id := range ids {
		if i.FTS {
			if _, err := tx.Exec("DELETE FROM memories_fts WHERE id = ?", id); err != nil {
				_ = tx.Rollback()
				i.disable(fmt.Sprintf("delete failed: %v", err))
				return
			}
		}
		for _, stmt := range []struct {
			sql  string
			args []any
		}{
			{"DELETE FROM memories WHERE id = ?", []any{id}},
			{"DELETE FROM tags WHERE memory_id = ?", []any{id}},
			// Both directions: a dangling dst would make reverse lookups
			// report neighbours that no longer exist.
			{"DELETE FROM links WHERE src = ? OR dst = ?", []any{id, id}},
		} {
			if _, err := tx.Exec(stmt.sql, stmt.args...); err != nil {
				_ = tx.Rollback()
				i.disable(fmt.Sprintf("delete failed: %v", err))
				return
			}
		}
	}
	if err := tx.Commit(); err != nil {
		i.disable(fmt.Sprintf("commit failed: %v", err))
	}
}

// Rebuild drops everything and re-indexes rows, restoring health on success.
func (i *Index) Rebuild(rows []Row) (int, error) {
	i.mu.Lock()
	defer i.mu.Unlock()
	tx, err := i.db.Begin()
	if err != nil {
		i.disable(fmt.Sprintf("rebuild failed: %v", err))
		return 0, err
	}
	stmts := []string{"DELETE FROM memories", "DELETE FROM tags", "DELETE FROM links"}
	if i.FTS {
		stmts = append(stmts, "DELETE FROM memories_fts")
	}
	for _, s := range stmts {
		if _, err := tx.Exec(s); err != nil {
			_ = tx.Rollback()
			i.disable(fmt.Sprintf("rebuild failed: %v", err))
			return 0, err
		}
	}
	for _, r := range rows {
		if err := upsertOne(tx, r, i.FTS); err != nil {
			_ = tx.Rollback()
			i.disable(fmt.Sprintf("rebuild failed: %v", err))
			return 0, err
		}
	}
	if err := tx.Commit(); err != nil {
		i.disable(fmt.Sprintf("rebuild commit failed: %v", err))
		return 0, err
	}
	i.healthy = true
	i.disabledReason = ""
	return len(rows), nil
}

// SearchLexical returns ids of the best full-corpus BM25 matches, best first.
//
// This is the piece the vector over-fetch pool cannot provide: an exact token
// match whose embedding is weak never enters the pool, so it can never be
// scored, at any corpus size.
func (i *Index) SearchLexical(query string, limit int) []string {
	if !i.Healthy() || !i.FTS {
		return nil
	}
	match := FTSQuery(query)
	if match == "" {
		return nil
	}
	i.mu.Lock()
	defer i.mu.Unlock()
	rows, err := i.db.Query(
		`SELECT id FROM memories_fts WHERE memories_fts MATCH ?
         ORDER BY bm25(memories_fts) LIMIT ?`, match, limit)
	if err != nil {
		// A malformed MATCH is a query problem, not corruption — return
		// nothing rather than disabling the whole index.
		return nil
	}
	defer rows.Close()
	var out []string
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err == nil {
			out = append(out, id)
		}
	}
	return out
}

// Incoming returns (src, rel) pairs pointing at dst — the reverse-link lookup.
func (i *Index) Incoming(dst, rel string) [][2]string {
	if !i.Healthy() {
		return nil
	}
	q := "SELECT src, rel FROM links WHERE dst = ?"
	args := []any{dst}
	if rel != "" {
		q += " AND rel = ?"
		args = append(args, rel)
	}
	i.mu.Lock()
	defer i.mu.Unlock()
	rows, err := i.db.Query(q+" ORDER BY src, rel", args...)
	if err != nil {
		return nil
	}
	defer rows.Close()
	var out [][2]string
	for rows.Next() {
		var src, r string
		if err := rows.Scan(&src, &r); err == nil {
			out = append(out, [2]string{src, r})
		}
	}
	return out
}

// RecentOpts bounds a RecentIDs query.
type RecentOpts struct {
	Limit             int
	Before            float64 // keyset cursor on created_at; 0 = from the top
	IncludeSuperseded bool
	IncludeExpired    bool
	Now               float64
	Type              string
}

// RecentIDs returns ids newest-first, with an optional created_at cursor.
func (i *Index) RecentIDs(o RecentOpts) []string {
	if !i.Healthy() {
		return nil
	}
	q := []string{"SELECT id FROM memories WHERE 1=1"}
	var args []any
	if !o.IncludeSuperseded {
		q = append(q, "AND superseded_by = ''")
	}
	if !o.IncludeExpired {
		q = append(q, "AND (expires_at = 0 OR expires_at > ?)")
		args = append(args, o.Now)
	}
	if o.Type != "" {
		q = append(q, "AND type = ?")
		args = append(args, o.Type)
	}
	if o.Before > 0 {
		q = append(q, "AND created_at < ?")
		args = append(args, o.Before)
	}
	q = append(q, "ORDER BY created_at DESC, id DESC")
	if o.Limit > 0 {
		q = append(q, "LIMIT ?")
		args = append(args, o.Limit)
	}
	i.mu.Lock()
	defer i.mu.Unlock()
	rows, err := i.db.Query(strings.Join(q, " "), args...)
	if err != nil {
		return nil
	}
	defer rows.Close()
	var out []string
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err == nil {
			out = append(out, id)
		}
	}
	return out
}

// ExpiredIDs returns ids whose TTL has passed — an index range scan.
func (i *Index) ExpiredIDs(now float64) []string {
	if !i.Healthy() {
		return nil
	}
	i.mu.Lock()
	defer i.mu.Unlock()
	rows, err := i.db.Query(
		"SELECT id FROM memories WHERE expires_at > 0 AND expires_at <= ?", now)
	if err != nil {
		return nil
	}
	defer rows.Close()
	var out []string
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err == nil {
			out = append(out, id)
		}
	}
	return out
}

// TagCounts reports how many memories carry each tag.
func (i *Index) TagCounts(includeSuperseded bool) map[string]int {
	if !i.Healthy() {
		return nil
	}
	q := "SELECT t.tag, COUNT(*) FROM tags t JOIN memories m ON m.id = t.memory_id"
	if !includeSuperseded {
		q += " WHERE m.superseded_by = ''"
	}
	q += " GROUP BY t.tag"
	return i.countQuery(q)
}

// TypeCounts reports how many memories carry each type.
func (i *Index) TypeCounts(includeSuperseded bool) map[string]int {
	if !i.Healthy() {
		return nil
	}
	q := "SELECT type, COUNT(*) FROM memories"
	if !includeSuperseded {
		q += " WHERE superseded_by = ''"
	}
	q += " GROUP BY type"
	return i.countQuery(q)
}

func (i *Index) countQuery(q string) map[string]int {
	i.mu.Lock()
	defer i.mu.Unlock()
	rows, err := i.db.Query(q)
	if err != nil {
		return nil
	}
	defer rows.Close()
	out := map[string]int{}
	for rows.Next() {
		var key string
		var n int
		if err := rows.Scan(&key, &n); err == nil {
			out[key] = n
		}
	}
	return out
}

// FTSQuery turns free text into a safe FTS5 MATCH expression.
//
// Every token is quoted and OR-ed: users type prose, not FTS5 syntax, and an
// unquoted '-' or '"' would otherwise be read as an operator (or a syntax
// error). OR rather than AND because this feeds a candidate pool — recall
// matters more than precision at this stage; BM25 does the ranking.
func FTSQuery(query string) string {
	var tokens []string
	var current []rune
	flush := func() {
		if len(current) > 0 {
			tokens = append(tokens, string(current))
			current = nil
		}
	}
	for _, ch := range query {
		if ch == '_' || isAlnum(ch) {
			current = append(current, ch)
		} else {
			flush()
		}
	}
	flush()
	if len(tokens) == 0 {
		return ""
	}
	quoted := make([]string, len(tokens))
	for i, t := range tokens {
		quoted[i] = `"` + strings.ReplaceAll(t, `"`, `""`) + `"`
	}
	return strings.Join(quoted, " OR ")
}

func isAlnum(r rune) bool {
	return (r >= '0' && r <= '9') || (r >= 'a' && r <= 'z') ||
		(r >= 'A' && r <= 'Z') || r > 127
}
