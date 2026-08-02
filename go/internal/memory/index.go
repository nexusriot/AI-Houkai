package memory

import (
	"context"
	"fmt"
	"path/filepath"

	"github.com/nexusriot/ai-houkai/internal/sidecar"
	"github.com/nexusriot/ai-houkai/internal/vector"
)

// Sidecar-index wiring. The index is a derived cache: every read below has a
// scan fallback, and an index that disagrees with the backend is disabled
// rather than trusted, so a stale index degrades to "slower", never "wrong".

// indexRow flattens a backend item into the row shape the index stores. It
// goes through MetadataToMemory so the index inherits exactly the same
// metadata decoding as every other read — an index that parsed tags or
// timestamps its own way would drift from the store it mirrors.
func indexRow(it vector.Item) sidecar.Row {
	m := MetadataToMemory(it.ID, it.Content, it.Metadata)
	links := make([]sidecar.Link, len(m.Links))
	for i, l := range m.Links {
		links[i] = sidecar.Link{To: l.To, Rel: l.Rel}
	}
	return sidecar.Row{
		ID: m.ID, Text: m.Text, Type: string(m.Type), Tags: m.Tags,
		Importance: m.Importance, CreatedAt: m.CreatedAt,
		LastAccessed: m.LastAccessed, AccessCount: m.AccessCount,
		Source: m.Source, SupersededBy: m.SupersededBy,
		ExpiresAt: m.ExpiresAt, Links: links,
	}
}

// DefaultIndexPath is where the sidecar lands when none is configured.
func DefaultIndexPath(storePath, collection string) string {
	return filepath.Join(filepath.Dir(storePath), collection+".index.sqlite3")
}

// EnableIndex attaches a sidecar index and wraps the backend so writes mirror
// into it. Off by default: an existing store has no index, so enabling it
// silently would make list/neighbors read an empty table — call Reindex first
// (or `houkai reindex`).
//
// Returns an error only if the index file cannot be opened; a stale index is
// reported by leaving it unhealthy, not by failing.
func (s *MemoryStore) EnableIndex(ctx context.Context, path string) error {
	if path == "" {
		path = DefaultIndexPath(s.cfg.Path, s.cfg.Collection)
	}
	idx, err := sidecar.Open(path, s.cfg.Collection)
	if err != nil {
		return err
	}
	if n, cerr := s.backend.Count(ctx); cerr == nil {
		idx.Verify(n)
	}
	s.index = idx
	s.backend = sidecar.Wrap(s.backend, idx, indexRow)
	return nil
}

// Index returns the sidecar index, or nil when none is configured.
func (s *MemoryStore) Index() *sidecar.Index { return s.index }

// ReindexResult summarises a rebuild.
type ReindexResult struct {
	Enabled bool   `json:"enabled"`
	Indexed int    `json:"indexed"`
	Healthy bool   `json:"healthy"`
	FTS     bool   `json:"fts"`
	Path    string `json:"path"`
	Error   string `json:"error,omitempty"`
}

// Reindex rebuilds the sidecar from the backend and re-enables it. The only
// way back from a disabled index.
func (s *MemoryStore) Reindex(ctx context.Context) (ReindexResult, error) {
	if s.index == nil {
		return ReindexResult{
			Error: "no sidecar index configured (enable it with index = \"sqlite\")",
		}, nil
	}
	// Read through the *undecorated* backend so a rebuild does not recurse
	// back through the mirror it is rebuilding.
	src := s.backend
	if d, ok := s.backend.(*sidecar.IndexedBackend); ok {
		src = d.Inner()
	}
	items, err := src.All(ctx)
	if err != nil {
		return ReindexResult{Enabled: true, Path: s.index.Path}, err
	}
	rows := make([]sidecar.Row, len(items))
	for i, it := range items {
		rows[i] = indexRow(it)
	}
	n, rerr := s.index.Rebuild(rows)
	res := ReindexResult{
		Enabled: true, Indexed: n, Healthy: s.index.Healthy(),
		FTS: s.index.FTS, Path: s.index.Path,
		Error: s.index.DisabledReason(),
	}
	if rerr != nil {
		return res, fmt.Errorf("reindex: %w", rerr)
	}
	return res, nil
}

// incomingCandidates returns memories with a link pointing at id.
//
// With the index this is a lookup on links(dst). Without it, the only way to
// answer "who points at me?" is to read every memory — and Neighbors asks once
// per frontier node per hop, so a depth-2 walk over ten neighbours means
// eleven full-collection loads.
func (s *MemoryStore) incomingCandidates(ctx context.Context, id, rel string) []Memory {
	if s.index != nil && s.index.Healthy() {
		pairs := s.index.Incoming(id, rel)
		if len(pairs) == 0 {
			return nil
		}
		seen := map[string]bool{}
		ids := make([]string, 0, len(pairs))
		for _, p := range pairs {
			if !seen[p[0]] {
				seen[p[0]] = true
				ids = append(ids, p[0])
			}
		}
		items, err := s.backend.Get(ctx, ids)
		if err != nil {
			return nil
		}
		out := make([]Memory, len(items))
		for i, it := range items {
			out[i] = MetadataToMemory(it.ID, it.Content, it.Metadata)
		}
		return out
	}
	items, err := s.backend.All(ctx)
	if err != nil {
		return nil
	}
	out := make([]Memory, len(items))
	for i, it := range items {
		out[i] = MetadataToMemory(it.ID, it.Content, it.Metadata)
	}
	return out
}

// unionLexical merges full-corpus BM25 hits into a vector query result.
//
// The newcomers keep their *real* similarity, computed here against the query
// vector. Fabricating one was tempting and wrong: a neutral value invents
// vector evidence a candidate has not earned, and a worst-case value buries it
// far below anything the lexical weight could recover — which would make the
// whole channel decorative.
//
// Returns hits unchanged when the index is off, unhealthy, FTS-less or has
// nothing to add, so setting the flag without an index is a no-op.
func (s *MemoryStore) unionLexical(
	ctx context.Context, hits []vector.Hit, query string,
	queryVec []float32, nFetch int,
) []vector.Hit {
	if s.index == nil || !s.index.Healthy() || !s.index.FTS {
		return hits
	}
	present := make(map[string]bool, len(hits))
	for _, h := range hits {
		present[h.ID] = true
	}
	var extra []string
	for _, id := range s.index.SearchLexical(query, nFetch) {
		if !present[id] {
			extra = append(extra, id)
		}
	}
	if len(extra) == 0 {
		return hits
	}
	items, err := s.backend.Get(ctx, extra)
	if err != nil {
		return hits
	}
	for _, it := range items {
		hits = append(hits, vector.Hit{
			Item:       it,
			Similarity: vector.CosineSim(queryVec, it.Embedding),
		})
	}
	return hits
}
