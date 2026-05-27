// Portable export to gzipped JSONL .ahkai files.
//
// The file format is line-oriented:
//   line 1     : header object  {format, version, exported_at, source, options}
//   line 2..N  : one memory row {id, text, meta, vector?}
//
// Memories are written in created_at-ascending order so two exports of
// the same store produce byte-identical files modulo the header timestamp.
package memory

import (
	"compress/gzip"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"time"
)

// ExportSummary is the result of MemoryStore.Export.
type ExportSummary struct {
	Path    string  `json:"path"`
	Count   int     `json:"count"`
	Bytes   int64   `json:"bytes"`
	Elapsed float64 `json:"elapsed"`
}

// ExportOpts customises an export.
type ExportOpts struct {
	IncludeVectors    bool
	IncludeSuperseded bool
	Types             []MemoryType
	Tags              []string
	Since             float64 // unix seconds; 0 = no filter
}

// Export streams the collection to a gzipped JSONL .ahkai file at path.
func (s *MemoryStore) Export(ctx context.Context, path string, opts ExportOpts) (ExportSummary, error) {
	t0 := time.Now()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return ExportSummary{}, err
	}

	items, err := s.backend.All(ctx)
	if err != nil {
		return ExportSummary{}, err
	}

	mems := make([]Memory, len(items))
	embs := make([][]float32, len(items))
	for i, it := range items {
		mems[i] = MetadataToMemory(it.ID, it.Content, it.Metadata)
		embs[i] = it.Embedding
	}

	// Sort by created_at ascending.
	idx := make([]int, len(mems))
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(i, j int) bool {
		return mems[idx[i]].CreatedAt < mems[idx[j]].CreatedAt
	})

	typeSet := make(map[MemoryType]struct{}, len(opts.Types))
	for _, t := range opts.Types {
		typeSet[t] = struct{}{}
	}
	tagSet := make(map[string]struct{}, len(opts.Tags))
	for _, t := range opts.Tags {
		tagSet[t] = struct{}{}
	}

	var kept []int
	for _, k := range idx {
		m := mems[k]
		if !opts.IncludeSuperseded && m.SupersededBy != "" {
			continue
		}
		if len(typeSet) > 0 {
			if _, ok := typeSet[m.Type]; !ok {
				continue
			}
		}
		if len(tagSet) > 0 {
			match := false
			for _, t := range m.Tags {
				if _, ok := tagSet[t]; ok {
					match = true
					break
				}
			}
			if !match {
				continue
			}
		}
		if opts.Since > 0 && m.CreatedAt < opts.Since {
			continue
		}
		kept = append(kept, k)
	}

	dim := 0
	if opts.IncludeVectors && len(kept) > 0 && len(embs[kept[0]]) > 0 {
		dim = len(embs[kept[0]])
	}

	typesOut := func() []string {
		if len(typeSet) == 0 {
			return nil
		}
		out := make([]string, 0, len(typeSet))
		for t := range typeSet {
			out = append(out, string(t))
		}
		sort.Strings(out)
		return out
	}()
	tagsOut := func() []string {
		if len(tagSet) == 0 {
			return nil
		}
		out := make([]string, 0, len(tagSet))
		for t := range tagSet {
			out = append(out, t)
		}
		sort.Strings(out)
		return out
	}()

	header := map[string]any{
		"format":      "ai-houkai/export",
		"version":     1,
		"exported_at": float64(t0.UnixNano()) / 1e9,
		"source": map[string]any{
			"collection":      s.cfg.Collection,
			"embedding_model": s.cfg.EmbeddingModel,
			"embedding_dim":   dim,
			"count":           len(kept),
		},
		"options": map[string]any{
			"include_vectors":    opts.IncludeVectors,
			"include_superseded": opts.IncludeSuperseded,
			"types":              typesOut,
			"tags":               tagsOut,
			"since":              opts.Since,
		},
	}

	f, err := os.Create(path)
	if err != nil {
		return ExportSummary{}, err
	}
	gw := gzip.NewWriter(f)

	enc := json.NewEncoder(gw)
	if err := enc.Encode(header); err != nil {
		_ = gw.Close()
		_ = f.Close()
		return ExportSummary{}, err
	}
	for _, k := range kept {
		m := mems[k]
		row := map[string]any{
			"id":   m.ID,
			"text": m.Text,
			"meta": m.ToDict(),
		}
		if opts.IncludeVectors && len(embs[k]) > 0 {
			row["vector"] = embs[k]
		}
		if err := enc.Encode(row); err != nil {
			_ = gw.Close()
			_ = f.Close()
			return ExportSummary{}, err
		}
	}
	if err := gw.Close(); err != nil {
		_ = f.Close()
		return ExportSummary{}, err
	}
	if err := f.Close(); err != nil {
		return ExportSummary{}, err
	}

	st, _ := os.Stat(path)
	size := int64(0)
	if st != nil {
		size = st.Size()
	}
	s.journalEntry("export", "", nil, nil, map[string]any{
		"path": path, "count": len(kept), "bytes": size,
	})
	return ExportSummary{
		Path:    path,
		Count:   len(kept),
		Bytes:   size,
		Elapsed: time.Since(t0).Seconds(),
	}, nil
}

// ExportHeader is a parsed .ahkai header (line 1 of the file).
type ExportHeader struct {
	Format     string         `json:"format"`
	Version    int            `json:"version"`
	ExportedAt float64        `json:"exported_at"`
	Source     map[string]any `json:"source"`
	Options    map[string]any `json:"options"`
}

// PeekExportHeader opens an .ahkai file, reads the header, and reports the
// number of memory rows behind it. Does not modify any store.
func PeekExportHeader(path string) (ExportHeader, int, error) {
	var hdr ExportHeader
	f, err := os.Open(path)
	if err != nil {
		return hdr, 0, err
	}
	defer f.Close()
	gr, err := gzip.NewReader(f)
	if err != nil {
		return hdr, 0, fmt.Errorf("%s: not gzipped (%w)", path, err)
	}
	defer gr.Close()
	dec := json.NewDecoder(gr)
	if err := dec.Decode(&hdr); err != nil {
		return hdr, 0, fmt.Errorf("%s: bad header: %w", path, err)
	}
	if hdr.Format != "ai-houkai/export" {
		return hdr, 0, fmt.Errorf("%s: missing/bad format header (got %q)", path, hdr.Format)
	}
	count := 0
	var tmp json.RawMessage
	for {
		if err := dec.Decode(&tmp); err != nil {
			break
		}
		count++
	}
	return hdr, count, nil
}
