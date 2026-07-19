// Portable import from gzipped JSONL .ahkai files. See export.go for format.
package memory

import (
	"compress/gzip"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"

	"github.com/google/uuid"
	"github.com/nexusriot/ai-houkai/internal/vector"
)

// ImportConflictPolicy controls what happens when an imported memory's
// id already exists in the destination store.
type ImportConflictPolicy string

const (
	ImportSkip        ImportConflictPolicy = "skip"
	ImportOverwrite   ImportConflictPolicy = "overwrite"
	ImportRename      ImportConflictPolicy = "rename"
	ImportErrorPolicy ImportConflictPolicy = "error"
)

// ImportOpts customises an import.
type ImportOpts struct {
	OnConflict        ImportConflictPolicy
	RegenerateVectors bool
	DryRun            bool
}

// ImportSummary is the result of MemoryStore.Import.
type ImportSummary struct {
	Imported           int              `json:"imported"`
	Skipped            int              `json:"skipped"`
	Overwritten        int              `json:"overwritten"`
	Renamed            int              `json:"renamed"`
	Errors             []ImportRowError `json:"errors,omitempty"`
	VectorsRegenerated bool             `json:"vectors_regenerated"`
}

// ImportRowError is one (id, message) pair from an import.
type ImportRowError struct {
	ID  string `json:"id"`
	Msg string `json:"msg"`
}

// ImportConflictError is returned when OnConflict=error and at least one id collided.
type ImportConflictError struct {
	Collisions [][2]string // (id, existing_text_preview)
}

func (e *ImportConflictError) Error() string {
	return fmt.Sprintf("%d id collision(s) on import", len(e.Collisions))
}

// Import loads memories from an .ahkai file.
func (s *MemoryStore) Import(ctx context.Context, path string, opts ImportOpts) (ImportSummary, error) {
	if opts.OnConflict == "" {
		opts.OnConflict = ImportSkip
	}
	var summary ImportSummary
	if err := validateChoice(string(opts.OnConflict), ImportPolicies, "on_conflict"); err != nil {
		return summary, err
	}

	f, err := os.Open(path)
	if err != nil {
		return summary, err
	}
	defer f.Close()
	s.recordCall("import")
	gr, err := gzip.NewReader(f)
	if err != nil {
		return summary, fmt.Errorf("%s: not gzipped (%w)", path, err)
	}
	defer gr.Close()
	dec := json.NewDecoder(gr)

	var header ExportHeader
	if err := dec.Decode(&header); err != nil {
		return summary, fmt.Errorf("%s: bad header: %w", path, err)
	}
	if header.Format != "ai-houkai/export" {
		return summary, fmt.Errorf("%s: missing/bad format header", path)
	}
	if header.Version > 1 {
		return summary, fmt.Errorf("%s: version %d > reader version 1", path, header.Version)
	}

	srcModel := ""
	srcDim := 0
	if header.Source != nil {
		if v, ok := header.Source["embedding_model"].(string); ok {
			srcModel = v
		}
		if v, ok := header.Source["embedding_dim"].(float64); ok {
			srcDim = int(v)
		}
	}
	modelMismatch := srcModel != "" && s.cfg.EmbeddingModel != "" && srcModel != s.cfg.EmbeddingModel
	if modelMismatch && !opts.RegenerateVectors {
		return summary, fmt.Errorf(
			"embedding model mismatch (file: %q, store: %q) — pass RegenerateVectors=true",
			srcModel, s.cfg.EmbeddingModel,
		)
	}
	summary.VectorsRegenerated = modelMismatch
	useVectors := !modelMismatch && srcDim > 0

	restore := s.AsActor("import")
	defer restore()

	var collisions [][2]string
	for {
		var row map[string]any
		if err := dec.Decode(&row); err != nil {
			if err == io.EOF {
				break
			}
			summary.Errors = append(summary.Errors, ImportRowError{ID: "?", Msg: err.Error()})
			break
		}
		if err := s.importOne(ctx, row, opts, useVectors, &summary, &collisions); err != nil {
			id, _ := row["id"].(string)
			summary.Errors = append(summary.Errors, ImportRowError{ID: id, Msg: err.Error()})
		}
	}

	if opts.OnConflict == ImportErrorPolicy && len(collisions) > 0 {
		return summary, &ImportConflictError{Collisions: collisions}
	}
	return summary, nil
}

func (s *MemoryStore) importOne(
	ctx context.Context,
	row map[string]any,
	opts ImportOpts,
	useVectors bool,
	summary *ImportSummary,
	collisions *[][2]string,
) error {
	// `meta` carries the full ToDict() snapshot; row.id / row.text override.
	meta, _ := row["meta"].(map[string]any)
	if meta == nil {
		meta = map[string]any{}
	}
	if id, ok := row["id"].(string); ok && id != "" {
		meta["id"] = id
	}
	if txt, ok := row["text"].(string); ok && txt != "" {
		meta["text"] = txt
	}
	mem := MemoryFromDict(meta)
	if mem.ID == "" {
		return fmt.Errorf("missing id")
	}

	var vec []float32
	if useVectors {
		if raw, ok := row["vector"].([]any); ok {
			vec = make([]float32, len(raw))
			for i, x := range raw {
				if f, ok := x.(float64); ok {
					vec[i] = float32(f)
				}
			}
		}
	}

	existing, err := s.GetByID(ctx, mem.ID)
	hasExisting := err == nil
	if hasExisting {
		switch opts.OnConflict {
		case ImportSkip:
			summary.Skipped++
			return nil
		case ImportErrorPolicy:
			preview := existing.Text
			if len(preview) > 80 {
				preview = preview[:80]
			}
			*collisions = append(*collisions, [2]string{mem.ID, preview})
			summary.Skipped++
			return nil
		case ImportRename:
			mem.ID = uuid.NewString()
			if !opts.DryRun {
				if err := s.addImported(ctx, mem, vec); err != nil {
					return err
				}
			}
			summary.Renamed++
			return nil
		case ImportOverwrite:
			if !opts.DryRun {
				if err := s.backend.Delete(ctx, []string{mem.ID}); err != nil {
					return err
				}
				if err := s.addImported(ctx, mem, vec); err != nil {
					return err
				}
			}
			summary.Overwritten++
			return nil
		}
	}

	if !opts.DryRun {
		if err := s.addImported(ctx, mem, vec); err != nil {
			return err
		}
	}
	summary.Imported++
	return nil
}

func (s *MemoryStore) addImported(ctx context.Context, mem Memory, vec []float32) error {
	// Record whether the FILE's vector was used before any re-embed, so the
	// journal reflects reality (a re-embedded row is not "preserved").
	vectorsPreserved := len(vec) > 0
	if len(vec) == 0 {
		// Re-embed from text.
		vecs, err := s.embedder.Embed(ctx, []string{mem.Text})
		if err != nil {
			return fmt.Errorf("embed: %w", err)
		}
		vec = vecs[0]
	}
	if mem.Links == nil {
		mem.Links = []Link{}
	}
	if mem.Tags == nil {
		mem.Tags = []string{}
	}
	if err := s.backend.Add(ctx, []vector.Item{{
		ID:        mem.ID,
		Content:   mem.Text,
		Embedding: vec,
		Metadata:  MemoryToMetadata(mem),
	}}); err != nil {
		return err
	}
	s.journalEntry("import", mem.ID, nil, mem.ToDict(), map[string]any{
		"vectors_preserved": vectorsPreserved,
	})
	return nil
}

// Undo reverses a single journal entry where possible. Returns true on success.
func (s *MemoryStore) Undo(ctx context.Context, e JournalEntry) (bool, error) {
	switch e.Op {
	case "remember":
		ok, err := s.Forget(ctx, e.ID)
		if err != nil || !ok {
			return ok, err
		}
		s.journalEntry("undo", e.ID, nil, nil, map[string]any{"of": e.TS, "of_op": e.Op})
		return true, nil

	case "forget":
		if e.Before == nil {
			return false, nil
		}
		if _, err := s.GetByID(ctx, e.ID); err == nil {
			// Already exists — refuse to clobber.
			return false, nil
		}
		mem := MemoryFromDict(e.Before)
		vecs, err := s.embedder.Embed(ctx, []string{mem.Text})
		if err != nil {
			return false, err
		}
		if err := s.backend.Add(ctx, []vector.Item{{
			ID:        mem.ID,
			Content:   mem.Text,
			Embedding: vecs[0],
			Metadata:  MemoryToMetadata(mem),
		}}); err != nil {
			return false, err
		}
		s.journalEntry("undo", mem.ID, nil, mem.ToDict(), map[string]any{"of": e.TS, "of_op": e.Op})
		return true, nil

	case "edit":
		if e.Before == nil {
			return false, nil
		}
		current, err := s.GetByID(ctx, e.ID)
		if err != nil {
			return false, nil // memory has since been forgotten
		}
		restored := MemoryFromDict(e.Before)
		// Always re-embed: the edit may have changed the text, and
		// re-embedding unchanged text is harmless (matches Python).
		if err := s.UpdateMemory(ctx, restored, true); err != nil {
			return false, err
		}
		s.journalEntry("undo", e.ID, current.ToDict(), restored.ToDict(),
			map[string]any{"of": e.TS, "of_op": e.Op})
		return true, nil

	case "supersede":
		ok, err := s.Restore(ctx, e.ID)
		if err != nil || !ok {
			return false, err
		}
		s.journalEntry("undo", e.ID, nil, nil, map[string]any{"of": e.TS, "of_op": e.Op})
		return true, nil

	case "restore":
		sid, _ := e.Meta["superseder_id"].(string)
		if sid == "" {
			return false, nil
		}
		if _, err := s.GetByID(ctx, sid); err != nil {
			return false, nil
		}
		if err := s.Supersede(ctx, e.ID, sid); err != nil {
			return false, err
		}
		s.journalEntry("undo", e.ID, nil, nil, map[string]any{"of": e.TS, "of_op": e.Op})
		return true, nil

	case "link":
		src, _ := e.Meta["src_id"].(string)
		dst, _ := e.Meta["dst_id"].(string)
		rel, _ := e.Meta["rel"].(string)
		if src == "" {
			src = e.ID
		}
		removed, err := s.unlinkRaw(ctx, src, dst, rel)
		if err != nil || len(removed) == 0 {
			return false, err
		}
		s.journalEntry("undo", e.ID, nil, nil, map[string]any{"of": e.TS, "of_op": e.Op})
		return true, nil

	case "unlink":
		src, _ := e.Meta["src_id"].(string)
		dst, _ := e.Meta["dst_id"].(string)
		if src == "" {
			src = e.ID
		}
		// A rel="" unlink may have removed several differently-typed edges;
		// meta["removed_rels"] records them. Entries written before that
		// field existed fall back to the single rel.
		var rels []string
		if raw, ok := e.Meta["removed_rels"].([]any); ok {
			for _, r := range raw {
				if s, ok := r.(string); ok {
					rels = append(rels, s)
				}
			}
		}
		if len(rels) == 0 {
			rel, _ := e.Meta["rel"].(string)
			if rel == "" {
				rel = RelRelated
			}
			rels = []string{rel}
		}
		addedAny := false
		for _, r := range rels {
			added, err := s.linkRaw(ctx, src, dst, r)
			if err != nil {
				if errors.Is(err, ErrNotFound) {
					return false, nil // an endpoint has since been forgotten
				}
				return false, err
			}
			if added {
				addedAny = true
			}
		}
		if !addedAny {
			return false, nil
		}
		s.journalEntry("undo", e.ID, nil, nil, map[string]any{"of": e.TS, "of_op": e.Op})
		return true, nil
	}
	// reflect / decay / import / export / undo themselves are not undoable.
	return false, nil
}
