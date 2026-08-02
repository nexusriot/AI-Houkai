package httpserver

import (
	"errors"
	"net/http"

	"github.com/nexusriot/ai-houkai/internal/memory"
)

// Curation routes graduated from ai-houkai-service (D).

func (s *Server) merge(r *http.Request) (int, any, error) {
	b, err := readBody(r)
	if err != nil {
		return 0, nil, err
	}
	targetID, err := requireStr(b, "target_id")
	if err != nil {
		return 0, nil, err
	}
	otherID, err := requireStr(b, "other_id")
	if err != nil {
		return 0, nil, err
	}
	mem, err := s.store.Merge(r.Context(), targetID, otherID,
		bodyStr(b, "separator", ""))
	if err != nil {
		if errors.Is(err, memory.ErrNotFound) {
			return 0, nil, errStatus(404, "%s", err.Error())
		}
		return 0, nil, err // self-merge and friends → 400 via wrap
	}
	return 200, memDict(mem), nil
}

func (s *Server) versions(r *http.Request) (int, any, error) {
	id := r.PathValue("id")
	mem, err := s.store.GetByID(r.Context(), id)
	if err != nil {
		return 0, nil, errStatus(404, "memory not found")
	}
	vs, err := s.store.Versions(mem.ID)
	if err != nil {
		return 0, nil, err
	}
	if vs == nil {
		vs = []memory.Version{}
	}
	return 200, map[string]any{"id": mem.ID, "versions": vs}, nil
}

func (s *Server) tags(r *http.Request) (int, any, error) {
	tags, err := s.store.ListTags(r.Context(),
		qsBool(r, "include_superseded"))
	if err != nil {
		return 0, nil, err
	}
	if tags == nil {
		tags = []memory.TagCount{}
	}
	return 200, map[string]any{"tags": tags}, nil
}

func (s *Server) renameTag(r *http.Request) (int, any, error) {
	b, err := readBody(r)
	if err != nil {
		return 0, nil, err
	}
	old, err := requireStr(b, "old")
	if err != nil {
		return 0, nil, err
	}
	updated, err := requireStr(b, "new")
	if err != nil {
		return 0, nil, err
	}
	n, err := s.store.RenameTag(r.Context(), old, updated)
	if err != nil {
		return 0, nil, err
	}
	return 200, map[string]any{"changed": n, "tag": updated}, nil
}

func (s *Server) mergeTags(r *http.Request) (int, any, error) {
	b, err := readBody(r)
	if err != nil {
		return 0, nil, err
	}
	sources := bodyStrSlice(b, "sources")
	if len(sources) == 0 {
		return 0, nil, errStatus(400, "missing required field: sources")
	}
	into, err := requireStr(b, "into")
	if err != nil {
		return 0, nil, err
	}
	n, err := s.store.MergeTags(r.Context(), sources, into)
	if err != nil {
		return 0, nil, err
	}
	return 200, map[string]any{"changed": n, "tag": into}, nil
}

func (s *Server) deleteTag(r *http.Request) (int, any, error) {
	tag := r.PathValue("tag")
	n, err := s.store.DeleteTag(r.Context(), tag)
	if err != nil {
		return 0, nil, err
	}
	return 200, map[string]any{"changed": n, "tag": tag}, nil
}

func (s *Server) findPath(r *http.Request) (int, any, error) {
	b, err := readBody(r)
	if err != nil {
		return 0, nil, err
	}
	fromID, err := requireStr(b, "from_id")
	if err != nil {
		return 0, nil, err
	}
	toID, err := requireStr(b, "to_id")
	if err != nil {
		return 0, nil, err
	}
	hops, err := s.store.FindPath(r.Context(), fromID, toID,
		int(bodyFloat(b, "max_depth", 6)))
	if err != nil {
		return 0, nil, err
	}
	path := make([]map[string]any, 0, len(hops))
	for _, h := range hops {
		entry := map[string]any{"id": h.ID, "rel": h.Rel}
		if mem, gerr := s.store.GetByID(r.Context(), h.ID); gerr == nil {
			entry["text"] = clip(mem.Text, 120)
		}
		path = append(path, entry)
	}
	length := 0
	if len(hops) > 0 {
		length = len(hops) - 1
	}
	return 200, map[string]any{
		"found": len(hops) > 0, "length": length, "path": path,
	}, nil
}

func (s *Server) trash(r *http.Request) (int, any, error) {
	b, err := readBody(r)
	if err != nil {
		return 0, nil, err
	}
	id, err := requireStr(b, "memory_id")
	if err != nil {
		return 0, nil, err
	}
	ok, err := s.store.Trash(r.Context(), id)
	if err != nil {
		return 0, nil, err
	}
	if !ok {
		return 0, nil, errStatus(404, "memory not found")
	}
	return 200, map[string]any{"trashed": true, "id": id}, nil
}

func (s *Server) trashList(r *http.Request) (int, any, error) {
	entries, err := s.store.TrashList()
	if err != nil {
		return 0, nil, err
	}
	out := make([]map[string]any, 0, len(entries))
	for _, e := range entries {
		text, _ := e.Memory["text"].(string)
		out = append(out, map[string]any{
			"memory_id": e.MemoryID, "deleted_at": e.DeletedAt,
			"actor": e.Actor, "text": clip(text, 200),
		})
	}
	return 200, map[string]any{"entries": out}, nil
}

func (s *Server) trashRestore(r *http.Request) (int, any, error) {
	b, err := readBody(r)
	if err != nil {
		return 0, nil, err
	}
	id, err := requireStr(b, "memory_id")
	if err != nil {
		return 0, nil, err
	}
	mem, ok, err := s.store.TrashRestore(r.Context(), id)
	if err != nil {
		return 0, nil, err
	}
	if !ok {
		return 0, nil, errStatus(404, "not in the trash")
	}
	return 200, memDict(mem), nil
}

func (s *Server) trashPurge(r *http.Request) (int, any, error) {
	b, err := readBody(r)
	if err != nil {
		return 0, nil, err
	}
	n, err := s.store.TrashPurge(bodyStr(b, "memory_id", ""))
	if err != nil {
		return 0, nil, err
	}
	return 200, map[string]any{"purged": n}, nil
}
