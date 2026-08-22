package memory

import (
	"compress/gzip"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Curation operations: merge, versions, tag management, path-finding, trash.
//
// These grew up in ai-houkai-service, which implemented them by reaching
// through the library's private API. Every one is a store-level primitive:
// re-pointing an incoming link, rewriting a tag across the collection and
// walking the link graph all need the store's own write path and journal, and
// a downstream consumer cannot do them correctly from outside.
//
// Trash fills the gap between Supersede (soft, but semantically "replaced by
// X") and Forget (irreversible): a recoverable delete. Decay pruning
// hard-deletes today, so a mis-tuned MinScore is unrecoverable.

// TrashFilename is where soft-deleted memories are parked, relative to the
// store directory's parent.
const TrashFilename = "trash.jsonl.gz"

// ErrSelfMerge is returned when Merge is asked to fold a memory into itself.
var ErrSelfMerge = errors.New("cannot merge a memory with itself")

// Version is one past text state of a memory, recovered from the journal.
type Version struct {
	TS         float64  `json:"ts"`
	Text       string   `json:"text"`
	Tags       []string `json:"tags"`
	Importance float32  `json:"importance"`
	Source     string   `json:"source,omitempty"`
	Type       string   `json:"type"`
}

// TagCount is one tag with its usage count.
type TagCount struct {
	Tag   string `json:"tag"`
	Count int    `json:"count"`
}

// PathHop is one step of a find-path result.
type PathHop struct {
	ID  string `json:"id"`
	Rel string `json:"rel"`
}

// TrashEntry is a soft-deleted memory parked in the trash file.
//
// Collection scopes the entry: the trash file lives beside the store path and
// is shared by every collection opened on it, so without the tag a restore
// from collection B would materialize collection A's memory into B. Entries
// written before the field existed have "" and stay visible from every
// collection — hiding them retroactively would strand them.
type TrashEntry struct {
	MemoryID   string         `json:"memory_id"`
	DeletedAt  float64        `json:"deleted_at"`
	Actor      string         `json:"actor"`
	Memory     map[string]any `json:"memory"`
	Collection string         `json:"collection"`
}

// Merge folds other into target and returns the updated target.
//
// Combines the text, transfers other's outgoing links, and — the part no
// caller can do from outside — re-points every incoming link x → other at
// x → target. Forget does not clean up incoming edges, so without this step
// merging would silently strand every relationship that pointed at the
// absorbed memory.
func (s *MemoryStore) Merge(ctx context.Context, targetID, otherID, separator string) (Memory, error) {
	target, err := s.GetByID(ctx, targetID)
	if err != nil {
		return Memory{}, fmt.Errorf("target: %w", err)
	}
	other, err := s.GetByID(ctx, otherID)
	if err != nil {
		return Memory{}, fmt.Errorf("other: %w", err)
	}
	if target.ID == other.ID {
		return Memory{}, ErrSelfMerge
	}
	if separator == "" {
		separator = "\n\n"
	}

	before := target.ToDict()
	target.Text = target.Text + separator + other.Text

	// The merged text now contains other's content, so the result can be no more
	// trustworthy than its least trustworthy half. Without this, merging is a
	// laundering path: absorb an untrusted memory into a trusted one and the
	// provenance label silently survives.
	target.Trust = WorstTrust(target.Trust, other.Trust)
	// The pin is the union of the two sides: `other` is deleted by the merge,
	// so a pin that does not travel is destroyed outright and the working set
	// silently loses a standing instruction.
	target.Pinned = target.Pinned || other.Pinned

	existing := map[string]bool{}
	for _, l := range target.Links {
		existing[l.To+"\x00"+l.Rel] = true
	}
	for _, l := range other.Links {
		// Skip self-loops and edges whose destination is already gone.
		if l.To == target.ID {
			continue
		}
		if _, err := s.GetByID(ctx, l.To); err != nil {
			continue
		}
		key := l.To + "\x00" + l.Rel
		if existing[key] {
			continue
		}
		existing[key] = true
		target.Links = append(target.Links, Link{To: l.To, Rel: l.Rel})
	}

	// The dedup hash has to move with the text, exactly as Edit moves it. Left
	// stale, the merged row still answers to its *pre-merge* text: the next
	// idempotent write of that text is absorbed as a duplicate and silently
	// lost, while the text the row now actually holds never matches at all.
	target.ContentHash = ContentHash(target.Text)

	// Text changed, so the vector must be recomputed — a merged memory that
	// kept the pre-merge embedding would not be findable by its new half.
	if err := s.UpdateMemory(ctx, target, true); err != nil {
		return Memory{}, err
	}
	s.journalEntry("edit", target.ID, before, target.ToDict(),
		map[string]any{"merged_from": other.ID})

	if err := s.repointIncoming(ctx, other.ID, target.ID); err != nil {
		return Memory{}, err
	}
	if _, err := s.Forget(ctx, other.ID); err != nil {
		return Memory{}, err
	}
	return s.GetByID(ctx, target.ID)
}

// repointIncoming rewrites every x → oldDst edge to x → newDst.
//
// Writes each source's link list directly rather than going through
// Unlink+Link: that path re-validates the rel vocabulary (rejecting a legacy
// custom rel outright) and costs two journal entries per edge. A pre-existing
// newDst → oldDst edge is dropped rather than turned into a self-loop.
func (s *MemoryStore) repointIncoming(ctx context.Context, oldDst, newDst string) error {
	for _, src := range s.incomingCandidates(ctx, oldDst, "") {
		if src.ID == oldDst {
			continue
		}
		touches := false
		for _, l := range src.Links {
			if l.To == oldDst {
				touches = true
				break
			}
		}
		if !touches {
			continue
		}
		srcBefore := src.ToDict()
		seen := map[string]bool{}
		var newLinks []Link
		for _, l := range src.Links {
			if l.To != oldDst {
				newLinks = append(newLinks, l)
				seen[l.To+"\x00"+l.Rel] = true
			}
		}
		if src.ID != newDst {
			for _, l := range src.Links {
				if l.To != oldDst {
					continue
				}
				key := newDst + "\x00" + l.Rel
				if !seen[key] {
					seen[key] = true
					newLinks = append(newLinks, Link{To: newDst, Rel: l.Rel})
				}
			}
		}
		src.Links = newLinks
		if err := s.UpdateMemory(ctx, src, false); err != nil {
			return err
		}
		s.journalEntry("edit", src.ID, srcBefore, src.ToDict(), nil)
	}
	return nil
}

// Versions returns past text states of a memory, oldest first.
//
// Each entry is the state BEFORE an edit; the current live state is excluded
// (fetch it with GetByID). Reads rotated journal segments too, so version
// history survives a rollover.
func (s *MemoryStore) Versions(memoryID string) ([]Version, error) {
	if s.journal == nil {
		return nil, nil
	}
	entries, err := s.journal.Read(ReadOpts{IncludeArchives: true})
	if err != nil {
		return nil, err
	}
	var out []Version
	for _, e := range entries {
		if e.Op != "edit" || e.ID != memoryID || e.Before == nil {
			continue
		}
		out = append(out, Version{
			TS:         e.TS,
			Text:       dictString(e.Before, "text"),
			Tags:       dictStrings(e.Before, "tags"),
			Importance: float32(dictFloat(e.Before, "importance", 0.5)),
			Source:     dictString(e.Before, "source"),
			Type:       dictStringOr(e.Before, "type", "semantic"),
		})
	}
	return out, nil
}

// ListTags returns every tag with its usage count, most-used first then
// alphabetical.
func (s *MemoryStore) ListTags(ctx context.Context, includeSuperseded bool) ([]TagCount, error) {
	counts := map[string]int{}
	{
		mems, err := s.ListRecent(ctx, 0, includeSuperseded, true)
		if err != nil {
			return nil, err
		}
		for _, m := range mems {
			for _, t := range m.Tags {
				counts[t]++
			}
		}
	}
	out := make([]TagCount, 0, len(counts))
	for tag, n := range counts {
		out = append(out, TagCount{Tag: tag, Count: n})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Count != out[j].Count {
			return out[i].Count > out[j].Count
		}
		return out[i].Tag < out[j].Tag
	})
	return out, nil
}

// rewriteTags applies fn to every memory's tag list, persisting + journaling
// changes. Includes superseded and expired memories: tag curation that skipped
// them would leave the old spelling alive in rows a later Restore brings back.
func (s *MemoryStore) rewriteTags(ctx context.Context, fn func([]string) []string) (int, error) {
	mems, err := s.ListRecent(ctx, 0, true, true)
	if err != nil {
		return 0, err
	}
	restore := s.AsActor("curation")
	defer restore()

	changed := 0
	for _, m := range mems {
		newTags := fn(append([]string(nil), m.Tags...))
		if newTags == nil || sameStrings(newTags, m.Tags) {
			continue
		}
		before := m.ToDict()
		m.Tags = newTags
		if err := s.UpdateMemory(ctx, m, false); err != nil {
			return changed, err
		}
		s.journalEntry("edit", m.ID, before, m.ToDict(), nil)
		changed++
	}
	return changed, nil
}

// RenameTag renames a tag across the collection, de-duplicating on collision.
func (s *MemoryStore) RenameTag(ctx context.Context, old, updated string) (int, error) {
	if err := validateTagName(updated); err != nil {
		return 0, err
	}
	return s.rewriteTags(ctx, func(tags []string) []string {
		if !contains(tags, old) {
			return nil
		}
		var out []string
		for _, t := range tags {
			t2 := t
			if t == old {
				t2 = updated
			}
			if !contains(out, t2) {
				out = append(out, t2)
			}
		}
		return out
	})
}

// MergeTags folds several tags into one across the collection.
func (s *MemoryStore) MergeTags(ctx context.Context, sources []string, into string) (int, error) {
	if err := validateTagName(into); err != nil {
		return 0, err
	}
	src := map[string]bool{}
	for _, t := range sources {
		src[t] = true
	}
	return s.rewriteTags(ctx, func(tags []string) []string {
		hit := false
		for _, t := range tags {
			if src[t] {
				hit = true
				break
			}
		}
		if !hit {
			return nil
		}
		var out []string
		for _, t := range tags {
			t2 := t
			if src[t] {
				t2 = into
			}
			if !contains(out, t2) {
				out = append(out, t2)
			}
		}
		return out
	})
}

// DeleteTag strips a tag from every memory that carries it.
func (s *MemoryStore) DeleteTag(ctx context.Context, tag string) (int, error) {
	return s.rewriteTags(ctx, func(tags []string) []string {
		if !contains(tags, tag) {
			return nil
		}
		out := []string{}
		for _, t := range tags {
			if t != tag {
				out = append(out, t)
			}
		}
		return out
	})
}

// FindPath returns the shortest undirected link path between two memories as
// (id, rel-used-to-reach-it) hops starting at fromID (whose rel is ""), or nil
// when no path exists within maxDepth.
//
// Undirected because "how are these two related?" does not care which way the
// author happened to draw the arrow.
func (s *MemoryStore) FindPath(ctx context.Context, fromID, toID string, maxDepth int) ([]PathHop, error) {
	from, err := s.GetByID(ctx, fromID)
	if err != nil {
		return nil, nil
	}
	to, err := s.GetByID(ctx, toID)
	if err != nil {
		return nil, nil
	}
	if from.ID == to.ID {
		return []PathHop{{ID: from.ID}}, nil
	}
	if maxDepth <= 0 {
		maxDepth = 6
	}

	adjacency, err := s.undirectedAdjacency(ctx)
	if err != nil {
		return nil, err
	}
	queue := [][]PathHop{{{ID: from.ID}}}
	visited := map[string]bool{from.ID: true}
	for len(queue) > 0 {
		path := queue[0]
		queue = queue[1:]
		if len(path) > maxDepth {
			break
		}
		for _, edge := range adjacency[path[len(path)-1].ID] {
			if visited[edge.ID] {
				continue
			}
			extended := append(append([]PathHop(nil), path...), edge)
			if edge.ID == to.ID {
				return extended, nil
			}
			visited[edge.ID] = true
			queue = append(queue, extended)
		}
	}
	return nil, nil
}

// undirectedAdjacency builds both-directions adjacency from one pass over the
// link graph — one scan, not one per hop.
func (s *MemoryStore) undirectedAdjacency(ctx context.Context) (map[string][]PathHop, error) {
	mems, err := s.ListRecent(ctx, 0, true, true)
	if err != nil {
		return nil, err
	}
	adjacency := map[string][]PathHop{}
	for _, m := range mems {
		for _, l := range m.Links {
			adjacency[m.ID] = append(adjacency[m.ID], PathHop{ID: l.To, Rel: l.Rel})
			adjacency[l.To] = append(adjacency[l.To], PathHop{ID: m.ID, Rel: l.Rel})
		}
	}
	for _, edges := range adjacency {
		sort.Slice(edges, func(i, j int) bool {
			if edges[i].ID != edges[j].ID {
				return edges[i].ID < edges[j].ID
			}
			return edges[i].Rel < edges[j].Rel
		})
	}
	return adjacency, nil
}

// TrashPath is where soft-deleted memories are parked.
func (s *MemoryStore) TrashPath() string {
	return filepath.Join(filepath.Dir(s.cfg.Path), TrashFilename)
}

// Trash soft-deletes a memory: it is parked in the trash file, then removed.
//
// The missing middle between Supersede (which asserts "replaced by X") and
// Forget (irreversible). TrashRestore brings it back with its id, tags, links
// and timestamps intact — but not its vector, which is recomputed on restore.
func (s *MemoryStore) Trash(ctx context.Context, memoryID string) (bool, error) {
	mem, err := s.GetByID(ctx, memoryID)
	if err != nil {
		return false, nil
	}
	entry := TrashEntry{
		MemoryID: mem.ID, DeletedAt: nowFloat(), Actor: s.actor,
		Memory: mem.ToDict(), Collection: s.cfg.Collection,
	}
	unlock := s.lockTrash()
	err = s.appendTrash(entry)
	unlock()
	if err != nil {
		return false, err
	}
	restore := s.AsActor("trash")
	defer restore()
	if _, err := s.Forget(ctx, mem.ID); err != nil {
		return false, err
	}
	return true, nil
}

// trashVisible reports whether a trash entry belongs to this store's
// collection. Legacy entries carry no collection tag and stay visible
// everywhere — see TrashEntry.
func (s *MemoryStore) trashVisible(e TrashEntry) bool {
	return e.Collection == "" || e.Collection == s.cfg.Collection
}

// readTrash returns every entry in the shared trash file, all collections
// included — mutation paths must write back what they do not own.
func (s *MemoryStore) readTrash() ([]TrashEntry, error) {
	f, err := os.Open(s.TrashPath())
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	defer f.Close()
	gz, err := gzip.NewReader(f)
	if err != nil {
		return nil, nil // an empty or truncated file is an empty trash
	}
	defer gz.Close()

	var out []TrashEntry
	dec := json.NewDecoder(gz)
	for dec.More() {
		var e TrashEntry
		if err := dec.Decode(&e); err != nil {
			// A truncated tail (crash mid-write) must not make the whole
			// trash unreadable.
			break
		}
		out = append(out, e)
	}
	return out, nil
}

// TrashList returns this collection's trash, oldest first.
func (s *MemoryStore) TrashList() ([]TrashEntry, error) {
	entries, err := s.readTrash()
	if err != nil {
		return nil, err
	}
	var out []TrashEntry
	for _, e := range entries {
		if s.trashVisible(e) {
			out = append(out, e)
		}
	}
	return out, nil
}

// TrashRestore brings a trashed memory back, or reports found=false.
//
// The row is re-added with its original id and metadata and re-embedded from
// its text; the restored entry is then dropped. found is false when the id is
// not in this collection's trash — or when it is live again (an export/import
// can resurrect a trashed id): restoring over a live row would clobber it, so
// the entry is kept instead. With several entries for one id (trash → import
// → trash), the newest snapshot is restored and older ones stay recoverable.
func (s *MemoryStore) TrashRestore(ctx context.Context, memoryID string) (Memory, bool, error) {
	// The read-filter-rewrite below must be atomic against other stores
	// sharing the trash file (one per collection, or another process) —
	// unlocked, two concurrent mutations lose whichever rewrite lands first.
	defer s.lockTrash()()
	entries, err := s.readTrash()
	if err != nil {
		return Memory{}, false, err
	}
	found := -1
	for i := range entries {
		if !s.trashVisible(entries[i]) || entries[i].MemoryID != memoryID {
			continue
		}
		if found < 0 || entries[i].DeletedAt > entries[found].DeletedAt {
			found = i
		}
	}
	if found < 0 {
		return Memory{}, false, nil
	}
	if _, err := s.GetByID(ctx, memoryID); err == nil {
		return Memory{}, false, nil
	}
	mem := MemoryFromDict(entries[found].Memory)
	if err := s.UpdateMemory(ctx, mem, true); err != nil {
		return Memory{}, false, err
	}
	restore := s.AsActor("trash")
	defer restore()
	s.journalEntry("restore", mem.ID, nil, mem.ToDict(),
		map[string]any{"from": "trash"})
	keep := append(entries[:found:found], entries[found+1:]...)
	if err := s.writeTrash(keep); err != nil {
		return mem, true, err
	}
	return mem, true, nil
}

// TrashPurge permanently drops one trashed memory, or empties this
// collection's trash when memoryID is empty. Irreversible — the only trash
// operation that loses data. Other collections' entries in the shared trash
// file are untouched.
func (s *MemoryStore) TrashPurge(memoryID string) (int, error) {
	defer s.lockTrash()()
	entries, err := s.readTrash()
	if err != nil {
		return 0, err
	}
	keep := make([]TrashEntry, 0, len(entries))
	for _, e := range entries {
		if s.trashVisible(e) && (memoryID == "" || e.MemoryID == memoryID) {
			continue
		}
		keep = append(keep, e)
	}
	purged := len(entries) - len(keep)
	if purged == 0 {
		return 0, nil
	}
	return purged, s.writeTrash(keep)
}

// TrashPurgeExpired drops trashed memories deleted more than ttlDays ago.
//
// Retention, not reclamation: without it a recoverable delete is really a
// permanent archive and the trash file grows without bound. Meant to be driven
// from a scheduler so the policy holds whether or not anyone runs a maintenance
// pass by hand.
//
// ttlDays <= 0 is a no-op rather than "purge everything" — a misconfigured or
// unset retention must never be read as "delete it all". now == 0 means the
// current time.
func (s *MemoryStore) TrashPurgeExpired(ttlDays float64, now float64) (int, error) {
	if ttlDays <= 0 {
		return 0, nil
	}
	if now == 0 {
		now = nowFloat()
	}
	defer s.lockTrash()()
	entries, err := s.readTrash()
	if err != nil {
		return 0, err
	}
	cutoff := now - ttlDays*86400
	keep := make([]TrashEntry, 0, len(entries))
	for _, e := range entries {
		if !(s.trashVisible(e) && e.DeletedAt < cutoff) {
			keep = append(keep, e)
		}
	}
	purged := len(entries) - len(keep)
	if purged == 0 {
		return 0, nil
	}
	return purged, s.writeTrash(keep)
}

// appendTrash adds one entry as a new gzip member.
//
// Read-all-then-rewrite made N deletes cost O(N²) — 400 calls took 2.16s
// against 137ms for the first 100 — and decay pruning routes through here in a
// loop, so a large prune could stall a maintenance tick. Both this port's
// reader and Python's gzip module read concatenated members transparently, so
// the file stays compatible either way.
func (s *MemoryStore) appendTrash(entry TrashEntry) error {
	path := s.TrashPath()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	gz := gzip.NewWriter(f)
	if err := json.NewEncoder(gz).Encode(entry); err != nil {
		gz.Close()
		f.Close()
		return err
	}
	if err := gz.Close(); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

func (s *MemoryStore) writeTrash(entries []TrashEntry) error {
	path := s.TrashPath()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp := path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	gz := gzip.NewWriter(f)
	enc := json.NewEncoder(gz)
	for _, e := range entries {
		if err := enc.Encode(e); err != nil {
			gz.Close()
			f.Close()
			return err
		}
	}
	if err := gz.Close(); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// validateTagName rejects a tag that would corrupt the comma-joined encoding.
func validateTagName(tag string) error {
	if tag == "" {
		return fmt.Errorf("tag must not be empty")
	}
	if strings.Contains(tag, ",") {
		return fmt.Errorf("tags must not contain commas — got %q", tag)
	}
	return nil
}

func contains(items []string, want string) bool {
	for _, s := range items {
		if s == want {
			return true
		}
	}
	return false
}

func sameStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func dictString(d map[string]any, key string) string {
	if v, ok := d[key].(string); ok {
		return v
	}
	return ""
}

func dictStringOr(d map[string]any, key, def string) string {
	if v, ok := d[key].(string); ok && v != "" {
		return v
	}
	return def
}

func dictFloat(d map[string]any, key string, def float64) float64 {
	switch v := d[key].(type) {
	case float64:
		return v
	case float32:
		return float64(v)
	case int:
		return float64(v)
	}
	return def
}

func dictStrings(d map[string]any, key string) []string {
	switch v := d[key].(type) {
	case []string:
		return v
	case []any:
		out := make([]string, 0, len(v))
		for _, x := range v {
			if s, ok := x.(string); ok {
				out = append(out, s)
			}
		}
		return out
	}
	return nil
}
