package memory

import (
	"context"
	"errors"
	"path/filepath"
	"testing"

	"github.com/nexusriot/ai-houkai/internal/vector"
)

// Curation operations graduated from ai-houkai-service (D). Merge in
// particular could not be done correctly from outside the store: Forget does
// not clean up incoming links, so folding one memory into another without
// re-pointing them silently strands every relationship pointing at the
// absorbed memory.

func TestMergeCombinesAndDeletes(t *testing.T) {
	ctx := context.Background()
	store := newJournaledMemStore(t)
	a, _, _, _ := store.Remember(ctx, "first half", RememberOpts{})
	b, _, _, _ := store.Remember(ctx, "second half", RememberOpts{})

	merged, err := store.Merge(ctx, a.ID, b.ID, "")
	if err != nil {
		t.Fatal(err)
	}
	if merged.Text != "first half\n\nsecond half" {
		t.Errorf("text = %q", merged.Text)
	}
	if merged.ID != a.ID {
		t.Errorf("merged id = %s, want the target %s", merged.ID, a.ID)
	}
	if _, err := store.GetByID(ctx, b.ID); err == nil {
		t.Error("the absorbed memory survived")
	}
}

func TestMergeCustomSeparator(t *testing.T) {
	ctx := context.Background()
	store := newJournaledMemStore(t)
	a, _, _, _ := store.Remember(ctx, "left", RememberOpts{})
	b, _, _, _ := store.Remember(ctx, "right", RememberOpts{})
	merged, err := store.Merge(ctx, a.ID, b.ID, " | ")
	if err != nil {
		t.Fatal(err)
	}
	if merged.Text != "left | right" {
		t.Errorf("text = %q", merged.Text)
	}
}

func TestMergeRepointsIncomingLinks(t *testing.T) {
	// The reason merge belongs in the store.
	ctx := context.Background()
	store := newJournaledMemStore(t)
	a, _, _, _ := store.Remember(ctx, "target", RememberOpts{})
	b, _, _, _ := store.Remember(ctx, "absorbed", RememberOpts{})
	pointer, _, _, _ := store.Remember(ctx, "points at the absorbed one", RememberOpts{})
	if err := store.Link(ctx, pointer.ID, b.ID, "refines"); err != nil {
		t.Fatal(err)
	}

	if _, err := store.Merge(ctx, a.ID, b.ID, ""); err != nil {
		t.Fatal(err)
	}
	got, err := store.GetByID(ctx, pointer.ID)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Links) != 1 || got.Links[0].To != a.ID || got.Links[0].Rel != "refines" {
		t.Errorf("links = %+v, want one edge to the merge target", got.Links)
	}
}

func TestMergeTransfersOutgoingLinksWithoutSelfLoops(t *testing.T) {
	ctx := context.Background()
	store := newJournaledMemStore(t)
	a, _, _, _ := store.Remember(ctx, "target", RememberOpts{})
	b, _, _, _ := store.Remember(ctx, "absorbed", RememberOpts{})
	c, _, _, _ := store.Remember(ctx, "b's neighbour", RememberOpts{})
	store.Link(ctx, b.ID, c.ID, "refines")
	// b also points back at a: transferring that edge verbatim would be a
	// self-loop on the merged memory.
	store.Link(ctx, b.ID, a.ID, "related")

	merged, err := store.Merge(ctx, a.ID, b.ID, "")
	if err != nil {
		t.Fatal(err)
	}
	for _, l := range merged.Links {
		if l.To == a.ID {
			t.Errorf("self-loop created: %+v", l)
		}
	}
	found := false
	for _, l := range merged.Links {
		if l.To == c.ID && l.Rel == "refines" {
			found = true
		}
	}
	if !found {
		t.Errorf("outgoing link not transferred: %+v", merged.Links)
	}
}

func TestMergeRejectsSelfMerge(t *testing.T) {
	ctx := context.Background()
	store := newJournaledMemStore(t)
	a, _, _, _ := store.Remember(ctx, "only one", RememberOpts{})
	if _, err := store.Merge(ctx, a.ID, a.ID, ""); !errors.Is(err, ErrSelfMerge) {
		t.Errorf("err = %v, want ErrSelfMerge", err)
	}
}

func TestMergeMissingMemory(t *testing.T) {
	ctx := context.Background()
	store := newJournaledMemStore(t)
	a, _, _, _ := store.Remember(ctx, "exists", RememberOpts{})
	_, err := store.Merge(ctx, a.ID, "ghost", "")
	if !errors.Is(err, ErrNotFound) {
		t.Errorf("err = %v, want ErrNotFound", err)
	}
}

func TestMergeIsJournaled(t *testing.T) {
	ctx := context.Background()
	store := newJournaledMemStore(t)
	a, _, _, _ := store.Remember(ctx, "target", RememberOpts{})
	b, _, _, _ := store.Remember(ctx, "absorbed", RememberOpts{})
	if _, err := store.Merge(ctx, a.ID, b.ID, ""); err != nil {
		t.Fatal(err)
	}
	entries, _ := store.Journal().Read(ReadOpts{Op: "edit"})
	found := false
	for _, e := range entries {
		if e.ID == a.ID && e.Meta["merged_from"] == b.ID {
			found = true
		}
	}
	if !found {
		t.Errorf("no merge journal entry with merged_from")
	}
}

func TestVersions(t *testing.T) {
	ctx := context.Background()
	store := newJournaledMemStore(t)
	m, _, _, _ := store.Remember(ctx, "v1 text", RememberOpts{Tags: []string{"a"}})
	v2, v3 := "v2 text", "v3 text"
	store.Edit(ctx, m.ID, EditOpts{Text: &v2})
	store.Edit(ctx, m.ID, EditOpts{Text: &v3})

	got, err := store.Versions(m.ID)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || got[0].Text != "v1 text" || got[1].Text != "v2 text" {
		t.Fatalf("versions = %+v", got)
	}
	if len(got[0].Tags) != 1 || got[0].Tags[0] != "a" {
		t.Errorf("tags = %v", got[0].Tags)
	}
}

func TestVersionsExcludesTheLiveState(t *testing.T) {
	ctx := context.Background()
	store := newJournaledMemStore(t)
	m, _, _, _ := store.Remember(ctx, "never edited", RememberOpts{})
	got, _ := store.Versions(m.ID)
	if len(got) != 0 {
		t.Errorf("versions = %+v, want none", got)
	}
}

func TestTagCuration(t *testing.T) {
	ctx := context.Background()
	store := newJournaledMemStore(t)
	store.Remember(ctx, "one", RememberOpts{Tags: []string{"common", "rare"}})
	store.Remember(ctx, "two", RememberOpts{Tags: []string{"common"}})

	tags, err := store.ListTags(ctx, false)
	if err != nil {
		t.Fatal(err)
	}
	if len(tags) != 2 || tags[0].Tag != "common" || tags[0].Count != 2 {
		t.Fatalf("tags = %+v", tags)
	}

	if n, err := store.RenameTag(ctx, "rare", "uncommon"); err != nil || n != 1 {
		t.Fatalf("rename = %d (%v)", n, err)
	}
	if n, err := store.MergeTags(ctx, []string{"uncommon"}, "common"); err != nil || n != 1 {
		t.Fatalf("merge = %d (%v)", n, err)
	}
	if n, err := store.DeleteTag(ctx, "common"); err != nil || n != 2 {
		t.Fatalf("delete = %d (%v)", n, err)
	}
	tags, _ = store.ListTags(ctx, false)
	if len(tags) != 0 {
		t.Errorf("tags after delete = %+v", tags)
	}
}

func TestRenameTagDeduplicates(t *testing.T) {
	ctx := context.Background()
	store := newJournaledMemStore(t)
	m, _, _, _ := store.Remember(ctx, "both", RememberOpts{Tags: []string{"old", "new"}})
	if _, err := store.RenameTag(ctx, "old", "new"); err != nil {
		t.Fatal(err)
	}
	got, _ := store.GetByID(ctx, m.ID)
	if len(got.Tags) != 1 || got.Tags[0] != "new" {
		t.Errorf("tags = %v, want [new]", got.Tags)
	}
}

func TestTagOpsRejectACommaAndNoOpCleanly(t *testing.T) {
	ctx := context.Background()
	store := newJournaledMemStore(t)
	store.Remember(ctx, "one", RememberOpts{Tags: []string{"a"}})

	// Tags are stored comma-joined; one with a comma splits into two.
	if _, err := store.RenameTag(ctx, "a", "b,c"); err == nil {
		t.Error("a comma in a tag must be rejected")
	}
	if n, _ := store.RenameTag(ctx, "absent", "x"); n != 0 {
		t.Errorf("renaming an absent tag changed %d", n)
	}
	if n, _ := store.DeleteTag(ctx, "absent"); n != 0 {
		t.Errorf("deleting an absent tag changed %d", n)
	}
}

func TestTagCurationCoversSupersededMemories(t *testing.T) {
	// Skipping them would leave the old spelling alive in rows a later
	// Restore brings back.
	ctx := context.Background()
	store := newJournaledMemStore(t)
	old, _, _, _ := store.Remember(ctx, "old", RememberOpts{Tags: []string{"typo"}})
	fresh, _, _, _ := store.Remember(ctx, "new", RememberOpts{})
	store.Supersede(ctx, old.ID, fresh.ID)

	if n, err := store.RenameTag(ctx, "typo", "fixed"); err != nil || n != 1 {
		t.Fatalf("rename = %d (%v)", n, err)
	}
	got, _ := store.GetByID(ctx, old.ID)
	if len(got.Tags) != 1 || got.Tags[0] != "fixed" {
		t.Errorf("superseded memory tags = %v", got.Tags)
	}
}

func TestFindPath(t *testing.T) {
	ctx := context.Background()
	store := newJournaledMemStore(t)
	a, _, _, _ := store.Remember(ctx, "a", RememberOpts{})
	b, _, _, _ := store.Remember(ctx, "b", RememberOpts{})
	c, _, _, _ := store.Remember(ctx, "c", RememberOpts{})
	store.Link(ctx, a.ID, b.ID, "refines")
	store.Link(ctx, b.ID, c.ID, "refines")

	path, err := store.FindPath(ctx, a.ID, c.ID, 6)
	if err != nil {
		t.Fatal(err)
	}
	if len(path) != 3 || path[0].ID != a.ID || path[2].ID != c.ID {
		t.Fatalf("path = %+v", path)
	}

	// Undirected: "how are these related?" ignores arrow direction.
	back, _ := store.FindPath(ctx, c.ID, a.ID, 6)
	if len(back) != 3 || back[0].ID != c.ID {
		t.Errorf("reverse path = %+v", back)
	}
}

func TestFindPathBoundaries(t *testing.T) {
	ctx := context.Background()
	store := newJournaledMemStore(t)
	ids := make([]string, 5)
	for i := range ids {
		m, _, _, _ := store.Remember(ctx, "chain", RememberOpts{})
		ids[i] = m.ID
	}
	for i := 0; i < 4; i++ {
		store.Link(ctx, ids[i], ids[i+1], "refines")
	}
	if p, _ := store.FindPath(ctx, ids[0], ids[4], 2); len(p) != 0 {
		t.Errorf("max_depth=2 should not reach 4 hops: %+v", p)
	}
	if p, _ := store.FindPath(ctx, ids[0], ids[4], 4); len(p) != 5 {
		t.Errorf("max_depth=4 path = %+v", p)
	}
	if p, _ := store.FindPath(ctx, ids[0], ids[0], 6); len(p) != 1 {
		t.Errorf("same-node path = %+v", p)
	}
	if p, _ := store.FindPath(ctx, ids[0], "ghost", 6); len(p) != 0 {
		t.Errorf("unknown id path = %+v", p)
	}
}

func TestFindPathNoRoute(t *testing.T) {
	ctx := context.Background()
	store := newJournaledMemStore(t)
	a, _, _, _ := store.Remember(ctx, "island a", RememberOpts{})
	b, _, _, _ := store.Remember(ctx, "island b", RememberOpts{})
	if p, _ := store.FindPath(ctx, a.ID, b.ID, 6); len(p) != 0 {
		t.Errorf("path between islands = %+v", p)
	}
}

func TestTrashRoundtrip(t *testing.T) {
	ctx := context.Background()
	store := newJournaledMemStore(t)
	other, _, _, _ := store.Remember(ctx, "link target", RememberOpts{})
	m, _, _, _ := store.Remember(ctx, "recoverable", RememberOpts{
		Tags: []string{"t"}, Importance: Float32Ptr(0.9),
	})
	store.Link(ctx, m.ID, other.ID, "refines")

	ok, err := store.Trash(ctx, m.ID)
	if err != nil || !ok {
		t.Fatalf("trash = %v (%v)", ok, err)
	}
	if _, err := store.GetByID(ctx, m.ID); err == nil {
		t.Error("trashed memory is still in the store")
	}

	restored, found, err := store.TrashRestore(ctx, m.ID)
	if err != nil || !found {
		t.Fatalf("restore = %v (%v)", found, err)
	}
	if restored.ID != m.ID || restored.Importance != 0.9 {
		t.Errorf("restored = %+v", restored)
	}
	if len(restored.Tags) != 1 || restored.Tags[0] != "t" {
		t.Errorf("tags = %v", restored.Tags)
	}
	if len(restored.Links) != 1 || restored.Links[0].To != other.ID {
		t.Errorf("links = %+v", restored.Links)
	}
	if _, err := store.GetByID(ctx, m.ID); err != nil {
		t.Errorf("restored memory not back in the store: %v", err)
	}
}

func TestTrashListAndPurge(t *testing.T) {
	ctx := context.Background()
	store := newJournaledMemStore(t)
	a, _, _, _ := store.Remember(ctx, "first", RememberOpts{})
	b, _, _, _ := store.Remember(ctx, "second", RememberOpts{})
	store.Trash(ctx, a.ID)
	store.Trash(ctx, b.ID)

	entries, err := store.TrashList()
	if err != nil || len(entries) != 2 {
		t.Fatalf("trash list = %d (%v)", len(entries), err)
	}
	if entries[0].MemoryID != a.ID {
		t.Errorf("not oldest-first: %+v", entries)
	}

	if n, _ := store.TrashPurge(a.ID); n != 1 {
		t.Errorf("purge one = %d", n)
	}
	if n, _ := store.TrashPurge(""); n != 1 {
		t.Errorf("purge all = %d", n)
	}
	entries, _ = store.TrashList()
	if len(entries) != 0 {
		t.Errorf("trash not empty: %+v", entries)
	}
}

func TestTrashEdgeCases(t *testing.T) {
	ctx := context.Background()
	store := newJournaledMemStore(t)
	if ok, _ := store.Trash(ctx, "nope"); ok {
		t.Error("trashing a missing memory reported success")
	}
	if _, found, _ := store.TrashRestore(ctx, "nope"); found {
		t.Error("restored something that was never trashed")
	}
	if n, _ := store.TrashPurge(""); n != 0 {
		t.Errorf("purging an empty trash = %d", n)
	}
	if entries, err := store.TrashList(); err != nil || len(entries) != 0 {
		t.Errorf("empty trash = %d (%v)", len(entries), err)
	}
}

// newJournaledMemStore builds a store with the journal ON (the shared
// newTestStore leaves DefaultStoreConfig's setting) so versions() and the
// merge journal assertions have a log to read.
func newJournaledMemStore(t *testing.T) *MemoryStore {
	t.Helper()
	dir := t.TempDir()
	backend, err := vector.NewChromem(filepath.Join(dir, "s"), "curation", 16)
	if err != nil {
		t.Fatalf("NewChromem: %v", err)
	}
	t.Cleanup(func() { _ = backend.Close() })
	cfg := DefaultStoreConfig(filepath.Join(dir, "s"), "curation")
	cfg.JournalEnabled = true
	cfg.JournalPath = filepath.Join(dir, "journal.log")
	return NewMemoryStore(backend, &stubEmbedder{dim: 16}, cfg)
}
