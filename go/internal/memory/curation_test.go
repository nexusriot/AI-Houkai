package memory

import (
	"bytes"
	"compress/gzip"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
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

// A merged memory must answer to its NEW text, not its pre-merge one — the
// same invariant Edit holds. Left stale, the hash makes the next idempotent
// write of the pre-merge text look like a repeat, so it is absorbed into the
// merged row and silently lost.
func TestMergeKeepsTheDedupHashInStep(t *testing.T) {
	ctx := context.Background()
	store := newJournaledMemStore(t)
	a, _, _, _ := store.Remember(ctx, "first half", RememberOpts{Idempotent: true})
	b, _, _, _ := store.Remember(ctx, "second half", RememberOpts{})

	merged, err := store.Merge(ctx, a.ID, b.ID, "")
	if err != nil {
		t.Fatal(err)
	}
	if merged.ContentHash != ContentHash(merged.Text) {
		t.Errorf("hash = %s, want the merged text's", merged.ContentHash)
	}

	again, _, _, _ := store.Remember(ctx, "first half", RememberOpts{Idempotent: true})
	if again.ID == a.ID {
		t.Error("the pre-merge text must no longer match the merged row")
	}
	same, _, _, _ := store.Remember(ctx, merged.Text, RememberOpts{Idempotent: true})
	if same.ID != a.ID {
		t.Error("the merged text should match the merged row")
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

// TrashPurgeExpired — retention, so a recoverable delete does not become a
// permanent archive. Mirrors Python's trash_purge_expired.

func agedTrash(t *testing.T, store *MemoryStore, text string, daysAgo float64) Memory {
	t.Helper()
	ctx := context.Background()
	m, _, _, err := store.Remember(ctx, text, RememberOpts{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Trash(ctx, m.ID); err != nil {
		t.Fatal(err)
	}
	entries, err := store.TrashList()
	if err != nil {
		t.Fatal(err)
	}
	for i := range entries {
		if entries[i].MemoryID == m.ID {
			entries[i].DeletedAt = nowFloat() - daysAgo*86400
		}
	}
	if err := store.writeTrash(entries); err != nil {
		t.Fatal(err)
	}
	return m
}

func TestTrashPurgeExpiredDropsOnlyPastCutoff(t *testing.T) {
	store := newTestStore(t)
	old := agedTrash(t, store, "trashed long ago", 40)
	recent := agedTrash(t, store, "trashed yesterday", 1)

	n, err := store.TrashPurgeExpired(30, 0)
	if err != nil || n != 1 {
		t.Fatalf("purged %d (%v), want 1", n, err)
	}
	entries, _ := store.TrashList()
	if len(entries) != 1 || entries[0].MemoryID != recent.ID {
		t.Errorf("remaining = %+v, want only %s", entries, recent.ID)
	}
	_ = old
}

func TestTrashPurgeExpiredZeroTTLIsNoop(t *testing.T) {
	store := newTestStore(t)
	agedTrash(t, store, "should survive", 999)
	for _, ttl := range []float64{0, -5} {
		if n, err := store.TrashPurgeExpired(ttl, 0); err != nil || n != 0 {
			t.Errorf("ttl=%v purged %d (%v), want 0 — a misconfigured retention "+
				"must not mean delete everything", ttl, n, err)
		}
	}
	if entries, _ := store.TrashList(); len(entries) != 1 {
		t.Errorf("entries = %d, want 1", len(entries))
	}
}

func TestTrashPurgeExpiredEmptyTrash(t *testing.T) {
	store := newTestStore(t)
	if n, err := store.TrashPurgeExpired(30, 0); err != nil || n != 0 {
		t.Errorf("purged %d (%v), want 0", n, err)
	}
}

func TestTrashPurgeExpiredHonoursExplicitNow(t *testing.T) {
	store := newTestStore(t)
	agedTrash(t, store, "aged five days", 5)
	// Ten days later the same 7-day retention should sweep it.
	n, err := store.TrashPurgeExpired(7, nowFloat()+10*86400)
	if err != nil || n != 1 {
		t.Errorf("purged %d (%v), want 1", n, err)
	}
}

func TestTrashRestorableRightUpToCutoff(t *testing.T) {
	store := newTestStore(t)
	kept := agedTrash(t, store, "just inside retention", 29)
	if _, err := store.TrashPurgeExpired(30, 0); err != nil {
		t.Fatal(err)
	}
	got, ok, err := store.TrashRestore(context.Background(), kept.ID)
	if err != nil || !ok {
		t.Errorf("restore after retention sweep = ok=%v (%v), want true", ok, err)
	}
	if ok && got.ID != kept.ID {
		t.Errorf("restored %s, want %s", got.ID, kept.ID)
	}
}

// A derived memory must inherit the least-trusted of its sources, or the trust
// tier has a laundering path: absorb untrusted content and the provenance label
// silently survives.

func TestWorstTrust(t *testing.T) {
	for _, tc := range []struct {
		name string
		in   []TrustLevel
		want TrustLevel
	}{
		{"all trusted", []TrustLevel{"trusted", "trusted"}, "trusted"},
		{"one reported", []TrustLevel{"trusted", "reported"}, "reported"},
		{"one untrusted", []TrustLevel{"trusted", "untrusted"}, "untrusted"},
		{"worst wins", []TrustLevel{"untrusted", "reported"}, "untrusted"},
		{"empty reads as trusted", []TrustLevel{"", "trusted"}, "trusted"},
		{"no levels", nil, "trusted"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := WorstTrust(tc.in...); got != tc.want {
				t.Errorf("WorstTrust(%v) = %q, want %q", tc.in, got, tc.want)
			}
		})
	}
}

func TestMergeInheritsLeastTrustedSide(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	target, _, _, err := store.Remember(ctx, "trusted target fact",
		RememberOpts{Trust: "trusted"})
	if err != nil {
		t.Fatal(err)
	}
	other, _, _, err := store.Remember(ctx, "untrusted absorbed fact",
		RememberOpts{Trust: "untrusted"})
	if err != nil {
		t.Fatal(err)
	}
	merged, err := store.Merge(ctx, target.ID, other.ID, "\n\n")
	if err != nil {
		t.Fatal(err)
	}
	if merged.Trust != "untrusted" {
		t.Errorf("merged trust = %q, want untrusted — absorbing untrusted text "+
			"must downgrade the target's provenance", merged.Trust)
	}
}

func TestMergeDoesNotUpgradeTrust(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	target, _, _, _ := store.Remember(ctx, "reported target",
		RememberOpts{Trust: "reported"})
	other, _, _, _ := store.Remember(ctx, "trusted addition",
		RememberOpts{Trust: "trusted"})
	merged, err := store.Merge(ctx, target.ID, other.ID, "\n\n")
	if err != nil {
		t.Fatal(err)
	}
	if merged.Trust != "reported" {
		t.Errorf("merged trust = %q, want reported (merge must not upgrade)",
			merged.Trust)
	}
}

// countTrashMembers reports how many gzip members the trash file holds.
// An append-based writer adds one per call; a rewrite-everything writer always
// leaves exactly one.
func countTrashMembers(t *testing.T, path string) int {
	t.Helper()
	// Read into a bytes.Reader: gzip.Reader wraps a plain *os.File in a
	// bufio.Reader and reads ahead past the member boundary, so Reset would
	// resume at the wrong offset. A bytes.Reader is an io.ByteReader, which
	// gzip consumes exactly.
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	src := bytes.NewReader(raw)
	gz, err := gzip.NewReader(src)
	if err != nil {
		t.Fatalf("gzip.NewReader: %v", err)
	}
	defer gz.Close()
	n := 0
	for {
		// Reset re-enables multistream, so this has to be inside the loop.
		gz.Multistream(false)
		if _, err := io.Copy(io.Discard, gz); err != nil {
			t.Fatalf("read member %d: %v", n, err)
		}
		n++
		if err := gz.Reset(src); err == io.EOF {
			return n
		} else if err != nil {
			t.Fatalf("reset after member %d: %v", n, err)
		}
	}
}

// Trash read the whole file and rewrote it on every call, so N deletes cost
// O(N²) — 100 calls took 137ms and 400 took 2.16s. Decay pruning now routes
// through here in a loop, so a large prune could stall a maintenance tick.
// Python appends a line; this port must too.
func TestTrashAppendsRatherThanRewriting(t *testing.T) {
	ctx := context.Background()
	store := newJournaledMemStore(t)

	var ids []string
	for i := 0; i < 5; i++ {
		m, _, _, err := store.Remember(ctx, fmt.Sprintf("row %d to bin", i), RememberOpts{})
		if err != nil {
			t.Fatalf("Remember: %v", err)
		}
		ids = append(ids, m.ID)
	}
	for _, id := range ids {
		if ok, err := store.Trash(ctx, id); err != nil || !ok {
			t.Fatalf("Trash(%s) = %v, %v", id, ok, err)
		}
	}

	if got := countTrashMembers(t, store.TrashPath()); got != len(ids) {
		t.Errorf("gzip members = %d, want %d (one appended per Trash call)",
			got, len(ids))
	}

	entries, err := store.TrashList()
	if err != nil {
		t.Fatalf("TrashList: %v", err)
	}
	if len(entries) != len(ids) {
		t.Fatalf("TrashList = %d entries, want %d — a multi-member file must "+
			"read back whole", len(entries), len(ids))
	}
	for i, e := range entries {
		if e.MemoryID != ids[i] {
			t.Errorf("entry %d = %s, want %s (oldest first)", i, e.MemoryID, ids[i])
		}
	}
}

// Restore and purge rewrite the file wholesale; they must handle a file that
// Trash appended to, and leave a readable one behind.
func TestTrashRestoreAndPurgeSurviveAppendedFiles(t *testing.T) {
	ctx := context.Background()
	store := newJournaledMemStore(t)

	var ids []string
	for i := 0; i < 3; i++ {
		m, _, _, err := store.Remember(ctx, fmt.Sprintf("appended row %d", i), RememberOpts{})
		if err != nil {
			t.Fatalf("Remember: %v", err)
		}
		ids = append(ids, m.ID)
		if _, err := store.Trash(ctx, m.ID); err != nil {
			t.Fatalf("Trash: %v", err)
		}
	}

	if _, ok, err := store.TrashRestore(ctx, ids[1]); err != nil || !ok {
		t.Fatalf("TrashRestore(%s) = %v, %v", ids[1], ok, err)
	}
	entries, err := store.TrashList()
	if err != nil {
		t.Fatalf("TrashList: %v", err)
	}
	if len(entries) != 2 {
		t.Fatalf("TrashList = %d, want 2 after one restore", len(entries))
	}
	if n, err := store.TrashPurge(ids[0]); err != nil || n != 1 {
		t.Fatalf("TrashPurge = %d, %v; want 1", n, err)
	}
	entries, err = store.TrashList()
	if err != nil {
		t.Fatalf("TrashList: %v", err)
	}
	if len(entries) != 1 || entries[0].MemoryID != ids[2] {
		t.Fatalf("TrashList = %+v, want just %s", entries, ids[2])
	}
}

// Trash deletes through Forget, so undoing that journal entry resurrects the
// row — but the trash entry stayed behind, listing a memory that is live again.
// The stale entry then makes TrashRestore a no-op on the newer row, and
// TrashList advertises a recovery point that recovers nothing.
func TestUndoOfATrashClearsTheTrashEntry(t *testing.T) {
	ctx := context.Background()
	store := newJournaledMemStore(t)

	m, _, _, err := store.Remember(ctx, "binned then reinstated", RememberOpts{})
	if err != nil {
		t.Fatalf("Remember: %v", err)
	}
	if _, err := store.Trash(ctx, m.ID); err != nil {
		t.Fatalf("Trash: %v", err)
	}
	entry := lastJournalEntry(t, store, "forget", m.ID)

	ok, err := store.Undo(ctx, entry)
	if err != nil || !ok {
		t.Fatalf("Undo = %v, %v", ok, err)
	}
	if _, err := store.GetByID(ctx, m.ID); err != nil {
		t.Fatalf("memory not restored: %v", err)
	}
	entries, err := store.TrashList()
	if err != nil {
		t.Fatalf("TrashList: %v", err)
	}
	if len(entries) != 0 {
		t.Fatalf("trash still holds %+v for a live memory", entries)
	}
}

// Only the undone memory's entry goes; anything else in the bin stays.
func TestUndoOfAPlainForgetLeavesTheTrashAlone(t *testing.T) {
	ctx := context.Background()
	store := newJournaledMemStore(t)

	kept, _, _, _ := store.Remember(ctx, "still in the bin", RememberOpts{})
	if _, err := store.Trash(ctx, kept.ID); err != nil {
		t.Fatalf("Trash: %v", err)
	}
	gone, _, _, _ := store.Remember(ctx, "plainly forgotten", RememberOpts{})
	if _, err := store.Forget(ctx, gone.ID); err != nil {
		t.Fatalf("Forget: %v", err)
	}
	entry := lastJournalEntry(t, store, "forget", gone.ID)

	if ok, err := store.Undo(ctx, entry); err != nil || !ok {
		t.Fatalf("Undo = %v, %v", ok, err)
	}
	entries, err := store.TrashList()
	if err != nil {
		t.Fatalf("TrashList: %v", err)
	}
	if len(entries) != 1 || entries[0].MemoryID != kept.ID {
		t.Fatalf("trash = %+v, want only %s", entries, kept.ID)
	}
}

func lastJournalEntry(t *testing.T, store *MemoryStore, op, id string) JournalEntry {
	t.Helper()
	j := store.Journal()
	if j == nil {
		t.Fatal("journal disabled")
	}
	entries, err := j.Read(ReadOpts{Op: op})
	if err != nil {
		t.Fatalf("journal read: %v", err)
	}
	for i := len(entries) - 1; i >= 0; i-- {
		if entries[i].ID == id {
			return entries[i]
		}
	}
	t.Fatalf("no %q entry for %s", op, id)
	return JournalEntry{}
}

// Merge folds other into target and deletes other. Trust already takes the
// worse of the two sides; the pin did not travel at all, so absorbing a
// standing instruction into an unpinned row emptied the working set and deleted
// the only copy of the flag.
func TestMergeKeepsThePin(t *testing.T) {
	ctx := context.Background()
	store := newJournaledMemStore(t)

	target, _, _, _ := store.Remember(ctx, "plain target", RememberOpts{})
	other, _, _, _ := store.Remember(ctx, "ALWAYS run make lint",
		RememberOpts{Pinned: true})

	merged, err := store.Merge(ctx, target.ID, other.ID, "\n\n")
	if err != nil {
		t.Fatalf("Merge: %v", err)
	}
	if !merged.Pinned {
		t.Error("merged row lost the pin it absorbed")
	}
	pinned, err := store.PinnedMemories(ctx)
	if err != nil {
		t.Fatalf("PinnedMemories: %v", err)
	}
	if len(pinned) != 1 || pinned[0].ID != target.ID {
		t.Fatalf("working set = %+v, want just the merged row", pinned)
	}
}

func TestMergeOfTwoUnpinnedRowsPinsNothing(t *testing.T) {
	ctx := context.Background()
	store := newJournaledMemStore(t)

	target, _, _, _ := store.Remember(ctx, "first half", RememberOpts{})
	other, _, _, _ := store.Remember(ctx, "second half", RememberOpts{})

	merged, err := store.Merge(ctx, target.ID, other.ID, "\n\n")
	if err != nil {
		t.Fatalf("Merge: %v", err)
	}
	if merged.Pinned {
		t.Error("merged row was pinned out of nowhere")
	}
}

// newCollectionStore builds a store on a shared dir under its own collection,
// the shape two collections opened on one store path have in production.
func newCollectionStore(t *testing.T, dir, collection string) *MemoryStore {
	t.Helper()
	backend, err := vector.NewChromem(filepath.Join(dir, "s-"+collection), collection, 16)
	if err != nil {
		t.Fatalf("NewChromem: %v", err)
	}
	t.Cleanup(func() { _ = backend.Close() })
	return NewMemoryStore(backend, &stubEmbedder{dim: 16}, DefaultStoreConfig(dir, collection))
}

func TestTrashScopedByCollection(t *testing.T) {
	dir := t.TempDir()
	colA := newCollectionStore(t, dir, "col_a")
	colB := newCollectionStore(t, dir, "col_b")
	ctx := context.Background()

	m, _, _, err := colA.Remember(ctx, "belongs to a", RememberOpts{})
	if err != nil {
		t.Fatal(err)
	}
	if ok, err := colA.Trash(ctx, m.ID); err != nil || !ok {
		t.Fatalf("Trash: ok=%v err=%v", ok, err)
	}

	if entries, _ := colB.TrashList(); len(entries) != 0 {
		t.Fatalf("col_b sees col_a's trash: %v", entries)
	}
	if _, found, _ := colB.TrashRestore(ctx, m.ID); found {
		t.Fatal("col_b restored col_a's memory — cross-collection leak")
	}
	if _, err := colB.GetByID(ctx, m.ID); err == nil {
		t.Fatal("col_a's memory materialized in col_b")
	}
	restored, found, err := colA.TrashRestore(ctx, m.ID)
	if err != nil || !found || restored.ID != m.ID {
		t.Fatalf("col_a restore: found=%v err=%v", found, err)
	}
}

func TestTrashPurgeLeavesOtherCollections(t *testing.T) {
	dir := t.TempDir()
	colA := newCollectionStore(t, dir, "col_a")
	colB := newCollectionStore(t, dir, "col_b")
	ctx := context.Background()

	a, _, _, _ := colA.Remember(ctx, "a's trash", RememberOpts{})
	b, _, _, _ := colB.Remember(ctx, "b's trash", RememberOpts{})
	colA.Trash(ctx, a.ID)
	colB.Trash(ctx, b.ID)

	if n, err := colB.TrashPurge(""); err != nil || n != 1 {
		t.Fatalf("purge-all in col_b: n=%d err=%v (want 1)", n, err)
	}
	entries, _ := colA.TrashList()
	if len(entries) != 1 || entries[0].MemoryID != a.ID {
		t.Fatalf("col_a's entry lost to col_b's purge: %v", entries)
	}
	if _, found, _ := colA.TrashRestore(ctx, a.ID); !found {
		t.Fatal("col_a can no longer restore after col_b purged")
	}
}

func TestTrashPurgeExpiredScopedToCollection(t *testing.T) {
	dir := t.TempDir()
	colA := newCollectionStore(t, dir, "col_a")
	colB := newCollectionStore(t, dir, "col_b")
	ctx := context.Background()

	a, _, _, _ := colA.Remember(ctx, "old in a", RememberOpts{})
	b, _, _, _ := colB.Remember(ctx, "old in b", RememberOpts{})
	colA.Trash(ctx, a.ID)
	colB.Trash(ctx, b.ID)

	// Both entries are ancient; only col_a runs retention.
	future := nowFloat() + 90*86400
	if n, err := colA.TrashPurgeExpired(30, future); err != nil || n != 1 {
		t.Fatalf("TrashPurgeExpired: n=%d err=%v (want 1)", n, err)
	}
	if entries, _ := colB.TrashList(); len(entries) != 1 {
		t.Fatalf("col_b's entry TTL-purged by col_a's maintenance: %v", entries)
	}
}

func TestLegacyUntaggedTrashEntryVisibleEverywhere(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	m, _, _, _ := store.Remember(ctx, "template row", RememberOpts{})
	store.Forget(ctx, m.ID)

	legacy := TrashEntry{
		MemoryID: m.ID, DeletedAt: nowFloat(), Actor: "lib",
		Memory: m.ToDict(), // Collection deliberately left ""
	}
	if err := store.appendTrash(legacy); err != nil {
		t.Fatal(err)
	}
	entries, _ := store.TrashList()
	if len(entries) != 1 {
		t.Fatalf("legacy entry invisible: %v", entries)
	}
	if _, found, err := store.TrashRestore(ctx, m.ID); err != nil || !found {
		t.Fatalf("legacy entry not restorable: found=%v err=%v", found, err)
	}
}

func TestTrashRestoreRefusesLiveID(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	m, _, _, _ := store.Remember(ctx, "original text", RememberOpts{})
	store.Trash(ctx, m.ID)

	// Resurrect the id out-of-band (an import can do this legitimately).
	live := m
	live.Text = "resurrected by import"
	if err := store.UpdateMemory(ctx, live, true); err != nil {
		t.Fatal(err)
	}

	if _, found, _ := store.TrashRestore(ctx, m.ID); found {
		t.Fatal("restore over a live id must be refused")
	}
	entries, _ := store.TrashList()
	if len(entries) != 1 {
		t.Fatal("refused restore must keep the snapshot recoverable")
	}
}

func TestTrashRestorePicksNewestDuplicate(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	m, _, _, _ := store.Remember(ctx, "version ONE", RememberOpts{})
	store.Trash(ctx, m.ID)

	entries, _ := store.TrashList()
	newer := entries[0]
	newer.DeletedAt += 10
	newer.Memory = m.ToDict()
	newer.Memory["text"] = "version TWO"
	if err := store.appendTrash(newer); err != nil {
		t.Fatal(err)
	}

	restored, found, err := store.TrashRestore(ctx, m.ID)
	if err != nil || !found {
		t.Fatalf("restore: found=%v err=%v", found, err)
	}
	if restored.Text != "version TWO" {
		t.Fatalf("restored %q, want the newest snapshot", restored.Text)
	}
	if entries, _ = store.TrashList(); len(entries) != 1 {
		t.Fatal("older snapshot must stay parked")
	}
	store.Forget(ctx, m.ID)
	restored, found, _ = store.TrashRestore(ctx, m.ID)
	if !found || restored.Text != "version ONE" {
		t.Fatalf("second restore got %q found=%v, want version ONE", restored.Text, found)
	}
}

func TestConcurrentTrashMutationsLoseNothing(t *testing.T) {
	// Two stores (one per collection) share the trash file; unlocked, their
	// read-filter-rewrite mutations raced and whichever rewrite landed last
	// silently dropped the other's entries.
	dir := t.TempDir()
	colA := newCollectionStore(t, dir, "col_a")
	colB := newCollectionStore(t, dir, "col_b")
	ctx := context.Background()

	const n = 8
	idsA := make([]string, n)
	idsB := make([]string, n)
	for i := 0; i < n; i++ {
		a, _, _, _ := colA.Remember(ctx, fmt.Sprintf("a doomed %d", i), RememberOpts{})
		b, _, _, _ := colB.Remember(ctx, fmt.Sprintf("b doomed %d", i), RememberOpts{})
		idsA[i], idsB[i] = a.ID, b.ID
	}

	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		for _, id := range idsA {
			if ok, err := colA.Trash(ctx, id); err != nil || !ok {
				t.Errorf("colA.Trash(%s): ok=%v err=%v", id, ok, err)
			}
		}
	}()
	go func() {
		defer wg.Done()
		for _, id := range idsB {
			if ok, err := colB.Trash(ctx, id); err != nil || !ok {
				t.Errorf("colB.Trash(%s): ok=%v err=%v", id, ok, err)
			}
			// Interleave a rewrite-path mutation with colA's appends.
			if _, _, err := colB.TrashRestore(ctx, id); err != nil {
				t.Errorf("colB.TrashRestore(%s): %v", id, err)
			}
			if ok, err := colB.Trash(ctx, id); err != nil || !ok {
				t.Errorf("colB re-Trash(%s): ok=%v err=%v", id, ok, err)
			}
		}
	}()
	wg.Wait()

	entriesA, err := colA.TrashList()
	if err != nil {
		t.Fatal(err)
	}
	entriesB, err := colB.TrashList()
	if err != nil {
		t.Fatal(err)
	}
	if len(entriesA) != n || len(entriesB) != n {
		t.Fatalf("lost trash entries under concurrency: colA=%d colB=%d, want %d each",
			len(entriesA), len(entriesB), n)
	}
}
