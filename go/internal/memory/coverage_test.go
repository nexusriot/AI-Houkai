package memory

import (
	"context"
	"encoding/json"
	"errors"
	"path/filepath"
	"strings"
	"testing"

	"github.com/nexusriot/ai-houkai/internal/vector"
)

// newConflictStore builds a store like newTestStore but lets the caller pin the
// conflict policy and threshold so the Remember conflict branches are reachable.
// It intentionally uses a distinct name so it does not collide with the shared
// newTestStore helper in store_test.go.
func newConflictStore(t *testing.T, policy ConflictPolicy, threshold float32) *MemoryStore {
	t.Helper()
	dir := t.TempDir()
	backend, err := vector.NewChromem(filepath.Join(dir, "s"), "test", 16)
	if err != nil {
		t.Fatalf("NewChromem: %v", err)
	}
	t.Cleanup(func() { _ = backend.Close() })

	cfg := DefaultStoreConfig(dir, "test")
	cfg.ConflictPolicy = policy
	cfg.ConflictThreshold = threshold
	return NewMemoryStore(backend, &stubEmbedder{dim: 16}, cfg)
}

func TestTokenizeASCIIAndApostrophe(t *testing.T) {
	// Plain ASCII lowercases and splits on word boundaries; no CJK bigrams.
	got := tokenize("The Cat SAT")
	want := []string{"the", "cat", "sat"}
	if len(got) != len(want) {
		t.Fatalf("ascii tokenize = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("ascii tokenize = %v, want %v", got, want)
		}
	}

	// Both ASCII and typographic apostrophes are stripped: "don't"/"don’t" → "dont".
	for _, in := range []string{"don't", "don’t"} {
		toks := tokenize(in)
		if len(toks) != 1 || toks[0] != "dont" {
			t.Errorf("tokenize(%q) = %v, want [dont]", in, toks)
		}
	}
}

func TestTokenizeCJKBigrams(t *testing.T) {
	// "日本語のテスト": the \w+ run splits at the hiragana boundary "の" is still a
	// letter so the whole thing is one \w token, then CJK bigrams are appended.
	toks := tokenize("日本語のテスト")
	set := map[string]bool{}
	for _, tk := range toks {
		set[tk] = true
	}
	// Adjacent-char bigrams must be emitted.
	for _, bg := range []string{"日本", "本語", "語の", "のテ", "テス", "スト"} {
		if !set[bg] {
			t.Errorf("expected CJK bigram %q in %v", bg, toks)
		}
	}
	// A lone CJK character is NOT re-emitted as its own extra token beyond the
	// original \w run; count how many times "日" appears as a standalone token.
	solo := 0
	for _, tk := range toks {
		if tk == "日" {
			solo++
		}
	}
	if solo != 0 {
		t.Errorf("single CJK char should not be re-emitted as a standalone bigram token, got %d", solo)
	}

	// A single lone CJK char has no bigram to emit, so it survives only as the
	// original word token (or as nothing extra). Assert no panic / no bigram.
	one := tokenize("犬")
	for _, tk := range one {
		if len([]rune(tk)) == 2 {
			t.Errorf("single CJK char must not produce a bigram, got %v", one)
		}
	}
}

func TestSubgraph(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	a, _, _, _ := mustRemember(t, store, "alpha node", RememberOpts{})
	b, _, _, _ := mustRemember(t, store, "beta node", RememberOpts{})
	c, _, _, _ := mustRemember(t, store, "gamma node", RememberOpts{})
	if err := store.Link(ctx, a, b, "refines"); err != nil {
		t.Fatal(err)
	}
	if err := store.Link(ctx, b, c, "refines"); err != nil {
		t.Fatal(err)
	}

	g, err := store.Subgraph(ctx, []string{a}, 2)
	if err != nil {
		t.Fatalf("Subgraph: %v", err)
	}
	// Nodes: a (seed) + b (hop1) + c (hop2) reachable via "both" neighbors.
	nodeIDs := map[string]bool{}
	for _, n := range g.Nodes {
		nodeIDs[n.ID] = true
	}
	for _, want := range []string{a, b, c} {
		if !nodeIDs[want] {
			t.Errorf("Subgraph missing node %s; got %v", want, nodeIDs)
		}
	}
	// Edges must include only intra-set links a->b and b->c.
	edges := map[string]string{}
	for _, e := range g.Edges {
		edges[e.From+"->"+e.To] = e.Rel
	}
	if edges[a+"->"+b] != "refines" {
		t.Errorf("expected edge a->b (refines), got %v", g.Edges)
	}
	if edges[b+"->"+c] != "refines" {
		t.Errorf("expected edge b->c (refines), got %v", g.Edges)
	}
}

func TestNeighborsInAndBoth(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	a, _, _, _ := mustRemember(t, store, "source node", RememberOpts{})
	b, _, _, _ := mustRemember(t, store, "target node", RememberOpts{})
	if err := store.Link(ctx, a, b, RelRelated); err != nil {
		t.Fatal(err)
	}

	// direction "in": neighbours of b should surface a (reverse scan).
	in, err := store.Neighbors(ctx, b, "", "in", 1)
	if err != nil {
		t.Fatalf("Neighbors in: %v", err)
	}
	if len(in) != 1 || in[0].ID != a {
		t.Fatalf("Neighbors(b, in) = %+v, want [a=%s]", in, a)
	}

	// direction "both" from b also reaches a via incoming edge.
	both, err := store.Neighbors(ctx, b, "", "both", 1)
	if err != nil {
		t.Fatalf("Neighbors both: %v", err)
	}
	foundA := false
	for _, nb := range both {
		if nb.ID == a {
			foundA = true
		}
	}
	if !foundA {
		t.Errorf("Neighbors(b, both) should include a, got %+v", both)
	}

	// A rel filter that does not match must exclude the incoming edge.
	none, err := store.Neighbors(ctx, b, "nonexistent", "in", 1)
	if err != nil {
		t.Fatalf("Neighbors in filtered: %v", err)
	}
	if len(none) != 0 {
		t.Errorf("rel filter should exclude non-matching incoming edge, got %+v", none)
	}
}

func TestUpdateMemoryMetadataOnly(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	id, _, _, _ := mustRemember(t, store, "original text here", RememberOpts{Importance: 0.4})

	m, err := store.GetByID(ctx, id)
	if err != nil {
		t.Fatalf("GetByID: %v", err)
	}
	m.Importance = 0.9
	m.Tags = []string{"updated"}
	if err := store.UpdateMemory(ctx, m, false); err != nil {
		t.Fatalf("UpdateMemory metadata-only: %v", err)
	}
	got, err := store.GetByID(ctx, id)
	if err != nil {
		t.Fatalf("GetByID after update: %v", err)
	}
	if got.Importance != 0.9 {
		t.Errorf("metadata update did not persist importance: got %v", got.Importance)
	}
	if got.Text != "original text here" {
		t.Errorf("metadata-only update must not change text, got %q", got.Text)
	}
	if !containsTag(got, "updated") {
		t.Errorf("metadata update did not persist tag, got %v", got.Tags)
	}
}

func TestUpdateMemoryTextChangedReembeds(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	id, _, _, _ := mustRemember(t, store, "before change", RememberOpts{})

	m, err := store.GetByID(ctx, id)
	if err != nil {
		t.Fatalf("GetByID: %v", err)
	}
	m.Text = "after the change entirely different"
	if err := store.UpdateMemory(ctx, m, true); err != nil {
		t.Fatalf("UpdateMemory textChanged: %v", err)
	}
	got, err := store.GetByID(ctx, id)
	if err != nil {
		t.Fatalf("GetByID after re-embed: %v", err)
	}
	if got.Text != "after the change entirely different" {
		t.Errorf("text change did not persist, got %q", got.Text)
	}
	// The re-embedded doc should be retrievable by its new text (proves re-embed
	// used the new content, not the stale vector).
	hits, err := store.Recall(ctx, "after the change entirely different", 5, RecallOpts{Mode: ModeSemantic})
	if err != nil {
		t.Fatalf("Recall: %v", err)
	}
	if len(hits) == 0 || hits[0].ID != id {
		t.Errorf("re-embedded memory should rank first for its new text, got %+v", hits)
	}
}

func TestRememberConflictRaise(t *testing.T) {
	store := newConflictStore(t, PolicyRaise, 0.1)
	ctx := context.Background()

	// First memory stores cleanly (no candidates yet).
	if _, stored, _, err := store.Remember(ctx, "the deploy pipeline is healthy", RememberOpts{Type: Semantic}); err != nil || !stored {
		t.Fatalf("first Remember: stored=%v err=%v", stored, err)
	}
	// A near-identical same-type memory triggers a conflict → Raise returns a
	// *ConflictError and stored=false.
	_, stored, conflicts, err := store.Remember(ctx, "the deploy pipeline is healthy", RememberOpts{Type: Semantic})
	if stored {
		t.Errorf("PolicyRaise should not store on conflict")
	}
	var ce *ConflictError
	if !errors.As(err, &ce) {
		t.Fatalf("PolicyRaise should return *ConflictError, got %T: %v", err, err)
	}
	if len(conflicts) == 0 || len(ce.Conflicts) == 0 {
		t.Errorf("expected conflicts to be reported, got %v", conflicts)
	}
	// Only the first memory should be persisted.
	if c, _ := store.Count(ctx); c != 1 {
		t.Errorf("after Raise, count = %d, want 1", c)
	}
}

func TestRememberConflictWarnStores(t *testing.T) {
	store := newConflictStore(t, PolicyWarn, 0.1)
	ctx := context.Background()

	store.Remember(ctx, "the release cadence is weekly", RememberOpts{Type: Semantic})
	_, stored, _, err := store.Remember(ctx, "the release cadence is weekly", RememberOpts{Type: Semantic})
	if err != nil {
		t.Fatalf("PolicyWarn Remember: %v", err)
	}
	if !stored {
		t.Errorf("PolicyWarn should store despite conflict")
	}
	if c, _ := store.Count(ctx); c != 2 {
		t.Errorf("after Warn, count = %d, want 2", c)
	}
}

func TestRememberConflictSupersede(t *testing.T) {
	store := newConflictStore(t, PolicySupersede, 0.1)
	ctx := context.Background()

	oldMem, _, _, err := store.Remember(ctx, "staging uses the blue cluster", RememberOpts{Type: Semantic})
	if err != nil {
		t.Fatalf("first Remember: %v", err)
	}
	newMem, stored, _, err := store.Remember(ctx, "staging uses the blue cluster", RememberOpts{Type: Semantic})
	if err != nil {
		t.Fatalf("second Remember: %v", err)
	}
	if !stored {
		t.Errorf("PolicySupersede should store the new memory")
	}
	// The older conflicting memory must now be superseded by the new one.
	got, err := store.GetByID(ctx, oldMem.ID)
	if err != nil {
		t.Fatalf("GetByID old: %v", err)
	}
	if got.SupersededBy != newMem.ID {
		t.Errorf("old memory superseded_by = %q, want %q", got.SupersededBy, newMem.ID)
	}
}

func TestCompressedGroupMarshalAndIDs(t *testing.T) {
	g := CompressedGroup{
		Memories: []Memory{{ID: "id-one"}, {ID: "id-two"}},
		Text:     "summary line",
		Tokens:   7,
	}
	if ids := g.IDs(); len(ids) != 2 || ids[0] != "id-one" || ids[1] != "id-two" {
		t.Fatalf("IDs() = %v, want [id-one id-two]", ids)
	}

	b, err := json.Marshal(g)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	var out map[string]any
	if err := json.Unmarshal(b, &out); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if out["count"].(float64) != 2 {
		t.Errorf("count = %v, want 2", out["count"])
	}
	if out["text"].(string) != "summary line" {
		t.Errorf("text = %v", out["text"])
	}
	if out["tokens"].(float64) != 7 {
		t.Errorf("tokens = %v, want 7", out["tokens"])
	}
	ids, ok := out["ids"].([]any)
	if !ok || len(ids) != 2 || ids[0].(string) != "id-one" {
		t.Errorf("ids = %v", out["ids"])
	}
	// Ensure the full source memories are NOT dumped into the JSON.
	if _, present := out["Memories"]; present {
		t.Errorf("MarshalJSON should not expose raw Memories field")
	}
}

func TestRecallRRFZeroLexicalWeight(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	for _, txt := range []string{"alpha signal one", "beta signal two", "gamma signal three"} {
		store.Remember(ctx, txt, RememberOpts{Type: Semantic})
	}
	w := DefaultWeights()
	w.Lexical = 0 // zero-weight signal must be skipped, not divide-by-something.

	out, err := store.Recall(ctx, "alpha signal one", 3, RecallOpts{Mode: ModeHybrid, Fusion: FusionRRF, Weights: w, Explain: true})
	if err != nil {
		t.Fatalf("Recall rrf: %v", err)
	}
	if len(out) == 0 {
		t.Fatalf("RRF with zeroed lexical weight returned no results")
	}
	// The explain payload must not carry a lexical contribution for the skipped
	// signal.
	if out[0].Explain != nil {
		if sig, ok := out[0].Explain["signals"].(map[string]any); ok {
			if _, present := sig["lexical"]; present {
				t.Errorf("zero-weight lexical signal should be skipped in RRF explain, got %v", sig)
			}
		}
	}
	// Results must still be ranked (non-increasing scores).
	for i := 1; i < len(out); i++ {
		if out[i].Score > out[i-1].Score {
			t.Errorf("RRF results not ranked: out[%d].Score=%v > out[%d].Score=%v", i, out[i].Score, i-1, out[i-1].Score)
		}
	}
}

func TestExpandResultsCapEnforced(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	a, _, _, _ := mustRemember(t, store, "chain root", RememberOpts{Type: Semantic})
	b, _, _, _ := mustRemember(t, store, "chain hop b", RememberOpts{Type: Semantic})
	c, _, _, _ := mustRemember(t, store, "chain hop c", RememberOpts{Type: Semantic})
	d, _, _, _ := mustRemember(t, store, "chain hop d", RememberOpts{Type: Semantic})
	// a -> b -> c -> d, all "refines".
	for _, pair := range [][2]string{{a, b}, {b, c}, {c, d}} {
		if err := store.Link(ctx, pair[0], pair[1], "refines"); err != nil {
			t.Fatal(err)
		}
	}

	spec := &ExpandSpec{Rels: []string{"refines"}, Depth: 3, Cap: 1, Score: 0.5, Decay: 0.5}
	out, err := store.Recall(ctx, "chain root", 1, RecallOpts{Mode: ModeSemantic, MinCosine: f32p(0.99), Expand: spec})
	if err != nil {
		t.Fatalf("Recall: %v", err)
	}
	// Seed a (1 hit) + exactly Cap=1 expanded node = 2 total.
	if len(out) != 2 {
		t.Fatalf("Cap=1 should add exactly one expanded node (total 2), got %d: %+v", len(out), out)
	}
	// The single expanded node must be the first hop (b).
	found := map[string]bool{}
	for _, r := range out {
		found[r.ID] = true
	}
	if !found[a] {
		t.Errorf("seed a missing from results")
	}
	if !found[b] {
		t.Errorf("hop-1 node b should be the single expanded node, got %v", found)
	}
	if found[c] || found[d] {
		t.Errorf("cap not enforced: deeper nodes c/d leaked in %v", found)
	}
}

func TestJournalRotationCreatesArchive(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "journal.log")
	// RotateMB=1 is the smallest positive rotate size (>=1 MiB). Each ~4KB entry,
	// written well past the 256-write rotateCheckEvery threshold with >1 MiB
	// accumulated, drives Append's own maybeRotate/rotate path.
	j := NewJournal(path, 1, 90)

	big := strings.Repeat("x", 4096) // ~4163 bytes/entry marshaled
	for i := 0; i < 400; i++ {       // > rotateCheckEvery (256) and > 1 MiB total
		j.Append(JournalEntry{TS: nowFloat(), Op: "remember", Actor: "test", ID: "z", Meta: map[string]any{"blob": big}})
	}

	// A direct maybeRotate is a safe no-op if the size is already below the
	// threshold (Append may have rotated at write 256) and exercises the size
	// gate explicitly regardless.
	if err := j.maybeRotate(); err != nil {
		t.Fatalf("maybeRotate: %v", err)
	}

	// Rotation (whether via Append's periodic check or the direct call) must have
	// produced at least one gzipped archive.
	matches, _ := filepath.Glob(filepath.Join(dir, "journal-*.log.gz"))
	if len(matches) == 0 {
		t.Fatalf("expected a rotated archive .log.gz, found none in %s", dir)
	}

	// The archive must be readable back as valid gzipped JSONL entries.
	entries, err := j.Read(ReadOpts{IncludeArchives: true, Op: "remember"})
	if err != nil {
		t.Fatalf("Read archives: %v", err)
	}
	if len(entries) == 0 {
		t.Errorf("rotated archive should still contain readable entries")
	}

	// pruneArchives with KeepDays far in the future keeps the fresh archive.
	j.KeepDays = 3650
	j.pruneArchives()
	matches2, _ := filepath.Glob(filepath.Join(dir, "journal-*.log.gz"))
	if len(matches2) != len(matches) {
		t.Errorf("pruneArchives removed a fresh archive; before=%d after=%d", len(matches), len(matches2))
	}
}

func TestLinkRejectsSelfLink(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	a, _, _, _ := store.Remember(ctx, "self link candidate", RememberOpts{Type: Semantic})
	// Matches Python's _link_raw: linking a memory to itself is an error.
	if err := store.Link(ctx, a.ID, a.ID, "related"); err == nil {
		t.Error("Link(a, a) should be rejected as a self-link")
	}
	// No dangling self-edge should have been written.
	got, _ := store.GetByID(ctx, a.ID)
	if len(got.Links) != 0 {
		t.Errorf("self-link must not be stored, got %d links", len(got.Links))
	}
}
