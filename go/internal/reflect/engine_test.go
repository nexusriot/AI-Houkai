package reflect

import (
	"context"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/nexusriot/ai-houkai/internal/memory"
	"github.com/nexusriot/ai-houkai/internal/vector"
)

// setPolarity rewrites the polarity in an episode item's stored metadata.
func setPolarity(it *vector.Item, p int) {
	it.Metadata["polarity"] = strconv.Itoa(p)
}

// fakeReflectStore implements Storable for unit tests.
type fakeReflectStore struct {
	items      []vector.Item
	added      []memory.Memory
	links      []struct{ src, dst, rel string }
	forgotten  []string
	superseded []struct{ old, new string }
}

func (f *fakeReflectStore) AllRaw(_ context.Context) ([]vector.Item, error) {
	return f.items, nil
}
func (f *fakeReflectStore) Remember(_ context.Context, text string, opts memory.RememberOpts) (memory.Memory, bool, []memory.Conflict, error) {
	// Carry every field the real store applies. A double that silently drops
	// one hides exactly the bugs these tests exist to catch — Trust was dropped
	// here, so trust laundering through reflection was invisible.
	m := memory.Memory{
		ID:     "new-" + text[:min(8, len(text))],
		Text:   text,
		Type:   opts.Type,
		Tags:   opts.Tags,
		Source: opts.Source,
		Pinned: opts.Pinned,
		Trust:  opts.Trust,
	}
	if opts.Importance != nil {
		m.Importance = *opts.Importance
	}
	f.added = append(f.added, m)
	return m, true, nil, nil
}
func (f *fakeReflectStore) Forget(_ context.Context, id string) (bool, error) {
	f.forgotten = append(f.forgotten, id)
	return true, nil
}
func (f *fakeReflectStore) Link(_ context.Context, src, dst, rel string) error {
	f.links = append(f.links, struct{ src, dst, rel string }{src, dst, rel})
	return nil
}
func (f *fakeReflectStore) Supersede(_ context.Context, oldID, newID string) error {
	f.superseded = append(f.superseded, struct{ old, new string }{oldID, newID})
	return nil
}

func episode(id string, vec []float32, imp float32) vector.Item {
	m := memory.Memory{
		ID:         id,
		Type:       memory.Episodic,
		Importance: imp,
		Tags:       []string{"t-" + id},
	}
	return vector.Item{
		ID:        id,
		Content:   "text-" + id,
		Embedding: vec,
		Metadata:  memory.MemoryToMetadata(m),
	}
}

func TestClustersGroupsSimilar(t *testing.T) {
	store := &fakeReflectStore{items: []vector.Item{
		episode("a", []float32{1, 0, 0}, 0.9),
		episode("b", []float32{0.99, 0.01, 0}, 0.5), // very close to a
		episode("c", []float32{0, 1, 0}, 0.5),       // far
	}}
	e := New(store, 0.9, 2, nil)
	clusters, err := e.Clusters(context.Background())
	if err != nil {
		t.Fatalf("Clusters: %v", err)
	}
	if len(clusters) != 1 {
		t.Fatalf("want 1 cluster, got %d", len(clusters))
	}
	if len(clusters[0]) != 2 {
		t.Errorf("cluster size: got %d, want 2", len(clusters[0]))
	}
}

func TestClustersSkipsOppositePolarity(t *testing.T) {
	// Two near-identical episodics with opposite polarity must NOT cluster;
	// a neutral third one may join the seed.
	pos := episode("a", []float32{1, 0, 0}, 0.9)
	neg := episode("b", []float32{0.99, 0.01, 0}, 0.8)
	neu := episode("c", []float32{0.98, 0.02, 0}, 0.7)
	setPolarity(&pos, 1)
	setPolarity(&neg, -1)
	setPolarity(&neu, 0)
	store := &fakeReflectStore{items: []vector.Item{pos, neg, neu}}
	e := New(store, 0.9, 2, nil)

	clusters, err := e.Clusters(context.Background())
	if err != nil {
		t.Fatalf("Clusters: %v", err)
	}
	if len(clusters) != 1 {
		t.Fatalf("want 1 cluster (seed+neutral), got %d", len(clusters))
	}
	if len(clusters[0]) != 2 {
		t.Errorf("opposite-polarity member must be excluded; cluster size=%d want 2", len(clusters[0]))
	}
	for _, m := range clusters[0] {
		if m.Polarity == -1 {
			t.Error("negative-polarity memory should not join a positive seed's cluster")
		}
	}
}

func TestReflectNoneLeavesSourcesUntouched(t *testing.T) {
	store := &fakeReflectStore{items: []vector.Item{
		episode("a", []float32{1, 0, 0}, 0.9),
		episode("b", []float32{0.99, 0.01, 0}, 0.5),
	}}
	e := New(store, 0.9, 2, nil)

	created, err := e.Reflect(context.Background(), false, ConsolidateNone)
	if err != nil {
		t.Fatalf("Reflect: %v", err)
	}
	if len(created) != 1 {
		t.Fatalf("want 1 semantic memory created, got %d", len(created))
	}
	if created[0].Type != memory.Semantic {
		t.Errorf("new memory type: got %q, want semantic", created[0].Type)
	}
	// none: sources are left entirely alone — no link, no supersede, no delete.
	if len(store.links) != 0 {
		t.Errorf("none mode should not create links, got %d", len(store.links))
	}
	if len(store.superseded) != 0 {
		t.Errorf("none mode should not supersede, got %d", len(store.superseded))
	}
	if len(store.forgotten) != 0 {
		t.Error("none mode should not delete sources")
	}
}

func TestReflectSoftSupersedesAndLinks(t *testing.T) {
	store := &fakeReflectStore{items: []vector.Item{
		episode("a", []float32{1, 0, 0}, 0.9),
		episode("b", []float32{0.99, 0.01, 0}, 0.5),
	}}
	e := New(store, 0.9, 2, nil)
	if _, err := e.Reflect(context.Background(), false, ConsolidateSoft); err != nil {
		t.Fatalf("Reflect: %v", err)
	}
	if len(store.superseded) != 2 {
		t.Errorf("soft should supersede 2 sources, got %d", len(store.superseded))
	}
	if len(store.links) != 2 {
		t.Errorf("soft should add 2 derived_from links, got %d", len(store.links))
	}
	for _, l := range store.links {
		if l.rel != memory.RelDerivedFrom {
			t.Errorf("link rel: got %q, want %q", l.rel, memory.RelDerivedFrom)
		}
	}
	if len(store.forgotten) != 0 {
		t.Error("soft must not forget sources")
	}
}

func TestReflectHardForgetsSources(t *testing.T) {
	store := &fakeReflectStore{items: []vector.Item{
		episode("a", []float32{1, 0, 0}, 0.9),
		episode("b", []float32{0.99, 0.01, 0}, 0.5),
	}}
	e := New(store, 0.9, 2, nil)
	if _, err := e.Reflect(context.Background(), false, ConsolidateHard); err != nil {
		t.Fatalf("Reflect: %v", err)
	}
	if len(store.forgotten) != 2 {
		t.Errorf("hard should forget 2 sources, got %d", len(store.forgotten))
	}
	if len(store.links) != 0 || len(store.superseded) != 0 {
		t.Errorf("hard should not link/supersede; links=%d superseded=%d", len(store.links), len(store.superseded))
	}
}

func TestReflectDryRunWritesNothing(t *testing.T) {
	store := &fakeReflectStore{items: []vector.Item{
		episode("a", []float32{1, 0, 0}, 0.9),
		episode("b", []float32{0.99, 0.01, 0}, 0.5),
	}}
	e := New(store, 0.9, 2, nil)
	created, err := e.Reflect(context.Background(), true, ConsolidateSoft)
	if err != nil {
		t.Fatalf("Reflect: %v", err)
	}
	if len(created) != 1 {
		t.Errorf("dry-run should still report 1 candidate, got %d", len(created))
	}
	if len(store.added) != 0 || len(store.links) != 0 || len(store.forgotten) != 0 {
		t.Errorf("dry-run should write nothing; added=%d links=%d forgotten=%d",
			len(store.added), len(store.links), len(store.forgotten))
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Tiered reflection (F1): the engine only ever clustered episodic memories, so
// summaries were never themselves consolidated and a long-lived store
// accumulated them without bound.

// typedItem builds a vector.Item of an arbitrary type and tag set, so the
// tier tests can seed existing summaries directly.
func typedItem(id string, t memory.MemoryType, vec []float32, tags ...string) vector.Item {
	m := memory.Memory{ID: id, Type: t, Importance: 0.5, Tags: tags}
	return vector.Item{
		ID: id, Content: "text-" + id, Embedding: vec,
		Metadata: memory.MemoryToMetadata(m),
	}
}

func TestReflectDefaultsToEpisodicOnly(t *testing.T) {
	store := &fakeReflectStore{items: []vector.Item{
		typedItem("e1", memory.Episodic, []float32{1, 0, 0}),
		typedItem("e2", memory.Episodic, []float32{1, 0, 0}),
		typedItem("s1", memory.Semantic, []float32{1, 0, 0}),
		typedItem("s2", memory.Semantic, []float32{1, 0, 0}),
	}}
	clusters, err := New(store, 0.9, 2, nil).Clusters(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(clusters) != 1 {
		t.Fatalf("clusters = %d, want 1", len(clusters))
	}
	for _, m := range clusters[0] {
		if m.Type != memory.Episodic {
			t.Errorf("clustered a %s memory by default", m.Type)
		}
	}
}

func TestReflectCanClusterOtherTypes(t *testing.T) {
	store := &fakeReflectStore{items: []vector.Item{
		typedItem("f1", memory.Feedback, []float32{1, 0, 0}),
		typedItem("f2", memory.Feedback, []float32{1, 0, 0}),
	}}
	e := New(store, 0.9, 2, nil)
	e.Types = []memory.MemoryType{memory.Feedback}
	clusters, err := e.Clusters(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(clusters) != 1 {
		t.Fatalf("feedback memories were not clustered: %d", len(clusters))
	}
}

func TestReflectTagsTheTier(t *testing.T) {
	store := &fakeReflectStore{items: []vector.Item{
		typedItem("e1", memory.Episodic, []float32{1, 0, 0}),
		typedItem("e2", memory.Episodic, []float32{1, 0, 0}),
	}}
	made, err := New(store, 0.9, 2, nil).Reflect(context.Background(), false, ConsolidateNone)
	if err != nil || len(made) == 0 {
		t.Fatalf("reflect = %d (%v)", len(made), err)
	}
	if LevelOf(made[0]) != 1 {
		t.Errorf("tags = %v, want level:1", made[0].Tags)
	}
}

func TestReflectMaxLevelCapsTheHierarchy(t *testing.T) {
	// The guard against runaway re-summarisation eating the store.
	seed := func() *fakeReflectStore {
		return &fakeReflectStore{items: []vector.Item{
			typedItem("s1", memory.Semantic, []float32{1, 0, 0}, "reflection", "level:1"),
			typedItem("s2", memory.Semantic, []float32{1, 0, 0}, "reflection", "level:1"),
		}}
	}
	capped := New(seed(), 0.9, 2, nil)
	capped.Types = []memory.MemoryType{memory.Semantic}
	capped.MaxLevel = 1
	if made, _ := capped.Reflect(context.Background(), false, ConsolidateNone); len(made) != 0 {
		t.Errorf("max_level=1 still produced %d summaries", len(made))
	}

	deeper := New(seed(), 0.9, 2, nil)
	deeper.Types = []memory.MemoryType{memory.Semantic}
	deeper.MaxLevel = 2
	made, err := deeper.Reflect(context.Background(), false, ConsolidateNone)
	if err != nil || len(made) == 0 {
		t.Fatalf("max_level=2 reflect = %d (%v)", len(made), err)
	}
	if LevelOf(made[0]) != 2 {
		t.Errorf("tags = %v, want level:2", made[0].Tags)
	}
	// The summary carries its own tier, not its members'.
	levels := 0
	for _, tag := range made[0].Tags {
		if strings.HasPrefix(tag, "level:") {
			levels++
		}
	}
	if levels != 1 {
		t.Errorf("tags = %v, want exactly one level tag", made[0].Tags)
	}
}

// episodeTrust is `episode` with an explicit provenance level.
func episodeTrust(id string, vec []float32, imp float32,
	trust memory.TrustLevel) vector.Item {
	m := memory.Memory{
		ID:         id,
		Type:       memory.Episodic,
		Importance: imp,
		Tags:       []string{"t-" + id},
		Trust:      trust,
	}
	return vector.Item{
		ID:        id,
		Content:   "text-" + id,
		Embedding: vec,
		Metadata:  memory.MemoryToMetadata(m),
	}
}

// A summary inherits the least-trusted source. Otherwise reflecting over
// content the agent did not author launders it into a "trusted" memory that
// MinTrust="trusted" will happily return — the hole the trust tier closes.

func TestReflectSummaryInheritsWorstSourceTrust(t *testing.T) {
	store := &fakeReflectStore{items: []vector.Item{
		episodeTrust("a", []float32{1, 0, 0}, 0.9, "untrusted"),
		episodeTrust("b", []float32{0.99, 0.01, 0}, 0.5, "untrusted"),
	}}
	created, err := New(store, 0.9, 2, nil).Reflect(
		context.Background(), false, ConsolidateNone)
	if err != nil {
		t.Fatal(err)
	}
	if len(created) != 1 {
		t.Fatalf("want 1 summary, got %d", len(created))
	}
	if created[0].Trust != "untrusted" {
		t.Errorf("summary trust = %q, want untrusted — a summary of untrusted "+
			"sources must not be born trusted", created[0].Trust)
	}
}

func TestReflectSummaryTakesTheWorstOfMixedSources(t *testing.T) {
	store := &fakeReflectStore{items: []vector.Item{
		episodeTrust("a", []float32{1, 0, 0}, 0.9, "trusted"),
		episodeTrust("b", []float32{0.99, 0.01, 0}, 0.5, "reported"),
	}}
	created, err := New(store, 0.9, 2, nil).Reflect(
		context.Background(), false, ConsolidateNone)
	if err != nil {
		t.Fatal(err)
	}
	if len(created) != 1 || created[0].Trust != "reported" {
		t.Errorf("summary trust = %q, want reported", created[0].Trust)
	}
}

func TestReflectTrustedSourcesStayTrusted(t *testing.T) {
	store := &fakeReflectStore{items: []vector.Item{
		episodeTrust("a", []float32{1, 0, 0}, 0.9, "trusted"),
		episodeTrust("b", []float32{0.99, 0.01, 0}, 0.5, "trusted"),
	}}
	created, err := New(store, 0.9, 2, nil).Reflect(
		context.Background(), false, ConsolidateNone)
	if err != nil {
		t.Fatal(err)
	}
	if len(created) != 1 || created[0].Trust != "trusted" {
		t.Errorf("summary trust = %q, want trusted", created[0].Trust)
	}
}

func TestReflectDryRunAlsoCarriesTrust(t *testing.T) {
	store := &fakeReflectStore{items: []vector.Item{
		episodeTrust("a", []float32{1, 0, 0}, 0.9, "untrusted"),
		episodeTrust("b", []float32{0.99, 0.01, 0}, 0.5, "trusted"),
	}}
	created, err := New(store, 0.9, 2, nil).Reflect(
		context.Background(), true, ConsolidateNone)
	if err != nil {
		t.Fatal(err)
	}
	if len(created) != 1 || created[0].Trust != "untrusted" {
		t.Errorf("dry-run summary trust = %q, want untrusted — a preview must "+
			"show what would actually be written", created[0].Trust)
	}
}

// episodePinned is `episode` with the standing-instruction flag set.
func episodePinned(id string, vec []float32, imp float32, pinned bool) vector.Item {
	m := memory.Memory{
		ID:         id,
		Type:       memory.Episodic,
		Importance: imp,
		Tags:       []string{"t-" + id},
		Pinned:     pinned,
	}
	return vector.Item{
		ID:        id,
		Content:   "text-" + id,
		Embedding: vec,
		Metadata:  memory.MemoryToMetadata(m),
	}
}

// A consolidated summary takes over its sources' standing-instruction slot:
// soft consolidate supersedes them (and a superseded row leaves the working
// set), hard consolidate deletes them outright — either way the pin was lost.

func TestReflectSoftConsolidateCarriesThePin(t *testing.T) {
	store := &fakeReflectStore{items: []vector.Item{
		episodePinned("a", []float32{1, 0, 0}, 0.9, true),
		episodePinned("b", []float32{0.99, 0.01, 0}, 0.5, false),
	}}
	created, err := New(store, 0.9, 2, nil).Reflect(
		context.Background(), false, ConsolidateSoft)
	if err != nil {
		t.Fatal(err)
	}
	if len(created) != 1 || !created[0].Pinned {
		t.Fatalf("summary = %+v, want it pinned", created)
	}
}

func TestReflectHardConsolidateCarriesThePin(t *testing.T) {
	store := &fakeReflectStore{items: []vector.Item{
		episodePinned("a", []float32{1, 0, 0}, 0.9, true),
		episodePinned("b", []float32{0.99, 0.01, 0}, 0.5, false),
	}}
	created, err := New(store, 0.9, 2, nil).Reflect(
		context.Background(), false, ConsolidateHard)
	if err != nil {
		t.Fatal(err)
	}
	if len(created) != 1 || !created[0].Pinned {
		t.Fatalf("summary = %+v, want it pinned — the sources are gone", created)
	}
}

// Without consolidation the sources stay live and pinned, so pinning the
// summary too would put two rows in the working set for one instruction.
func TestReflectWithoutConsolidationDoesNotPin(t *testing.T) {
	store := &fakeReflectStore{items: []vector.Item{
		episodePinned("a", []float32{1, 0, 0}, 0.9, true),
		episodePinned("b", []float32{0.99, 0.01, 0}, 0.5, false),
	}}
	created, err := New(store, 0.9, 2, nil).Reflect(
		context.Background(), false, ConsolidateNone)
	if err != nil {
		t.Fatal(err)
	}
	if len(created) != 1 || created[0].Pinned {
		t.Fatalf("summary = %+v, want it unpinned", created)
	}
}

func TestReflectUnpinnedSourcesStayUnpinned(t *testing.T) {
	store := &fakeReflectStore{items: []vector.Item{
		episodePinned("a", []float32{1, 0, 0}, 0.9, false),
		episodePinned("b", []float32{0.99, 0.01, 0}, 0.5, false),
	}}
	created, err := New(store, 0.9, 2, nil).Reflect(
		context.Background(), false, ConsolidateSoft)
	if err != nil {
		t.Fatal(err)
	}
	if len(created) != 1 || created[0].Pinned {
		t.Fatalf("summary = %+v, want it unpinned", created)
	}
}

func TestReflectDryRunShowsThePinItWouldSet(t *testing.T) {
	store := &fakeReflectStore{items: []vector.Item{
		episodePinned("a", []float32{1, 0, 0}, 0.9, true),
		episodePinned("b", []float32{0.99, 0.01, 0}, 0.5, false),
	}}
	created, err := New(store, 0.9, 2, nil).Reflect(
		context.Background(), true, ConsolidateSoft)
	if err != nil {
		t.Fatal(err)
	}
	if len(created) != 1 || !created[0].Pinned {
		t.Fatalf("dry-run summary = %+v, want it pinned — a preview must show "+
			"what would actually be written", created)
	}
}

// episodeExpiring builds an episodic item whose TTL deadline is expiresAt.
func episodeExpiring(id string, vec []float32, imp float32, expiresAt float64) vector.Item {
	m := memory.Memory{
		ID:         id,
		Type:       memory.Episodic,
		Importance: imp,
		Tags:       []string{"t-" + id},
		ExpiresAt:  expiresAt,
	}
	return vector.Item{
		ID:        id,
		Content:   "text-" + id,
		Embedding: vec,
		Metadata:  memory.MemoryToMetadata(m),
	}
}

// A TTL means the content stops being available: recall, list and stats hide it
// and PurgeExpired eventually reclaims it. Reflect reads the collection
// directly, so without an expiry filter it clusters lapsed rows and writes
// their text into a fresh, NON-expiring summary — resurrecting permanently what
// the caller gave a lifetime. Same laundering shape as the trust rule.
func TestReflectSkipsExpiredSources(t *testing.T) {
	past := float64(time.Now().Add(-time.Hour).Unix())
	store := &fakeReflectStore{items: []vector.Item{
		episode("live", []float32{1, 0, 0}, 0.9),
		episodeExpiring("lapsed", []float32{0.99, 0.01, 0}, 0.8, past),
	}}
	e := New(store, 0.5, 2, nil)

	clusters, err := e.Clusters(context.Background())
	if err != nil {
		t.Fatalf("Clusters: %v", err)
	}
	// Only one live candidate remains, below minClusterSize, so nothing
	// summarises at all.
	if len(clusters) != 0 {
		t.Errorf("Clusters() = %d clusters, want 0 once the lapsed row is excluded", len(clusters))
	}

	created, err := e.Reflect(context.Background(), false, "")
	if err != nil {
		t.Fatalf("Reflect: %v", err)
	}
	for _, m := range created {
		if strings.Contains(m.Text, "text-lapsed") {
			t.Errorf("summary %q carries an expired source's text", m.Text)
		}
	}
}

// Only *lapsed* rows are excluded — a deadline in the future is still live.
func TestReflectKeepsUnexpiredTTLSources(t *testing.T) {
	future := float64(time.Now().Add(time.Hour).Unix())
	store := &fakeReflectStore{items: []vector.Item{
		episode("a", []float32{1, 0, 0}, 0.9),
		episodeExpiring("b", []float32{0.99, 0.01, 0}, 0.8, future),
	}}
	e := New(store, 0.5, 2, nil)

	clusters, err := e.Clusters(context.Background())
	if err != nil {
		t.Fatalf("Clusters: %v", err)
	}
	if len(clusters) != 1 {
		t.Errorf("Clusters() = %d, want 1 (a future TTL is still eligible)", len(clusters))
	}
}
