package cli

import (
	"math"
	"strings"
	"testing"
	"time"

	"github.com/nexusriot/ai-houkai/internal/memory"
)

func TestHealthDecayScore(t *testing.T) {
	now := float64(time.Now().Unix())

	// Zero elapsed days: score == importance when no frequency term.
	if got := healthDecayScore(1.0, now, 0.1, 0, 0); math.Abs(float64(got)-1.0) > 1e-4 {
		t.Errorf("fresh score = %v, want ~1.0", got)
	}

	// Ten days at rate 0.1 -> importance * e^-1.
	tenDaysAgo := now - 10*86400
	want := math.Exp(-1.0)
	if got := healthDecayScore(1.0, tenDaysAgo, 0.1, 0, 0); math.Abs(float64(got)-want) > 1e-3 {
		t.Errorf("decayed score = %v, want ~%v", got, want)
	}

	// Future timestamp clamps elapsed days to 0 (no boost above importance).
	future := now + 5*86400
	if got := healthDecayScore(0.8, future, 0.1, 0, 0); math.Abs(float64(got)-0.8) > 1e-4 {
		t.Errorf("future score = %v, want ~0.8 (days clamped to 0)", got)
	}

	// Frequency reinforcement raises the score: base * (1 + w*ln(1+count)).
	base := healthDecayScore(0.5, tenDaysAgo, 0.1, 0, 0)
	boosted := healthDecayScore(0.5, tenDaysAgo, 0.1, 4, 0.3)
	if boosted <= base {
		t.Errorf("frequency weight should raise score: base=%v boosted=%v", base, boosted)
	}
	expectMul := 1.0 + 0.3*math.Log1p(4)
	if math.Abs(float64(boosted)-float64(base)*expectMul) > 1e-3 {
		t.Errorf("boosted = %v, want base*%v = %v", boosted, expectMul, float64(base)*expectMul)
	}

	// Negative access_count is clamped to 0, so no boost is applied.
	if got := healthDecayScore(0.5, tenDaysAgo, 0.1, -3, 0.3); math.Abs(float64(got)-float64(base)) > 1e-4 {
		t.Errorf("negative access_count should clamp: got %v want %v", got, base)
	}
}

func TestDecayBucket(t *testing.T) {
	cases := map[float32]int{
		0.0:  0,
		0.1:  0,
		0.2:  1,
		0.39: 1,
		0.4:  2,
		0.6:  3,
		0.8:  4,
		1.0:  4,
		1.5:  4, // clamps to 4
		-0.5: 0, // clamps to 0
	}
	for score, want := range cases {
		if got := decayBucket(score); got != want {
			t.Errorf("decayBucket(%v) = %d, want %d", score, got, want)
		}
	}
}

func TestComputeHealth(t *testing.T) {
	now := float64(time.Now().Unix())
	staleTS := now - 100*86400 // older than a 30-day stale window

	active := []memory.Memory{
		// High importance, frequently recalled, fresh -> high decay bucket, top-recalled.
		{ID: "aaaaaaaa1111", Text: "well maintained procedural note", Type: memory.Procedural,
			Importance: 0.95, LastAccessed: now, AccessCount: 20, Links: []memory.Link{{To: "x", Rel: "related"}}},
		// Low importance, never recalled, stale, episodic -> at risk + never + stale + episodic.
		{ID: "bbbbbbbb2222", Text: "forgotten episodic note", Type: memory.Episodic,
			Importance: 0.02, LastAccessed: staleTS, AccessCount: 0},
		// Mid importance, recalled a few times, fresh, has links.
		{ID: "cccccccc3333", Text: "semantic fact recalled sometimes", Type: memory.Semantic,
			Importance: 0.6, LastAccessed: now, AccessCount: 5,
			Links: []memory.Link{{To: "y", Rel: "related"}, {To: "z", Rel: "related"}}},
	}

	h := computeHealth(active, 30, 0.1, 0.05, []string{"procedural"}, 0)

	// at_risk: only the low-importance episodic scores below 0.05; the
	// procedural is protected even if it dipped.
	if got := h["at_risk_count"].(int); got != 1 {
		t.Errorf("at_risk_count = %d, want 1", got)
	}
	if got := h["never_recalled_count"].(int); got != 1 {
		t.Errorf("never_recalled_count = %d, want 1", got)
	}
	if got := h["stale_count"].(int); got != 1 {
		t.Errorf("stale_count = %d, want 1", got)
	}
	if got := h["episodic_active_count"].(int); got != 1 {
		t.Errorf("episodic_active_count = %d, want 1", got)
	}
	if got := h["total_links"].(int); got != 3 {
		t.Errorf("total_links = %d, want 3", got)
	}

	// link_density = 3 links / 3 memories = 1.0.
	if got := h["link_density"].(float64); math.Abs(got-1.0) > 1e-6 {
		t.Errorf("link_density = %v, want 1.0", got)
	}
	// avg_importance = (0.95 + 0.02 + 0.6)/3 rounded to 3 dp.
	wantAvg := math.Round((0.95+0.02+0.6)/3*1000) / 1000
	if got := h["avg_importance"].(float64); math.Abs(got-wantAvg) > 1e-6 {
		t.Errorf("avg_importance = %v, want %v", got, wantAvg)
	}

	// decay_histogram is a label->count map summing to len(active).
	hist := h["decay_histogram"].(map[string]int)
	sum := 0
	for _, c := range hist {
		sum += c
	}
	if sum != len(active) {
		t.Errorf("histogram sums to %d, want %d", sum, len(active))
	}
	if _, ok := hist["0.0–0.2"]; !ok {
		t.Errorf("histogram missing expected band labels: %v", hist)
	}

	// top_recalled excludes never-recalled memories and is ordered by count desc.
	top := h["top_recalled"].([]map[string]any)
	if len(top) != 2 {
		t.Fatalf("top_recalled len = %d, want 2 (never-recalled excluded)", len(top))
	}
	if top[0]["access_count"].(int) < top[1]["access_count"].(int) {
		t.Errorf("top_recalled not sorted desc: %v", top)
	}
	if top[0]["id"] != "aaaaaaaa" { // truncated to 8 chars
		t.Errorf("top_recalled[0] id = %v, want 8-char prefix aaaaaaaa", top[0]["id"])
	}
}

func TestComputeHealthEmpty(t *testing.T) {
	h := computeHealth(nil, 30, 0.1, 0.05, nil, 0)
	if got := h["link_density"].(float64); got != 0 {
		t.Errorf("empty link_density = %v, want 0", got)
	}
	if got := h["avg_importance"].(float64); got != 0 {
		t.Errorf("empty avg_importance = %v, want 0", got)
	}
	if got := h["at_risk_count"].(int); got != 0 {
		t.Errorf("empty at_risk_count = %d, want 0", got)
	}
	if len(h["top_recalled"].([]map[string]any)) != 0 {
		t.Errorf("empty top_recalled should be empty")
	}
}

func TestGraphToASCII(t *testing.T) {
	g := memory.Graph{
		Nodes: []memory.Memory{
			{ID: "0123456789abcdef", Text: "first node text", Type: memory.Semantic},
			{ID: "fedcba9876543210", Text: "second node text", Type: memory.Procedural},
		},
	}
	g.Edges = append(g.Edges, struct {
		From string
		To   string
		Rel  string
	}{From: "0123456789abcdef", To: "fedcba9876543210", Rel: "related"})

	out := graphToASCII(g)
	for _, want := range []string{"01234567", "fedcba98", "semantic", "procedural", "first node text",
		"01234567 --related--> fedcba98"} {
		if !strings.Contains(out, want) {
			t.Errorf("ascii output missing %q in:\n%s", want, out)
		}
	}
}

func TestGraphToASCIIEmpty(t *testing.T) {
	if got := graphToASCII(memory.Graph{}); !strings.Contains(got, "(empty)") {
		t.Errorf("empty graph ascii = %q, want it to contain (empty)", got)
	}
}

func TestGraphToDOT(t *testing.T) {
	g := memory.Graph{
		Nodes: []memory.Memory{
			{ID: "0123456789abcdef", Text: `text with "quotes" inside`, Type: memory.Semantic},
		},
	}
	g.Edges = append(g.Edges, struct {
		From string
		To   string
		Rel  string
	}{From: "0123456789abcdef", To: "0123456789abcdef", Rel: "self"})

	out := graphToDOT(g)
	if !strings.HasPrefix(out, "digraph memory {") {
		t.Errorf("dot output should start with digraph header:\n%s", out)
	}
	if !strings.Contains(out, "01234567") {
		t.Errorf("dot output missing node id prefix:\n%s", out)
	}
	if !strings.Contains(out, `"01234567" -> "01234567"`) {
		t.Errorf("dot output missing edge:\n%s", out)
	}
	// Double-quotes in the label must be escaped to single quotes.
	if strings.Contains(out, `"quotes"`) {
		t.Errorf("dot output leaked unescaped quotes:\n%s", out)
	}
	if !strings.HasSuffix(strings.TrimSpace(out), "}") {
		t.Errorf("dot output should end with closing brace:\n%s", out)
	}
}

func TestSortByImportance(t *testing.T) {
	mems := []memory.Memory{
		{ID: "low", Importance: 0.2, CreatedAt: 100},
		{ID: "high", Importance: 0.9, CreatedAt: 100},
		{ID: "mid-old", Importance: 0.5, CreatedAt: 50},
		{ID: "mid-new", Importance: 0.5, CreatedAt: 200},
	}
	sortByImportance(mems)

	order := []string{mems[0].ID, mems[1].ID, mems[2].ID, mems[3].ID}
	// Descending by importance; equal importance breaks ties by created_at desc.
	want := []string{"high", "mid-new", "mid-old", "low"}
	for i := range want {
		if order[i] != want[i] {
			t.Fatalf("sort order = %v, want %v", order, want)
		}
	}
}

func TestFmtAge(t *testing.T) {
	now := time.Now()
	cases := []struct {
		ts   float64
		want string
	}{
		{float64(now.Unix()), "just now"},
		{float64(now.Add(-30 * time.Minute).Unix()), "m ago"},
		{float64(now.Add(-5 * time.Hour).Unix()), "h ago"},
		{float64(now.Add(-3 * 24 * time.Hour).Unix()), "d ago"},
	}
	for _, c := range cases {
		if got := fmtAge(c.ts); !strings.Contains(got, c.want) {
			t.Errorf("fmtAge(%v) = %q, want to contain %q", c.ts, got, c.want)
		}
	}
	// Far in the past falls back to an absolute date (YYYY-MM-DD).
	old := float64(now.Add(-100 * 24 * time.Hour).Unix())
	if got := fmtAge(old); !strings.Contains(got, "-") || len(got) != 10 {
		t.Errorf("fmtAge(old) = %q, want an absolute YYYY-MM-DD date", got)
	}
}

func TestFmtID(t *testing.T) {
	if got := fmtID("0123456789abcdef"); got != "01234567" {
		t.Errorf("fmtID long = %q, want 01234567", got)
	}
	if got := fmtID("abc"); got != "abc" {
		t.Errorf("fmtID short = %q, want abc (unchanged)", got)
	}
	if got := fmtID("12345678"); got != "12345678" {
		t.Errorf("fmtID exactly-8 = %q, want 12345678", got)
	}
}

func TestTruncate(t *testing.T) {
	if got := truncate("short", 10); got != "short" {
		t.Errorf("truncate within limit = %q, want short", got)
	}
	got := truncate("abcdefghij", 5)
	// truncate keeps n-1 runes then appends the ellipsis.
	if got != "abcd…" {
		t.Errorf("truncate over limit = %q, want abcd…", got)
	}
}

func TestStripAnsi(t *testing.T) {
	// A red-colored word wrapped in SGR escapes.
	colored := "\x1b[31mred\x1b[0m plain"
	if got := stripAnsi(colored); got != "red plain" {
		t.Errorf("stripAnsi = %q, want %q", got, "red plain")
	}
	// No escapes -> unchanged.
	if got := stripAnsi("nothing here"); got != "nothing here" {
		t.Errorf("stripAnsi plain = %q", got)
	}
}

func TestMaintPathsDefaults(t *testing.T) {
	cfg := Config{StorePath: "/data/store/.chroma"}
	state, pid, logp := cfg.MaintPaths()
	if state != "/data/store/maintenance_state.json" {
		t.Errorf("state = %q", state)
	}
	if pid != "/data/store/maintenance.pid" {
		t.Errorf("pid = %q", pid)
	}
	if logp != "/data/store/maintenance.log" {
		t.Errorf("log = %q", logp)
	}
}

func TestMaintPathsOverrides(t *testing.T) {
	cfg := Config{StorePath: "/data/store/.chroma"}
	cfg.Maintenance.StatePath = "/custom/state.json"
	cfg.Maintenance.PidPath = "/custom/run.pid"
	cfg.Maintenance.LogPath = "/custom/out.log"
	state, pid, logp := cfg.MaintPaths()
	if state != "/custom/state.json" || pid != "/custom/run.pid" || logp != "/custom/out.log" {
		t.Errorf("overrides not honored: %q %q %q", state, pid, logp)
	}
}

func TestFmtTS(t *testing.T) {
	if got := fmtTS(0); got != "never" {
		t.Errorf("fmtTS(0) = %q, want never", got)
	}
	if got := fmtTS(-5); got != "never" {
		t.Errorf("fmtTS(negative) = %q, want never", got)
	}
	// A concrete timestamp formats as "YYYY-MM-DD HH:MM:SS" in local time.
	ts := float64(time.Date(2024, 3, 4, 5, 6, 7, 0, time.Local).Unix())
	if got := fmtTS(ts); got != "2024-03-04 05:06:07" {
		t.Errorf("fmtTS(ts) = %q, want 2024-03-04 05:06:07", got)
	}
}
