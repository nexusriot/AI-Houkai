package eval

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The metric functions must match Python's ai_houkai/eval.py exactly — the two
// ports are expected to report the same numbers for the same gold set.

func approx(t *testing.T, got, want float64, label string) {
	t.Helper()
	if math.Abs(got-want) > 1e-9 {
		t.Errorf("%s = %v, want %v", label, got, want)
	}
}

func TestRecallAtK(t *testing.T) {
	approx(t, RecallAtK([]string{"a", "b", "c"}, []string{"b"}, 2), 1.0, "hit in top-2")
	approx(t, RecallAtK([]string{"a", "b", "c"}, []string{"c"}, 2), 0.0, "miss outside k")
	approx(t, RecallAtK([]string{"a", "b"}, []string{"a", "b"}, 2), 1.0, "both")
	approx(t, RecallAtK([]string{"a", "b"}, []string{"a", "z"}, 2), 0.5, "half")
	approx(t, RecallAtK([]string{"a"}, nil, 2), 0.0, "no relevant ids")
}

func TestRecallAtKCountsDistinct(t *testing.T) {
	// A duplicated retrieved id must not push recall above 1.0.
	approx(t, RecallAtK([]string{"a", "a", "a"}, []string{"a"}, 3), 1.0, "duplicates")
}

func TestPrecisionAtK(t *testing.T) {
	approx(t, PrecisionAtK([]string{"a", "b"}, []string{"a"}, 2), 0.5, "one of two")
	approx(t, PrecisionAtK(nil, []string{"a"}, 2), 0.0, "nothing retrieved")
	approx(t, PrecisionAtK([]string{"a", "b"}, []string{"a", "b"}, 2), 1.0, "all")
}

func TestReciprocalRank(t *testing.T) {
	approx(t, ReciprocalRank([]string{"x", "a"}, []string{"a"}), 0.5, "second")
	approx(t, ReciprocalRank([]string{"a", "x"}, []string{"a"}), 1.0, "first")
	approx(t, ReciprocalRank([]string{"x", "y"}, []string{"a"}), 0.0, "absent")
}

func TestAveragePrecision(t *testing.T) {
	// Relevant at ranks 1 and 3: (1/1 + 2/3) / 2
	approx(t, AveragePrecision([]string{"a", "x", "b"}, []string{"a", "b"}),
		(1.0+2.0/3.0)/2.0, "two hits")
	approx(t, AveragePrecision([]string{"a", "a"}, []string{"a"}), 1.0,
		"duplicate credited once")
	approx(t, AveragePrecision([]string{"x"}, nil), 0.0, "no relevant ids")
}

func TestNDCGAtK(t *testing.T) {
	approx(t, NDCGAtK([]string{"a", "b"}, []string{"a", "b"}, 2), 1.0, "perfect")
	approx(t, NDCGAtK([]string{"x", "y"}, []string{"a"}, 2), 0.0, "nothing found")
	approx(t, NDCGAtK([]string{"x"}, nil, 2), 0.0, "no relevant ids")

	// One relevant id at rank 2: DCG = 1/log2(3), IDCG = 1/log2(2) = 1.
	approx(t, NDCGAtK([]string{"x", "a"}, []string{"a"}, 2), 1.0/math.Log2(3),
		"single hit at rank 2")
}

func TestNDCGCreditsEachIDOnce(t *testing.T) {
	// Duplicates must not inflate the gain above the perfect score.
	got := NDCGAtK([]string{"a", "a", "b"}, []string{"a", "b"}, 3)
	if got > 1.0 {
		t.Errorf("nDCG = %v, must not exceed 1.0", got)
	}
}

// fakeRecaller returns a canned ranking per query, so Evaluate can be tested
// without a store or an embedder.
type fakeRecaller struct {
	byQuery map[string][]string
	calls   int
	lastK   int
}

func (f *fakeRecaller) Recall(_ context.Context, query string, k int, _ RecallOpts) ([]Hit, error) {
	f.calls++
	f.lastK = k
	ids := f.byQuery[query]
	if len(ids) > k {
		ids = ids[:k]
	}
	hits := make([]Hit, len(ids))
	for i, id := range ids {
		hits[i] = Hit{ID: id}
	}
	return hits, nil
}

func TestEvaluateAggregates(t *testing.T) {
	store := &fakeRecaller{byQuery: map[string][]string{
		"perfect": {"a", "b"},
		"missed":  {"x", "y"},
	}}
	res, err := Evaluate(context.Background(), store, []Case{
		{Query: "perfect", RelevantIDs: []string{"a"}},
		{Query: "missed", RelevantIDs: []string{"z"}},
	}, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if res.N != 2 {
		t.Fatalf("n = %d", res.N)
	}
	approx(t, res.RecallAtK, 0.5, "mean recall")
	approx(t, res.MRR, 0.5, "mean RR")
	if len(res.PerCase) != 2 {
		t.Errorf("per-case rows = %d", len(res.PerCase))
	}
}

func TestEvaluateDefaults(t *testing.T) {
	store := &fakeRecaller{byQuery: map[string][]string{"q": {"a"}}}
	res, _ := Evaluate(context.Background(), store,
		[]Case{{Query: "q", RelevantIDs: []string{"a"}}}, Options{})
	if store.lastK != 5 {
		t.Errorf("default k = %d, want 5", store.lastK)
	}
	if res.K != 5 {
		t.Errorf("result k = %d, want 5", res.K)
	}
}

func TestEvaluatePerCaseKOverridesDefault(t *testing.T) {
	store := &fakeRecaller{byQuery: map[string][]string{"q": {"a"}}}
	Evaluate(context.Background(), store,
		[]Case{{Query: "q", RelevantIDs: []string{"a"}, K: 3}}, Options{DefaultK: 5})
	if store.lastK != 3 {
		t.Errorf("k = %d, want the per-case 3", store.lastK)
	}
}

func TestEvaluateMixedKIsMinusOne(t *testing.T) {
	store := &fakeRecaller{byQuery: map[string][]string{"a": {"1"}, "b": {"2"}}}
	res, _ := Evaluate(context.Background(), store, []Case{
		{Query: "a", RelevantIDs: []string{"1"}, K: 2},
		{Query: "b", RelevantIDs: []string{"2"}, K: 5},
	}, Options{})
	if res.K != -1 {
		t.Errorf("mixed k = %d, want -1", res.K)
	}
}

func TestEvaluateEmptyCases(t *testing.T) {
	store := &fakeRecaller{byQuery: map[string][]string{}}
	res, err := Evaluate(context.Background(), store, nil, Options{})
	if err != nil {
		t.Fatal(err)
	}
	// No division by zero, and no fabricated scores.
	if res.N != 0 || res.RecallAtK != 0 {
		t.Errorf("empty run = %+v", res)
	}
}

func TestLoadGoldset(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "gold.jsonl")
	os.WriteFile(path, []byte(
		"# a comment\n"+
			`{"query": "alpha", "relevant_ids": ["a"]}`+"\n"+
			"\n"+
			`{"query": "beta", "relevant_ids": ["b","c"], "k": 3, "mode": "semantic"}`+"\n",
	), 0o644)

	cases, err := LoadGoldset(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(cases) != 2 {
		t.Fatalf("cases = %d, want 2", len(cases))
	}
	if cases[0].Query != "alpha" || len(cases[0].RelevantIDs) != 1 {
		t.Errorf("case 0 = %+v", cases[0])
	}
	if cases[1].K != 3 || cases[1].Mode != "semantic" {
		t.Errorf("case 1 = %+v", cases[1])
	}
}

func TestLoadGoldsetErrors(t *testing.T) {
	dir := t.TempDir()
	for _, tc := range []struct{ name, body, want string }{
		{"bad json", "not json\n", "invalid JSON"},
		{"no query", `{"relevant_ids":["a"]}` + "\n", "missing 'query'"},
		{"no ids", `{"query":"q"}` + "\n", "non-empty list"},
		{"empty", "# nothing\n", "no evaluation cases"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(dir, tc.name+".jsonl")
			os.WriteFile(path, []byte(tc.body), 0o644)
			_, err := LoadGoldset(path)
			if err == nil {
				t.Fatal("expected an error")
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("error = %q, want it to mention %q", err, tc.want)
			}
		})
	}
}

func TestLoadGoldsetMissingFile(t *testing.T) {
	if _, err := LoadGoldset(filepath.Join(t.TempDir(), "absent.jsonl")); err == nil {
		t.Error("expected an error for a missing file")
	}
}
