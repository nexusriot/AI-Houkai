package memory

import (
	"context"
	"math"
	"testing"
)

func score(text string) float32 {
	return ScoreImportance(text, Semantic, nil)
}

func TestImportanceTiers(t *testing.T) {
	// Standing instructions score high.
	for _, s := range []string{
		"Always run make lint before committing",
		"Never push directly to main",
		"From now on use uv instead of pip",
		// Corrections.
		"Actually, the API key lives in vault, not env",
		"That's wrong — the timeout is 30s",
		// Preferences.
		"vlad prefers tabs over spaces",
		"I hate auto-formatting on save",
	} {
		if got := score(s); got < 0.9 {
			t.Errorf("score(%q) = %v, want >= 0.9", s, got)
		}
	}

	// Decisions score mid-high.
	for _, s := range []string{
		"We decided to target Python 3.11 minimum",
		"The team convention is one module per command",
	} {
		if got := score(s); got < 0.7 || got >= 0.9 {
			t.Errorf("score(%q) = %v, want [0.7, 0.9)", s, got)
		}
	}

	// Completions score mid.
	for _, s := range []string{
		"Fixed the race condition in the journal writer",
		"Deployed version 0.5.5 to PyPI yesterday",
	} {
		if got := score(s); got < 0.55 || got >= 0.7 {
			t.Errorf("score(%q) = %v, want [0.55, 0.7)", s, got)
		}
	}

	// Observations score low.
	for _, s := range []string{
		"It seems the cache is sometimes stale",
		"Noticed the tests take a while on CI",
	} {
		if got := score(s); got >= 0.5 {
			t.Errorf("score(%q) = %v, want < 0.5", s, got)
		}
	}

	// Neutral text gets the default.
	if got := score("The store uses ChromaDB for persistence"); got != 0.5 {
		t.Errorf("neutral score = %v, want 0.5", got)
	}

	// Strongest tier wins: "never" (0.9) beats "maybe" (0.35).
	if got := score("Never use eval, maybe except in the REPL"); got < 0.9 {
		t.Errorf("strongest tier: got %v, want >= 0.9", got)
	}
}

func approxEq(a, b float32) bool {
	return math.Abs(float64(a-b)) < 1e-3
}

func TestImportanceModifiers(t *testing.T) {
	// Procedural / feedback type bonus.
	base := ScoreImportance("Deploy with make release", Semantic, nil)
	proc := ScoreImportance("Deploy with make release", Procedural, nil)
	if !approxEq(proc, base+0.10) {
		t.Errorf("procedural bonus: base=%v proc=%v", base, proc)
	}
	base = ScoreImportance("The output was truncated", Semantic, nil)
	fb := ScoreImportance("The output was truncated", Feedback, nil)
	if !approxEq(fb, base+0.10) {
		t.Errorf("feedback bonus: base=%v fb=%v", base, fb)
	}

	// Question penalty.
	if score("Is the deploy target staging?") >= score("The deploy target is staging") {
		t.Error("question should score below plain statement")
	}

	// Short fragment penalty.
	if got := score("ok then"); got >= 0.5 {
		t.Errorf("short fragment score = %v, want < 0.5", got)
	}

	// Clamped to bounds.
	if got := ScoreImportance("Always always never must", Procedural, nil); got > 0.98 {
		t.Errorf("ceiling: got %v", got)
	}
	if got := ScoreImportance("hm, maybe?", Semantic, nil); got < 0.05 {
		t.Errorf("floor: got %v", got)
	}

	// Deterministic.
	text := "We decided to use sqlite for the cache"
	if score(text) != score(text) {
		t.Error("not deterministic")
	}
}

func TestRememberUsesImportanceFn(t *testing.T) {
	store := newTestStore(t)
	store.cfg.ImportanceFn = ScoreImportance
	ctx := context.Background()

	m, _, _, err := store.Remember(ctx, "Never commit secrets to the repo", RememberOpts{Type: Procedural})
	if err != nil {
		t.Fatal(err)
	}
	if m.Importance < 0.9 {
		t.Errorf("auto importance = %v, want >= 0.9", m.Importance)
	}

	// Explicit value still wins.
	m2, _, _, err := store.Remember(ctx, "Never do that other thing", RememberOpts{Importance: Float32Ptr(0.2)})
	if err != nil {
		t.Fatal(err)
	}
	if !approxEq(m2.Importance, 0.2) {
		t.Errorf("explicit importance = %v, want 0.2", m2.Importance)
	}
}

func TestRememberWithoutFnKeepsDefault(t *testing.T) {
	store := newTestStore(t)
	m, _, _, err := store.Remember(context.Background(), "Never commit secrets to the repo", RememberOpts{})
	if err != nil {
		t.Fatal(err)
	}
	if !approxEq(m.Importance, 0.5) {
		t.Errorf("default importance = %v, want 0.5", m.Importance)
	}
}
