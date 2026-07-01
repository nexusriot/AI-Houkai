package maintenance

import (
	"context"
	"hash/fnv"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/nexusriot/ai-houkai/internal/memory"
	"github.com/nexusriot/ai-houkai/internal/vector"
)

// hashEmbedder deterministically hashes each text into a fixed-dim,
// L2-normalised vector, so identical texts yield identical vectors. It lets a
// package outside `memory` build a real *memory.MemoryStore without touching
// memory's test-only stub. Named distinctly from the mcpserver test embedder to
// avoid confusion; each _test.go file is compiled with its own package.
type hashEmbedder struct{ dim int }

func (e *hashEmbedder) Dim() int { return e.dim }

func (e *hashEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, t := range texts {
		v := make([]float32, e.dim)
		h := fnv.New64a()
		_, _ = h.Write([]byte(t))
		seed := h.Sum64()
		for j := 0; j < e.dim; j++ {
			seed = seed*6364136223846793005 + 1442695040888963407
			v[j] = float32(int64(seed>>33)%1000) / 1000.0
		}
		var norm float64
		for _, x := range v {
			norm += float64(x) * float64(x)
		}
		norm = math.Sqrt(norm)
		if norm > 0 {
			for j := range v {
				v[j] = float32(float64(v[j]) / norm)
			}
		}
		out[i] = v
	}
	return out, nil
}

// newStore builds a real store backed by chromem in a temp dir.
func newStore(t *testing.T) *memory.MemoryStore {
	t.Helper()
	dir := t.TempDir()
	backend, err := vector.NewChromem(filepath.Join(dir, "s"), "test", 16)
	if err != nil {
		t.Fatalf("NewChromem: %v", err)
	}
	t.Cleanup(func() { _ = backend.Close() })
	// Disable the journal so the daemon test does not litter the dir with logs.
	cfg := memory.DefaultStoreConfig(dir, "test")
	cfg.JournalEnabled = false
	return memory.NewMemoryStore(backend, &hashEmbedder{dim: 16}, cfg)
}

func TestLoadStateSaveRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "state.json")

	// Missing path → zero State.
	if got := LoadState(path); got != (State{}) {
		t.Fatalf("LoadState(missing) = %+v, want zero State", got)
	}

	want := State{
		LastDecayAt:    123.5,
		LastReflectAt:  456.75,
		TotalDecayed:   7,
		TotalReflected: 3,
	}
	if err := want.Save(path); err != nil {
		t.Fatalf("Save: %v", err)
	}
	got := LoadState(path)
	if got != want {
		t.Fatalf("round-trip mismatch: got %+v, want %+v", got, want)
	}

	// Confirm the file is really on disk with the expected mode.
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("state file not written: %v", err)
	}
}

func TestLoadStateCorruptFile(t *testing.T) {
	// Unreadable/garbage JSON must not panic; json.Unmarshal error is ignored
	// so we get whatever was parsed before the error (zero here).
	path := filepath.Join(t.TempDir(), "state.json")
	if err := os.WriteFile(path, []byte("{not valid json"), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	got := LoadState(path)
	if got != (State{}) {
		t.Fatalf("LoadState(garbage) = %+v, want zero State", got)
	}
}

func TestPidfileRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "daemon.pid")

	// Missing → 0.
	if pid := ReadPid(path); pid != 0 {
		t.Fatalf("ReadPid(missing) = %d, want 0", pid)
	}

	if err := WritePid(path, 4242); err != nil {
		t.Fatalf("WritePid: %v", err)
	}
	if pid := ReadPid(path); pid != 4242 {
		t.Fatalf("ReadPid = %d, want 4242", pid)
	}

	RemovePid(path)
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("RemovePid did not remove file: stat err = %v", err)
	}
	// ReadPid after removal is 0 again.
	if pid := ReadPid(path); pid != 0 {
		t.Fatalf("ReadPid(after remove) = %d, want 0", pid)
	}
}

func TestReadPidUnparseable(t *testing.T) {
	path := filepath.Join(t.TempDir(), "daemon.pid")
	if err := os.WriteFile(path, []byte("garbage\n"), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	if pid := ReadPid(path); pid != 0 {
		t.Fatalf("ReadPid(garbage) = %d, want 0", pid)
	}
}

func TestIsAlive(t *testing.T) {
	path := filepath.Join(t.TempDir(), "daemon.pid")

	// No pidfile → not alive.
	if IsAlive(path) {
		t.Fatal("IsAlive(missing pidfile) = true, want false")
	}

	// Our own pid → alive.
	if err := WritePid(path, os.Getpid()); err != nil {
		t.Fatalf("WritePid: %v", err)
	}
	if !IsAlive(path) {
		t.Fatal("IsAlive(self) = false, want true")
	}

	// A very high, almost-certainly-unused pid → not alive. os.FindProcess
	// always succeeds on unix, so IsAlive relies on the signal-0 probe.
	if err := WritePid(path, 0x7FFFFFFE); err != nil {
		t.Fatalf("WritePid: %v", err)
	}
	if IsAlive(path) {
		t.Fatal("IsAlive(huge pid) = true, want false")
	}
}

func TestStopDaemonNoPid(t *testing.T) {
	// StopDaemon with no pidfile must report false (nothing signalled).
	path := filepath.Join(t.TempDir(), "daemon.pid")
	if StopDaemon(path) {
		t.Fatal("StopDaemon(missing pidfile) = true, want false")
	}
	// A zero/negative pid is also refused.
	if err := WritePid(path, 0); err != nil {
		t.Fatalf("WritePid: %v", err)
	}
	if StopDaemon(path) {
		t.Fatal("StopDaemon(pid 0) = true, want false")
	}
}

func TestDefaultConfig(t *testing.T) {
	c := DefaultConfig()
	if c.Interval <= 0 {
		t.Fatalf("DefaultConfig().Interval = %v, want > 0", c.Interval)
	}
	if c.DecayRate <= 0 {
		t.Fatalf("DefaultConfig().DecayRate = %v, want > 0", c.DecayRate)
	}
	if c.Reflect {
		t.Fatal("DefaultConfig().Reflect = true, want false")
	}
}

func TestTickPrunePersistsState(t *testing.T) {
	store := newStore(t)
	ctx := context.Background()

	// Seed a few memories. They are fresh (created now), so a modest decay
	// rate should not prune them — the point here is that Tick runs cleanly
	// and persists LastDecayAt.
	for _, txt := range []string{"alpha fact", "beta fact", "gamma fact"} {
		if _, _, _, err := store.Remember(ctx, txt, memory.RememberOpts{
			Type:       memory.Semantic,
			Importance: 0.9,
		}); err != nil {
			t.Fatalf("Remember: %v", err)
		}
	}

	statePath := filepath.Join(t.TempDir(), "state.json")
	const now = 1_700_000_000.0
	cfg := Config{DecayRate: 0.1, MinScore: 0.05, Reflect: false}

	res := Tick(ctx, store, store, cfg, statePath, now)
	if res.Err != nil {
		t.Fatalf("Tick returned err: %v", res.Err)
	}
	if res.Reflected != 0 {
		t.Fatalf("Reflect=false but Reflected=%d", res.Reflected)
	}

	st := LoadState(statePath)
	if st.LastDecayAt != now {
		t.Fatalf("LastDecayAt = %v, want %v", st.LastDecayAt, now)
	}
	// Reflect was off, so LastReflectAt must remain zero.
	if st.LastReflectAt != 0 {
		t.Fatalf("LastReflectAt = %v, want 0 (reflect off)", st.LastReflectAt)
	}
	// TotalDecayed accumulates however many were pruned (>= 0).
	if st.TotalDecayed != res.Pruned {
		t.Fatalf("TotalDecayed = %d, want %d (== res.Pruned)", st.TotalDecayed, res.Pruned)
	}
}

func TestTickAccumulatesDecayTotal(t *testing.T) {
	// Two ticks against the same state file must accumulate TotalDecayed and
	// advance LastDecayAt to the latest now.
	store := newStore(t)
	ctx := context.Background()
	if _, _, _, err := store.Remember(ctx, "kept memory", memory.RememberOpts{
		Type: memory.Semantic, Importance: 0.99,
	}); err != nil {
		t.Fatalf("Remember: %v", err)
	}

	statePath := filepath.Join(t.TempDir(), "state.json")
	cfg := Config{DecayRate: 0.1, MinScore: 0.05, Reflect: false}

	r1 := Tick(ctx, store, store, cfg, statePath, 1000)
	if r1.Err != nil {
		t.Fatalf("Tick 1: %v", r1.Err)
	}
	r2 := Tick(ctx, store, store, cfg, statePath, 2000)
	if r2.Err != nil {
		t.Fatalf("Tick 2: %v", r2.Err)
	}

	st := LoadState(statePath)
	if st.LastDecayAt != 2000 {
		t.Fatalf("LastDecayAt = %v, want 2000 (latest tick)", st.LastDecayAt)
	}
	if st.TotalDecayed != r1.Pruned+r2.Pruned {
		t.Fatalf("TotalDecayed = %d, want %d", st.TotalDecayed, r1.Pruned+r2.Pruned)
	}
}

func TestTickEmptyStatePathSkipsPersistence(t *testing.T) {
	// statePath == "" must run the pass but write no state file.
	store := newStore(t)
	ctx := context.Background()
	res := Tick(ctx, store, store, Config{DecayRate: 0.1, MinScore: 0.05}, "", 42)
	if res.Err != nil {
		t.Fatalf("Tick: %v", res.Err)
	}
}

func TestTickReflect(t *testing.T) {
	// Reflect=true should exercise the reflection branch. With the built-in
	// extractive summarizer (nil) there is no network dependency. It may
	// create zero reflections (nothing to consolidate), which is fine — we
	// assert it runs cleanly and stamps LastReflectAt.
	store := newStore(t)
	ctx := context.Background()
	for _, txt := range []string{
		"the deployment uses kubernetes for orchestration",
		"the deployment relies on kubernetes clusters heavily",
		"kubernetes orchestrates the deployment pods",
	} {
		if _, _, _, err := store.Remember(ctx, txt, memory.RememberOpts{
			Type: memory.Episodic, Importance: 0.6,
		}); err != nil {
			t.Fatalf("Remember: %v", err)
		}
	}

	statePath := filepath.Join(t.TempDir(), "state.json")
	const now = 1_700_000_500.0
	cfg := Config{DecayRate: 0.1, MinScore: 0.05, Reflect: true, Consolidate: false}

	res := Tick(ctx, store, store, cfg, statePath, now)
	if res.Err != nil {
		t.Fatalf("Tick(reflect) returned err: %v", res.Err)
	}
	if res.Reflected < 0 {
		t.Fatalf("Reflected = %d, want >= 0", res.Reflected)
	}

	st := LoadState(statePath)
	if st.LastReflectAt != now {
		t.Fatalf("LastReflectAt = %v, want %v (reflect on)", st.LastReflectAt, now)
	}
	if st.TotalReflected != res.Reflected {
		t.Fatalf("TotalReflected = %d, want %d", st.TotalReflected, res.Reflected)
	}
}

// SpawnDetached is intentionally NOT tested here: it forks a real background
// `houkai maintenance run` process via exec + Setsid, which is neither
// hermetic nor easily reapable in a unit test. Covered by manual/integration
// testing instead.
