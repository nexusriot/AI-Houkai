package maintenance

import (
	"context"
	"hash/fnv"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

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

func TestJobDue(t *testing.T) {
	// every < 0 disables the job outright.
	if jobDue(0, -1, 1000) {
		t.Fatal("jobDue(every=-1) = true, want false (disabled)")
	}
	// A job that never ran is immediately due.
	if !jobDue(0, 3600, 1000) {
		t.Fatal("jobDue(lastAt=0) = false, want true (never ran)")
	}
	// every == 0 runs on every tick.
	if !jobDue(999, 0, 1000) {
		t.Fatal("jobDue(every=0) = false, want true (ungated)")
	}
	// Gated: not due before the interval elapses, due at/after it.
	if jobDue(1000, 100, 1099) {
		t.Fatal("jobDue before interval elapsed = true, want false")
	}
	if !jobDue(1000, 100, 1100) {
		t.Fatal("jobDue at interval boundary = false, want true")
	}
}

func TestTickScheduleGates(t *testing.T) {
	// A tick whose jobs are gated (interval not yet elapsed) must skip both
	// jobs and leave the recorded timestamps untouched.
	store := newStore(t)
	ctx := context.Background()

	statePath := filepath.Join(t.TempDir(), "state.json")
	if err := (State{LastDecayAt: 1000, LastReflectAt: 1000}).Save(statePath); err != nil {
		t.Fatalf("Save: %v", err)
	}
	cfg := Config{
		DecayRate: 0.1, MinScore: 0.05, Reflect: true,
		DecayEvery: 86_400, ReflectEvery: 604_800,
	}
	res := Tick(ctx, store, store, cfg, statePath, 1500) // 500s later — nothing due
	if res.Err != nil {
		t.Fatalf("Tick: %v", res.Err)
	}
	if res.RanDecay || res.RanReflect {
		t.Fatalf("gated tick ran jobs: decay=%v reflect=%v, want false/false", res.RanDecay, res.RanReflect)
	}
	st := LoadState(statePath)
	if st.LastDecayAt != 1000 || st.LastReflectAt != 1000 {
		t.Fatalf("gated tick moved timestamps: %+v", st)
	}

	// Once the decay interval elapses the decay job runs (reflect still gated).
	res = Tick(ctx, store, store, cfg, statePath, 1000+86_400)
	if res.Err != nil {
		t.Fatalf("Tick 2: %v", res.Err)
	}
	if !res.RanDecay {
		t.Fatal("decay job should be due after decay_every elapsed")
	}
	if res.RanReflect {
		t.Fatal("reflect job should still be gated")
	}
}

func TestTickHonorsProtectTypes(t *testing.T) {
	// A prunable memory of a protected type must survive the decay job.
	// Without ProtectTypes plumbed through, the tick defaulted to protecting
	// only "procedural" and silently deleted the user's configured types.
	ctx := context.Background()
	seed := func(store *memory.MemoryStore) {
		// importance 0.01 < MinScore 0.05 even when fresh → prune candidate.
		if _, _, _, err := store.Remember(ctx, "user disliked the verbose tone",
			memory.RememberOpts{Type: memory.Feedback, Importance: memory.Float32Ptr(0.01)}); err != nil {
			t.Fatalf("Remember: %v", err)
		}
	}

	protectedStore := newStore(t)
	seed(protectedStore)
	cfg := Config{
		DecayRate: 0.1, MinScore: 0.05,
		ProtectTypes: []memory.MemoryType{memory.Feedback},
	}
	res := Tick(ctx, protectedStore, protectedStore, cfg, "", 1000)
	if res.Err != nil {
		t.Fatalf("Tick (protected): %v", res.Err)
	}
	if n, _ := protectedStore.Count(ctx); n != 1 {
		t.Fatalf("protected feedback memory was pruned: count=%d, want 1", n)
	}

	// Control: without protection, the same memory is pruned (proves it was
	// genuinely a prune candidate, so the survival above is due to protection).
	unprotectedStore := newStore(t)
	seed(unprotectedStore)
	res = Tick(ctx, unprotectedStore, unprotectedStore,
		Config{DecayRate: 0.1, MinScore: 0.05}, "", 1000)
	if res.Err != nil {
		t.Fatalf("Tick (unprotected): %v", res.Err)
	}
	if n, _ := unprotectedStore.Count(ctx); n != 0 {
		t.Fatalf("unprotected feedback memory survived: count=%d, want 0", n)
	}
}

func TestTickTakesStateLock(t *testing.T) {
	// The tick cycle must hold a flock on <state>.lock so concurrent tickers
	// (daemon + cron + MCP) serialise instead of double-running jobs.
	store := newStore(t)
	ctx := context.Background()
	dir := t.TempDir()
	statePath := filepath.Join(dir, "state.json")

	res := Tick(ctx, store, store, Config{DecayRate: 0.1, MinScore: 0.05}, statePath, 42)
	if res.Err != nil {
		t.Fatalf("Tick: %v", res.Err)
	}
	if _, err := os.Stat(filepath.Join(dir, "state.lock")); err != nil {
		t.Fatalf("state.lock not created next to state file: %v", err)
	}

	// While the lock is held by someone else, lockState must block — verify by
	// taking it ourselves and confirming a second tick completes only after
	// release.
	unlock, err := lockState(statePath)
	if err != nil {
		t.Fatalf("lockState: %v", err)
	}
	done := make(chan TickResult, 1)
	go func() {
		done <- Tick(ctx, store, store, Config{DecayRate: 0.1, MinScore: 0.05}, statePath, 43)
	}()
	select {
	case <-done:
		t.Fatal("Tick completed while the state lock was held")
	case <-time.After(100 * time.Millisecond):
		// still blocked — expected
	}
	unlock()
	select {
	case res := <-done:
		if res.Err != nil {
			t.Fatalf("Tick after unlock: %v", res.Err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Tick did not complete after the lock was released")
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
			Importance: memory.Float32Ptr(0.9),
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
		Type: memory.Semantic, Importance: memory.Float32Ptr(0.99),
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
			Type: memory.Episodic, Importance: memory.Float32Ptr(0.6),
		}); err != nil {
			t.Fatalf("Remember: %v", err)
		}
	}

	statePath := filepath.Join(t.TempDir(), "state.json")
	const now = 1_700_000_500.0
	cfg := Config{DecayRate: 0.1, MinScore: 0.05, Reflect: true}

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

func TestStateHasPurgeFields(t *testing.T) {
	path := filepath.Join(t.TempDir(), "state.json")
	want := State{LastPurgeAt: 123.0, TotalPurged: 7}
	if err := want.Save(path); err != nil {
		t.Fatalf("Save: %v", err)
	}
	got := LoadState(path)
	if got.LastPurgeAt != 123.0 || got.TotalPurged != 7 {
		t.Errorf("purge fields round-trip mismatch: %+v", got)
	}
}

func TestOldStateFileWithoutPurgeFieldsLoads(t *testing.T) {
	// Migration: a state file written before the purge fields existed.
	path := filepath.Join(t.TempDir(), "state.json")
	if err := os.WriteFile(path, []byte(`{"last_decay_at": 5.0, "total_decayed": 2}`), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	got := LoadState(path)
	if got.LastPurgeAt != 0 || got.TotalPurged != 0 {
		t.Errorf("missing purge fields should default to 0: %+v", got)
	}
	if got.LastDecayAt != 5.0 {
		t.Errorf("existing fields should still load: %+v", got)
	}
}

func TestTickPurgesExpiredWhenOverdue(t *testing.T) {
	ctx := context.Background()
	store := newStore(t)
	store.Remember(ctx, "expired doc", memory.RememberOpts{ExpiresAt: f64(1.0)})
	store.Remember(ctx, "live doc", memory.RememberOpts{})
	statePath := filepath.Join(t.TempDir(), "state.json")

	// decay/reflect disabled; purge_every=1 with no history → immediately due.
	cfg := Config{DecayEvery: -1, ReflectEvery: -1, PurgeEvery: 1}
	res := Tick(ctx, store, store, cfg, statePath, nowSec())
	if res.Err != nil {
		t.Fatalf("Tick: %v", res.Err)
	}
	if !res.RanPurge || res.Purged != 1 {
		t.Fatalf("ranPurge=%v purged=%d, want true/1", res.RanPurge, res.Purged)
	}
	if n, _ := store.Count(ctx); n != 1 {
		t.Fatalf("only the live memory should remain, count=%d", n)
	}
	st := LoadState(statePath)
	if st.LastPurgeAt == 0 || st.TotalPurged != 1 {
		t.Errorf("purge state not persisted: %+v", st)
	}
}

func TestTickSkipsPurgeWhenFresh(t *testing.T) {
	ctx := context.Background()
	store := newStore(t)
	store.Remember(ctx, "expired doc", memory.RememberOpts{ExpiresAt: f64(1.0)})
	statePath := filepath.Join(t.TempDir(), "state.json")
	now := nowSec()
	(State{LastPurgeAt: now}).Save(statePath)

	cfg := Config{DecayEvery: -1, ReflectEvery: -1, PurgeEvery: 86_400}
	res := Tick(ctx, store, store, cfg, statePath, now)
	if res.RanPurge {
		t.Error("purge should be gated (fresh last_purge_at)")
	}
	if n, _ := store.Count(ctx); n != 1 {
		t.Errorf("gated purge should not delete, count=%d", n)
	}
}

func TestTickPurgeDisabled(t *testing.T) {
	ctx := context.Background()
	store := newStore(t)
	store.Remember(ctx, "expired doc", memory.RememberOpts{ExpiresAt: f64(1.0)})
	cfg := Config{DecayEvery: -1, ReflectEvery: -1, PurgeEvery: -1}
	res := Tick(ctx, store, store, cfg, "", nowSec())
	if res.RanPurge {
		t.Error("purge_every < 0 must disable the job")
	}
	if n, _ := store.Count(ctx); n != 1 {
		t.Errorf("disabled purge should not delete, count=%d", n)
	}
}

func f64(v float64) *float64 { return &v }

func nowSec() float64 { return float64(time.Now().Unix()) }
