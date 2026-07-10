package maintenance

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/nexusriot/ai-houkai/internal/decay"
	"github.com/nexusriot/ai-houkai/internal/memory"
	reflectpkg "github.com/nexusriot/ai-houkai/internal/reflect"
)

// Config controls a maintenance tick (used by the CLI tick/run commands and
// the MCP maintenance_tick tool).
type Config struct {
	Interval    time.Duration // cadence of the foreground `maintenance run` loop
	DecayRate   float32
	MinScore    float32
	// ProtectTypes are never pruned by the decay job. Without this the tick
	// defaulted to decay's built-in ["procedural"], silently ignoring the
	// user's configured protect_types and deleting memories they meant to keep.
	ProtectTypes []memory.MemoryType
	Reflect      bool                       // run the reflection job at all
	Consolidate  reflectpkg.ConsolidateMode // what happens to reflected sources ("" = none)

	// DecayEvery / ReflectEvery gate the jobs on a schedule: a job only runs
	// when at least this many seconds have passed since its last recorded run
	// in the state file (mirrors Python's MaintenanceScheduler). 0 = no gate
	// (run on every tick); < 0 = job disabled. With an empty state path there
	// is no history, so a non-disabled job is always due.
	DecayEvery   float64
	ReflectEvery float64
	// PurgeEvery gates the TTL-purge job on the same schedule semantics.
	PurgeEvery float64

	// FrequencyWeight > 0 makes frequently-recalled memories resist decay
	// (forwarded to decay.Engine; 0 = recency-only, the default).
	FrequencyWeight float32

	// Summarizer is forwarded to the reflection engine (e.g. from
	// reflect.BuildSummarizer("ollama:llama3.1")). Nil → the built-in
	// extractive summarizer.
	Summarizer reflectpkg.Summarizer
}

// State is the persisted maintenance state, shared between the daemon,
// cron ticks, the MCP maintenance_tick tool, and `maintenance status`.
type State struct {
	LastDecayAt    float64 `json:"last_decay_at"`
	LastReflectAt  float64 `json:"last_reflect_at"`
	LastPurgeAt    float64 `json:"last_purge_at"`
	TotalDecayed   int     `json:"total_decayed"`
	TotalReflected int     `json:"total_reflected"`
	TotalPurged    int     `json:"total_purged"`
}

// LoadState reads the state file (a zero State if it is missing/unreadable).
func LoadState(path string) State {
	var s State
	data, err := os.ReadFile(path)
	if err != nil {
		return s
	}
	_ = json.Unmarshal(data, &s)
	return s
}

// Save writes the state file atomically-ish (best-effort).
func (s State) Save(path string) error {
	data, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

// TickResult summarises one maintenance tick. RanDecay/RanReflect report
// whether each job was due (a gated-out job leaves them false).
type TickResult struct {
	RanDecay   bool
	RanReflect bool
	RanPurge   bool
	Pruned     int
	Reflected  int
	Purged     int
	Err        error
}

// lockState takes an exclusive cross-process flock guarding the
// load→run→save tick cycle (mirrors Python's _state_lock). The daemon loop, a
// cron `houkai maintenance tick`, and the MCP maintenance_tick tool may all
// target the same state file; without the lock two concurrent tickers could
// both observe a job as due (double run) and the later save would clobber the
// earlier one's timestamps. Returns an unlock func.
func lockState(statePath string) (func(), error) {
	ext := filepath.Ext(statePath)
	lockPath := strings.TrimSuffix(statePath, ext) + ".lock"
	if err := os.MkdirAll(filepath.Dir(lockPath), 0o755); err != nil {
		return nil, err
	}
	f, err := os.OpenFile(lockPath, os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return nil, err
	}
	if err := syscall.Flock(int(f.Fd()), syscall.LOCK_EX); err != nil {
		_ = f.Close()
		return nil, err
	}
	return func() {
		_ = syscall.Flock(int(f.Fd()), syscall.LOCK_UN)
		_ = f.Close()
	}, nil
}

// Tick runs one synchronous maintenance pass (prune + optional reflect),
// updating the state at `statePath` (empty = skip persistence and schedule
// gating). `nowUnix` is the current time in Unix seconds (passed in so
// callers control the clock). The whole read-modify-write cycle holds an
// exclusive flock on <state>.lock so concurrent tickers serialise instead of
// double-running jobs.
func Tick(ctx context.Context, store decay.Storable, reflStore reflectpkg.Storable, cfg Config, statePath string, nowUnix float64) TickResult {
	if statePath != "" {
		unlock, err := lockState(statePath)
		if err != nil {
			return TickResult{Err: err}
		}
		defer unlock()
	}
	return tickLocked(ctx, store, reflStore, cfg, statePath, nowUnix)
}

// jobDue reports whether a job gated by `every` should run given its last
// recorded run. every < 0 disables the job; 0 = every tick; lastAt 0 = never
// ran → immediately due (mirrors Python's next_run_at).
func jobDue(lastAt, every, nowUnix float64) bool {
	if every < 0 {
		return false
	}
	if lastAt == 0 {
		return true
	}
	return nowUnix >= lastAt+every
}

func tickLocked(ctx context.Context, store decay.Storable, reflStore reflectpkg.Storable, cfg Config, statePath string, nowUnix float64) TickResult {
	var res TickResult
	var st State
	if statePath != "" {
		st = LoadState(statePath)
	}

	if jobDue(st.LastDecayAt, cfg.DecayEvery, nowUnix) {
		res.RanDecay = true
		de := decay.New(store, cfg.DecayRate, cfg.MinScore, cfg.ProtectTypes, cfg.FrequencyWeight)
		pruned, err := de.Prune(ctx, false)
		if err != nil {
			// One failing job must not prevent the other from running
			// (mirrors Python's per-job try/except).
			res.Err = errors.Join(res.Err, err)
		} else {
			res.Pruned = len(pruned)
			st.LastDecayAt = nowUnix
			st.TotalDecayed += res.Pruned
		}
	}

	if cfg.Reflect && jobDue(st.LastReflectAt, cfg.ReflectEvery, nowUnix) {
		res.RanReflect = true
		re := reflectpkg.New(reflStore, 0, 0, cfg.Summarizer)
		mode := cfg.Consolidate
		if mode == "" {
			mode = reflectpkg.ConsolidateNone
		}
		created, err := re.Reflect(ctx, false, mode)
		if err != nil {
			res.Err = errors.Join(res.Err, err)
		} else {
			res.Reflected = len(created)
			// The schedule gates the WORK, not the writes — last_reflect_at
			// is stamped whenever the job ran (Python does the same even for
			// dry-run reflection).
			st.LastReflectAt = nowUnix
			st.TotalReflected += res.Reflected
		}
	}

	// TTL purge. The tick's store is the decay.Storable interface, which lacks
	// PurgeExpired; type-assert to reach it (mirrors decay's actorScoped probe).
	// A store without the method silently skips the job.
	if px, ok := store.(expirable); ok && jobDue(st.LastPurgeAt, cfg.PurgeEvery, nowUnix) {
		res.RanPurge = true
		purged, err := px.PurgeExpired(ctx, nowUnix, false)
		if err != nil {
			res.Err = errors.Join(res.Err, err)
		} else {
			res.Purged = len(purged)
			st.LastPurgeAt = nowUnix
			st.TotalPurged += res.Purged
		}
	}

	if statePath != "" {
		_ = st.Save(statePath)
	}
	return res
}

// expirable is the optional store capability the purge job needs; the concrete
// *memory.MemoryStore satisfies it, while a bare decay.Storable fake does not.
type expirable interface {
	PurgeExpired(ctx context.Context, now float64, dryRun bool) ([]memory.Memory, error)
}

// ReadPid returns the pid recorded in path (0 if absent/unparseable).
func ReadPid(path string) int {
	data, err := os.ReadFile(path)
	if err != nil {
		return 0
	}
	pid, _ := strconv.Atoi(strings.TrimSpace(string(data)))
	return pid
}

// WritePid records pid in path.
func WritePid(path string, pid int) error {
	return os.WriteFile(path, []byte(strconv.Itoa(pid)), 0o644)
}

// RemovePid deletes the pidfile (best-effort).
func RemovePid(path string) { _ = os.Remove(path) }

// IsAlive reports whether the pid in path refers to a live process.
func IsAlive(path string) bool {
	pid := ReadPid(path)
	if pid <= 0 {
		return false
	}
	proc, err := os.FindProcess(pid)
	if err != nil {
		return false
	}
	// Signal 0 probes existence without affecting the process.
	return proc.Signal(syscall.Signal(0)) == nil
}

// StopDaemon sends SIGTERM to the pid in path. Returns true if a signal was sent.
func StopDaemon(path string) bool {
	pid := ReadPid(path)
	if pid <= 0 {
		return false
	}
	proc, err := os.FindProcess(pid)
	if err != nil {
		return false
	}
	return proc.Signal(syscall.SIGTERM) == nil
}

// SpawnDetached launches `houkai maintenance run` as a detached background
// process, redirecting its output to logPath and recording its pid in pidPath.
// Returns the child pid.
func SpawnDetached(exe, storePath, collection, logPath, pidPath string) (int, error) {
	logFile, err := os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return 0, err
	}
	args := []string{"--store", storePath, "--collection", collection, "maintenance", "run"}
	cmd := exec.Command(exe, args...)
	cmd.Stdout = logFile
	cmd.Stderr = logFile
	cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true} // detach into its own session
	if err := cmd.Start(); err != nil {
		_ = logFile.Close()
		return 0, err
	}
	pid := cmd.Process.Pid
	if err := WritePid(pidPath, pid); err != nil {
		return pid, err
	}
	// The parent no longer needs the log handle; the child inherited its fd.
	_ = logFile.Close()
	return pid, nil
}
