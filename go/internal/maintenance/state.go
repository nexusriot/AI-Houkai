package maintenance

import (
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"syscall"

	"github.com/nexusriot/ai-houkai/internal/decay"
	reflectpkg "github.com/nexusriot/ai-houkai/internal/reflect"
)

// State is the persisted maintenance-daemon state, shared between the
// foreground/background daemon and `maintenance status`.
type State struct {
	LastDecayAt    float64 `json:"last_decay_at"`
	LastReflectAt  float64 `json:"last_reflect_at"`
	TotalDecayed   int     `json:"total_decayed"`
	TotalReflected int     `json:"total_reflected"`
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

// TickResult summarises one maintenance tick.
type TickResult struct {
	Pruned    int
	Reflected int
	Err       error
}

// Tick runs one synchronous maintenance pass (prune + optional reflect),
// updating the state at `statePath` (empty = skip persistence). `nowUnix` is
// the current time in Unix seconds (passed in so callers control the clock).
func Tick(ctx context.Context, store decay.Storable, reflStore reflectpkg.Storable, cfg Config, statePath string, nowUnix float64) TickResult {
	var res TickResult
	de := decay.New(store, cfg.DecayRate, cfg.MinScore, nil, cfg.FrequencyWeight)
	pruned, err := de.Prune(ctx, false)
	if err != nil {
		res.Err = err
		return res
	}
	res.Pruned = len(pruned)

	if cfg.Reflect {
		re := reflectpkg.New(reflStore, 0, 0, cfg.Summarizer)
		mode := reflectpkg.ConsolidateNone
		if cfg.Consolidate {
			mode = reflectpkg.ConsolidateSoft
		}
		created, err := re.Reflect(ctx, false, mode)
		if err != nil {
			res.Err = err
			return res
		}
		res.Reflected = len(created)
	}

	if statePath != "" {
		st := LoadState(statePath)
		st.LastDecayAt = nowUnix
		st.TotalDecayed += res.Pruned
		if cfg.Reflect {
			st.LastReflectAt = nowUnix
			st.TotalReflected += res.Reflected
		}
		_ = st.Save(statePath)
	}
	return res
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
