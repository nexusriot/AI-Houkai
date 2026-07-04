package cli

import (
	"fmt"
	"log"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/nexusriot/ai-houkai/internal/maintenance"
	reflectpkg "github.com/nexusriot/ai-houkai/internal/reflect"
	"github.com/spf13/cobra"
)

// maintCfg builds a maintenance.Config from the resolved CLI config.
func maintCfg(cfg Config) maintenance.Config {
	interval := time.Duration(cfg.Maintenance.IntervalSecs) * time.Second
	if interval <= 0 {
		interval = time.Hour
	}
	summ, err := reflectpkg.BuildSummarizer(cfg.Summarizer, true)
	if err != nil {
		summ = nil // reflect.New falls back to the extractive summarizer
	}
	consolidate := reflectpkg.ConsolidateNone
	if cfg.Maintenance.Consolidate {
		consolidate = reflectpkg.ConsolidateSoft
	}
	return maintenance.Config{
		Interval:        interval,
		DecayRate:       cfg.Maintenance.Decay.DecayRate,
		MinScore:        cfg.Maintenance.Decay.MinScore,
		Reflect:         cfg.Maintenance.Reflect,
		Consolidate:     consolidate,
		DecayEvery:      float64(cfg.Maintenance.DecayEverySecs),
		ReflectEvery:    float64(cfg.Maintenance.ReflectEverySecs),
		FrequencyWeight: cfg.Maintenance.Decay.FrequencyWeight,
		Summarizer:      summ,
	}
}

// MaintenanceRuntime resolves the maintenance.Config and state path for cfg —
// used by the MCP entry point to wire the schedule-gated maintenance_tick.
func MaintenanceRuntime(cfg Config) (maintenance.Config, string) {
	statePath, _, _ := cfg.MaintPaths()
	return maintCfg(cfg), statePath
}

// tickSummary renders a TickResult as a one-line human summary, mirroring
// Python's TickResult.summary(): gated-out jobs are reported as skipped.
func tickSummary(res maintenance.TickResult) string {
	var parts []string
	if res.RanDecay {
		parts = append(parts, fmt.Sprintf("decay pruned %d", res.Pruned))
	}
	if res.RanReflect {
		parts = append(parts, fmt.Sprintf("reflect created %d", res.Reflected))
	}
	if len(parts) == 0 {
		return "nothing to do (jobs not due yet)"
	}
	return strings.Join(parts, " | ")
}

func fmtTS(ts float64) string {
	if ts <= 0 {
		return "never"
	}
	return time.Unix(int64(ts), 0).Format("2006-01-02 15:04:05")
}

func newMaintenanceCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "maintenance",
		Short: "Scheduled decay + reflection daemon (tick|run|start|stop|status)",
	}
	cmd.AddCommand(
		newMaintTickCmd(),
		newMaintRunCmd(),
		newMaintStartCmd(),
		newMaintStopCmd(),
		newMaintStatusCmd(),
	)
	return cmd
}

func newMaintTickCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "tick",
		Short: "Run one maintenance tick synchronously (cron-friendly)",
		RunE: func(cmd *cobra.Command, _ []string) error {
			store := storeFromCtx(cmd.Context())
			cfg := cfgFromCtx(cmd.Context())
			statePath, _, _ := cfg.MaintPaths()
			res := maintenance.Tick(cmd.Context(), store, store, maintCfg(cfg), statePath, float64(time.Now().Unix()))
			fmt.Printf("Tick complete: %s\n", tickSummary(res))
			return res.Err
		},
	}
}

func newMaintRunCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "run",
		Short: "Run the maintenance scheduler in the foreground (Ctrl-C to stop)",
		RunE: func(cmd *cobra.Command, _ []string) error {
			store := storeFromCtx(cmd.Context())
			cfg := cfgFromCtx(cmd.Context())
			statePath, _, _ := cfg.MaintPaths()
			mc := maintCfg(cfg)

			ctx, stop := signal.NotifyContext(cmd.Context(), syscall.SIGTERM, syscall.SIGINT)
			defer stop()

			log.Printf("maintenance: running every %s (reflect=%v)", mc.Interval, mc.Reflect)
			runOne := func() {
				res := maintenance.Tick(ctx, store, store, mc, statePath, float64(time.Now().Unix()))
				if res.Err != nil {
					log.Printf("maintenance tick error: %v", res.Err)
				} else {
					log.Printf("maintenance tick: %s", tickSummary(res))
				}
			}
			runOne() // one immediate pass on startup
			ticker := time.NewTicker(mc.Interval)
			defer ticker.Stop()
			for {
				select {
				case <-ctx.Done():
					log.Print("maintenance: stopping")
					return nil
				case <-ticker.C:
					runOne()
				}
			}
		},
	}
}

func newMaintStartCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "start",
		Short: "Detach the maintenance daemon into the background",
		RunE: func(cmd *cobra.Command, _ []string) error {
			cfg := cfgFromCtx(cmd.Context())
			_, pidPath, logPath := cfg.MaintPaths()
			if maintenance.IsAlive(pidPath) {
				fmt.Printf("Daemon is already running (pid %d).\n", maintenance.ReadPid(pidPath))
				return nil
			}
			exe, err := os.Executable()
			if err != nil {
				return err
			}
			pid, err := maintenance.SpawnDetached(exe, cfg.StorePath, cfg.Collection, logPath, pidPath)
			if err != nil {
				return err
			}
			fmt.Printf("Maintenance daemon started (pid %d).\n", pid)
			fmt.Printf("Logs → %s\nStop → houkai maintenance stop\n", logPath)
			return nil
		},
	}
}

func newMaintStopCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "stop",
		Short: "Send SIGTERM to the background daemon",
		RunE: func(cmd *cobra.Command, _ []string) error {
			cfg := cfgFromCtx(cmd.Context())
			_, pidPath, _ := cfg.MaintPaths()
			if !maintenance.IsAlive(pidPath) {
				if pid := maintenance.ReadPid(pidPath); pid > 0 {
					fmt.Printf("Daemon (pid %d) is not running. Cleaning up stale pid file.\n", pid)
					maintenance.RemovePid(pidPath)
				} else {
					fmt.Println("No daemon is running.")
				}
				return nil
			}
			pid := maintenance.ReadPid(pidPath)
			if maintenance.StopDaemon(pidPath) {
				fmt.Printf("SIGTERM sent to daemon (pid %d).\n", pid)
				maintenance.RemovePid(pidPath)
			} else {
				fmt.Println("Failed to send signal — daemon may have already exited.")
			}
			return nil
		},
	}
}

func newMaintStatusCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "status",
		Short: "Show daemon state, last runs, and next schedules",
		RunE: func(cmd *cobra.Command, _ []string) error {
			cfg := cfgFromCtx(cmd.Context())
			statePath, pidPath, logPath := cfg.MaintPaths()
			state := maintenance.LoadState(statePath)
			pid := maintenance.ReadPid(pidPath)
			alive := maintenance.IsAlive(pidPath)

			daemon := "stopped"
			if alive {
				daemon = fmt.Sprintf("running (pid %d)", pid)
			} else if pid > 0 {
				daemon = "stopped (stale pid file)"
			}

			line := "────────────────────────────────────────────────────────"
			fmt.Println(line)
			fmt.Printf("  Daemon:          %s\n", daemon)
			fmt.Println(line)
			fmt.Printf("  Last decay:      %s\n", fmtTS(state.LastDecayAt))
			fmt.Printf("  Last reflect:    %s\n", fmtTS(state.LastReflectAt))
			fmt.Printf("  Total decayed:   %d\n", state.TotalDecayed)
			fmt.Printf("  Total reflected: %d\n", state.TotalReflected)
			fmt.Println(line)
			fmt.Printf("  Interval:        %ds\n", cfg.Maintenance.IntervalSecs)
			fmt.Printf("  Reflect:         %v\n", cfg.Maintenance.Reflect)
			fmt.Printf("  State file:      %s\n", statePath)
			fmt.Printf("  Log file:        %s\n", logPath)
			reinf := "off"
			if cfg.Maintenance.Decay.FrequencyWeight != 0 {
				reinf = fmt.Sprintf("on (frequency_weight=%g)", cfg.Maintenance.Decay.FrequencyWeight)
			}
			fmt.Printf("  Reinforcement:   %s\n", reinf)
			return nil
		},
	}
}
