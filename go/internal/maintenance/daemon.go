package maintenance

import (
	"context"
	"log"
	"time"

	"github.com/nexusriot/ai-houkai/internal/decay"
	reflectpkg "github.com/nexusriot/ai-houkai/internal/reflect"
)

// Config controls the background maintenance daemon.
type Config struct {
	Interval    time.Duration
	DecayRate   float32
	MinScore    float32
	Reflect     bool
	Consolidate bool

	// FrequencyWeight > 0 makes frequently-recalled memories resist decay
	// (forwarded to decay.Engine; 0 = recency-only, the default).
	FrequencyWeight float32

	// Summarizer is forwarded to the reflection engine (e.g. from
	// reflect.BuildSummarizer("ollama:llama3.1")). Nil → the built-in
	// extractive summarizer.
	Summarizer reflectpkg.Summarizer
}

func DefaultConfig() Config {
	return Config{
		Interval:  time.Hour,
		DecayRate: 0.1,
		MinScore:  0.05,
		Reflect:   false,
	}
}

// Start launches a background goroutine that periodically prunes and reflects.
// Cancel ctx to stop it.
func Start(ctx context.Context, store decay.Storable, reflStore reflectpkg.Storable, cfg Config) {
	go func() {
		ticker := time.NewTicker(cfg.Interval)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				runTick(ctx, store, reflStore, cfg)
			}
		}
	}()
}

func runTick(ctx context.Context, store decay.Storable, reflStore reflectpkg.Storable, cfg Config) {
	de := decay.New(store, cfg.DecayRate, cfg.MinScore, nil, cfg.FrequencyWeight)
	pruned, err := de.Prune(ctx, false)
	if err != nil {
		log.Printf("maintenance prune: %v", err)
	} else {
		log.Printf("maintenance: pruned %d memories", len(pruned))
	}

	if cfg.Reflect {
		re := reflectpkg.New(reflStore, 0, 0, cfg.Summarizer)
		mode := reflectpkg.ConsolidateNone
		if cfg.Consolidate {
			mode = reflectpkg.ConsolidateSoft
		}
		created, err := re.Reflect(ctx, false, mode)
		if err != nil {
			log.Printf("maintenance reflect: %v", err)
		} else {
			log.Printf("maintenance: created %d reflection memories", len(created))
		}
	}
}
