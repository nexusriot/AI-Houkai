package version

// Version and BuildTime are injected at link time via -ldflags.
var (
	Version   = "dev"
	BuildTime = "unknown"
)
