package version

// Version and BuildTime are injected at link time via -ldflags. The
// defaults below are the in-source fallback for `go run` / `go build`
// without -ldflags; the Makefile / release scripts override both.
var (
	Version   = "0.10.0"
	BuildTime = "unknown"
)
