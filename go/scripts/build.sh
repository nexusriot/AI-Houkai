#!/usr/bin/env bash
# Build ai-houkai-mcp and houkai for all target platforms.
#
# Usage:
#   ./scripts/build.sh                 # all targets
#   ./scripts/build.sh linux/amd64     # single target
#   VERSION=1.0.0 ./scripts/build.sh   # override version

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

VERSION="${VERSION:-$(git -C "$REPO_ROOT" describe --tags --always --dirty 2>/dev/null || echo "dev")}"
BUILD_TIME="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
LDFLAGS="-s -w -X github.com/nexusriot/ai-houkai/internal/version.Version=${VERSION} -X github.com/nexusriot/ai-houkai/internal/version.BuildTime=${BUILD_TIME}"

# Targets: GOOS/GOARCH
TARGETS=(
    "linux/amd64"
    "linux/arm64"
    "darwin/amd64"
    "darwin/arm64"
)

# If a single target was passed, use only that.
if [[ $# -ge 1 ]]; then
    TARGETS=("$1")
fi

DIST="$REPO_ROOT/dist"
mkdir -p "$DIST"

echo "Building ai-houkai ${VERSION}"
echo "---"

for target in "${TARGETS[@]}"; do
    GOOS="${target%%/*}"
    GOARCH="${target##*/}"
    suffix=""
    [[ "$GOOS" == "windows" ]] && suffix=".exe"

    out_dir="$DIST/${GOOS}_${GOARCH}"
    mkdir -p "$out_dir"

    echo -n "  ${GOOS}/${GOARCH} ... "

    GOOS="$GOOS" GOARCH="$GOARCH" CGO_ENABLED=0 \
        go build -trimpath -ldflags "$LDFLAGS" \
        -o "$out_dir/ai-houkai-mcp${suffix}" \
        ./cmd/ai-houkai-mcp

    GOOS="$GOOS" GOARCH="$GOARCH" CGO_ENABLED=0 \
        go build -trimpath -ldflags "$LDFLAGS" \
        -o "$out_dir/ai-houkai-serve${suffix}" \
        ./cmd/ai-houkai-serve

    GOOS="$GOOS" GOARCH="$GOARCH" CGO_ENABLED=0 \
        go build -trimpath -ldflags "$LDFLAGS" \
        -o "$out_dir/houkai${suffix}" \
        ./cmd/houkai

    sizes=$(du -sh "$out_dir"/* | awk '{printf "%s %s  ", $1, $2}')
    echo "OK  (${sizes% })"
done

echo "---"
echo "Artifacts in: $DIST/"
ls -1 "$DIST"/
