#!/usr/bin/env bash
# Build macOS tarballs (.tar.gz) for ai-houkai.
#
# Produces self-contained archives suitable for manual install or Homebrew
# tap distribution. No code signing / notarisation is performed — users will
# need to run `xattr -d com.apple.quarantine` on the binaries if downloaded
# via a browser (Gatekeeper attaches the quarantine bit on download).
#
# Usage:
#   ./scripts/build-macos.sh                  # both arches (amd64 + arm64)
#   ./scripts/build-macos.sh arm64            # Apple Silicon only
#   ./scripts/build-macos.sh amd64            # Intel only
#   VERSION=1.2.3 ./scripts/build-macos.sh    # override version
#
# Cross-compiles cleanly from Linux — CGO is disabled by build.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

VERSION="${VERSION:-$(git -C "$REPO_ROOT" describe --tags --always --dirty 2>/dev/null || echo "0.0.0")}"
PKG_VERSION="${VERSION#v}"

# Requested architectures.
if [[ $# -ge 1 ]]; then
    ARCHS=("$1")
else
    ARCHS=("amd64" "arm64")
fi

DIST="$REPO_ROOT/dist"
mkdir -p "$DIST"

# --------------------------------------------------------------------------- #
# Build binaries for all requested archs
# --------------------------------------------------------------------------- #
echo "Building macOS binaries for: ${ARCHS[*]}"
for arch in "${ARCHS[@]}"; do
    VERSION="$VERSION" bash "$SCRIPT_DIR/build.sh" "darwin/${arch}"
done

# --------------------------------------------------------------------------- #
# Package each arch as a .tar.gz
# --------------------------------------------------------------------------- #
for arch in "${ARCHS[@]}"; do
    bin_dir="$DIST/darwin_${arch}"
    pretty_arch="$arch"
    [[ "$arch" == "arm64" ]] && pretty_arch="arm64"   # Apple Silicon
    [[ "$arch" == "amd64" ]] && pretty_arch="x86_64"  # Intel — Homebrew naming

    pkg_name="ai-houkai_${PKG_VERSION}_darwin_${pretty_arch}"
    stage="$DIST/${pkg_name}"

    echo ""
    echo "Packaging ${pkg_name}.tar.gz ..."

    rm -rf "$stage"
    install -d "$stage/bin"
    install -d "$stage/share/ai-houkai"

    install -m 0755 "$bin_dir/ai-houkai-mcp" "$stage/bin/ai-houkai-mcp"
    install -m 0755 "$bin_dir/houkai"        "$stage/bin/houkai"
    install -m 0644 "$REPO_ROOT/packaging/etc/ai-houkai/config.toml" \
        "$stage/share/ai-houkai/config.toml.example"

    # Lightweight README inside the tarball.
    cat > "$stage/README.txt" <<EOF
ai-houkai ${PKG_VERSION} — macOS (${pretty_arch})

Install:
  1. Copy bin/ai-houkai-mcp and bin/houkai into your PATH, e.g.:
         sudo install -m 0755 bin/* /usr/local/bin/

  2. If you downloaded this tarball via a browser, macOS Gatekeeper will
     have attached a quarantine attribute. Strip it before first run:
         xattr -d com.apple.quarantine /usr/local/bin/ai-houkai-mcp /usr/local/bin/houkai

  3. (Optional) Copy share/ai-houkai/config.toml.example to:
         /usr/local/etc/ai-houkai/config.toml      (system-wide), or
         ~/.config/ai_houkai/config.toml           (per-user)

  4. Install Ollama (https://ollama.com) and pull the embedding model:
         ollama pull all-minilm
     ...or switch to OpenAI / DigitalOcean in the config.

  5. Register with an MCP client:
         houkai install            # Claude Code
         houkai install cursor     # Cursor
         houkai install opencode   # OpenCode

Homepage: https://github.com/nexusriot/AI-Houkai
EOF

    deb_file="$DIST/${pkg_name}.tar.gz"
    tar -C "$DIST" \
        --owner=0 --group=0 \
        -czf "$deb_file" \
        "${pkg_name}"
    rm -rf "$stage"

    size=$(du -sh "$deb_file" | cut -f1)
    echo "  -> $deb_file  (${size})"

    # SHA-256 alongside the artefact for Homebrew formulae.
    if command -v shasum >/dev/null 2>&1; then
        (cd "$DIST" && shasum -a 256 "${pkg_name}.tar.gz" > "${pkg_name}.tar.gz.sha256")
    elif command -v sha256sum >/dev/null 2>&1; then
        (cd "$DIST" && sha256sum "${pkg_name}.tar.gz" > "${pkg_name}.tar.gz.sha256")
    fi
done

echo ""
echo "Done. Archives:"
ls -lh "$DIST"/*.tar.gz 2>/dev/null
