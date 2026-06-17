#!/usr/bin/env bash
# Build Debian packages (.deb) for ai-houkai.
#
# Prerequisites: dpkg-deb, fakeroot  (both available on any Debian/Ubuntu host)
#
# Usage:
#   ./scripts/build-deb.sh                         # build for amd64 + arm64
#   ./scripts/build-deb.sh amd64                   # single arch
#   VERSION=1.2.3 ./scripts/build-deb.sh           # override version
#
# The script first calls build.sh to produce compiled binaries, then packages
# them into .deb files under dist/.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# --------------------------------------------------------------------------- #
# Resolve version
# --------------------------------------------------------------------------- #
VERSION="${VERSION:-$(git -C "$REPO_ROOT" describe --tags --always --dirty 2>/dev/null || echo "0.0.0")}"
# Strip leading 'v' for dpkg (dpkg requires: epoch:upstream_version[-debian_revision])
DEB_VERSION="${VERSION#v}"
# Replace any non-numeric/dot/tilde chars with ~  (e.g. git "0.3.4-dirty")
# Note: tilde in the replacement of ${var//pat/repl} undergoes tilde expansion,
# so escape it to keep a literal '~'.
DEB_VERSION="${DEB_VERSION//-/\~}"

MAINTAINER="${MAINTAINER:-Vlad Ananyev <vlananyev@gmail.com>}"
DESCRIPTION_SHORT="Long-term memory system for AI agents"
DESCRIPTION_LONG=" AI-Houkai gives LLM agents persistent, searchable memory that survives
 across sessions — without requiring cloud services or API keys.
 .
 Includes:
  * ai-houkai-mcp   — MCP server (Claude Code / Claude Desktop integration)
  * houkai          — CLI for direct memory management"

HOMEPAGE="https://github.com/nexusriot/AI-Houkai"

# Debian arch names differ from Go arch names.
declare -A GOARCH_TO_DEB=(
    ["amd64"]="amd64"
    ["arm64"]="arm64"
)
declare -A GOARCH_TO_GOARCH=(
    ["amd64"]="amd64"
    ["arm64"]="arm64"
)

# Resolve requested architectures.
if [[ $# -ge 1 ]]; then
    DEB_ARCHS=("$1")
else
    DEB_ARCHS=("amd64" "arm64")
fi

DIST="$REPO_ROOT/dist"
mkdir -p "$DIST"

# --------------------------------------------------------------------------- #
# Build binaries for all requested archs
# --------------------------------------------------------------------------- #
echo "Building binaries for: ${DEB_ARCHS[*]}"
for deb_arch in "${DEB_ARCHS[@]}"; do
    go_arch="${GOARCH_TO_GOARCH[$deb_arch]}"
    VERSION="$VERSION" bash "$SCRIPT_DIR/build.sh" "linux/${go_arch}"
done

# --------------------------------------------------------------------------- #
# Package each arch
# --------------------------------------------------------------------------- #
for deb_arch in "${DEB_ARCHS[@]}"; do
    go_arch="${GOARCH_TO_GOARCH[$deb_arch]}"
    bin_dir="$DIST/linux_${go_arch}"

    pkg_name="ai-houkai_${DEB_VERSION}_${deb_arch}"
    pkg_root="$DIST/${pkg_name}"

    echo ""
    echo "Packaging ${pkg_name}.deb ..."

    # ---- directory tree -------------------------------------------------- #
    rm -rf "$pkg_root"
    install -d "$pkg_root/DEBIAN"
    install -d "$pkg_root/usr/bin"
    install -d "$pkg_root/usr/share/doc/ai-houkai"
    install -d "$pkg_root/lib/systemd/system"
    install -d "$pkg_root/etc/ai-houkai"

    # ---- binaries --------------------------------------------------------- #
    install -m 0755 "$bin_dir/ai-houkai-mcp"   "$pkg_root/usr/bin/ai-houkai-mcp"
    install -m 0755 "$bin_dir/ai-houkai-serve" "$pkg_root/usr/bin/ai-houkai-serve"
    install -m 0755 "$bin_dir/houkai"          "$pkg_root/usr/bin/houkai"

    # ---- default config (marked as conffile below) ----------------------- #
    install -m 0644 "$REPO_ROOT/packaging/etc/ai-houkai/config.toml" \
        "$pkg_root/etc/ai-houkai/config.toml"

    # Tell dpkg to treat this file as user-editable configuration.
    cat > "$pkg_root/DEBIAN/conffiles" <<EOF
/etc/ai-houkai/config.toml
EOF

    # ---- control ---------------------------------------------------------- #
    installed_size=$(du -sk "$pkg_root/usr" | cut -f1)
    cat > "$pkg_root/DEBIAN/control" <<EOF
Package: ai-houkai
Version: ${DEB_VERSION}
Architecture: ${deb_arch}
Maintainer: ${MAINTAINER}
Installed-Size: ${installed_size}
Homepage: ${HOMEPAGE}
Section: utils
Priority: optional
Description: ${DESCRIPTION_SHORT}
${DESCRIPTION_LONG}
EOF

    # ---- copyright -------------------------------------------------------- #
    cat > "$pkg_root/usr/share/doc/ai-houkai/copyright" <<EOF
Format: https://www.debian.org/doc/packaging-manuals/copyright-format/1.0/
Upstream-Name: ai-houkai
Upstream-Contact: ${MAINTAINER}
Source: ${HOMEPAGE}

Files: *
Copyright: $(date +%Y) ${MAINTAINER}
License: MIT
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 .
 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.
 .
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
EOF
    gzip -9 -n -c "$pkg_root/usr/share/doc/ai-houkai/copyright" \
        > "$pkg_root/usr/share/doc/ai-houkai/copyright.gz" 2>/dev/null || true

    # ---- changelog -------------------------------------------------------- #
    cat > "$pkg_root/usr/share/doc/ai-houkai/changelog.Debian" <<EOF
ai-houkai (${DEB_VERSION}) unstable; urgency=low

  * Release ${VERSION}

 -- ${MAINTAINER}  $(date -R)
EOF
    gzip -9 -n "$pkg_root/usr/share/doc/ai-houkai/changelog.Debian"

    # ---- systemd unit for the MCP server (optional, user can enable) ------ #
    cat > "$pkg_root/lib/systemd/system/ai-houkai-mcp.service" <<EOF
[Unit]
Description=AI-Houkai MCP memory server
After=network.target

[Service]
Type=simple
ExecStart=/usr/bin/ai-houkai-mcp
Restart=on-failure
RestartSec=5
Environment=AI_HOUKAI_PATH=%h/.ai_houkai
Environment=AI_HOUKAI_COLLECTION=ai_houkai

[Install]
WantedBy=multi-user.target
EOF

    # ---- postinst --------------------------------------------------------- #
    cat > "$pkg_root/DEBIAN/postinst" <<'EOF'
#!/bin/sh
set -e
if [ "$1" = "configure" ]; then
    echo "ai-houkai: system config at /etc/ai-houkai/config.toml"
    echo "  Per-user override: ~/.config/ai_houkai/config.toml"

    if command -v ollama >/dev/null 2>&1; then
        echo "ai-houkai: pulling required Ollama model (all-minilm) ..."
        ollama pull all-minilm 2>/dev/null || \
            echo "  Note: 'ollama pull all-minilm' failed — run it manually before use."
    else
        echo "ai-houkai: Ollama not found."
        echo "  Install Ollama (https://ollama.com) and run: ollama pull all-minilm,"
        echo "  or switch to a cloud provider by editing /etc/ai-houkai/config.toml:"
        echo "    embed_provider = \"openai\"        # OPENAI_API_KEY"
        echo "    embed_provider = \"digitalocean\"  # DIGITALOCEAN_TOKEN"
    fi
fi
EOF
    chmod 0755 "$pkg_root/DEBIAN/postinst"

    # ---- postrm ----------------------------------------------------------- #
    cat > "$pkg_root/DEBIAN/postrm" <<'EOF'
#!/bin/sh
set -e
# Nothing to clean up automatically; user data lives in ~/.ai_houkai
EOF
    chmod 0755 "$pkg_root/DEBIAN/postrm"

    # ---- build the .deb --------------------------------------------------- #
    deb_file="$DIST/${pkg_name}.deb"
    fakeroot dpkg-deb --build --root-owner-group "$pkg_root" "$deb_file"
    rm -rf "$pkg_root"

    size=$(du -sh "$deb_file" | cut -f1)
    echo "  -> $deb_file  (${size})"

    # Sanity check.
    echo "  Contents:"
    dpkg-deb --contents "$deb_file" | awk '{print "    " $NF}' | grep -v '/$'
done

echo ""
echo "Done. Packages:"
ls -lh "$DIST"/*.deb 2>/dev/null
