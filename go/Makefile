VERSION  ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")
BIN_DIR  := bin
DIST_DIR := dist

LDFLAGS  := -s -w \
	-X github.com/nexusriot/ai-houkai/internal/version.Version=$(VERSION) \
	-X github.com/nexusriot/ai-houkai/internal/version.BuildTime=$(shell date -u +%Y-%m-%dT%H:%M:%SZ)

.PHONY: build release deb deb-amd64 deb-arm64 macos macos-arm64 macos-amd64 test vet clean install install-user help

## build: compile for the current host
build:
	go build -trimpath -ldflags "$(LDFLAGS)" -o $(BIN_DIR)/ai-houkai-mcp ./cmd/ai-houkai-mcp
	go build -trimpath -ldflags "$(LDFLAGS)" -o $(BIN_DIR)/ai-houkai-serve ./cmd/ai-houkai-serve
	go build -trimpath -ldflags "$(LDFLAGS)" -o $(BIN_DIR)/houkai ./cmd/houkai

## release: cross-compile for linux/{amd64,arm64} and darwin/{amd64,arm64}
release:
	VERSION=$(VERSION) bash scripts/build.sh

## deb: build .deb packages for amd64 and arm64
deb:
	VERSION=$(VERSION) bash scripts/build-deb.sh

## deb-amd64: build .deb package for amd64 only
deb-amd64:
	VERSION=$(VERSION) bash scripts/build-deb.sh amd64

## deb-arm64: build .deb package for arm64 only
deb-arm64:
	VERSION=$(VERSION) bash scripts/build-deb.sh arm64

## macos: build macOS tarballs for arm64 (Apple Silicon) + amd64 (Intel)
macos:
	VERSION=$(VERSION) bash scripts/build-macos.sh

## macos-arm64: build macOS tarball for Apple Silicon (M1/M2/M3/M4) only
macos-arm64:
	VERSION=$(VERSION) bash scripts/build-macos.sh arm64

## macos-amd64: build macOS tarball for Intel only
macos-amd64:
	VERSION=$(VERSION) bash scripts/build-macos.sh amd64

## test: run all tests
test:
	go test ./...

## vet: run go vet
vet:
	go vet ./...

## install: install binaries system-wide (requires sudo)
install: build
	install -m 0755 $(BIN_DIR)/ai-houkai-mcp /usr/local/bin/
	install -m 0755 $(BIN_DIR)/ai-houkai-serve /usr/local/bin/
	install -m 0755 $(BIN_DIR)/houkai /usr/local/bin/

## install-user: install binaries to ~/.local/bin
install-user: build
	mkdir -p $(HOME)/.local/bin
	install -m 0755 $(BIN_DIR)/ai-houkai-mcp $(HOME)/.local/bin/
	install -m 0755 $(BIN_DIR)/ai-houkai-serve $(HOME)/.local/bin/
	install -m 0755 $(BIN_DIR)/houkai $(HOME)/.local/bin/

## clean: remove build artefacts
clean:
	rm -rf $(BIN_DIR) $(DIST_DIR)

help:
	@grep -E '^## ' Makefile | sed 's/^## //'
