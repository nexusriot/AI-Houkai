# AI-Houkai (Go)

Persistent long-term memory for AI agents — a Go port of the Python
[ai-houkai](../README.md) project. Two static binaries, zero Python runtime,
embeddable on a fresh Debian/Ubuntu box with a single `.deb`.

| Binary           | Purpose                                                |
| ---------------- | ------------------------------------------------------ |
| `ai-houkai-mcp`  | MCP server over stdio for Claude Code / Claude Desktop |
| `houkai`         | CLI for direct terminal use of the memory store        |

Both binaries share the same configuration, on-disk format, and embedder
plumbing — they are two front-ends over the same `MemoryStore`.

---

## Why a Go fork?

The Python version (`ai_houkai/`) is the reference implementation. The Go fork
exists to:

- **Ship a single static binary.** No `pip`, no venv, no ChromaDB native
  install — just `apt install ./ai-houkai_*.deb` and you're done.
- **Reduce footprint.** ~8 MB per binary vs. ~hundreds of MB of Python +
  PyTorch dependencies pulled in by `sentence-transformers`.
- **Run on small boxes.** Targets amd64 and arm64 (Raspberry Pi, uConsole,
  cheap VPS) without compiler toolchains.

Embeddings are delegated to either a local **Ollama** instance or **OpenAI** —
the Go binary itself does no model inference.

---

## Quick start

### Install (Debian/Ubuntu)

```bash
make deb-amd64                          # or deb-arm64
sudo apt install ./dist/ai-houkai_*.deb
ollama pull all-minilm                  # embedding model
```

### Install (macOS — Apple Silicon or Intel)

Cross-compile produces a self-contained `.tar.gz` per arch. Both Apple
Silicon (M1/M2/M3/M4) and Intel are supported:

```bash
make macos-arm64                        # Apple Silicon (M-chips)
make macos-amd64                        # Intel
make macos                              # both

# install (on the Mac)
tar -xzf ai-houkai_*_darwin_arm64.tar.gz
cd ai-houkai_*_darwin_arm64
sudo install -m 0755 bin/* /usr/local/bin/

# if the tarball was downloaded via a browser, strip Gatekeeper quarantine:
xattr -d com.apple.quarantine /usr/local/bin/ai-houkai-mcp /usr/local/bin/houkai

# embedding model + register with Claude Code
brew install ollama && ollama pull all-minilm
houkai install
```

The archive includes the binaries, a sample config under
`share/ai-houkai/config.toml.example`, and a `README.txt` with the same
steps. A `.sha256` file is emitted alongside each tarball for Homebrew
formula authoring.

### Install (any platform)

```bash
make build           # produces ./bin/{ai-houkai-mcp,houkai}
make install-user    # → ~/.local/bin
```

### Register with Claude Code

The easy way:

```bash
houkai install                          # patches ~/.claude/settings.json
houkai install --project                # → ./.claude/settings.json
```

After restarting Claude Code, eleven `mcp__ai-houkai__*` tools become
available.

#### Manual registration

If you'd rather not run the installer (sandboxed environment, you want to
review the change first, or you're managing `settings.json` in dotfiles),
add the block below to `~/.claude/settings.json` by hand. Merge it with any
existing `mcpServers` object — don't overwrite it.

```json
{
  "mcpServers": {
    "ai-houkai": {
      "command": "ai-houkai-mcp",
      "args": [],
      "env": {
        "AI_HOUKAI_PATH": "/home/YOU/.ai_houkai",
        "AI_HOUKAI_COLLECTION": "ai_houkai"
      }
    }
  }
}
```

Notes:

- `command` must resolve on `PATH` from Claude Code's environment. If
  `ai-houkai-mcp` isn't on the default `PATH` (e.g. you installed via
  `make install-user` to `~/.local/bin`), use an absolute path:
  `"command": "/home/YOU/.local/bin/ai-houkai-mcp"`.
- `AI_HOUKAI_PATH` points at the data directory (the store lives under
  `<path>/.chroma`). `AI_HOUKAI_COLLECTION` is the chromem-go collection
  name — keep `ai_houkai` unless you want multiple isolated stores.
- For per-project memory, put the same block in `./.claude/settings.json`
  inside the project root with a project-local `AI_HOUKAI_PATH` (e.g.
  `"$PWD/.ai_houkai"` resolved to absolute) and a distinct collection.
- Extra env vars (`AI_HOUKAI_EMBED_PROVIDER`, `AI_HOUKAI_OLLAMA_URL`,
  `AI_HOUKAI_OLLAMA_MODEL`, `OPENAI_API_KEY`, `AI_HOUKAI_CONFIG`) can be
  added to the same `env` map if you want to override the resolved config
  per-MCP-instance. See [config resolution](#configuration) for the full
  list.
- Restart Claude Code (or run `/mcp` and re-add the server) for it to pick
  up the change. To verify the binary works in isolation, run
  `echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | ai-houkai-mcp`
  — you should see the eleven tool definitions on stdout.

To print the exact block the installer would write without touching any
file, run `houkai install --settings /dev/null` then copy from the script,
or read [`internal/installer/claude_code.go`](internal/installer/claude_code.go).

---

## CLI overview

`houkai --help` lists 23 subcommands. Most-used:

```
remember <text>        Store a new memory
recall <query>         Semantic / hybrid search
list                   Browse recent memories
show <id>              Full record (8-char prefix OK)
edit <id>              Open in $EDITOR
forget <id>            Delete
tag / bump             Edit tags / importance
link / unlink          Build the knowledge graph
neighbors / graph      Traverse the graph
conflicts              Detect contradictions/duplicates
supersede / restore    Soft-delete + revive
prune                  Decay-based pruning (dry-run by default)
reflect                Cluster episodics into semantic memories
export / import        JSONL round-trip
backup                 Snapshot the chromem-go directory
stats                  Aggregate counters
config                 Show resolved config + search order
install                Register MCP server in Claude Code
```

All commands accept `--store`, `--collection`, and `--format auto|json|tsv`.

---

## CLI cookbook

### Capturing memories

```bash
# simplest — defaults to type=episodic, importance=0.5
houkai remember "Vlad prefers tabs over spaces in Go"

# tagged + higher importance
houkai remember "deploy script lives at ops/deploy.sh" \
    -t devops -t infra -i 0.8 --type procedural

# from stdin (so you can pipe long content)
git log -1 --format=%B | houkai remember -t commit --source git

# with provenance
houkai remember "user said the spinner blocks the UI" \
    --source slack --type feedback -i 0.9
```

### Searching

```bash
# semantic recall — top 5 by default
houkai recall "how to deploy"

# narrow to a tag, raise the limit
houkai recall "rate limit" -k 20 --tag api

# hybrid mode (cosine + BM25 + recency + importance)
houkai recall "auth bug" --mode hybrid

# JSON output for piping into jq
houkai --format json recall "deploy" -k 3 | jq '.[].text'

# only high-importance memories
houkai recall "decisions" --min-importance 0.7
```

### Browsing and inspecting

```bash
# 20 most recent
houkai list

# filter by type + tag, include soft-deleted
houkai list --type semantic --tag api --include-superseded --limit 50

# full record (8-char prefix is enough if unambiguous)
houkai show a1b2c3d4

# pretty-print via fzf for interactive picking
houkai --format tsv list --limit 200 | fzf | cut -f1 | xargs houkai show
```

### Editing in place

```bash
# launches $EDITOR with the memory text; saves + re-embeds on exit
houkai edit a1b2c3d4

# add/remove tags without re-embedding
houkai tag a1b2c3d4 --add backend,critical --remove draft

# nudge importance
houkai bump a1b2c3d4 0.9
```

### The knowledge graph

```bash
# link two memories with a typed relation
houkai link a1b2c3d4 e5f6g7h8 --rel refines

# walk 2 hops outwards
houkai neighbors a1b2c3d4 --depth 2 --direction out

# only follow one relation type
houkai neighbors a1b2c3d4 --rel derived_from

# dump the subgraph as JSON for visualisation
houkai graph a1b2c3d4 e5f6g7h8 --depth 2 > graph.json

# remove a specific link (or all links if --rel omitted)
houkai unlink a1b2c3d4 e5f6g7h8 --rel related
```

Relation vocabulary: `supersedes`, `refines`, `derived_from`, `example_of`,
`contradicts`, `related`.

### Conflicts and supersession

```bash
# scan the whole store for contradictions/duplicates
houkai conflicts --threshold 0.85

# check a specific memory
houkai conflicts a1b2c3d4

# replace one memory with another (soft-delete; reversible)
houkai supersede a1b2c3d4 e5f6g7h8

# bring a superseded memory back
houkai restore a1b2c3d4
```

### Maintenance — pruning and reflection

```bash
# dry-run: show what decay would remove
houkai prune

# actually delete
houkai prune --apply

# tune the decay
houkai prune --decay-rate 0.05 --min-score 0.02 --apply

# cluster episodic memories into semantic summaries (dry-run)
houkai reflect

# create the summaries for real
houkai reflect --apply

# create summaries AND delete the source episodics
houkai reflect --apply --consolidate

# tighter clustering
houkai reflect --threshold 0.85 --min-cluster 3 --apply
```

### Backup, export, import

```bash
# binary snapshot of the chromem-go directory
houkai backup
# → ~/.ai_houkai/backups/20260524T120415/

# portable JSONL export (works across stores + Python ↔ Go)
houkai export -o memories.jsonl

# pipe straight to import on another host
houkai export | ssh other-host "houkai import"

# import into a fresh store
houkai import < memories.jsonl
```

### Stats and config introspection

```bash
# counts, by-type breakdown, top tags, avg importance
houkai stats

# see what config the binary actually resolved (and from where)
houkai config
```

### Multiple stores / projects

```bash
# pin a specific store + collection (overrides config + env)
houkai --store /srv/team-memory --collection backend list

# project-local memory via env vars
AI_HOUKAI_PATH="$PWD/.ai_houkai" AI_HOUKAI_COLLECTION=this-project \
    houkai remember "TODO: drop legacy v1 endpoint"
```

A handy shell function for per-project memory:

```bash
# ~/.bashrc
houkai-here() {
    AI_HOUKAI_PATH="$PWD/.ai_houkai" \
    AI_HOUKAI_COLLECTION="$(basename "$PWD")" \
    houkai "$@"
}
# usage:  houkai-here recall "the api thing"
```

### Switching embedding providers ad hoc

```bash
# one-off recall against OpenAI without touching config files
AI_HOUKAI_EMBED_PROVIDER=openai OPENAI_API_KEY=sk-... houkai recall "x"

# same for DigitalOcean
AI_HOUKAI_EMBED_PROVIDER=digitalocean DIGITALOCEAN_TOKEN=dop_v1_... \
    houkai recall "x"
```

(Caveat: a store built with one provider's embeddings shouldn't be queried
with another's — the cosine scores will be meaningless. Use this only for
fresh stores or after a re-embed via `export` → wipe → `import`.)

### Piping and scripting

```bash
# top 3 memories about "auth" as a bash array
mapfile -t hits < <(houkai --format tsv recall auth -k 3 | cut -f1)
for id in "${hits[@]}"; do houkai bump "$id" 0.9; done

# every memory tagged 'todo', oldest first
houkai --format json list --tag todo --limit 1000 \
  | jq -r 'sort_by(.created_at) | .[] | "\(.id)\t\(.text)"'

# bulk-tag everything from a particular source
houkai --format json list --limit 1000 \
  | jq -r '.[] | select(.source=="slack") | .id' \
  | xargs -I{} houkai tag {} --add from-slack
```

### Claude Code integration

```bash
# user-scope (default)
houkai install

# project-scope, with a project-local store
houkai install --project --memory-path "$PWD/.ai_houkai"

# point at a non-PATH binary
houkai install --binary /opt/ai-houkai/ai-houkai-mcp

# write to a custom settings file (useful for testing)
houkai install --settings /tmp/settings.json
```

---

## Configuration

Resolution order (later wins):

1. Built-in defaults
2. `/etc/ai-houkai/config.toml` (shipped as a dpkg conffile)
3. `~/.config/ai_houkai/config.toml`
4. `$AI_HOUKAI_CONFIG` (explicit override)
5. `AI_HOUKAI_*` env vars
6. CLI flags

```toml
collection         = "ai_houkai"
default_importance = 0.5
embed_provider     = "ollama"        # "ollama" | "openai" | "digitalocean"
embed_dim          = 384             # 384/768/1024/1536 — must match the model
ollama_url         = "http://localhost:11434"
ollama_model       = "all-minilm"
openai_model       = "text-embedding-3-small"
do_model           = "qwen3-embedding-0.6b"
```

### Embedding providers

| Provider       | Default model              | Dim   | Auth env var          | Local? |
| -------------- | -------------------------- | ----- | --------------------- | ------ |
| `ollama`       | `all-minilm`               | 384   | —                     | yes    |
| `openai`       | `text-embedding-3-small`   | 1536  | `OPENAI_API_KEY`      | no     |
| `digitalocean` | `qwen3-embedding-0.6b`     | 1024  | `DIGITALOCEAN_TOKEN`  | no     |

DigitalOcean's [Serverless Inference](https://docs.digitalocean.com/reference/api/reference/embeddings/index.html.md)
is wire-compatible with OpenAI's `/v1/embeddings` — it's a thin wrapper over
the same client code, just pointed at `https://inference.do-ai.run`. To use
it:

```bash
export DIGITALOCEAN_TOKEN="dop_v1_..."
# either:
houkai --format json config | grep -i embed         # check
AI_HOUKAI_EMBED_PROVIDER=digitalocean houkai recall "test"
# or set it persistently:
cat >> ~/.config/ai_houkai/config.toml <<EOF
embed_provider = "digitalocean"
embed_dim      = 1024
do_model       = "qwen3-embedding-0.6b"
EOF
```

**Important:** `embed_dim` must match the chosen model — switching providers
on an existing store will produce nonsense scores until you re-embed
(`export` → wipe → `import`).

Run `houkai config` to see what was actually resolved.

---

## Layout

```
cmd/
  ai-houkai-mcp/    MCP server entry point (stdio)
  houkai/           CLI entry point
internal/
  memory/           MemoryStore, hybrid scoring, BM25, conflicts, links
  vector/           Backend interface + chromem-go implementation
  embed/            Embedder interface + Ollama + OpenAI clients
  decay/            Time-based pruning engine
  reflect/          Episodic → semantic clustering
  maintenance/      Background daemon (prune + reflect on a ticker)
  mcpserver/        11 MCP tool definitions
  cli/              cobra commands, config resolver, output formatting
  installer/        settings.json patcher for Claude Code
  version/          ldflags-injected build info
scripts/            build.sh, build-deb.sh
packaging/          conffile defaults, systemd unit, postinst hints
```

Source files are kept small and unit-testable; see [DESIGN.md](DESIGN.md) for
architecture details.

---

## Building

```bash
make build         # current host
make release       # cross-compile for linux/{amd64,arm64} + darwin/{amd64,arm64}
make deb           # Debian/Ubuntu .deb for both Linux arches
make macos         # macOS .tar.gz for Apple Silicon + Intel
make test          # go test ./...
make vet           # go vet ./...
```

Per-arch shortcuts: `make deb-amd64`, `make deb-arm64`, `make macos-arm64`,
`make macos-amd64`.

Version is derived from `git describe --tags --always --dirty` and injected at
link time. Override with `VERSION=1.2.3 make build`.

---

## Compatibility with the Python version

The on-disk store is **not** binary-compatible: the Python version uses a
ChromaDB SQLite store, the Go version uses
[`chromem-go`](https://github.com/philippgille/chromem-go)'s persistent format.
Use `houkai export` / `import` (JSONL) to migrate between them.

The MCP tool surface is similar but **trimmed**: 11 Go tools vs. 14 Python
tools (no separate `update`/`get_by_id`/`subgraph` MCP endpoints — those are
CLI-only). Tool names match (`remember`, `recall`, `forget`, …) so clients
that don't depend on the missing tools work unchanged.

---

## License

MIT. See [LICENSE](../LICENSE) at the repo root.
