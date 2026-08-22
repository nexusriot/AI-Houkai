# AI-Houkai — Roadmap & Recommendations

Forward-looking feature recommendations for AI-Houkai, grounded in a deep read
of the current codebase (Python canonical + Go port) and a competitive scan of
comparable agent-memory systems (Mem0, Zep/Graphiti, Letta/MemGPT, LangMem,
cognee).

This is a **living backlog**, not a commitment. Every item was checked against
the shipped code to avoid re-proposing existing behaviour; see
[DESIGN.md](DESIGN.md) for what already exists.

## Guiding constraints

Any item below is expected to honour the same three constraints that shaped the
original [PROPOSALS.md](../PROPOSALS.md):

- **No new infra by default** — stay on ChromaDB + sentence-transformers +
  stdlib. New dependencies must be optional and justified.
- **Backwards compatible** — old stores load unchanged; new metadata fields
  default to neutral values.
- **Off by default** — anything that changes ranking or write behaviour is
  opt-in, and the Python and Go ports stay at parity.

Effort key: **S** ≈ ½–1 day · **M** ≈ 2–4 days · **L** ≈ ≥1 week (roughly
doubled by the two-port parity requirement).

---

## ✅ Shipped

### This cycle (rationale for each lives in the matching [DESIGN.md](DESIGN.md) section)

| Feature | Notes |
|---|---|
| **Full MCP ranking surface** | `recall`/`recall_pack`/`auto_context` now expose `fusion`, `diversity`, `dedup_threshold`, `min_cosine`, `graph`, `lexical_index`, `min_trust`, `touch`, `header` and flat `expand_*` knobs in **both** ports. Three cycles of retrieval work had been invisible to MCP clients. |
| **Public `get()`** | `_get_by_id` promoted (alias kept), plus `AsyncMemoryStore.get` and an MCP `get` tool. |
| **Surface coverage** | MCP 23→41 tools, HTTP 24→41 routes, CLI 30→39 commands. `undo` was the sharpest gap: the append-only journal is the project's differentiator and undo was reachable only from a local shell. |
| **Pluggable embedder** (§24) | `embedding_function=` seam + `AI_HOUKAI_EMBEDDER`, stdlib OpenAI-compatible/Ollama backends, `sentence-transformers` moved to a `[local]` extra, `ai_houkai.testing.FakeEmbedder`. |
| **CI + enforced parity** (§23) | `.github/workflows/{ci,release}.yml`; `parity.json` asserted by both ports; a fast suite that needs no torch. |
| **Eval harness wired** (§28) | `houkai eval`, the `eval_recall` MCP tool, and a Go port. Ranking constants are now measurable. |
| **Full-corpus lexical recall** (§25) | `lexical_index="corpus"` via Chroma's `where_document`, plus a Chroma-native range query for `purge_expired`. A SQLite FTS index was measured against it and rejected — see §25 for the numbers and the trade. |
| **Curation + trash** (§26) | `merge`/`versions`/tag ops/`find_path` graduated out of ai-houkai-service; recoverable delete between `supersede` and `forget`, with decay pruning routed through it. |
| **Tiered reflection** | `ReflectionEngine(types=…)` instead of episodic-only, with a `level` tag and `max_level` guard so reflections-of-reflections form a hierarchy. |
| **Pinned / trust / idempotent** (§27) | A standing-instruction slot, a provenance tier, and content-hash dedupe on write. |

### Earlier cycles

| Feature | Notes |
|---|---|
| **Graph-proximity fusion** (`HybridWeights.graph`) | PPR-lite spread over intra-pool links, fused into both weighted and RRF scoring; `graph=0.0` default is a byte-for-byte no-op. |
| **Gated graph expansion** (`ExpandSpec.rerank`) | Expanded neighbours can be merged into the pool *before* `min_cosine`/dedup/MMR/top-k. |
| **`houkai doctor` + `GET /ready`** | Active embedder probe (latency + dim), embed-dim guardrail, store/journal checks. |
| **Batch write & embed** (`remember_many`) | `ceil(N/batch_size)` encode passes instead of N. |
| **Metrics** | Real p50/p95/p99 percentiles over a bounded sample ring, plus counters on every mutator. Only the Prometheus exposition format and per-stage latency remain (below). |

---

## Tier 1 — highest leverage

### 1. Multi-tenant scope metadata (`owner_id` / `agent_id` / `session_id`) · value 5 · fit 5 · M
Today isolation exists only by using N separate collections. Add three optional
string metadata fields (default `""` = global), thread them through
`remember`/`edit`/`ingest` and the `recall*` filters, and push them into the
`_build_where` machinery exactly as `type`/`source` already are. One shared HNSW
index then serves many users/agents (Mem0-style scoping) — and it unlocks
per-scope quotas, stats, GDPR erasure, and agent hand-off.
- **Watch-out:** Chroma equality won't match legacy rows lacking the key —
  backfill once or filter client-side (as the Go port already does); keep both
  ports' "missing key = unscoped" semantics identical.
- **Sequencing:** worth pairing with Tier 1 #2 — scoping multiplies whatever
  scans remain, so denormalising the link edges first keeps the cost bounded.

### 2. Denormalise link edges so reverse lookups stop scanning · value 4 · fit 5 · S
Chroma stores `links` as an opaque metadata string, so "who points at me?" is
not expressible as a `where` clause. `neighbors(direction="in"|"both")` therefore
reads every memory — **once per frontier node per hop**, so a depth-2 walk over
ten neighbours is eleven full loads. `merge`'s incoming-link re-pointing and
`find_path` pay the same cost.

Record the edge on both sides (`a → b` also stored as an inbound entry on `b`)
and the reverse lookup becomes a plain `get`. No index, no second database, and
both ports get it from the same data-model change.
- **Watch-out:** the two halves must stay consistent under `unlink`, `forget`,
  `merge` and `undo`; a migration pass has to backfill existing stores, and the
  scan fallback should stay for a store whose inbound fields are absent.

---

## Tier 2 — retrieval quality (Mem0 / Zep / HippoRAG-inspired)

| Feature | V/Fit/Eff | Sketch |
|---|---|---|
| **Conversation fact-distillation ingest** | 4/4/M | `distill_turns()` extracts atomic facts from raw dialogue, then routes each via ADD/UPDATE/MERGE/NOOP with a dry-run plan. ~70% new extraction; MERGE reuses `remember(on_conflict="supersede")`. |
| **Bi-temporal validity** (`valid_from`/`valid_until` + `recall(as_of=T)`) | 4/5/M | "What was true at T", distinct from the transaction-time journal. Critical subtlety: `as_of` on a past T must override superseded-hiding for in-interval memories. Contrast clearly with `state_at` ("as of when we *knew*"). |
| **Entity extraction + entity-overlap channel** | 4/4/M | Store extracted entities like `tags`; add an entity-overlap term (`weight_entity=0.0` default → no-op) and a `recall(entity=…)` filter. |
| **Full graph-proximity tuning** (follow-up to the shipped fusion) | 3/4/M | Empirically tune damping / iteration count / default `graph` weight — now actually possible, since `houkai eval` / `eval_recall` exist (§28). Document a recommended profile. |

---

## Tier 3 — observability & ops

| Feature | V/Fit/Eff | Sketch |
|---|---|---|
| **Prometheus metrics** | 3/5/S | p50/p95/p99 and the full mutator counters already ship; what remains is per-*stage* latency and a dependency-free `GET /metrics?format=prometheus`. |
| **Quota + eviction** | 4/5/M | `max_memories` + policy (`decay`/`lru`/`reject`), batch-evict to a headroom watermark, never evict `procedural` (nor `pinned`, §27), journal evictions under `as_actor("quota")`. **The highest-value item in this tier**: it is the one ops feature with no workaround today, because decay pruning is the only bound on store growth and it is time-based, not size-based — a store that grows faster than it ages has nothing holding it. |
| **Quiesced backup + restore** | 4/4/M | Today's `houkai backup` is an unlocked `copytree` with no real restore. Add a manifest (embedding-model / collection / checksums), a real `houkai restore`, and integrity verification. **ai-houkai-service's `backup.py` already has manifests and a working restore** — another graduation candidate, like the curation set (§26). |

---

## Tier 4 — safety & privacy

| Feature | V/Fit/Eff | Sketch |
|---|---|---|
| **PII scanner + privacy tier** | 4/4/M | Stdlib regex + Luhn; `off`/`tag`/`redact`. In redact mode, scrub *before* `collection.add` so vectors/journal/exports carry only placeholders. Best-effort, not a compliance guarantee. |
| **At-rest encryption** | 3/4/M | Optional encryption for the plaintext journal and `.ahkai` export archives. |
| **GDPR cascade forget-me** | 4/3/L | Per-scope right-to-erasure with journal redaction + an erasure receipt (depends on Tier 1 scoping). |
| **Journal integrity + compaction** | 3/5/M | Sequence numbers + checksums, and a `journal compact` command. |

---

## Tier 5 — developer experience & ecosystem

| Feature | V/Fit/Eff | Sketch |
|---|---|---|
| **Typed remote SDK client** | 3/5/L | A thin stdlib `HoukaiClient` mirroring the REST surface. |
| **LangChain / LlamaIndex adapters** | 4/4/M | Memory/retriever adapters — the primitives already match. |
| **Contradiction-surfacing context packing** | 4/5/M | Opt-in `surface_conflicts` on `recall_pack` that *annotates* (does not resolve) conflicts among packed items — reuses two shipped-but-disconnected features. |
| **Gold sets in CI** | 3/5/S | `houkai eval` exists now, but no gold set is checked in, so CI still cannot fail on a ranking regression. Commit a small fixture corpus + gold set and add a threshold job. |

---

## Novel / higher-risk bets

- **Memory-utility learning from retrieval outcomes** — credit memories by
  usefulness, not frequency (the store only tracks `access_count` today).
- **Known-gap memories** — first-class tracked unknowns with auto-resolution.
- **Hebbian co-recall links** — build an associative `related` graph from actual
  retrieval co-occurrence (a natural feeder for the shipped graph-proximity
  signal). Namespace derived keywords so they don't pollute curated `tags`.

---

## Latent issues worth a quick correctness pass

Both previously listed here (the `metrics()` doc/impl mismatch and the
incomplete op counters) were already fixed in the shipped code — percentiles
and every mutator counter are live, so only the Prometheus exposition format
and per-*stage* latency remain open (see [DESIGN.md](DESIGN.md) §22).

- **`/ready` has no probe deadline** — the 5 s is a cache TTL, not a timeout,
  and the route is auth-exempt and never caches a failure.
- **`mcp` is capped at `<2`** — mcp 2.0.0 removed `mcp.server.fastmcp` (the
  FastMCP server was renamed to `mcp.server.mcpserver`), so an unbounded
  requirement made every fresh install resolve to a version the code cannot
  import. The cap restores reproducible installs; porting
  `mcp_server/server.py` to the 2.x API is real work across 41 tools and should
  lift the cap in the same change.
- **`houkai list --format json` on an empty store** prints a human message to
  stderr and nothing to stdout, so the output is not parseable JSON. Any script
  piping it has to tolerate the empty case (see `list_ids` in
  `functional_tests/test_e2e.py`).
- **`neighbors(direction="in")` still scans** — see Tier 1 #2. Callers on large
  stores should prefer `direction="out"` until the edges are denormalised.
- **Corpus-lexical recall is a linear scan** — ~4.5 ms at 25k rows, growing with
  the collection. Fine at agent scale, and off by default; if a deployment needs
  a flat lookup curve, the index belongs in the consuming service's database
  rather than back in the library (§25).
