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

## ✅ Shipped in this cycle

Everything in [RESEARCH-2026-08-01.md](RESEARCH-2026-08-01.md) — A1–A3, B, C,
F2, E, D, F1, F3, F4 and G — landed together (2026-08-02), in both ports.

| Feature | Notes |
|---|---|
| **A1 — retrieval knobs on MCP** | Python MCP `recall`/`recall_pack`/`auto_context` gained `fusion`, `diversity`, `dedup_threshold`, `min_cosine`, `graph`, `touch`, `header` and flat `expand_*`; Go gained `graph`/`expand_*`/`header` and `AutoContextOpts.NoTouch`. |
| **A2 — public `get()`** | `_get_by_id` promoted (alias kept one release), `AsyncMemoryStore.get`, MCP `get` tool. |
| **A3 — surface coverage** | MCP +`get`/`subgraph`/`restore`/`undo`/`nuke`/`ready`; HTTP +`/restore`, `/subgraph`, `/undo`, `/nuke`, `/journal`, `/export`, `/import`; CLI +`metrics`/`history`/`state-at`/`get-at`/`journal undo-last`. |
| **B — pluggable embedder** | `ai_houkai/embed.py` (stdlib OpenAI-compatible + Ollama), `MemoryStore(embedding_function=)`, `AI_HOUKAI_EMBEDDER=provider:model`, `ai_houkai/testing.py` (`FakeEmbedder` + fixtures). sentence-transformers moved out of core into a `[local]` extra. |
| **C — CI + parity guard** | `.github/workflows/{ci,release}.yml` and repo-root `parity.json`, asserted by **both** ports (`tests/test_parity.py`, `go/internal/parity/`). |
| **F2 — eval wired up** | `houkai eval goldset.jsonl`, `eval_recall` MCP tool, and a full Go port of the harness (`go/internal/eval`). |
| **E — SQLite sidecar index** | Opt-in `index="sqlite"`: full-corpus FTS5 BM25 (`lexical_index="fts"`), cursor pagination, O(1) reverse links, tag/type counts, indexed expiry sweep, `houkai reindex`. Derived cache with a scan fallback everywhere. |
| **D — curation graduated** | `merge`, `versions`, `list_tags`/`rename_tag`/`merge_tags`/`delete_tag`, `find_path`, and the `trash`/`trash_list`/`trash_restore`/`trash_purge` trio — all previously implemented in ai-houkai-service against library privates. |
| **F1 — tiered reflection** | `ReflectionEngine(types=…, max_level=…)`; summaries tagged `level:N` so reflections-of-reflections form a capped hierarchy. |
| **F3 — pinned tier** | `Memory.pinned`: always offered to `recall_pack(include_pinned=True)`, never pruned by decay. |
| **F4 — idempotent writes** | `remember(idempotent=True)` + `content_hash` (sha256, byte-identical across ports). |
| **G — provenance trust** | `Memory.trust` (`trusted`/`reported`/`untrusted`), `recall(min_trust=…)`, untrusted lines marked in packed context. |
| **Graph-proximity fusion** (`HybridWeights.graph`) | PPR-lite spread over intra-pool links, fused into both weighted and RRF scoring; `graph=0.0` default is a byte-for-byte no-op. |
| **Gated graph expansion** (`ExpandSpec.rerank`) | Expanded neighbours can now be merged into the pool *before* `min_cosine`/dedup/MMR/top-k so they can't inject near-duplicates or overflow `k`. `rerank=False` keeps the legacy append-after behaviour. |
| **`houkai doctor` + `GET /ready`** | Active embedder probe (latency + dim), embed-dim guardrail, store/journal checks; readiness endpoint returns 200/503 and is auth-exempt like `/health`. |
| **Batch write & embed** (`remember_many`) | Bulk store with batched embedding — N docs cost `ceil(N/batch_size)` encode passes instead of N; one journal entry per id (undo stays per-id); wired into `houkai ingest`, `POST /memories/batch`, and the `remember_many` MCP tool. Intra-batch semantics are explicit (earlier items win under `supersede`); `on_conflict="raise"` is rejected in bulk. |

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

### 2. Tune the lexical weight for full-corpus BM25 · value 4 · fit 5 · S
The FTS5 sidecar shipped (see above), but `β=0.20` was tuned against
*pool-relative* normalisation. Global IDF changes the term's magnitude, so
re-tune it against a gold set with `houkai eval --lexical-index fts` now that
there is a ruler.

---

## Tier 2 — retrieval quality (Mem0 / Zep / HippoRAG-inspired)

| Feature | V/Fit/Eff | Sketch |
|---|---|---|
| **Conversation fact-distillation ingest** | 4/4/M | `distill_turns()` extracts atomic facts from raw dialogue, then routes each via ADD/UPDATE/MERGE/NOOP with a dry-run plan. ~70% new extraction; MERGE reuses `remember(on_conflict="supersede")`. |
| **Bi-temporal validity** (`valid_from`/`valid_until` + `recall(as_of=T)`) | 4/5/M | "What was true at T", distinct from the transaction-time journal. Critical subtlety: `as_of` on a past T must override superseded-hiding for in-interval memories. Contrast clearly with `state_at` ("as of when we *knew*"). |
| **Entity extraction + entity-overlap channel** | 4/4/M | Store extracted entities like `tags`; add an entity-overlap term (`weight_entity=0.0` default → no-op) and a `recall(entity=…)` filter. |
| **Full graph-proximity tuning** (follow-up to the shipped fusion) | 3/4/M | Empirically tune damping / iteration count / default `graph` weight via `eval_recall`; expose the knobs and document a recommended profile. |

---

## Tier 3 — observability & ops

| Feature | V/Fit/Eff | Sketch |
|---|---|---|
| **Prometheus metrics** | 3/5/S | p50/p95/p99 and the full mutator counters already ship; what remains is per-*stage* latency and a dependency-free `GET /metrics?format=prometheus`. |
| **Quota + eviction** | 4/5/M | `max_memories` + policy (`decay`/`lru`/`reject`), batch-evict to a headroom watermark, never evict `procedural`, journal evictions under `as_actor("quota")`. |
| **Quiesced backup + restore** | 4/4/M | Today's `houkai backup` is an unlocked `copytree` with no real restore. Add a manifest (embedding-model / collection / checksums), a real `houkai restore`, and integrity verification. |

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
incomplete op counters) were already fixed in the shipped code — see
[RESEARCH-2026-08-01.md](RESEARCH-2026-08-01.md) §1.

- **`/ready` has no probe deadline** — the 5 s is a cache TTL, not a timeout,
  and the route is auth-exempt and never caches a failure.
