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

| Feature | Notes |
|---|---|
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

### 2. Persistent full-corpus BM25 (opt-in SQLite FTS5 sidecar) · value 5 · fit 4 · L
`_bm25_score_pool` scores lexical relevance only over the vector over-fetch
pool, so a strong exact-token match with a weak embedding is *never surfaced at
any corpus size*. Add an optional FTS5 sidecar (`lexical_index="fts"`, default
`"pool"` = today's behaviour) that indexes the same pre-tokenised text, UNIONs
its top-N ids into the pool before scoring, plus a `houkai reindex` command.
- **Watch-out:** the `β=0.20` lexical weight was tuned against pool-relative
  normalisation; global IDF changes magnitude and may need re-tuning.

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
| **Metrics overhaul** | 4/5/M | `metrics()` docstring promises "recall-latency percentiles" it never computes — add real p50/p95/p99 (fixed-bucket histogram), per-stage latency, counters on the *uncounted* mutators (`link`/`unlink`/`import`/`export`/`restore`/`purge`), and an optional dependency-free `GET /metrics?format=prometheus`. |
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
| **Wire the eval harness into CLI + MCP** | 4/5/M | `ai_houkai.eval` exists but is unreachable from any surface. Add `houkai eval <goldset.jsonl>` + a read-only `eval_recall` MCP tool + a Go port, so ranking changes are measurable in CI. Keep gold sets JSONL-only (preserve the stdlib-only ethos). |
| **`ai_houkai.testing`** | 4/4/M | A hash-based `FakeEmbedder` + pytest fixtures. Requires first adding an `embedding_function=` injection seam to the Python constructor (Go already has the seam). |
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

- **`metrics()` doc/impl mismatch** — the docstring advertises percentiles that
  are not computed (folded into the Metrics overhaul above).
- **Incomplete op counters** — `link`/`unlink`/`import`/`export`/`restore`/
  `purge_expired` are invisible to `metrics()`.
