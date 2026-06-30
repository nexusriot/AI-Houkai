# AI-Houkai — Feature Design Proposals

> **Status: all three shipped.** This document is the original design record
> for hybrid retrieval, conflict/contradiction detection, and memory linking —
> all now implemented and live. Kept for historical context; see
> [DESIGN.md](DESIGN.md) for the current behaviour.

Designs for three (now-implemented) features.

1. [Hybrid retrieval](#1-hybrid-retrieval)
2. [Conflict / contradiction detection](#2-conflict--contradiction-detection)
3. [Memory linking (lightweight graph)](#3-memory-linking-lightweight-graph)

The proposals share three constraints:

- **No new infra.** Stay on ChromaDB + sentence-transformers; no extra
  service, no second database.
- **Backwards compatible.** Existing memories (created without the new
  metadata) must keep working — new fields default to neutral values.
- **Off by default.** Each feature is a toggleable strategy / flag so
  current behaviour is unchanged for callers who don't opt in.

---

## 1. Hybrid retrieval

### Goal

`recall()` today is pure cosine similarity.  This misses three signals
the memory store already tracks:

- **Lexical match** — agents often query with exact identifiers
  (`PR #441`, `MemoryStore.recall`) where keyword overlap > embedding sim.
- **Recency** — a one-day-old memory is usually more relevant than a
  one-year-old paraphrase.
- **Importance** — `importance=0.95` should outrank `importance=0.3`.

### Score formula

```
final = α · cosine_sim
      + β · bm25_score_normalized
      + γ · recency_decay(last_accessed)
      + δ · importance
```

Defaults:

| Weight | Default | Rationale |
|---|---|---|
| α (cosine)    | 0.55 | Primary signal, must dominate |
| β (lexical)   | 0.20 | Catches identifiers / exact tokens |
| γ (recency)   | 0.15 | Mirrors `DecayEngine` half-life |
| δ (importance)| 0.10 | Already explicit, weaker tiebreak |

`recency_decay(t) = exp(−λ · age_days)` reusing the `DecayEngine` formula
with the **same** `λ` so the two systems agree on what "fresh" means.

> **Editor's note (as shipped):** the default measures `created_at`, not
> `last_accessed` (`HybridWeights.recency_basis` defaults to `"created"`;
> `recency_basis="accessed"` restores the `last_accessed` behaviour
> described here).

`bm25_score_normalized` is computed lazily over the query result set
only (top-k from cosine + a small over-fetch), not the whole corpus —
no second index needed.

### API additions

```python
def recall(
    self,
    query: str,
    k: int = 5,
    type: MemoryType | None = None,
    tag: str | None = None,
    min_importance: float | None = None,
    *,
    mode: Literal["semantic", "hybrid"] = "semantic",
    weights: HybridWeights | None = None,
    overfetch: int = 4,                  # multiplier for cosine first stage
) -> list[tuple[Memory, float]]: ...
```

```python
@dataclass(frozen=True)
class HybridWeights:
    cosine:     float = 0.55
    lexical:    float = 0.20
    recency:    float = 0.15
    importance: float = 0.10
    decay_rate: float = 0.1            # shared with DecayEngine
```

### Algorithm

```
1. cosine_pool = collection.query(query, n_results=k * overfetch, where=…)
2. for each candidate:
       cosine     = 1 - distance
       lexical    = bm25(query, doc, idf_over_pool)   # local IDF
       recency    = exp(-λ · age_days_since_last_access)
       importance = stored importance
       final = α·cosine + β·lexical + γ·recency + δ·importance
3. apply tag post-filter (existing logic)
4. sort by final, take top k
5. _touch() the survivors
6. return [(memory, final_score), …]
```

**Why local IDF?** A real BM25 index over the whole collection would
require either a Chroma full-text index (not available) or a parallel
SQLite FTS5 table.  Local IDF — computed over the cosine top-N — is a
well-known approximation (used by Elasticsearch's `rescore` phase) and
is good enough for the scale of a single agent's memory (~10⁴ entries).

### Edge cases

| Case | Behaviour |
|---|---|
| Empty pool | Return `[]` (current behaviour) |
| All weights zero | `ValueError` at `HybridWeights.__post_init__` |
| `mode="semantic"` (default) | Identical to current `recall()` — zero risk |
| `last_accessed` in the future | Clamp `age_days >= 0` |
| Ties | Break by `importance`, then `created_at` desc |

### Tests

- `test_hybrid_dispatch.py` — `mode="hybrid"` returns the same shape.
- Synthetic corpus where cosine and BM25 disagree: query "PR #441"
  matches the lexical doc even when its embedding is mediocre.
- Recency: two near-identical memories, the newer one ranks higher.
- Weight equivalence: with `lexical=recency=importance=0`, hybrid ===
  semantic up to floating-point noise.
- Stability: `_touch()` mutates `last_accessed`, but a second call with
  the same query returns identical ordering on the *same* second.

### Out of scope (follow-ups)

- Cross-encoder reranking (heavy model).  Could land later as a
  pluggable `reranker=` argument.
- Persistent BM25 index.  Only worth it past ~100k memories.

---

## 2. Conflict / contradiction detection

### Goal

When the agent stores `"Use ruff for linting"` and later
`"Use flake8 for linting"`, both sit happily in the vector store and
`recall("linting")` returns the contradiction without flagging it.
The agent should be able to detect this **at write time** and **at
recall time**, then choose between supersede / forget / keep-both.

### Data model

Add three optional metadata fields:

| Field | Type | Default | Meaning |
|---|---|---|---|
| `superseded_by` | `str` (memory_id) | `""` | Set when another memory replaces this one |
| `superseded_at` | `float` (epoch) | `0.0` | When the supersede happened |
| `polarity` | `int` (-1 / 0 / +1) | `0` | Optional explicit polarity hint from the writer |

A "superseded" memory is **soft-deleted**: it stays in storage (so
forensics / undo are possible) but is filtered out of `recall()` by
default.  Hard `forget()` still works.

### Conflict definition

Two memories `a, b` are candidate conflicts iff:

1. `cosine_sim(a, b) ≥ similarity_threshold` (default 0.80)
2. `a.type == b.type` (apples-to-apples)
3. Tag overlap is non-empty *or* both have empty tags
4. Neither is already `superseded_by` the other

Cosine alone catches paraphrases of the *same* fact too — that's
desirable for dedup.  Distinguishing "same" from "opposite" is the next
layer.

### Polarity check (cheap, no LLM)

```
contradicts(a, b) =
    sim(a, b) ≥ 0.80
    AND (
        a.polarity * b.polarity == -1                      # explicit
        OR negation_diff(a.text, b.text)                   # heuristic
    )
```

`negation_diff` is a tiny rule:

```python
NEG = {"not", "never", "no", "don't", "doesn't", "won't", "shouldn't",
       "can't", "without", "avoid"}

def negation_diff(s1, s2):
    return (count_neg(s1) % 2) != (count_neg(s2) % 2)
```

This catches the common case (`"never use X"` vs `"use X"`) without
calling an LLM.  An optional `contradiction_fn=Callable[[Memory, Memory], bool]`
hook lets users plug in an LLM judge.

### API additions

```python
class MemoryStore:
    def remember(
        self, text, type="semantic", tags=(), importance=0.5,
        source=None,
        *,
        on_conflict: Literal["ignore", "warn", "supersede", "raise"] = "warn",
        contradiction_fn: ConflictFn | None = None,
    ) -> RememberResult: ...

    def find_conflicts(
        self, memory_id: str | None = None,
        *, threshold: float = 0.80,
    ) -> list[Conflict]: ...

    def supersede(self, old_id: str, new_id: str) -> None: ...
    def restore(self, memory_id: str) -> bool: ...   # un-supersede
```

```python
@dataclass
class RememberResult:
    memory:    Memory
    conflicts: list[Conflict] = field(default_factory=list)
    action:    Literal["stored", "superseded", "rejected"] = "stored"

@dataclass
class Conflict:
    a: Memory
    b: Memory
    similarity: float
    kind: Literal["duplicate", "contradiction"]
    reason: str            # "polarity_diff", "negation_diff", "custom_fn", "similarity" (near-dup, kind="duplicate")
```

### Behaviour matrix

| `on_conflict` | Duplicate (sim ≥ 0.95, same polarity) | Contradiction (negation_diff) |
|---|---|---|
| `ignore`     | store anyway, no signal | store anyway, no signal |
| `warn`       | store, return Conflict in result | store, return Conflict |
| `supersede`  | merge into existing (touch, raise importance) | mark old as `superseded_by` new |
| `raise`      | `ConflictError` | `ConflictError` |

### Recall integration

`recall(..., include_superseded=False)` (default) filters out any memory
with non-empty `superseded_by`.  Hybrid scoring naturally penalises
superseded entries because they're excluded from the candidate pool
before scoring.

### MCP tools

Two new tools, parallel to existing ones:

| Tool | Inputs | Returns |
|---|---|---|
| `find_conflicts` | `memory_id?, threshold?` | `[{a, b, similarity, kind, reason}]` |
| `supersede`      | `old_id, new_id`         | `{ok, old_id, new_id}` |

`remember` gains an `on_conflict` parameter mirroring the Python API.
The MCP server returns conflicts in its result so the LLM can decide
what to do — exposing rather than auto-resolving.

### Edge cases

- **Cycles**: `supersede(a → b)` then `supersede(b → a)` — reject the
  second call with a clear error.
- **Chains**: `a → b → c`. `recall()` follows `superseded_by` chains
  iff `include_superseded=False` so users see only the head.
- **Self-supersede**: `supersede(x, x)` is a no-op error.
- **Reflection products**: `ReflectionEngine` already creates summaries
  from clusters of similar memories; flag those as `kind="duplicate"`
  not `"contradiction"` even if negation differs (the summary is by
  definition the new canonical).

### Tests

- Two near-identical positives → `kind="duplicate"`.
- "Use X" + "Never use X" → `kind="contradiction", reason="negation_diff"`.
- `on_conflict="supersede"` → old memory has `superseded_by` set.
- `recall()` excludes superseded; `recall(..., include_superseded=True)`
  includes them.
- Custom `contradiction_fn` is consulted only when `negation_diff`
  returns `False` (negation is a fast-path).

---

## 3. Memory linking (lightweight graph)

### Goal

Today every memory is independent.  Reflection summaries lose track of
their sources after `consolidate=True`.  Supersede chains (§2) need a
typed pointer.  Users want to express "this refines that" or "this is
an example of that".

### Data model

One new metadata field on `Memory`:

```python
links: list[Link]
```

Stored in ChromaDB metadata (which is scalar-only) using the same
trick as `tags` — a JSON-encoded string:

```python
# write
{"links": '[{"to":"a8...","rel":"refines"},{"to":"d2...","rel":"example_of"}]'}

# read
links = json.loads(meta.get("links") or "[]")
```

### Link relations

A small fixed vocabulary keeps the graph queryable without a schema
service:

| `rel` | Meaning | Created by |
|---|---|---|
| `supersedes` | replaces another memory (§2) | `supersede()` |
| `refines`    | adds detail to another memory | manual / agent |
| `derived_from` | reflection produced this from sources | `ReflectionEngine` |
| `example_of` | concrete instance of a procedural rule | manual / agent |
| `contradicts` | disagrees with another (kept intentionally) | `find_conflicts` if user chooses keep-both |
| `related`    | catch-all weak link | manual / agent |

`rel` is a string, not an enum, so users may add their own.  The vocab
above is what the engine itself emits.

### API additions

```python
class MemoryStore:
    def link(self, src_id: str, dst_id: str, rel: str = "related") -> None: ...
    def unlink(self, src_id: str, dst_id: str, rel: str | None = None) -> int: ...
    def neighbors(
        self, memory_id: str,
        *, rel: str | None = None, direction: Literal["out", "in", "both"] = "both",
        depth: int = 1,
    ) -> list[tuple[Memory, str]]: ...
    def subgraph(
        self, memory_ids: Iterable[str], *, depth: int = 1,
    ) -> Graph: ...
```

```python
@dataclass(frozen=True)
class Link:
    to:  str
    rel: str

@dataclass
class Graph:
    nodes: dict[str, Memory]
    edges: list[tuple[str, str, str]]   # (src, dst, rel)
```

`Link` is one-directional and stored on the source.  Reverse lookups
require iterating the collection — fine at this scale; if it ever
matters, materialise an in-memory adjacency on `__init__`.

### Recall integration

```python
recall(query, k, *, expand: ExpandSpec | None = None) -> [...]
```

```python
@dataclass
class ExpandSpec:
    rels: tuple[str, ...] = ("refines", "example_of")
    depth: int = 1
    cap: int = 5                  # max expanded neighbours added
    score: float = 0.7            # score assigned to expanded hits
```

After ranking the top-k, optionally walk `rels` outward up to `depth`
hops and append unseen neighbours with the configured score.  This is
the "fetch the rule that this example points to" pattern — useful when
the query hits an `example_of` and the agent really needs the procedural
parent.

### Reflection integration

`ReflectionEngine.reflect(consolidate=True)` currently deletes source
episodics.  New default: instead of deleting, mark them as
`superseded_by=<summary_id>` and add a `derived_from` link from the
summary back to each source.  Add a flag `consolidate="hard"` for the
old destructive behaviour.

This solves a long-standing concern: reflection summaries become
auditable — you can walk `derived_from` to see what raw events the
condensed insight came from.

### MCP tools

| Tool | Inputs | Returns |
|---|---|---|
| `link`      | `src_id, dst_id, rel?`  | `{ok}` |
| `unlink`    | `src_id, dst_id, rel?`  | `{removed}` |
| `neighbors` | `memory_id, rel?, depth?` | `[{memory, rel}]` |

### Storage cost

`links` is a single ChromaDB metadata string per memory.  Empirically a
typical agent's memories accumulate ≪ 10 links each → < 1 KB per
memory.  No measurable impact on HNSW or query latency; the field
isn't used as a `where` filter.

### Edge cases

- **Dangling link**: target deleted via `forget()`. Tolerated — `link()`
  doesn't validate; `neighbors()` skips IDs that no longer resolve and
  returns a `dangling=[ids]` audit list.
- **Self-link**: `link(a, a)` rejected.
- **Duplicate link**: same `(src, dst, rel)` is idempotent.
- **Cycles**: allowed (the graph is a DAG only by convention, not
  enforcement).  `neighbors(depth=N)` uses a visited-set to avoid
  infinite loops.
- **Migration**: existing memories have no `links` metadata key —
  `from_record()` defaults to `[]`.

### Tests

- Round-trip: `link`, `neighbors`, `unlink`.
- Reflection now sets `derived_from` + `superseded_by` and source
  memories survive in storage.
- `recall(..., expand=ExpandSpec(rels=("refines",)))` returns expanded
  neighbours with the configured score.
- Cycle: `link(a,b,rel)` + `link(b,a,rel)` + `neighbors(a, depth=5)`
  terminates and visits each node once.
- Migration: load a pre-existing `.chroma` directory written without the
  `links` field, confirm `from_record` returns `links=[]`.

---

## Cross-cutting concerns

### Order of implementation

Implement **in this order** so each feature can lean on the previous:

1. **Memory linking** first — the data model change is small, and both
   other features want to emit links (`supersedes`, `derived_from`).
2. **Conflict detection** second — uses `link(rel="supersedes")` for
   its soft-delete representation instead of inventing a parallel
   metadata field.
3. **Hybrid retrieval** last — purely additive on the read path, no
   data-model coupling.

### Schema migration

All three features add metadata fields with safe defaults:

```python
class Memory:
    # existing fields …
    links:           list[Link] = field(default_factory=list)
    superseded_by:   str = ""
    superseded_at:   float = 0.0
    polarity:        int = 0
```

`Memory.from_record` reads them with `meta.get(..., default)`.  Old
ChromaDB rows continue to load without re-encoding.  No migration
script required.

### Configuration surface

Per-store defaults belong on `MemoryStore.__init__` so they're
discoverable and overridable in one place:

```python
MemoryStore(
    path=…, collection=…,
    hybrid_weights:    HybridWeights | None = None,    # None → semantic only
    conflict_policy:   Literal["ignore","warn","supersede","raise"] = "warn",
    contradiction_fn:  ConflictFn | None = None,
)
```

Per-call kwargs override per-store defaults — same pattern the existing
codebase already uses for `recall(type=…)`.

### Documentation deltas

When implementing, update:

- `DESIGN.md` §3 (Data Model) — new fields, polarity, links serialisation.
- `DESIGN.md` §5 (Memory Lifecycle) — supersede branch in the diagram.
- `DESIGN.md` §7 (Reflection Engine) — new default `consolidate` mode.
- `DESIGN.md` §8 (MCP Server) — five new tools.
- `README.md` — quick examples for each.
- `tests/` — new files `test_hybrid.py`, `test_conflicts.py`, `test_links.py`.
