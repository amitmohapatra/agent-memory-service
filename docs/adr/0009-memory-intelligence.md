# ADR 0009: Memory intelligence — native rules first, LLM providers as benchmarkable adapters

**Status:** accepted · **Date:** 2026-09-15

## Decision
- **Observations in, canonical memories out.** Every chat message, file, agent/tool result,
  decision, feedback or event is an `Observation` (durably committed with its
  `memory.process_observation` outbox job). The `ObservationPipeline` turns it into zero or
  more `CanonicalMemory` rows: extract → classify → consolidate → persist → `memory.index`.
  All writes for one observation, the index job and the revision bumps commit in one unit
  of work; replays are no-ops (`processed_at` guard), so retries never double-create.
- **Native provider is deterministic and conservative.** No LLM (`llm.enabled=false` is the
  default and the gates run that way). Sentence and first-person clause splitting, then
  ordered rules: attributes (`my timezone is …`), identity/employer/location, preferences
  and instructions, favourites, decisions (`we decided …`, `Decision:`), procedures, tasks,
  `Subject is/has/costs Object` facts, and episodic events; questions and chit-chat produce
  nothing. Each candidate carries subject/predicate/object, a category, temporal hints
  (`since 2026-09-01`, `until …`, `Q3 2026`) and an explicit-replacement flag (`no longer`,
  `instead of`, `actually`, `now`).
- **Classification defaults** (hints override): USER/PREFERENCE → `USER` visibility, LONG_TERM;
  AGENT/TOOL/WORKING → `PRIVATE`; decisions/facts/events → THREAD when a thread is present,
  else WORKSPACE/USER/TENANT; TASK/TOOL/AGENT are SHORT_TERM with a 7-day `expires_at`;
  EPHEMERAL candidates never reach PostgreSQL (cache-only `EphemeralMemory`, folded into the
  bundle for the same thread/agent run while they last).
- **Scope vs visibility.** `scope_for` anchors a memory (USER, THREAD, AGENT, WORK, …) and
  `keys_for` computes the audience keys (ADR 0005). The owner principal is always in the
  audience except for PRIVATE memories, whose only audience is the owner.
- **Consolidation needs positive evidence.** Candidates are compared with CURRENT memories of
  the same scope that share the normalized hash or the subject (bounded by
  `dedup_candidate_k`): identical normalized text → REINFORCE; same subject+predicate+object
  → REINFORCE; same single-valued slot (timezone, name, role, employer, favourite_x, …) with a
  new value → SUPERSEDE (old row gets `SUPERSEDED`, `superseded_by`, `valid_to`; new row
  `supersedes`); a replacement signal on the same slot with ≥ 0.3 token overlap → SUPERSEDE;
  lexical Jaccard ≥ 0.92 → MERGE/REINFORCE; otherwise CREATE. **Differing numbers or
  negation always block a merge** — those are the classic false merges. Dense similarity is
  used only with a real embedding model (never with the hash stand-in).
- **Only CURRENT memories are searchable.** `Indexer.index_memories` upserts CURRENT rows
  into the memories collection (payload: visibility keys, type, lifetime, subject/predicate/
  object, owner, importance) and deletes every other state; superseded/expired/retracted
  rows stay in PostgreSQL for history (`GET /v1/memories?include_superseded=true`) and M8's
  temporal queries. `mem_…` identifiers resolve through the engine's exact path with the
  same visibility check.
- **Forget** is a soft delete by the owner (or a tenant admin): RETRACTED, index removal via
  the same job, revisions bumped so cached bundles drop it immediately.
- **External providers are adapters, not the core.** `Mem0MemoryIntelligence` (per-tenant
  namespace, events ADD/UPDATE/NONE mapped to CREATE/SUPERSEDE/REINFORCE),
  `LangMemIntelligence` (memory manager inserts/updates by provider id) and the experimental
  `CogneeMemoryIntelligence` implement the same port, declare `requires_llm=True`, refuse to
  start without `models.llm.enabled=true`, and keep PostgreSQL canonical. Their mapping is
  unit-tested with fakes; their quality is benchmarked by `benchmark/memory.py` only when an
  LLM is configured (otherwise recorded as skipped, never as a number).

## Evidence
- `tests/eval/golden/memory_pairs.json` (35 labelled pairs) →
  `benchmark/results/memory_gate.json`: false-merge rate 0.00 (threshold 0.01), dedup recall
  1.00.
- `tests/integration/test_memory.py`: observation → memories → recall; reinforce/supersede
  history; thread-shared decisions; agent-private isolation (other agents and the user see
  nothing; hints widen deliberately); forget and expiry; replay idempotence.
- `tests/e2e/test_memory_flow.py`: `/v1/observations` (idempotent), `/v1/memories`,
  `/v1/recall`, `/v1/context` with memories; SDK `observe/remember/recall/get_memory/forget`.
- `benchmark/results/memory.json`: native pipeline ≈ 34 observations/s single-threaded
  against PostgreSQL + Qdrant local; accept p95 ≈ 10 ms, process p95 ≈ 21 ms.

## Consequences
- Recall of memory-worthy statements that the rules do not recognise is the known limit of the
  native provider; the LLM `uses` list (`ambiguous_extraction`, `ambiguous_worthiness`) is the
  designed escalation path and stays off by default.
- Duplicate *preferences on the same topic* stated without a replacement signal are kept as
  two memories (no false merge, at the cost of redundancy); M11 may add topic clustering.
