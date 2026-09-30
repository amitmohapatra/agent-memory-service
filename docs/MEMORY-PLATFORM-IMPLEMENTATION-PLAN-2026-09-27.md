# Memory platform: finalized implementation plan and handoff

Date: 2026-09-27. Status: design and execution order, not implemented or deployed.
Scope: the capabilities in the user's supplied Hindsight documentation, plus SDK
and MCP integration, with our authorization, tool memory and document RAG retained.
This document creates no new accuracy, latency, feature-parity or cost claim.

## Decision

Extend our service with an optional, pinned Hindsight backend through its async
Python SDK. Keep our canonical source records, authorization and lifecycle as the
authority. Reuse Hindsight's knowledge-maintenance capabilities instead of building
a second complete implementation of every missing feature. Existing native
extraction/consolidation work receives isolated evaluation, not automatic promotion.

First establish the measurement and authorization/deletion contracts. Then test
native improvements and integrate the Hindsight capabilities behind those contracts.
The agent benchmark and LoCoMo measure different requirements; neither replaces the
other. A failed native mechanism does not establish that Hindsight's counterpart
will fail, and API feature parity does not establish accuracy parity.

## Verified starting point

- Main: `efaf008`. Retrieval stack: `a121dc7`. Write stack: `f988c26`.
- `adapters/wiring.py` in main does not construct LandingReflection. The write
  stack does construct it when `contextual_extraction` is allowed. Extraction and
  landing therefore need separate controls before an extraction-only arm exists.
- `modules/memory/derived.py` assigns processing time to both `observed_at` and
  `valid_from`. `_memory_line` renders `observed_at` as a date. Correct the distinction
  between synthesis time and source-event chronology before enabling derived output.
- Temporal validity/filtering and graph as-of behavior exist; a dedicated temporal
  ranking policy is a separate change, not a feature already implemented by those fields.
- QueryType already affects retrieval routing. The missing branch is the explicit
  deep-reasoning policy and provider integration, not all query-dependent behavior.
- Existing tests include `tests/integration/test_multi_agent.py`,
  `tests/e2e/test_visibility_surface.py`, `tests/security`, and `benchmark/security.py`.
  The missing evaluation is longitudinal task quality/cost/freshness, not every
  measurement of agent visibility.
- The 226-question fragmentation estimate is a token-overlap diagnosis. Do not treat
  it as 226 confirmed recoverable answers or tune ingestion using test questions.
- Full no-LLM Hindsight run: upstream `ccfe85b4851957ac2adf88b4a9ddf9668b2882f1`,
  package 0.10.1, raw chunks, local E5, RRF, no observations/consolidation/reflect.
  It did not measure Hindsight's LLM-assisted capabilities.

See `HINDSIGHT-CAPABILITY-AUDIT-2026-09-27.md` for the complete page-by-page audit.
Benchmark artifacts and comparison live in `/Users/ricky/usage_data/ams-hindsight-benchmark`.

## Operating modes

LLM permission and latency policy must be separate from the storage provider.
The following names describe proposed profiles, not currently shipped API values.

| Profile | Ingestion/background work | Query path | Honest claim |
| --- | --- | --- | --- |
| Native / fully LLM-free | Rules, merge, supersession, deterministic aggregates; local embeddings | Local hybrid, graph, temporal ranking, stored deterministic/manual knowledge | No generative calls at any stage; benchmark on a clean corpus with generated derivatives excluded |
| Enriched-fast | Selective LLM extraction, batched observations and mental-model/page refresh | Local retrieval plus validated stored derivatives; no synchronous LLM generation | LLM-assisted ingestion with LLM-free reads, not a fully LLM-free system |
| Deep, explicitly enabled | Same selectable ingestion policy | Budgeted Hindsight reflect for complex tasks | Query-time reasoning with separate cost and latency reporting |

Local embeddings and any future neural reranker are models; avoid calling their
use literally model-free. Track generative calls by ingestion, consolidation,
refresh, query reasoning and benchmark reader/judge separately.

Native mode cannot honestly offer fresh LLM-generated insights. It can expose the
same knowledge-resource interfaces using manual or deterministic content and
explicitly report unsupported generation operations. Reading previously generated
material without new calls belongs to enriched-fast, unless its provenance is
explicitly disclosed; it is not evidence for a clean LLM-free benchmark.

The fast path targets p99 below 300 ms. The user's accepted approximately 551 ms
tradeoff may be evaluated as a separate configuration; it is not a universal SLA.
Deep mode gets a separate deadline after pilot measurements, not the same p99 promise.
Never silently escalate a fast request into paid reasoning. QueryType is a routing
signal; caller permission, deadline, budget and backend health also govern escalation.

## Feature ownership and SDK map

Hindsight's client is an HTTP client. Most features need its server and its own
retained bank corpus; they cannot operate directly over our PostgreSQL/Qdrant schema.
Use a separate Hindsight database. Pin client/server together and contract-test them.
Names below were inspected in the pinned Python client; generated API methods are
already async and do not necessarily use the convenience methods' `a` prefix.

| Capability | Integration decision | SDK surface / prerequisite |
| --- | --- | --- |
| Contextual extraction | Evaluate native selective narrative extraction; compare external extraction on matched inputs | `client.memory.dry_run_extract_memories` returns candidate facts/chunks/usage without persistence, embeddings, entity resolution or graph construction. Requires Hindsight runtime and enabled route; normal extraction uses LLM calls |
| Retain, entity resolution, typed/causal relations | Offer Hindsight retain as an opt-in backend for selected authorized source streams | `aretain`, `aretain_batch`, source timestamps, stable `document_id`, metadata, supplied entities/`resolve_entities`; server extraction/config controls the richer processing |
| Observations / incremental consolidation | Reuse Hindsight for the richer maintenance capability; benchmark against native landing | `client.banks.trigger_consolidation`; scoped batching and auto-consolidation settings; `arecall(types=["observation"], include_source_facts=True)` for export with lineage |
| Mental models | Reuse managed resources for user profile, project state, decisions, commitments and run-start context | `acreate_mental_model`, `aget_mental_model`, `arefresh_mental_model`, update/history methods; delta refresh, rate limits and configured triggers |
| Knowledge pages | Reuse mental-model-backed pages and folder/search/export lifecycle | `acreate_knowledge_page`, `aget_knowledge_page`, `asearch_knowledge_base`, tree/update/delete/export methods |
| Reflect / complex reasoning | Explicit deep mode; preserve source references and verify outputs | `areflect`, `include_facts`, optional schema and trace; query-time LLM calls; structured-output errors handled explicitly |
| Recall, temporal ranking, graph and exploration budgets | Keep native fast retrieval; evaluate Hindsight as an optional provider under common budgets | `arecall`, query timestamp, temporal window, types, tags/tag groups, source facts, chunks and trace. Temporal window is a ranking hint, not a hard date restriction |
| Reranking | Configurable provider feature, off until an arm earns promotion | Hindsight server configuration, not a standalone SDK reranker over our candidates. Preserve the measured CPU latency warning and test bounded candidate sets |
| Memory editing, invalidation, history, entity inspection | Expose controlled lifecycle operations through our authority | Generated `client.memory` and `client.entities` APIs; canonical-source updates and provider synchronization must remain consistent |
| Documents/chunks, reprocessing, deletion | Preserve Docling/document RAG and bridge approved text plus source lineage | `client.documents`, `aretain_files` where appropriate, document/chunk listing and reprocessing; avoid parsing/embedding the same source twice without a measured reason |
| Banks, missions, directives, configuration | Map trusted service-owned policies to provider banks | `acreate_bank`, config methods, directive methods. Bank identity and permitted configuration come from our service, never an arbitrary caller-supplied bank ID |
| Bank templates and transfer | Expose repeatable provisioning separately from data export/import | Generated `BankTemplatesApi` for template import/export/schema; `aexport_bank`/`aimport_bank` are data-transfer operations, not interchangeable template helpers |
| Operations and webhooks | Reuse backend status/events and connect them to our existing outbox/jobs | `client.operations`, `client.webhooks`; authenticated handlers, deduplication and reconciliation. Events trigger synchronization; they are not proof of freshness by themselves |
| Multilingual behavior | Add a separately evaluated model/extraction profile | Server model and tokenizer configuration plus SDK language-preserving inputs. SDK installation does not establish cross-language quality |
| MCP, HTTP/OpenAPI and SDK coverage | Expose our policy-enforcing retain/recall/reflect/knowledge/tool interfaces | Hindsight's MCP server is a separate server surface, not an SDK wrapper for our APIs. Do not expose a shared provider credential or bypass our scope resolver |
| Storage/performance/observability | Keep provider storage isolated and make its work visible in our telemetry | Track SDK latency, queues, LLM usage, retries and dependency health; no need to migrate our canonical schema to Hindsight's storage to use its features |

All rows remain in the scope. Native and Hindsight providers do not need separate
implementations of every generation algorithm. Interface availability, backend
support, safe lifecycle behavior and workload validation define completion.

## Architecture and data flow

1. Our source write commits canonical content, stable source identity, revision,
   policy identity and an outbox entry together. Reuse the existing job infrastructure.
2. A provider worker processes a bounded, deduplicated batch. Persist a mapping from
   source ID/revision to Hindsight bank/document/fact IDs and provider operation IDs.
3. The provider performs extraction/consolidation outside the application's write
   transaction. No LLM/network wait under admission locks.
4. Import completed observations/models/pages only after validating lineage,
   authorization, source revisions, deletion state and refresh status. The imported
   artifact records its provider/version and complete dependency set.
5. Index/cache validated derived artifacts locally for enriched-fast reads. They are
   retrievable evidence or run-start context, not an unconditional answer to any query.
6. Deep requests may call reflect against eligible, synchronized provider banks.
   Preserve citations, token usage and errors; verify claims with our existing surface.

Use small domain ports for extraction/knowledge maintenance/deep reasoning as needed;
extend existing ports when their contracts genuinely match. Keep SDK DTOs and provider
exceptions inside `adapters`. One shared lifecycle/dependency service handles native
and external derived content. Do not duplicate extraction under both providers for
every production message. Dual execution is an explicitly budgeted evaluation mode.

Bound batch size, bank fan-out, queue depth, retrieved items and token budgets. Reuse
an async client with explicit per-operation timeout/retry budgets rather than the
SDK's generous default timeout. Batch authorization checks; avoid per-fact network
lookups. Invalidation work should follow indexed dependency edges proportional to
affected artifacts, not scan the complete corpus. Refresh jobs coalesce repeated
source changes and recheck revisions before publishing.

Route Hindsight's generative model calls through the configured Bifrost gateway,
with server-side credentials and a compatible, tested model. Provider and SDK keys
are distinct concerns. Apply a gateway/provider dollar cap as well as request/token
budgets; an SDK request can produce multiple internal LLM calls. Check the remaining
account balance before a paid pilot. Complex or poorly handled source inputs may
use contextual extraction; routine known facts stay native. Batch downstream
knowledge refreshes and skip unchanged scopes. These are proposed cost policies,
not a claim that consolidation is free or that a call-count limit guarantees spend.

## Authorization design: required before persistent integration

The SDK is accessible only to our service identities. Clients continue to authenticate
to our API; they cannot choose a provider bank or call the backend with our credentials.

- Resolve PRIVATE, RUN, THREAD, AGENT_GROUP, USER and TENANT through the existing
  server-side visibility policy, including the distinct `run:` and `runup:` semantics.
- Initially partition banks by tenant and an exact canonical source-audience policy,
  including owner exceptions where the current policy permits them. Do not consolidate
  sources with different read policies in one bank. Opaque bank IDs are not authorization.
- A caller may query a bank only if the current policy grants access to its source
  audience. Check revocations and current thread/run context before dispatch.
- A derived result may be read only if every contributing source is readable. Use
  conjunction of source policies, not an ordinal minimum such as “most private”.
- First release keeps Hindsight consolidation/reflect within homogeneous banks.
  Cross-bank joins need explicit authorization and bounded orchestration; do not merge
  banks merely to make a multi-hop question easier. Local authorized retrieval can
  combine evidence across allowed banks without changing stored sharing rights.
- No cross-tenant, sibling-run or cross-thread source may enter a request's synthesis
  context. Output filtering alone is insufficient once synthesis has read forbidden data.
- Recheck policy/source revisions before accepting an asynchronous result. Policy
  changes invalidate local derived caches and quarantine affected provider operations
  until their projections are reconciled.

A provider result without reliable, complete dependency lineage cannot be promoted
to durable local knowledge. Track bank generation as a conservative fallback and
invalidate the whole affected bank's derived artifacts when precise lineage is absent.
This trades availability for correctness; measure that cost rather than concealing it.

## Updates, deletion and revocation

The local transaction tombstones/revises the source, increments its generation,
invalidates dependent read caches and emits an outbox event. From that point our
reads suppress invalid derived content, even if Hindsight is unavailable.

The worker removes or updates the provider document/facts and invalidates affected
observations, mental models and pages. Affected banks are unavailable for provider
reasoning until reconciliation makes their contents eligible again. Late refreshes
cannot republish a deleted or superseded source revision. Use idempotent retries,
durable operation receipts and a reconciliation sweep. Distinguish immediate logical
read denial from eventual provider purge; do not report physical deletion as complete
while it is pending. Include history/export/trace retention in the deletion contract.

Do not rely only on Hindsight's mental-model staleness flag: its documentation notes
that source deletion alone does not advance that write-based freshness check. Test
deletion explicitly rather than assuming the refresh trigger will notice it.

## Evaluation first, then isolated changes

### Agent evaluation

Extend the existing security tests with a separately reported longitudinal benchmark:

- Run-start profiles and project summaries over 10, 50 and 100 sequential runs;
  relevant context survives without irrelevant history consuming the context budget.
- Parallel children and sibling runs; downward handoff and upward reporting; retries
  and later runs of the same agent identity; positive and negative visibility checks.
- User preferences shared across eligible threads, thread-local facts kept local,
  group membership changes, private memories and tenant boundaries.
- New facts, corrections, contradictory evidence, event time versus ingestion time,
  out-of-order arrival, delete/TTL/revoke while refresh is in flight, provider outage.
- Tool successes/failures and procedure reuse; task completion without repeating
  known failed actions; memory poisoning and unsupported-summary rejection.
- Novel questions over generic maintained summaries and separately repeated questions;
  do not bake evaluation questions or answers into mental-model definitions.

Report task success, source/claim correctness, stale/wrong-memory rate, unauthorized
disclosures, read-after-delete behavior, memory-service and total agent calls per
task, tokens per turn, ingestion/refresh cost, startup-context latency, p50/p95/p99,
refresh lag and context size. Keep isolation results separate from quality averages.
Use independent expected-state/permission oracles and held-out realistic traces;
passing only fixtures authored alongside the implementation is insufficient.

### Native experiments

First correct event-time handling, source lineage and permission/deletion guards.
Give native extraction and landing consolidation independent controls. Evaluate:

| Arm | Contextual extraction | Landing consolidation | Purpose |
| --- | --- | --- | --- |
| N0 | Off | Off | Frozen baseline |
| N1 | On | Off | Extraction contribution |
| N2 | Off | On | Consolidation contribution |
| N3 | On | On | Interaction between the two |

Test temporal ranking separately against frozen identical corpora. Preserve historical
validity for “as of” queries; a universal recency boost is not a temporal solution.
The write stack changes schema and input corpus: per-arm database, vector namespace,
tenant/cache state and matching source checkout are mandatory. Do not migrate the
shared baseline to 0009–0011. Reuse identical corpus snapshots for retrieval-only arms.

### Hindsight and common accuracy evaluation

Run extraction preview on a small development sample with a hard provider spend cap
and recorded tokens, then compare a separately ingested Hindsight enriched arm. Expose
one new capability at a time: observations, stored mental models, then optional reflect.
Keep model choice fixed during mechanism comparisons; change models in separate arms.

LoCoMo remains the full ten-conversation external regression: exact source-ID
recall/complete@10/20/50, actual returned-context coverage, spans/provenance and separate
answer correctness. An ID match proves source presence, not necessarily full evidence
content in a split turn. Use the same reader, prompt, judge, evidence budget and
adversarial policy for end-to-end answer comparisons. Failed calls remain unmeasured,
not wrong answers or lexical fallback successes. Record all denominators and missing
annotations. Preserve the strict ruler; changing the ruler is not an accuracy gain.

Generation-free retrieval runs can measure evidence and latency without paid calls.
Reader/judge evaluation still costs calls, even when the memory backend's read path
is LLM-free. Do not confuse the agent/benchmark reader with a hidden memory-service call.

Retain SciFact and the multi-chunk golden document corpus for RAG; test multilingual
queries and sources separately. Neither document retrieval nor agent-task improvement
is established by LoCoMo alone.

Latency runs use the same HTTP boundary, controlled host load, declared hardware,
cold/warm states and concurrency. Test simultaneous ingestion/refresh so background
work cannot silently ruin p99. Reuse recorded contexts for reader comparisons where
valid, checkpoint responses, pace requests and account for retries. No “two hours”
or “cents” promise until a pilot establishes throughput and total cost.

## Implementation milestones and promotion gates

1. **Measurement and contracts:** exact provenance capture, agent evaluation,
   operating-mode enforcement, bank-policy mapping and deletion state-machine tests.
2. **Native corrections and arms:** timestamps, independent flags, N0–N3 and temporal
   ranking. Promote only changes supported by the intended workload and regression suite.
3. **SDK adapter and synchronization:** pinned client/runtime, async outbox projection,
   source mappings, quarantine/invalidation, contract tests, failure injection and cost
   accounting. Start with extraction preview and a controlled persistent bank.
4. **Knowledge features:** observations, mental models, knowledge pages and local
   validated read projections. Demonstrate recurring-run and novel-question behavior.
5. **Complete product surface:** memory/document lifecycle, bank configuration,
   templates, operations/webhooks, authorized MCP and multilingual validation.
6. **Deep mode:** opt-in reflect, per-request deadline/cost controls and matched answer
   evaluation. Product feature completion and benchmark accuracy remain separate gates.

Zero unauthorized results and no post-invalidation exposure are release conditions
for the tested scenarios, not statistical quality targets. Report quality deltas with
paired counts and uncertainty; retain per-category regressions and actual costs.
90%+ answer accuracy is an objective to measure, not a promised consequence of adoption.

Follow the current layer boundaries, typed ports, PEP8, lint/type/architecture checks,
and focused behavioral regression tests. Remove superseded paths after migration,
reuse shared source/lifecycle logic, and avoid keeping two active implementations of
the same mechanism without an explicit provider/evaluation purpose.

Each milestone updates this handoff with commit, flags, schema/corpus versions,
provider/model versions, test commands, benchmark manifests, counts, cost, failures
and remaining limitations. This plan does not authorize silent shared-DB migrations,
runtime default changes, paid runs beyond the remaining budget, or publication of
unmeasured accuracy claims.

## Sources

- Existing implementation audit: `HINDSIGHT-CAPABILITY-AUDIT-2026-09-27.md`.
- Existing branch/migration handoff: `CODEX-STACK-HANDOFF-2026-09-27.md`.
- [Python SDK](https://hindsight.vectorize.io/sdks/python), including async client use.
- [Observations](https://hindsight.vectorize.io/developer/observations), scoped consolidation.
- [Mental models](https://hindsight.vectorize.io/developer/api/mental-models), refresh,
  stored content and deletion/staleness limits.
- [Reflect](https://hindsight.vectorize.io/developer/api/reflect), reasoning and provenance.
- [Knowledge pages](https://hindsight.vectorize.io/developer/api/knowledge-pages).
- Pinned local SDK: `/Users/ricky/usage_data/hindsight-benchmark-upstream/hindsight-clients/python`.

No production code changed and no paid model calls were made to finalize this plan.
