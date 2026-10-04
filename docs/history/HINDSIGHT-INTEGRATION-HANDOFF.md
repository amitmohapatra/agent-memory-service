# Integrated memory work: implementation and verification handoff

> Historical integration/benchmark stack. The later combined code is in `ams-agent-capabilities`;
> read `AGENT-CAPABILITIES-HANDOFF-20260927.md` for credentials, briefs and migrations 0012–0013.

Worktree: `ams-hindsight-integration`; branch: `codex/hindsight-integration-20260927`.
Base: `f988c26` (write-path stack, including migrations 0009–0011). This is **not**
the deployed baseline. Do not migrate a shared benchmark database or compare this
corpus to an older arm without an independently ingested database.

## Authorization and constraints

The user authorized implementation and testing, CPU model replacement, independent
LLM ingestion/retrieval policies, integrated Hindsight reuse, semantic caching,
multilingual evaluation, and source-backed continuous memory. No paid model calls
while credit is unavailable. Never put the key supplied in chat in a report, log,
fixture, command output or commit. Agents' model calls from memory must exclude MCP
tools. Existing service authorization and deletion semantics remain authoritative.

## Implemented in this worktree

- Source chronology for native beliefs/entity summaries, with separate processing
  timestamps; independent consolidation wiring. Consolidation remains unpromoted
  pending a measured corpus-changing arm.
- Hindsight 0.10.1 extraction-preview SDK as a core dependency, automatically used
  for eligible contextual extraction when that LLM use is enabled. One bounded
  attempt, bounded concurrency, service-owned instructions, original source
  evidence, native authorization, no remote persistence. This is not full Hindsight
  retain/reflect/mental-model integration.
- Real SDK HTTP contract and isolated PostgreSQL ingestion/replay/deletion tests.
  Responses in these tests are fixtures; they do not prove model accuracy.
- Native Bifrost calls explicitly deny MCP clients/tools and reject tool-call
  responses. The Hindsight server's own model requests still need the same policy
  and per-agent virtual-key propagation before customer keys can use that path.
- Unicode sparse tokenization, with a new index fingerprint requiring reindexing.
- Conservative semantic context cache: scope/revision/configuration/budget/document
  boundaries, embedding similarity plus ordered lexical guards, query-vector reuse
  on misses, bounded entries and TTL. It does not cache generated answers or claim
  safe arbitrary paraphrase equivalence.
- Explicit bundle replay now checks live content and authorization revisions too. A
  handle cannot replay the old bundle after forgetting or a membership change.
- Qualified graph assertion identity is shared by SQL-before-LIMIT and cross-hop
  deduplication. Equal subject/predicate/object with different period, validity,
  document, layer or attributes survives. Equivalent repeated supports still collapse.
- CPU reranking uses the existing serial executor, an explicit input-token bound and
  a profile fingerprint included in context cache namespaces. No neural reranker has
  been promoted to the shipped defaults.
- Dense encoders support E5's asymmetric query/document prefixes; profile changes
  receive separate index fingerprints. Fingerprints are computed once per encoder.
- Both real embedding adapters release the serial executor between indexing batches.
  A queued interactive query no longer waits for the entire document. Batch order and
  vectors remain covered by arithmetic tests; a controlled concurrency test verifies
  that the query runs between batches. `benchmark/embedding_interference.py` measures
  this with real weights. Initial runs finished; a smaller per-turn batch cap is
  now under validation (see the latency notes below).

## Measurements so far

`benchmark/results/multilingual_sparse_{baseline,unicode}.json` use all 1,190 queries
in each of 12 XQuAD languages. Public data is pinned and hashed in the ignored
dataset manifest. This repurposes QA data for **paragraph retrieval** and is not an
official XQuAD QA score, full RAG score, or LoCoMo answer accuracy. Same-language
recall@10 improved substantially for Arabic, Greek, Hindi, Russian, Thai, Vietnamese
and Chinese; English is unchanged. German/Spanish/Romanian/Turkish have small
regressions that must remain visible. Cross-language retrieval needs a multilingual
dense encoder; sparse tokenization does not solve it.

The earlier regression run passed 922 tests with one skip, before the newest Unicode,
MCP and semantic-cache changes. Do not report that count as validating the final tree.
The complexity ratchet subsequently passed after separating cache lookup, revision
reading and semantic bookkeeping; its limits were not increased.

MiniLM, E5-small int8 and the English Granite baseline have completed all 12 languages.
See `MULTILINGUAL-CPU-EVALUATION-20260927.md` for the table and limits. Mean dense
paragraph R@10 is 65.96% for English Granite, 99.15% for E5, and 93.52% for MiniLM.
These are component retrieval results, not LoCoMo or answer scores. E5 hybrid fusion
is slightly worse than its dense-only result, which needs separate task calibration.
The 97M multilingual Granite challenger has also completed all 12 languages (mean dense R@10 96.37%, hybrid 98.15%). No embedding replacement
has been promoted. Timings encountered other test/process activity; do not quote them
as a controlled comparison or application p99.

E5 weights: `intfloat/multilingual-e5-small` revision
`614241f622f53c4eeff9890bdc4f31cfecc418b3`, locally quantized to QInt8, per-channel,
reduced range for AVX2; 118,308,812 bytes. Specs and file hashes live in ignored
`.bench_data/models/`. MiniLM revision is
`e8f8c211226b894fcb81acc59f3b34ba3efd5f42`.

`benchmark/native_source_retrieval.py` now evaluates exact observation-source IDs at
fixed memory depths, with graph/derived companions consuming positions. It explicitly
does not score generated answers. Its first E5 trial was stopped before a completed
conversation because contended graph deadlines changed the candidate set. Restart on
an idle machine, rather than treating this trial as a quality result. No new full
LoCoMo accuracy or whole-service p99 has been measured. No paid calls were made.

## Regression discoveries

- Both the graph and tool gates failed against unchanged `f988c26` as well as this
  tree. Graph dedup discarded qualified facts; the production fix above resolves it.
  The tool gate stored training trajectories as private RUN memories then expected
  another run to use them. Training data is now explicitly AGENT_GROUP-shared; rival
  runs remain private. Product access rules were not weakened. Both gates plus graph
  query units passed 13 focused cases after these corrections.
- The worker failure test imported another editable checkout in its subprocess.
  It now prepends this worktree's `src`; the real crash/recovery test passed.
- Queue contract tests never requested the fixture that creates their isolated DB,
  silently skipping each real queue case and leaking failed clients. They now create
  it and close clients on every outcome: 10 passed, two intentional worker-contract
  skips, in 3.11 seconds.
- The search-rebuild failure test still passed a removed `workspaces` keyword to
  `grant_membership`. That obsolete argument was removed; the actual rebuild
  comparison remains. The focused rebuild/cache/profile/complexity run passed 58 tests.
- A stale sparse fingerprint expectation was updated to the intentional new version.
- Full suite logs are under `.bench_data/hindsight-integration/`. Earlier interrupted
  runs are not validation. Complete regression counts and later focused checks are
  recorded below; do not reuse historical counts for subsequent edits.

## Hindsight and agent-key boundary

The pinned SDK's extraction request has **no per-request model credential field**.
Its API key authenticates the memory server, not Bifrost. Upstream explicitly excludes
model credentials from bank/tenant config API overrides. Do not forward a customer VK
as Hindsight's API key, mutate a global key between agents, or claim per-agent routing
is implemented. The current extraction adapter requires an operator-controlled server;
its own model calls still need end-to-end gateway/MCP enforcement before customer use.

The SDK also offers `banks.preview_prompt` for retain/consolidation/reflect without
model calls or source data. Reusing validated prompt components through our own Bifrost
adapter is a possible integration design, but **not implemented**. Preview includes
runtime placeholders and a rendered clock; blindly replacing strings would introduce
another chronology bug. Persistent mental models/knowledge pages and per-agent
credential lifecycle are also outstanding, not covered by the extraction fixture.

## Outstanding work, in dependency order

1. Finish cache integration/security tests, complexity checks and regression suite.
2. Measure CPU multilingual dense candidates and a small reranker with pinned model
   revisions; report per-language quality, cross-language quality, RAM, and latency.
   Promote only measured improvements. Do not use a GPU throughput claim as CPU data.
3. Add protected per-agent credential references and independent ingestion/read
   policies, binding credentials to tenant and owner principal. Test concurrent
   agents, rotation, revocation, background jobs, logs and MCP exclusion end to end.
4. Integrate persistent Hindsight features only behind source/audience mapping,
   provenance, revision and deletion guarantees. Cover observations, mental models,
   maintained knowledge pages and opt-in LLM deep reflection. No-LLM reads must stay
   model-call-free; precomputed synthesis may be read without a generation call.
5. Evaluate native consolidation and extraction separately on isolated corpora;
   evaluate combined changes only after individual effects are known. Cached reader
   predictions cannot establish accuracy for newly changed retrieval contexts.
6. Add agent-task evaluation across runs/threads, including memory correctness,
   contradictions/corrections, access boundaries, forgetting, calls and tokens.
7. Run full LoCoMo retrieval and document RAG comparisons, with consistent depth caps,
   corpus/model hashes, paired results and separately measured end-to-end latency.
8. Review changes, update capability coverage, deployment and migration instructions,
   and retain attributable implementation/test artifacts. Never claim 90% without
   a matching, completed evaluation.

## Isolated test commands

Use the shared interpreter but this worktree's source and isolated SDK install:

```sh
export PYTHONPATH=src:.sdk-test-deps
export MEMORY_TEST_DATABASE_URL=postgresql+psycopg://memory:memory@localhost:5432/memory_hi_20260927
export MEMORY_TEST_APP_DB=memory_hi_failure_20260927
export MEMORY_TEST_QUEUE_DB=memory_hi_queue_20260927
export MEMORY_TEST_ADMIN_URL=postgresql://memory:memory@localhost:5432/postgres
/Users/ricky/usage_data/agent-memory-service/.venv/bin/python -m pytest tests/unit tests/security tests/contract/test_hindsight_sdk.py tests/integration/test_hindsight_ingestion.py
```

Do not use `make bench-db`: it addresses the shared benchmark database. The SDK test
dependency directory and dataset/model downloads are ignored and must not be committed.

The disposable Qdrant for this work is `memory-hi-qdrant-20260927`, REST port 16333,
gRPC port 16334. Own benchmark databases are `memory_hi_perf_base_20260927` and
`memory_hi_perf_changed_20260927`. They may be reset by these arms. Run arms sequentially:
they otherwise share vector tenant `bench_conv` even though SQL databases differ.

## Review repairs and operational notes (latest pass)

- A generated extraction's category/provider now survives SQL, search-index, `/recall`,
  `/context` and rendering projections. It is labelled unverified, cannot reinforce or
  fuzzy-merge into an asserted source, and cannot prove itself through `/verify`.
  Ordinal citations retain their positions. Generated contextual facts are excluded
  from graph enrichment and higher-order reflection source discovery.
- Audience-aware revision invalidation is shared by ingestion, reflection, forgetting,
  archive/restore, expiration and indexing completion. Shared TENANT, AGENT_GROUP,
  RUN and THREAD audiences conservatively invalidate the tenant revision; threadless
  readers may read multiple granted threads. PRIVATE principal audiences invalidate
  their user/agent independent of their storage anchor. Deleted rows missing from the
  completion lookup still invalidate the tenant. Broad invalidation is a correctness
  tradeoff; measure cache hit rate under write load before refining its granularity.
- Model download provenance is now pinned before download. Cached bytes retain their
  recorded revision; checkpoint refresh invalidates exported ONNX graphs. An incomplete
  marker survives download/publication failure and is cleared only after atomic
  manifest replacement. Tests cover interrupted refresh and publication.
- Native adversarial review found and drove those repairs; its last source pass found
  no additional actionable defect. Fixtures were reviewed in summary, not independently
  executed by the reviewer. Paid outside Claude review was not run under the no-credit
  constraint. This is not a completed whole-product parity certification.
- Latest focused checks: 75 provenance/cache/consolidation/grounding tests; 21 shared-thread
  cache/provenance/OpenAPI tests; 56 principal-cache/reflection/download-provenance tests.
  Ruff checked/formatted 409 files. Pyright reported zero errors and 21 missing optional
  heavy-dependency warnings, including the final downloader publication changes.
- Earlier complete non-unit run passed 360, skipped 18, deselected 12. A subsequent full
  unit run passed 941 with one failure from an obsolete reflection fake; the production
  path now reuses inserted objects instead of an unnecessary second SELECT. A fresh full
  unit run passed **954 with one skip** (`unit-final-reviewed.log`, 93.56 seconds);
  fresh non-unit run passed **377 with 18 skips and 12 deselections**
  (`nonunit-reviewed.log`, 635.32 seconds). These full runs predate the later expiry,
  shared-Unicode and read-policy changes; their focused coverage is listed below.
  Historical gate JSONs were restored; new outputs live in gates-reviewed/.
- The prior regression hang was CPython faulthandler cancellation on macOS, confirmed
  with a process sample. Runs use `-o faulthandler_timeout=0 --timeout-method=thread`,
  preserving the actual test timeout rather than allowing indefinite tests.
- One earlier focused queue test omitted MEMORY_TEST_QUEUE_DB and created two uniquely
  marked testcontract.noop jobs in memory_queue_tests. Only those two still-TODO jobs
  were deleted with exact ID/task/status/marker predicates after backing them up in
  `.bench_data/hindsight-integration/default-queue-cleanup.json`. No benchmark corpus,
  database schema or other jobs were removed. Always source test-env.sh now.

## Isolated benchmark queue

`.bench_data/hindsight-integration/serial-benchmarks.sh` runs encoders sequentially:

1. Ettin 17M English reranker screening on 100 XQuAD queries, fixed 20-candidate pools,
   CPU-only Docker with network disabled. This screen cannot establish multilingual
   quality or full-corpus improvements.
2. Before/after embedding interference with identical Granite weights.
3. Full LoCoMo source coverage against f988c26's source, current source with Granite,
   then current source with E5, each re-ingesting its isolated corpus.

LoCoMo uses exact observation IDs; direct and transitive evidence lineage are reported
separately at fixed depths. Graph/derived companions consume slots. These metrics do
not establish whether a fragment actually answers the question. The harness disables
all LLM calls regardless of environment. It measures context-builder latency, not HTTP
or production p99. Stop on any error; inspect logs before restarting.

An additional isolated unit DB is memory_hi_unit_20260927. The main/failure/queue test
DBs and benchmark DBs remain distinct. The public Ettin weights are pinned at
9e4aa35321a6dd1a43ca313f500c4b4f7cfb5cc6 with hashes under .bench_data/models/.

## Latest lifecycle, read-policy and lexical changes

- Expiry formerly left active graph assertions behind. A real PostgreSQL regression
  failed before the repair. Expiry now enqueues `memory.index` in its SQL transaction,
  reusing search/graph removal and revision updates through the durable outbox. Cleanup
  survives a crash after expiry commits. The lifecycle/batch focused run passed 70.
- Shared Unicode primitives in `domain/text.py` serve sparse indexing, source splitting,
  document summaries and evidence terms. Existing text sanitization is preserved.
  Combining marks remain attached; non-Latin sentences are no longer scored with an
  ASCII-only word regex. 73 focused tests passed, including the unchanged complexity
  budget. This is functional language support, not multilingual NLI validation.
- `/context`, `/recall` and `/verify` accept `use_llm` (default false). It permits only
  configured model uses, never turns a disabled provider on. The task-local policy wraps
  the complete read, including verification. Nested operations cannot widen a denial.
  Ingestion settings remain independent. Python SDK methods expose the same keyword.
  This changes prior implicit read assistance: existing clients wanting it must now
  send `use_llm=true`. Exact and semantic caches include the request policy.
- Read policy: 43 focused unit/HTTP/OpenAPI checks and eight SDK tests passed. The HTTP
  tests use the actual Bifrost adapter with mocked responses: zero calls by default,
  one opted-in expansion, no later call for a native read. No provider was billed.
- `unit-policy-final.log`: 970 passed, one skipped, before the final provenance repairs.
  A later combined regression (`regression-policy-provenance-verbose.log`) passed
  **1,376, with 19 skips and 12 deselections**, in 493.30 seconds. It imported source
  before the final SQL provider/summary and reparse repairs. The focused fresh-import
  run `trust-and-index-final.log` passed 15 covering those repairs and complexity.
  `final-unit-and-repairs.log` is the fresh unit/API/SDK/repair rerun after selecting
  the one-document executor turn; inspect its completion before quoting a result.

## Completed CPU screens and their decisions

- Ettin 17M, 100 English XQuAD queries, fixed 20-passage pool: R@1 94% → 99%, MRR@10
  0.9637 → 0.995. Component median 2,095 ms and p99 6,334 ms on the tested CPU/Docker
  profile. Keep neural reranking off on the normal read path; this is too slow and
  does not establish multilingual or LoCoMo gains. Artifact: `ettin_cpu_screen.json`.
- Embedding interference before/after releasing the runner between batches: worst
  observed query stall 11,196 ms → 2,832 ms. The small-sample p99 rose from 45 ms to
  2,775 ms because four shorter stalls replaced one long stall. Do not call this a
  p99 win. This finding drove the separate document-turn experiment. These synthetic
  component measurements exclude HTTP, authorization, SQL and search.

## Latest review and measured scheduling decision

- SDK inline verification now preserves provenance attributes through bundle, typed-item
  and dictionary inputs; unsupported generated text is blanked without moving citation
  positions. Native LLM rewrites, reflection and contextual extraction share one trust
  predicate. The corresponding SQL predicate uses the dedicated `MemoryRow.provider`
  column, not the JSON metadata from which that field is removed on insert.
- Generated document summaries carry `provider=llm` in search. SQL node summaries stay
  extractive, so parent expansion cannot accidentally promote generated text to evidence.
  Legacy SQL/index summaries need reindexing; current code does not rewrite old data at boot.
- Reparse formerly purged old chunk vectors but retained old summary vectors. The expanded
  real PostgreSQL regression failed with three orphan summary IDs before repair; cleanup
  now retains exactly the current chunk and summary generations.
- The first combined regression hit the 120-second timeout in the worker-crash case. It
  passed alone in 5.87 seconds and passed in the complete verbose rerun. The intermittent
  first-run timeout was not conclusively attributed; do not call it a repaired product bug.
- Latest native review of the two provenance repairs found no additional supported defect.
  This was a focused read-only pass, not complete parity certification or outside review.

Controlled sequential Granite screens used 128 passages and 100 interactive queries, with
the same source/weights, two model threads and no other real encoder. Small samples remain
exploratory:

| Documents per executor turn | Concurrent query p99 | Worst query | Whole workload |
|---:|---:|---:|---:|
| 1 | 232.06 ms | 334.93 ms | 14.68 s |
| 2 | 446.91 ms | 722.48 ms | 15.13 s |
| 4 | 2,225.84 ms | 2,349.76 ms | 26.64 s |

Selected `document_batch_size=1`. This bounds an indexing turn, not total request
concurrency or production latency. Artifacts: `embedding_interference_batch_{1,2,4}.json`.

## Completed full LoCoMo source baseline

`locomo_source_baseline.json`: old imported source `f988c26`, English Granite, own database
and Qdrant. All 1,986 questions processed; 1,536 answerable questions have usable source
annotations. Four answerable questions lack annotations; 446 adversarial questions are
not part of source-recall means. No model API calls and no generated answer scoring.

| Category | n | Source recall@50 | Complete@50 | Source recall@100 | Complete@100 |
|---|---:|---:|---:|---:|---:|
| All annotated answerable | 1,536 | 77.26% | 70.77% | 84.22% | 78.06% |
| Temporal | 321 | 83.77% | 81.93% | 89.90% | 87.85% |
| Open domain | 92 | 51.59% | 42.39% | 59.61% | 51.09% |
| Multi-hop | 282 | 58.23% | 32.27% | 68.63% | 43.62% |
| Single-hop | 841 | 83.97% | 82.52% | 89.97% | 88.82% |

Baseline context-builder timing: p50 115.08 ms, p95 256.54 ms, p99 367.62 ms. This
excludes HTTP and uses in-process authorization/cache and lexical NLI, with real embeddings,
PostgreSQL and own Qdrant. Some focused host tests overlapped this baseline; do not present
it as controlled production p99. Inspect the imported-source manifest: generic Git metadata
names the harness checkout, whereas the source manifest records the old application files.

The changed-Granite and E5 arms are complete. See
`LOCOMO-OFFLINE-COMPARISON-20260927.md` for the paired results and category regressions.
Full SciFact encoder screening is complete; see `SCIFACT-CPU-EVALUATION-20260927.md`.
E5 regressed document hybrid nDCG@10 (0.7055 versus Granite 0.7409) and recall@10
(0.8482 versus 0.8912), so the global embedding default remains Granite. No paid calls.
`HINDSIGHT-CAPABILITY-STATUS-20260927.md` records implemented counterparts and explicit gaps.
