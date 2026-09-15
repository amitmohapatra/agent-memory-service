# Final report — Enterprise Multi-Agent Memory Service (M0–M13)

Date: 2026-09-15 · Commit: see `benchmark/results/*.json` → `provenance.git_commit`

## Verdict, in one paragraph

Every hard gate passes in the build environment, and the release-gate evaluator says
`RELEASE GATE: PASS` — with two caveats it prints itself: retrieval quality was measured
with the deterministic hash embedding and lexical reranker (the Granite/MiniLM weights
could not be downloaded in the build sandbox), and latency was measured over in-process
ASGI with local Qdrant, in-memory cache and blob providers. Those numbers bound the
service's own logic and overhead; they are not a production measurement. **This service
is not declared production-ready.** Production readiness requires re-running `make gates`
with the real embedding/reranker weights in `models/`, real Qdrant/Dragonfly/GCS, and a
network hop (`benchmark/load/locustfile.py`), and every gate passing again. Everything
needed to do that is in the repository; nothing in the gates is relaxed for the sandbox.

## What was built

A FastAPI service with a hexagonal core (domain → ports → application/modules →
adapters; enforced by ruff banned imports and an architecture test), PostgreSQL as the
only source of truth, Qdrant as a rebuildable index, Dragonfly/Redis as a cache that is
never load-bearing, a transactional outbox into Procrastinate, GCS/filesystem archive
segments with checksums and a reconciler, Docling-backed document parsing into a context
graph, hybrid retrieval (dense + BM25, RRF, bounded rerank) with graph, expansion and
verification stages, native rule-based memory intelligence with temporal consolidation,
a PostgreSQL knowledge graph, multi-agent visibility semantics, a Python SDK
(`universal-memory`) and a LangGraph adapter. Fifteen ADRs record the decisions
(`docs/adr/`); `docs/MILESTONES.md` lists what each milestone delivered.

## Gate evidence (from `benchmark/results/`)

| Gate | Requirement | Measured | Artifact |
|---|---|---|---|
| Acknowledged data loss | 0 | **0** over 200 messages, 59 observations, 6 uploads acknowledged during cache/blob/queue outages and 285 injected worker crashes; 0 duplicate memories after recovery | `durability.json` |
| Unauthorized retrieval | 0 cross-tenant, cross-user, private-agent | **0 / 0 / 0** (property-based oracle, exhaustive tenant matrix, retrieval- and graph-level reader matrices incl. run lineage; 10 suites) | `security.json` |
| Critical Recall@20 | 1.00 | **1.00** (9 critical questions, golden set `acme_fy26_critical`) | `retrieval_gate.json` |
| Critical Evidence-Group Recall | 1.00 | **1.00**; evidence-complete rate 1.00 | `retrieval_gate.json` |
| False-merge rate | ≤ 0.01 | **0.00** on 35 labelled pairs; dedup recall 1.00 | `memory_gate.json` |
| p95 latency | chat 100 / cached ctx 75 / recall 300 / bundle 400 / file 200 ms | **32.7 / 6.2 / 71.3 / 78.3 / 25.3 ms** (in-process, 50-document corpus) | `performance.json` |
| Failure recovery | all scenarios pass | **worker_kill, cache_flush, blob_outage, search_rebuild, authz_denial: pass** | `failure_injection.json` |
| All tests pass | 0 failed, 0 errors | **274 passed, 0 failed, 0 errors** (unit, contract, integration, security, e2e, eval, failure, SDK, LangGraph) | `tests.json` |

Advanced retrieval strategies (`advanced_retrieval.json`): PageIndex, RAPTOR-style summary
fusion and graph personalised PageRank are *adoptable* (no critical gate below baseline,
within the latency budget); SPLADE, miniCOIL, ColBERT and late chunking are *skipped* —
their adapters refuse to run without local weights rather than degrade silently.

## What the chaos run found (and fixed)

The durability benchmark (`benchmark/durability.py`) found two real defects before it
reported zero loss, both now covered by `tests/failure`:

1. Concurrent first messages of a new thread raced on thread creation (unique-key
   violation → 500). Fixed with a transaction-scoped advisory lock per thread
   (`UnitOfWork.serialize`), which also serializes session/turn creation.
2. A message acknowledged while the cache was down was hidden from the thread listing once
   the cache returned: the hot thread list was trusted without checking it was current.
   The list is now validated against the thread revision (and against a truncated head)
   and refilled from PostgreSQL on a miss.

Also added in M13: stalled-job recovery for workers that die mid-job (Procrastinate
heartbeats + periodic requeue), `tools.reindex` to rebuild the search index from
PostgreSQL, a per-tenant rate limit with a shared counter that fails open, and the
`pytest_results` plugin so "all tests pass" is evidence rather than a claim.

## Known limitations and what remains before production

- **Model weights.** Granite/MiniLM embeddings, the cross-encoder reranker, SPLADE,
  ColBERT and the Docling PDF models were not downloadable in the build sandbox. The
  adapters exist and are contract-tested with fakes; quality gates must be re-run with
  real weights (`-m models` tests, `make gates`). Until then no retrieval-quality figure
  in this repository is representative.
- **Real infrastructure.** Qdrant ran in local mode, the cache in memory, blobs on the
  filesystem, and the task queue in the in-process recording mode for the chaos run
  (the SIGKILL scenario uses real Procrastinate on real PostgreSQL). Docker-gated tests
  (`-m docker`: OpenFGA, Qdrant server) need a Docker host.
- **Latency.** Figures are in-process; run the Locust file against a deployed instance
  and compare with `performance.json` — the difference is the network and server hop.
- **LLM-backed features** (Mem0/LangMem/Cognee providers, GraphRAG community summaries)
  are off by default and documented as such; the native providers carry the gates.
- **Language.** The native memory rules and the query router are English-first.

## How to re-validate

```bash
make validate            # lint, types, every suite, all gate artifacts, then the evaluator
make gates               # only the artifacts under benchmark/results/
uv run python -m memory_service.tools.release_gate
```

With real weights in `models/` and the compose stack up (`make dev-up`), set the
`MEMORY__MODELS__*`, `MEMORY__SEARCH__*`, `MEMORY__CACHE__*` and `MEMORY__BLOB__*`
variables and run the same commands; the artifacts will record `representative: true`
and the evaluator will print a PASS without caveats — or a FAIL with the reason.
