# ADR 0015: Release gates are produced by code, evaluated by code, and caveated honestly

**Status:** accepted · **Date:** 2026-09-15

## Context
The specification's hard gates (acknowledged data loss = 0, unauthorized retrieval = 0,
critical Recall@20 = 1.00, critical Evidence-Group Recall = 1.00, all tests pass,
false-merge rate and p95 latency within budget, failure recovery passes) are only worth
something if the evidence behind them cannot be hand-written and if a PASS says exactly
what was measured.

## Decision
- **Every gate has a producer under `benchmark/` and an artifact under
  `benchmark/results/`**, each with provenance (commit, platform, package versions):
  `durability.py` (chaos run over the real API: cache, blob and queue outages plus worker
  crashes, then every acknowledgement is verified), `security.py` and
  `failure_injection.py` (run the marked suites in-process and reduce outcomes — a missing
  or skipped test is a failed counter, never a zero), `performance.py` (the five budgeted
  operations over in-process ASGI), the eval suites (`retrieval_gate.json`,
  `memory_gate.json`) and the `pytest_results` plugin (`tests.json`). `make gates`
  produces all of them; `make validate` adds lint, types and the evaluator.
- **`memory_service.tools.release_gate` fails on missing evidence** and never relaxes a
  threshold. It also reports *representativeness*: quality measured with the hash
  embedding or latency measured without a network hop is flagged on every PASS, with the
  sentence that production readiness additionally requires the same gates with
  representative providers. A PASS in this repository is a statement about the service
  logic, not about a deployment.
- **Failure scenarios are real, not mocked:** a Procrastinate worker subprocess is
  SIGKILLed mid-job (recovery = stalled-job requeue from the periodic reconcile, replay
  guarded by idempotency); the cache is flushed and taken down mid-conversation; the blob
  store is down during archiving; the search index is dropped and rebuilt from PostgreSQL
  (`tools.reindex`); the authorization provider is down (fail closed).
- **Hardening that the chaos run forced:** concurrent first messages of a new thread are
  serialized with a transaction-scoped advisory lock (`UnitOfWork.serialize`), and the hot
  thread cache is validated against the thread revision and refilled from PostgreSQL on a
  miss, so a message acknowledged during a cache outage is never hidden by a stale cached
  list. Plus a per-tenant rate limit (shared counter in the cache, fail-open, 429 with
  `Retry-After`) and the existing body-size limit and error envelopes.

## Consequences
- The gate artifacts are committed with their provenance so a reviewer can see what was
  measured where; re-running `make gates` overwrites them.
- Two bugs found by the durability run (thread-creation race, stale hot cache after an
  outage) have regression tests in `tests/failure`.
- Representative gates (Granite/MiniLM embeddings, real Qdrant/Dragonfly, network) are
  the remaining step before any production claim; the harness records them as skipped
  or non-representative until the weights and services are present.
