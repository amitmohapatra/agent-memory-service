# ADR 0031: Runtime: freshness, overload, pools, readiness, workers

Date: 2026-10-04. Status: accepted. Amends 0004 (the worker's lifecycle), 0021 (the tenant
registry), and the cache-invalidation scheme of `domain/revisions.py`.

## Context

An audit of the runtime found the service correct on the request path and fragile around it.

- **Stale context.** A cached bundle is keyed on the revisions it depends on (TENANT, USER,
  THREAD, AGENT, GRAPH, MEMBERSHIP). Three writes moved none of them. A working-memory item
  lives only in the cache but is folded into the bundle. A document becomes searchable when
  its index job runs, and that job bumped nothing a bundle reads (ingestion bumps DOCUMENT,
  which no lookup subscribes to). The bundle cached before either write was served until its
  five-minute TTL. In the other direction, GRAPH was one counter per tenant, bumped by every
  memory-index job and every document enrichment, so each write dropped every cached bundle
  of the tenant, including bundles of users who could read none of the new facts.
- **No overload bound.** One in-process model sits behind a one-permit semaphore with
  unlimited waiters, and nothing bounded a request's duration. On the 4-core box the earlier
  load run (`benchmark/results/load_4core_10rps_cold.json`) shows the result: `/v1/context`
  p95 35 s and a verified context at 96 s, with every caller waiting and none refused. The
  verified-context judge was bounded only by 30 s × 3 attempts per claim.
- **Connections.** The pools were sized per process with no account of the whole pod, the
  task queue's psycopg pool had no statement or connect timeout, and nothing allowed a
  transaction pooler in front of PostgreSQL.
- **Readiness** failed when Qdrant, OpenFGA, the blob store or the task queue was down. A
  shared dependency blinking took every pod out of rotation at once, turning a partial outage
  into a total one, and every probe pinged every store.
- **Workers.** The job worker ran with Procrastinate's signal handling off: SIGTERM killed it
  mid-job and the job sat in `doing` until the stalled-job reconciler noticed. The worker
  exposed no metrics. The API's `/metrics` was one worker's share of the traffic.
- **CPU on the loop.** The builtin parser and the document fact extraction ran pure-Python
  work over a whole document on the event loop. Thread authorization ran under the thread's
  advisory lock.
- **The tenant registry** reloaded every live key id every minute in every process. A
  suspension or revocation reached other processes only on that refresh.

## Decision

**Freshness.**

1. A successful working-memory write (an EPHEMERAL candidate, or a DEFER verdict) bumps the
   thread's revision, or TENANT when the item is anchored only by an agent run, in the
   observation's unit of work (`EphemeralMemory.revision_key`).
2. `index_document` and `delete_document` bump the revisions of the document's audience:
   its visibility keys read like a memory's, plus its thread (`document_revision_keys`). This
   happens after the previous generation is purged, whether or not graph enrichment is on.
3. Graph changes bump the audience of the facts they wrote: the memory's
   (`memory_revision_keys`) or the document's, plus the readers of every entity whose summary
   was rewritten; tool edges bump their calls' audiences. A relation is readable exactly by
   its visibility keys, so this invalidates every reader who can see the change and no one
   else. GRAPH remains for a change whose audience is unknown (the facts of a deleted
   memory). Kept as before: an entity's ranking signals (mention counts) can change for
   readers outside the new fact's audience without invalidating their bundles. Their
   bundles keep the old order until the TTL; no fact they cannot read reaches them.
4. Query vectors are cached per dense space, keyed
   `emb:<encoder fingerprint>:q:<sha256(query)>` (one day), and the three forms of a bundle
   are written in one pipelined `mset`.
5. The revision counters live only in PostgreSQL. The docstring that promised a Dragonfly
   read cache of them is corrected.

**Overload** (`constants.OVERLOAD`; constants, not settings, by the repository's
environment-surface rule).

6. At most 32 callers wait for one in-process model. A caller past that gets
   `DependencyUnavailable` (a retryable 503) at once.
7. A per-request deadline (`api/deadline.py`): 5 s for reads (GET, `/v1/context`,
   `/v1/recall`, `/v1/tools/hints`), 15 s for writes and for `/v1/verify`. Uploads and probes
   have no deadline. A request past its deadline is answered `504` with `TIMEOUT` problem
   details (retryable), built by the shared problem builder.
8. The judge gets one deadline per answer (`LLM.judge_deadline_seconds`, 8 s). A claim still
   undecided when it runs out stays `borderline`.
9. `memory-api` passes uvicorn `limit_concurrency` (128 per worker), `backlog` (2048), a
   graceful-shutdown timeout, and a 65 s `timeout_keep_alive`: above the SDK's 30 s keep-alive
   and the usual 60 s load-balancer idle timeout, where uvicorn's 5 s default closed
   connections clients were about to reuse.

**Connections** ([deploy/database.md](../deploy/database.md)).

10. One per-pod `MEMORY__DATABASE__CONNECTION_BUDGET` replaces `pool_size`/`max_overflow`.
    Each process takes `budget // processes` and splits it 4:2:1 between the request pool,
    the graph traversal and the task queue. Unset, it keeps the old sizes (8+8, 4+4, 4). The
    task queue's pool gets a size, a pool timeout, a statement timeout and a connect timeout.
11. A transaction-mode PgBouncer may front the request path (`TRANSACTION_POOLER=true`:
    psycopg prepares no server-side statements and sends no startup `options`; PgBouncer's
    `connect_query` sets the statement timeout). Work that needs a session goes direct
    (`DIRECT_URL`): Procrastinate's LISTEN/NOTIFY and job locks, the graph traversal's session
    timeout and prepared plan, and migrations. docker-compose runs PgBouncer this way.
12. Migrations are online. `migrations/env.py` sets `lock_timeout` 5 s and
    `statement_timeout` 15 min. Indexes on live tables are built `CONCURRENTLY` in an
    autocommit block. Migration 0024 adds the consolidation candidate index
    `(tenant_id, scope_key, temporal_status, updated_at DESC) WHERE deleted_at IS NULL`.
    `list_idle` is already served by `ix_memories_recent_updates`.

**Readiness and workers.**

13. Readiness is PostgreSQL and the process: `process` reads false once shutdown has begun,
    so a load balancer drains the pod before its pools close. Qdrant, OpenFGA, blob, the task
    queue, the cache and the gateway are reported (`degraded`) without failing readiness. An
    answer is reused for 3 s, and the pings run concurrently, each bounded at 2 s. Liveness
    checks nothing outside the process.
14. The job worker installs Procrastinate's signal handlers. On SIGTERM or SIGINT it stops
    fetching and gives running jobs 30 s; then it aborts them, and Procrastinate releases them
    for a retry (the handlers are idempotent). It serves Prometheus on
    `MEMORY__TASKS__METRICS_PORT`: queue depth and oldest-job lag per queue, outbox backlog,
    terminally failed jobs, running jobs, and the time of its last sample (the compose
    healthcheck).
15. With more than one API worker, `memory-api` sets `PROMETHEUS_MULTIPROC_DIR` (emptied at
    start) and `/metrics` sums every worker's values (`MultiProcessCollector`). It adds pool
    and model-queue gauges.
16. API workers and job concurrency default to the CPUs the container may use (cgroup quota
    and affinity), clamped to 1–8. The image still pins `WEB_CONCURRENCY=3` beside two math
    threads per worker.
17. The builtin parser and the document fact extraction run in a thread. Thread write
    authorization is asked before the unit of work opens (`may_write_thread`). Only a refusal
    is asked again under the lock, because the thread may have been created concurrently in
    the meantime.

**Search scale-out** ([deploy/search.md](../deploy/search.md)).

18. New collections take `shard_number`, `replication_factor` and `write_consistency_factor`
    from `SearchSettings`. `tenant_id` is indexed with `is_tenant=True`, and an existing
    plain index is re-created with the flag. Memory payloads stay in RAM up to 500 000 points,
    then move to disk in place. A new layout reaches existing collections only through
    `reindex --drop`.

**The tenant registry.**

19. Live keys are learned one by one (issued, announced or verified; bounded at 100 000), and
    the full key list is no longer reloaded. Changes are published on the cache's pub/sub
    channel and applied by every process. The small quota and suspension maps are still
    refreshed every minute, and after a lost subscription, so with the cache down nothing is
    worse than before.

**The read audit** stays an operational record with a stated loss window, not a compliance
ledger. A graceful stop flushes it. A process killed outright loses at most one flush
interval (1 s) plus a batch in flight. A full queue or an unstorable row drops the entry and
counts it (`memory_read_audit_dropped_total`). Making it a ledger would put a transactional
write, a WAL flush, on every recall. The docs promised the loss window, and the code keeps it.

## Consequences

- The environment surface grows from 36 to 41 fields. All five are topology: `connection_budget`
  replaces two pool sizes, plus `direct_url`, `transaction_pooler`, `tasks.metrics_port` and
  three Qdrant layout fields.
- `CacheProvider` gains `publish` and `subscribe`, and the cache contract covers them.
- Under overload the service now answers fast 503 and 504 responses, and clients retry them
  (the SDK honours `Retry-After`). It no longer queues every caller behind the slowest. The
  load sanity check in [MEASUREMENTS.md §9](../MEASUREMENTS.md) records the numbers measured
  on this box.
- Not changed: other services still authorize inside an open unit of work when they hold no
  lock (ingestion's thread check, reads). Each such call holds one pooled connection for one
  authorization round trip, which is bounded by the authorization timeout.
- Not done: Qdrant replication and write consistency are not raised on live collections by
  the service. That is a cluster operation.
