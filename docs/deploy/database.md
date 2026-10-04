# PostgreSQL: connections, PgBouncer, online migrations

PostgreSQL is the source of truth and the job queue (ADR 0004). This page is how a deployment
sizes its connections to it, where a transaction-mode PgBouncer fits, and how a schema change
is applied without blocking a live service. The decisions are ADR 0031.

## One connection budget per pod

Every process opens three pools:

| pool | connects to | used by |
|---|---|---|
| request pool (SQLAlchemy, size + overflow) | `MEMORY__DATABASE__URL` | every request and job: units of work, reads |
| graph traversal (SQLAlchemy, size + overflow) | `MEMORY__DATABASE__DIRECT_URL` | the budgeted graph prefetch (150 ms session `statement_timeout`, a prepared plan) |
| task queue (psycopg pool, max) | `MEMORY__DATABASE__DIRECT_URL` | Procrastinate: deferring jobs, the worker's fetch, LISTEN/NOTIFY |

`MEMORY__DATABASE__CONNECTION_BUDGET` is how many connections one pod (container) may open
across all of its processes and pools. A pod of the API runs `service.workers` processes; the
job worker's pod runs one. `DatabaseSettings.pool_plan` divides it:

```text
per_process = connection_budget // processes          (unset: 28)
queue       = max(2, per_process // 7)
graph       = max(2, 2 * per_process // 7)            size ceil(g/2), overflow floor(g/2)
main        = max(2, per_process - graph - queue)     size ceil(m/2), overflow floor(m/2)
```

Unset, a process keeps the pools it had before the budget existed: requests 8 + 8, graph
4 + 4, queue 4 (28). Overflow counts against the budget because a burst is exactly when the
budget matters. `tests/unit/test_process_model.py` asserts that a plan never exceeds its
share. The pools' live use is on `/metrics`: `memory_db_pool_checked_out{pool}` against
`memory_db_pool_capacity{pool}`.

Every connection is bounded in time: `connect_timeout` 5 s, a pool checkout 5 s, a statement
15 s (`constants.DATABASE`), including the task queue's pool, which had neither a statement
nor a connect timeout before.

### Adding it up

What PostgreSQL sees is the sum over pods of everything that connects to it directly, plus
PgBouncer's server pool when the request path goes through one. The compose stack, with three
API workers and one job worker and no budget set:

| | API pod (3 processes) | worker pod | total |
|---|---|---|---|
| request pools | 3 × 16 = 48 to PgBouncer | 16 to PgBouncer | 64 client connections |
| graph pools, direct | 3 × 8 = 24 | 8 | 32 |
| queue pools, direct | 3 × 4 = 12 | 4 | 16 |
| PgBouncer server pool | | | 40 + 10 reserve |

PostgreSQL holds at most 32 + 16 + 50 = 98 of its 200 `max_connections` for the service,
leaving room for OpenFGA, migrations and an operator's `psql`. Without PgBouncer the request
pools connect directly and the total is 64 + 48 = 112.

## PgBouncer in transaction mode

`docker-compose.yml` runs PgBouncer (`deploy/pgbouncer/pgbouncer.ini`) and points the request
path at it:

```yaml
MEMORY__DATABASE__URL: postgresql+psycopg://memory:memory@pgbouncer:6432/memory
MEMORY__DATABASE__DIRECT_URL: postgresql+psycopg://memory:memory@postgres:5432/memory
MEMORY__DATABASE__TRANSACTION_POOLER: "true"
```

Transaction pooling lends a client a server connection for one transaction. Many short
transactions from many processes then share a few backends, which is what lets the API scale
out without scaling PostgreSQL's connection count. The cost is that nothing may outlive a
transaction, and that decides what goes where.

**Through PgBouncer: the request pool.** With `TRANSACTION_POOLER=true`, psycopg prepares no
server-side statements (`prepare_threshold=None`; by default it prepares a query on its sixth
run, and a later run could land on a server connection that never saw it), and no `options`
startup parameter is sent (PgBouncer refuses it). PgBouncer's `connect_query` sets
`statement_timeout=15000` on every server connection instead. The service's own locks are
`pg_advisory_xact_lock`, scoped to the transaction and released with it, so they are safe.

**Direct: everything that needs a session.**

- *Procrastinate.* The worker LISTENs for new jobs on a connection it keeps, and its job
  locks span statements. Through a transaction pooler the LISTEN would be lost the moment
  the connection went back to the pool. `procrastinate_dsn` is always `DIRECT_URL`.
- *The graph traversal.* Its 150 ms `statement_timeout` is a session parameter, and its
  traversal is prepared on first use. Both would leak to the next client of a pooled
  connection.
- *Migrations.* Alembic takes locks and sets `lock_timeout` and `statement_timeout` for its
  session (below). `DatabaseSettings.sync_url` is `DIRECT_URL`.

A PgBouncer in **session** mode would also do for the direct paths. It saves nothing for a
connection that is held for the life of the process.

Without a pooler, leave `DIRECT_URL` unset and `TRANSACTION_POOLER` false: everything uses
`URL`, as before.

## Online migrations

A migration runs against a live service. The rule for every revision:

1. **Fail fast instead of queueing.** `migrations/env.py` sets `lock_timeout = 5s` and
   `statement_timeout = 15min` for the migration's session. An `ALTER TABLE` waiting for a
   lock that a long transaction holds blocks every query behind it, so a migration that
   cannot get its lock in five seconds fails and is retried later. It must not stall the
   service. Override with `MEMORY_MIGRATION_LOCK_TIMEOUT` /
   `MEMORY_MIGRATION_STATEMENT_TIMEOUT` (PostgreSQL interval strings) for a maintenance
   window.
2. **Indexes on live tables are built `CONCURRENTLY`**, inside
   `op.get_context().autocommit_block()`, since `CREATE INDEX CONCURRENTLY` cannot run in a
   transaction. Use `IF NOT EXISTS` so a retried migration is harmless. The downgrade drops
   it `CONCURRENTLY` too. Migration 0023 is the pattern.
3. **Expand, then contract.** Add a nullable column or a new table and deploy code that
   writes both, then backfill in batches from a job, and only drop the old shape in a later
   release. A `NOT NULL` column with a volatile default rewrites the table.
4. **Test upgrade and downgrade** on a scratch database: `alembic upgrade head`,
   `alembic downgrade -1`, `alembic upgrade head`.
