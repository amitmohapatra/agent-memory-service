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
listeners   = the job worker's Procrastinate workers  (4; 0 in the API)
pooled      = per_process - listeners
queue       = max(2, pooled // 7, 2 * listeners)
graph       = max(2, 2 * pooled // 7)                 size ceil(g/2), overflow floor(g/2)
jobs        = the job worker's slots, every pool's     (0 in the API)
main        = max(2, jobs, per_process - graph - queue - listeners)
                                                      size ceil(m/2), overflow floor(m/2)
```

The job worker runs four Procrastinate workers (one per pool: indexing, graph, chat, the rest) on
one queue pool; each fetches, finishes, heartbeats, polls for aborts and defers periodic jobs,
so the pool gets two connections per worker, and each worker's LISTEN connection, opened
outside the pool, counts against the budget too. Every slot of its pools runs a job at once,
and a job writes through the request pool, so that pool is never smaller than the slots
(`jobs`: 4 / 4 / 5 / 8 at 1 / 2 / 4 / 8 CPUs, 5 / 5 / 6 / 12 with a model; sized from the
budget alone, a request pool of 10 once served 17 slots). A worker pod's share is at least what that split needs
(21 for 4 workers and 5 jobs, 31 for 12 jobs): in `staging` and `prod` a smaller budget is
refused at start (`ConnectionBudgetError`, naming both numbers); elsewhere it is raised to
the floor, and the worker logs a warning, once, naming both. Unset, a worker whose pools need
more than 28 takes what they need.

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
   it `CONCURRENTLY` too. Migration 0024 is the pattern. A concurrent build still runs
   under the session's `statement_timeout`: migration 0027 builds an index over the whole
   `memories` table (`ix_memories_scope_subject_lower`), which on a large deployment can
   outlast 15 minutes - run it with a larger `MEMORY_MIGRATION_STATEMENT_TIMEOUT`, or the
   upgrade fails and leaves an INVALID index that the next attempt drops and rebuilds.
3. **Expand, then contract.** Add a nullable column or a new table and deploy code that
   writes both, then backfill in batches from a job, and only drop the old shape in a later
   release. A `NOT NULL` column with a volatile default rewrites the table.
4. **Test upgrade and downgrade** on a scratch database: `alembic upgrade head`,
   `alembic downgrade -1`, `alembic upgrade head`.

## Upgrading a store an earlier build wrote (no clearing)

Memory is never cleared, and nothing in an upgrade needs it to be. A store written before
ADR 0038/0040/0041 holds a user's old and new statements of one thing both current, and index
points without the `statement_kind` payload the rules section filters on. Both are read
correctly as soon as the new build serves (the read path's latest-value rules apply to old
and new memories alike, ADR 0041), and the backfill brings the stored state up to the new
build, in place, while it serves.

**With compose, upgrading is: pull, then `docker compose up -d`.** Nothing else (`--build`
because the stack builds its image from the checkout rather than pulling one):

```bash
git pull
docker compose up -d --build
```

`db-migrate` applies the schema first, then the API and the worker start and the one-shot
`memory-backfill` service runs the backfill beside them
(`python -m memory_service.tools.backfill --state /data/backfill/state.json`). The API and
the worker do not wait for it: it is live-safe, and they serve while it runs. Its checkpoint
lives on the `backfill` volume and records the backfill's revision (`BACKFILL_REVISION`) and
every user finished, so:

* the next `docker compose up` of the same release skips every user finished, and with none
  left exits 0 before loading a model;
* a release that changes what the backfill applies bumps the revision, and a checkpoint
  from an earlier revision starts over (every user is brought up to the new rules);
* a fresh install (no users yet) exits 0 at once;
* a run stopped midway resumes where it stopped on the next `up`.

Its result is in `docker compose logs memory-backfill` (a JSON report: users, messages,
replaced, relabelled, standing, failures); a user that failed makes it exit 1, and the next `up` runs
that user again.

**Without compose**, run the same command after `alembic upgrade head` (`make migrate`),
with a state file that outlives the run:

```bash
make backfill BACKFILL_ARGS="--state /var/lib/memory/backfill.json"           # every user
python -m memory_service.tools.backfill --state /tmp/b.json --tenant acme --user u1  # one user
make backfill BACKFILL_ARGS="--state /var/lib/memory/backfill.json --reindex-all"
```

* **What it does** (`memory_service.tools.backfill`): for every user, their own statements,
  message by message in the order they were said (event time), go through the same
  cross-conversation replacement the write path runs (ADR 0038): an older value becomes an
  earlier value (history) or is superseded, exactly as if it had been said after the upgrade.
  Then every memory of that user is indexed again (an upsert of the same ids, with the new
  payload) and its graph enrichment runs again, as the write path's index job does, so the
  graph facts of a memory now replaced are retired with it, and the user's `user` profile
  block is composed again from what is current (the block an earlier build composed listed
  what this one reads as replaced); `--reindex-all` rebuilds the
  whole index afterwards (`tools.reindex`).
* **Each statement is read again first, as the write path reads it** (`read_again`), so
  it is compared and served as a fresh store's would be:
  * a sentence that gives an instruction for every time ("Setz die Auftragsnummer immer in
    den E-Mail-Betreff ...", "Pon siempre ...", "ضع دائمًا ...") is marked a standing
    instruction (`standing_instruction`), in every deployment: the words alone decide it.
    An earlier build marked none, so a rule restated with a new value was two current rules
    to the read path (`standing` in the report);
  * in a deployment without a model (every kind there came from the lexicon and the NLI
    head), a kind this build's labeller cannot give a statement is read again by the
    labeller, with the head for what the lexicon leaves open: Hindi "आज से X संभालेंगे" was a
    rule by an earlier lexicon and is a dated fact (or a correction) by this one; a rule
    reading this build reads no rule in becomes a fact, or, when it was all its message
    left (no turn kept beside it), the turn this build keeps in its place. A kind it can still give (the
    lexicon's, or the one it leaves to the head) is the same head's verdict and is kept
    (`relabelled` in the report).
* **Nothing is deleted.** A replacement stays reversible (forgetting or retracting the newer
  statement reinstates the older one).
* **Live-safe.** One message per transaction, under the user's serialization, bumping the
  revisions its changes touch, so no cached bundle serves a value it marked.
* **Idempotent and resumable.** A statement already marked is not marked again; a second run
  changes nothing. `--state` records each finished user and, for the one in progress, the
  last message done (every `--batch`, default 50); rerun the same command to resume. The
  state also records `BACKFILL_REVISION`: a state file from another revision starts over,
  so keeping one file across releases is safe.
* **Needs the NLI head** (the replacement's decider, as on the write path). Without it the
  command only re-indexes.
