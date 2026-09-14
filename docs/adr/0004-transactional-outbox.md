# ADR 0004: Transactional outbox in front of Procrastinate

**Status:** accepted · **Date:** 2026-09-14

## Context
The acknowledgement rule is: a `2xx/202` is returned only after the source record **and** a
retryable processing job are durably committed. Procrastinate defers jobs through its own
connection pool, so a `defer_async()` cannot join the request's PostgreSQL transaction.
Procrastinate's sync SQLAlchemy connector could share a connection, but the service is
fully async (psycopg3), and `procrastinate.contrib` offers no async shared-connection path.

## Decision
`job_outbox` is written in the same transaction as the source rows (Unit of Work). After
`COMMIT`, the `OutboxRelay` dispatches the rows to Procrastinate immediately (fast path) and
marks them dispatched. A periodic sweep re-dispatches rows that are still pending (crash or
queue outage between `COMMIT` and `defer`). Job idempotency keys map to Procrastinate
`queueing_lock` and to a unique partial index on the outbox, so retries never duplicate work.

## Consequences
- Acknowledgement is exactly the commit point; the job exists durably even if the queue is
  unreachable at that moment (tested: `test_outbox_sweep_recovers_when_dispatch_failed`).
- Job ids returned in API responses may be empty when the relay lagged; clients poll the
  outbox-derived status through `/v1/jobs/{id}` once dispatched.
- Procrastinate remains replaceable (Kafka, Cloud Tasks): only the relay's `enqueue` changes.
- `retries=N` means N additional attempts; `retries=0` disables retry (Procrastinate treats
  `max_attempts=0` as unlimited, so the adapter maps 0 to `retry=False`).
- psycopg3 reports `rowcount=-1` for `INSERT ... ON CONFLICT DO NOTHING`; every conditional
  insert uses `RETURNING` to detect whether a row was written.
