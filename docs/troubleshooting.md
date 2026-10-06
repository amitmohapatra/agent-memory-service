# Troubleshooting and FAQ

The errors and surprises a first integration meets, what each means, and what to do. Every
error the API returns is an RFC 9457 problem (`application/problem+json`) with a `code`, a
`retryable` flag, a `trace_id` and a `request_id`; the SDK raises one exception class per code
([api/README.md](api/README.md#what-every-call-shares)). Quote the `trace_id` when you ask
for help: it names the trace and the log lines of that request.

| Status | `code` | SDK exception | Retry? |
|---|---|---|---|
| 401 | `AUTHENTICATION` | `AuthenticationError` | no: the key is unknown, revoked or expired |
| 403 | `AUTHORIZATION`, `SCOPE_DENIED` | `AuthorizationError` | no |
| 404 | `NOT_FOUND` | `NotFoundError` | no |
| 409 | `CONFLICT` | `ConflictError` | no |
| 413 | `PAYLOAD_TOO_LARGE` | `PayloadTooLargeError` | no |
| 422 | `VALIDATION`, `CORRUPT_SOURCE` | `ValidationError` | no |
| 429 | `RATE_LIMIT` | `RateLimitedError` | yes, after `Retry-After` |
| 503 | `DEPENDENCY_UNAVAILABLE` | `DependencyUnavailableError` (`CircuitOpenError` from the SDK's breaker) | yes, after `Retry-After` |
| 504 | `TIMEOUT` | `TimeoutError` | yes, after `Retry-After` |

---

## Writing

**I get `Workspace not found` when I write a WORKSPACE-visible memory.**
A workspace is a team the tenant's administrator created, not a label: onboard the tenant
(`POST /v1/admin/tenants`), create the workspace (`POST /v1/workspaces`) and admit the member
(`PUT /v1/workspaces/{id}/members/{principal_ref}` with `user:u1`) first. Example
[09](../examples/09_admin_onboarding_and_keys.py) does all three;
[api/tenancy.md](api/tenancy.md#onboarding-a-team-in-full) has the full sequence. A workspace
id that names no team grants nothing.

**The write answered `202`, but search does not find it yet.**
A `2xx` means the record and its processing job are committed, not that the result is
retrievable: extraction, embedding and indexing run in the job worker. Follow the job the
acknowledgement names (`ack.job_ids`, `GET /v1/jobs/{id}`, `ctx.advanced.job(id)`), or the
document's own status. If jobs never leave `PENDING`, the worker is not running
(`docker compose ps memory-worker`, or `uv run memory-worker`); the outbox sweep re-dispatches
anything the fast path missed every minute.

**The same message sent twice became two messages.**
Without a `turn_id`, every `USER` message opens the thread's next turn, so the second one is a
new turn saying the same thing, which is what a user who repeats themselves did. Name the
turn (`bind(..., turn_id=...)`) or send an `Idempotency-Key` to make a retry a replay
(example [02](../examples/02_conversation_history.py)).

**A retry answers `409`.**
The same `Idempotency-Key` with a different body is a conflict, not a replay. A replay of
the same key and body returns the first response with `Idempotent-Replayed: true`.

**`413 PAYLOAD_TOO_LARGE`.**
A body or file over the size limit (`MAX_BODY_BYTES`, 25 MB, a constant). Split the upload;
the limit is not a setting.

## Reading

**`evidence_status` is `INSUFFICIENT`.**
That is an answer, not an error: the service did not find what an answer to this question
needs, and says so rather than guess. Answer that you do not know, or ask for the missing
document. `format="full"` adds `missing_evidence`, the companion passages that are not there.

**Search still returns the old value of a fact.**
A superseded memory is never served as current. The verbatim turn that stated the old value
is still searchable, as what was said when (a temporal question needs it); forget that turn
too if it must go. `as_of` and `known_at` read memories as they were.

**`403 SCOPE_DENIED` on a thread or a memory.**
The principal the request acts for is not in the record's audience: another user's thread,
an agent's `RUN` note read by a sibling, a tenant header a key may not use. An issued key with
`may_act_as` may act only for the principals it names, agents included (ADR 0032).

**`verify` answers `404 bundle not found`.**
A bundle record is kept for 30 minutes and only for the scope that built it. Verify soon
after `context()`, under the same scope.

**My `feedback` changed nothing.**
A vote from a person or an agent waits in the review queue until the tenant's administrator
approves it (`review.state == "pending"`, ADR 0028). Applied at once: the service's own judge,
a run reporting its own status (`source="system"`, citing no memories), an owner correcting
their memory, a tool-call verdict, and anything the tenant admin says. The projection is a
later fact either way: read it back with `ctx.feedback.get(id)` (example
[08](../examples/08_feedback_and_review.py)).

## Authentication and keys

**`401` with a key that used to work.**
It was revoked or has expired; revocation takes effect on the next request, from any
instance. A suspended tenant's keys get `403`, not `401`, and work again when it is resumed.

**I set `MEMORY__AUTHENTICATION__BOOTSTRAP_ADMIN_KEY` and `dev-key` stopped working.**
The authentication mode follows from what is configured: with a bootstrap key the service is in
`api_key` mode, where every caller presents a key the service issued, and development keys
are ignored. Onboard a tenant with the bootstrap key and use the keys it issues
([configuration.md](configuration.md#authentication)).

**`429` and `Retry-After`.**
The tenant's `rate_limit_per_minute` (set per tenant through the admin API). The SDK waits
`Retry-After` and retries; a 429 never opens its circuit breaker.

## The service

**It refuses to start in `staging` or `prod`.**
The production guards: development keys alone (`trusted_dev` mode), the `filesystem` blob
store, and a bootstrap key under 32 characters are refused when deployed. The error names the
setting ([configuration.md](configuration.md#production-guards)).

**`/health/ready` says `degraded`.**
A dependency other than PostgreSQL is down (search, authorization, blob, the queue, the cache
or the gateway); the service still serves what it can, and `/version` lists what fell back.
`503 not_ready` means PostgreSQL is down or the process is stopping. Liveness checks nothing
outside the process ([api/admin.md](api/admin.md)).

**`503 DEPENDENCY_UNAVAILABLE` or `504 TIMEOUT`.**
PostgreSQL unreachable or its pool exhausted, Qdrant or OpenFGA away (503), or a database
statement stopped at its budget (504). Both carry `Retry-After`. Under sustained 503s, check
the connection budget against the server's `max_connections`
([deploy/database.md](deploy/database.md)). When the SDK's breaker is open every call raises
`CircuitOpenError` at once for 30 seconds: catch `DependencyUnavailableError` and answer
without memory for that turn.

**The first start is slow, or the API reports unhealthy while starting.**
The first `docker compose up` downloads about 1 GB of model weights and loads them; the API's
healthcheck allows ten minutes. A model missing from `./models` (or `/models`) is a startup
error, never a silent fall back: run `make models`.

**It stops at start with a pinned OpenFGA model.**
`MEMORY__AUTHORIZATION__OPENFGA_MODEL_ID` names a model that is not this build's. Unset it, or
pin the id the `openfga.model_written` log line names.

**I changed the Qdrant shard or replica settings and nothing happened.**
They apply when a collection is created. Rebuild: `make reindex REINDEX_ARGS=--drop`
([deploy/search.md](deploy/search.md)).

**The job worker's metrics port is taken.**
`MEMORY__TASKS__METRICS_PORT` cannot be switched off from the environment; give it a free
port.

## Models

**No model call is ever made.**
A use runs only when all four hold: `BIFROST_URL` is set, the tenant's policy names the use (no
policy row: every use except the opt-in `memory_restatement`), a key can pay (the agent's,
else the tenant's, else the operator's `BIFROST_VIRTUAL_KEY`), and, on a read, the policy's
`read_assist` is on ([chapter 8](guide/08-models.md#when-a-call-is-allowed)). Any model failure
falls back to the deterministic path, so the service still answers.

**Registering a model key is refused.**
In `staging` and `prod` the envelope keys (`MEMORY__AGENT_CREDENTIALS__ACTIVE_KEY_ID` and
`__ENCRYPTION_KEYS`) must be set first. `dev` and `test` derive a development key with a
warning, which protects nothing.

## Development

**`uv sync` fails with `Distribution not found at: .../vendor/bifrost-sdk`, or
`test_vendored_sdk` fails as stale.**
`vendor/bifrost-sdk` is generated from a bifrost-sdk checkout and git-ignored. Run
`make vendor` (with `BIFROST_SDK=/path/to/bifrost-sdk` when it is not at `../bifrost-sdk`).
Every Makefile test target runs it first; a bare `pytest` does not.

**The examples say PostgreSQL is not reachable.**
They need a PostgreSQL 16 with the user and password `memory` on `localhost:5432` (`make
dev-up` starts one), or `MEMORY_EXAMPLES_DATABASE_URL` naming another. The database name must
end in `examples`: the examples empty it on every run ([examples/README.md](../examples/README.md)).

**Tests skip with "PostgreSQL not reachable".**
The suite creates its own databases (`memory_tests` and two more) on the server at
`MEMORY_TEST_ADMIN_URL`; nothing it needs is shared with the dev stack
([chapter 12](guide/12-testing-and-gates.md)).
