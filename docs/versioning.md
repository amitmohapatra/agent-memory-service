# Versioning and compatibility

What carries a version here, what may change between versions, and which versions of the five
Trellis packages work together. The history itself is the [CHANGELOG](../CHANGELOG.md).

## What is versioned

| Thing | Where the version is | Now |
|---|---|---|
| The service, `trellis-memory-service` | `pyproject.toml`, `src/memory_service/__about__.py`, `GET /version`, the OpenAPI `info.version` | 0.3.0 |
| The SDK, `trellis-memory` (`trellis.memory`) | `sdk/python/pyproject.toml` | 0.4.0 |
| The HTTP API | the path prefix | `/v1` |
| The database schema | Alembic revisions in `migrations/versions/` | `0024_online_candidate_index` |
| The search index | the collection name: the fingerprint of every encoder and the key layout | changes with a model |

The service and the SDK are versioned separately: the SDK depends on `httpx` and `pydantic`
only and is released when its surface changes. The service's version string has not moved
since 0.3.0, although `/v1` responses changed in place on 2026-10-04 (lean responses,
ADR 0029; the API surface, ADR 0030). SDK 0.4.0 is the client for that API; an SDK older than
0.4.0 does not read its responses.

## The rules before 1.0

The service is pre-1.0 and its only consumers are the Trellis repositories, so `/v1` has
changed in place, each time with an ADR and a CHANGELOG entry:

- **0.2.0** renamed the package and the headers and changed errors to RFC 9457 problems
  (ADR 0022). The old spellings were kept as aliases for one release.
- **0.3.0** removed those aliases: one route per operation, one spelling per header
  (`tests/e2e/test_removed_aliases.py` keeps them gone).

What does not change without an ADR: the `MEMORY__` environment prefix and the platform's
unprefixed names, the `X-Trellis-*` headers, the problem `code` values, the operation ids
(`<tag>.<function>`), the scope grammar, and the rule that every write takes an
`Idempotency-Key`. What CI holds:

- `docs/openapi.json` must equal the contract the code exports (the build diffs them), so any
  change to the HTTP surface is a reviewed change to that file;
- the SDK's `Literal` vocabularies must equal the service's enums;
- every setting must be in [configuration.md](configuration.md) and `.env.example`.

There is no deprecation window promised before 1.0: read the CHANGELOG before upgrading.

## Upgrading the service

1. **Migrate first**: `alembic upgrade head` (`make migrate`, or the compose `db-migrate`
   step). Migrations take a lock with a short timeout and fail rather than queue behind a long
   transaction; retry them ([deploy/database.md](deploy/database.md)). CI runs every migration
   up, down to the base and up again.
2. **Re-index after a model change**: a new encoder or key layout is a new collection, never a
   mixed one. `make reindex` builds it from PostgreSQL ([deploy/search.md](deploy/search.md)).
3. **Then roll the API and the worker together.** Every task name the API enqueues must have a
   handler in the worker's registry (`tests/unit/test_job_registration.py`).

## Which versions work together

| Package | Version | Needs |
|---|---|---|
| `trellis-memory-service` (this repository) | 0.3.0 | `bifrost-sdk>=0.3` (the gateway's deny-all MCP scope) |
| `trellis-memory` (this repository's SDK) | 0.4.0 | a service with the 2026-10-04 `/v1` responses |
| `trellis-harness` ([agent-harness](https://github.com/amitmohapatra/agent-harness)) | 0.4.0 | `trellis-memory>=0.4`, `trellis-runs>=0.4.0`, `trellis-contracts>=0.6.0,<0.7`, `bifrost-sdk>=0.3` |
| `agent-runs` and `trellis-runs` ([agent-runs](https://github.com/amitmohapatra/agent-runs)) | 0.4.0 | `trellis-contracts>=0.6.1,<0.7` |
| `trellis-contracts` ([agent-contracts](https://github.com/amitmohapatra/agent-contracts)) | 0.6.1 | nothing of the others |
| `bifrost-sdk` ([bifrost-sdk](https://github.com/amitmohapatra/bifrost-sdk)) | 0.3.0 | a Bifrost gateway |

The memory service does not import `trellis-contracts`: `POST /v1/feedback` accepts the
`Feedback` record's shape, and the SDK sends a contracts record as it is
(`ctx.feedback(record)`). The versions above are the ones in each repository's
`pyproject.toml` on the date of this page; each repository's own CHANGELOG is the authority
for its package.
