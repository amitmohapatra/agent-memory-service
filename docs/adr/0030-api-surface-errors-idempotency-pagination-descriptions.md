# ADR 0030: API surface — errors, idempotency, pagination, descriptions

Date: 2026-10-04. Status: accepted. Amends ADR 0022 (problem details, OpenAPI conventions) and
ADR 0023 (pagination).

## Context

An audit of the public surface against what the code does found the contract promising more
than it kept, and saying less than a client needs:

- **Errors.** PostgreSQL going away (`OperationalError`, `InterfaceError`, an invalidated
  connection, an exhausted pool) was an unhandled 500 a client gives up on; no 503 carried
  `Retry-After`; a Qdrant failure quoted the client library's exception text (server URL,
  collection names) in `detail`; the `StarletteHTTPException` handler dropped the exception's
  headers (`Allow` on a 405); `RETRYABLE_PROCESSING` was in the enum and the docs and nothing
  produced it; a 413 said `VALIDATION`, where agent-runs says `PAYLOAD_TOO_LARGE`.
- **Idempotency.** `Idempotency-Key` was documented on every write and honoured on eleven: a
  retried `DELETE /v1/threads/{id}` was a 404, a retried profile edit a 409, a retried key
  revocation, membership change or catalog upsert a second effect. The read-only POSTs
  advertised it too.
- **Locations.** No 201 said where the created resource lives, no 202 where its job is.
- **Pagination.** Workspace members, the tool catalog, approval suggestions and graph entities
  had no cursor; `GET /v1/admin/tenants` kept a legacy `after`; `GET /v1/reads` used `after`
  for a since-filter, the opposite of what `after` means everywhere else.
- **Two routes for one list** (`GET /v1/feedback` and `/v1/feedback/pending`); `DELETE` of a
  model key answered 200 with a body.
- **Descriptions.** 389 schema properties and 199 parameters had none; fixed-value strings
  (`EvidenceRef.source_type` - whose description listed six values and missed `memory` and
  `statement` - `KeySelfResponse.role`, `SearchItem.kind`, `JobResponse.queue`, ...) were
  typed `str`.
- **Hygiene.** The operational routes inherited the document's key requirement; twelve GETs
  documented a 409 they cannot raise; naive request datetimes were refused by one model,
  compared against an aware one (a 500) by another; a chunked body had no size limit and an
  upload was read whole before its size was checked; `GET /v1/agent-tools` rebuilt fixed
  schemas on every call and nothing could be validated with `If-None-Match`; the drift test
  skipped when `docs/openapi.json` was missing.

## Decision

1. **Errors.** `adapters/db/errors.py` reads driver failures (the API may not import the
   driver): a statement cancelled at `statement_timeout` is `TIMEOUT` (504,
   `OperationTimedOut`); a pool timeout, `OperationalError`, `InterfaceError` or an
   invalidated connection is `DEPENDENCY_UNAVAILABLE` (503); anything else stays a 500. Every
   retryable 503/504 carries `Retry-After` (the failure's own `retry_after_seconds`, else 5 s),
   as the 429 always did. A `detail` never quotes a driver's or a server's message - it is
   logged. The HTTP-exception handler keeps the exception's headers. `RETRYABLE_PROCESSING`
   is removed: every `ErrorCode` is one the service produces (a test holds the enum to the
   domain errors). Every 413 is `PAYLOAD_TOO_LARGE` ("Payload too large", not retryable),
   the platform's code; the SDK raises `PayloadTooLargeError`, a `ValidationError`.
2. **Every write honours `Idempotency-Key`** through `api/idempotent.run_idempotent`: a
   retry with the same key and body is the first response - status, body, `Location` - with
   `Idempotent-Replayed: true`; a 204 is replayed as a 204. The read-only POSTs
   (`/v1/recall`, `/v1/context`, `/v1/verify` - whose verdict id is derived from run,
   bundle and answer - and `/v1/tools/hints`) no longer advertise it. A key is one value of
   at most 255 characters, checked at the edge.
3. **`Location`** on every 201 (the created resource) and on the 202s that queue a job
   (`/v1/jobs/{id}` of the first job). `POST /v1/tools/invocations` records the call in the
   request and queues nothing, so its 202 has none.
4. **One pagination convention, everywhere** (ADR 0023): workspace members (by principal),
   the tool catalog (by name; the default page is the whole 500-entry catalog), approval
   suggestions (by support, tool, shape) and graph entities (a position in the ranking,
   bounded at 100) take `cursor` + `limit` and send `Link: rel="next"`; envelopes also
   `next_cursor`. `GET /v1/admin/tenants` takes the cursor only; `GET /v1/reads` names its
   filter `since`.
5. **One route per list.** `GET /v1/feedback?review=pending` is the review queue;
   `/v1/feedback/pending` stays as an alias marked `deprecated` in OpenAPI and answered with
   `Deprecation: true` and `Link: rel="successor-version"`. `DELETE /v1/model-key` and
   `DELETE /v1/agents/model-key` answer 204; the SDK reads the status back, so its methods
   still return `AgentKeyStatus`.
6. **Every property, parameter and operation is described**: what it is, what for, its
   format or unit, its allowed values. The scope query parameters and path ids are shared
   `Annotated` aliases (`api/params.py`), so the wording lives once;
   `tests/contract/test_openapi_descriptions.py` fails on any field without a description.
7. **Fixed values are enums**, the ones the code writes: `EvidenceSource` (message, file,
   document_chunk, agent_result, tool_result, import, observation, statement, memory,
   graph_fact, summary, episode, feedback) on both evidence shapes; `KeySelfResponse.role`
   (platform|admin|service|trusted_dev|jwt); `SearchItem.kind` (`SearchKind`);
   `ContextPassage.kind` (relation|memory, absent for a chunk); `WindowMessageBody.role`
   (`MessageRole`); `UsageDayOut.use` (`LLMUse`); `JobResponse.queue` (`Queue`, null when the
   queue forgot the job); the catalog's `source` (`ToolSource`: manual plus the contracts'
   local|mcp|memory|openapi|a2a); `LiveResponse.status`, `VersionResponse.api_version` and
   `.environment`. The SDK models mirror them as `Literal`s.
8. **Hygiene.** The operational routes declare `security: []`; reads document no 409;
   `PATCH /v1/admin/tenants/{id}` has body examples; the document names its contact and
   licence. A request instant without an offset is read as UTC (`domain/instants.UtcDateTime`)
   and documented so. The correlation middleware counts a streamed body as it arrives and
   stops it at `MAX_BODY_BYTES` with the 413 problem; the upload is read a megabyte at a time
   against `max_file_bytes`.
9. **Conditional GETs.** `GET /v1/agent-tools` builds its listing once per process and sends
   `ETag` and `Cache-Control: private, max-age=300`; `GET /v1/tools` sends an `ETag` of the
   page (`private, no-cache`), which is how a harness refreshes approval tiers cheaply. Both
   answer a matching `If-None-Match` with 304. The tag is a weak digest of the bytes, so it
   changes exactly when the answer does and needs no revision kept in step with every write.
10. **The drift test fails, not skips,** when `docs/openapi.json` is missing.

## Breaking changes

- `GET /v1/reads?after=` is `?since=`; SDK `reads(after=)` / `reads_page(after=)` are
  `since=`.
- `GET /v1/admin/tenants?after=` is gone (send `cursor`); SDK `tenants(after=)` /
  `tenants_page(after=)` lose the argument.
- `DELETE /v1/model-key` and `DELETE /v1/agents/model-key` answer 204 without a body
  (the SDK methods are unchanged).
- A 413 says `code: PAYLOAD_TOO_LARGE` (was `VALIDATION`); an upload over `max_file_bytes`
  is a 413 (was 422).
- `ErrorCode.RETRYABLE_PROCESSING` is removed (it was never sent).
- `EvidenceRef.source_type`, the catalog's `source`, `KeySelfResponse.role` and the other
  fields of decision 7 refuse values outside their enum (requests) and are typed so in
  responses; a client that sent `source_type: "chunk"` sends `"document_chunk"`.
- A database outage is a 503 (or a statement timeout a 504) instead of a 500.
- Writes that ignored `Idempotency-Key` now honour it: a retry with the same key gets the
  first response instead of a second effect.

## Consequences

`docs/openapi.json` changes on most operations (descriptions, headers, enums). Generated
clients are regenerated once. The harness can poll the catalog with `If-None-Match`, and a
retrying client (SDK X3/X8) gets `Retry-After` on every retryable status.
