# ADR 0022: Trellis names and API conventions

**Status:** accepted · **Date:** 2026-09-28 · **Amends:** ADR 0005 and ADR 0021 (header names)

## Context
The memory service is the first package of the trellis platform; `trellis-harness`,
`trellis-contracts`, `trellis-core`, `trellis-runs`, `trellis-a2a`, `trellis-eval` and the
rest code against this API and this SDK, so its public surface is settled before any of
them is written. Until now the SDK was `universal_memory` (distribution
`universal-memory 0.1.0`), the trusted context headers were `X-Memory-*`, errors were a
bespoke `{"error": {...}}` envelope, the trace id was whatever `X-Trace-ID` said or else the
request id, operation ids were path-derived (`create_tenant_v1_admin_tenants_post`) and two
nouns were wrong: `POST /v1/files` (removed in 0.3.0) next to `GET /v1/documents/{id}`, and
`POST /v1/tools/record` (removed since) for a resource every other route calls an invocation. The only
consumers are the owner's own repositories.

## Decision
- **Package.** The SDK is `trellis.memory`: `trellis` is a namespace package (no
  `__init__.py`), so `trellis-harness` and the others share it. Distribution
  `trellis-memory 0.2.0`; the service is `trellis-memory-service 0.2.0`. `universal_memory`
  is removed, not shimmed: a clean break costs the two owner repositories one import edit
  each and leaves no dead module. The optional extra `otel` adds the OpenTelemetry API.
- **Headers.** `X-Trellis-Tenant`, `X-Trellis-Workspace`, `X-Trellis-User`; the response
  header `X-Trellis-LLM-Tokens`. The `X-Memory-*` spellings are read for one release through
  one helper (`api/headers.py`, the only reader of `DEPRECATED_HEADER_ALIASES`), the response
  header is sent under both spellings, and both are removed in 0.3.0. A request that carries both
  spellings of one header with different values, or any scope or credential header more than once
  with different values, is refused (422) by the correlation middleware, before the credential is
  read and before the rate limiter counts it: a gateway that still stamps `X-Memory-*` strips only
  `X-Memory-*`, and a client
  behind it must not choose its own tenant or user by adding `X-Trellis-*`; a gateway that
  stamps these headers must strip both spellings. The credential binding still refuses a
  tenant the key contradicts, whichever spelling carried it. OpenAPI documents the new names
  on every public operation and marks the old ones deprecated.
- **Tracing.** W3C Trace Context. An incoming `traceparent` is continued; every response
  carries `traceparent` and `X-Trace-ID` with the same 32-hex trace id. Precedence: the
  active OpenTelemetry span (when the process is tracing, its server span has already
  continued the incoming header, and the exported trace is what the client must be told),
  else the incoming `traceparent` (the first when a proxy sent two, as the OpenTelemetry
  propagator reads it; stripping duplicates is the ingress's job), else a fresh id. `X-Trace-ID`
  is a response header only: the one request header that names a
  trace is the one the OpenTelemetry propagator reads too, so a request behaves the same
  whether or not the process is tracing. Whether an external caller's `traceparent` is
  trusted at all is the ingress's decision: the service continues what reaches it, so an
  edge that must not let clients attach requests to foreign traces or force sampling strips
  or regenerates the header there. Never the request id, and never the request body:
  `ScopeBody.trace_id` is accepted but ignored (removed in 0.3.0), so headers, logs, rows and
  problems name one trace. Opaque
  ids belong in `X-Correlation-ID`. When this process is not tracing, the span id in the response
  `traceparent` is synthetic: it names no exported span. The SDK's `traceparent` built from
  `scope.trace_id` is synthetic in the same way and marked sampled, so a parent-based sampler on
  the service keeps the request being correlated. A body `correlation_id` is still honoured over
  the header, as before 0.2; the effective id is written back to the request state and into the
  log context, so the response and the logs name the same correlation id (rows store the trace id,
  not the correlation id). The alias routes are recognised by method and route path, so a service
  mounted under a root path still marks them and links to the successor under that root. The SDK
  sends
  `X-Request-ID` on every call (one id per logical call, kept across its retries), sends an
  opaque `scope.trace_id` as the correlation id, and sends `traceparent` from the active span
  through the W3C propagator only (never baggage) when the `otel` extra is installed, else
  built from `scope.trace_id` when that is a W3C id.
- **Errors.** RFC 9457 problem details, `application/problem+json`, one shape for every
  error from the middleware's early 413 and 429 to the unhandled 500: `type`
  (`urn:trellis:problem:<code in kebab case>`), `title` (one per code, `PROBLEM_TITLES`),
  `status`, `detail`, `instance` (the request path), and the extensions `code`,
  `retryable`, `trace_id`, `request_id`, `details`. The SDK maps `code` to its exception
  classes and still reads a 0.1 server's nested envelope.
- **Operation ids.** `<tag>.<function>`, generated by `api/openapi.py:operation_id`; a
  route without a tag keeps its function name, and the contract test holds every public
  route to a tag. They are stable across path edits and readable in generated clients; the
  form-body schema a dotted id would misname is renamed to `Body_<tag>_<function>`.
- **Nouns.** `POST /v1/documents` and `POST /v1/tools/invocations` are canonical.
  `POST /v1/files` and `POST /v1/tools/record` (the latter removed since) are the same handlers, marked deprecated in
  OpenAPI, answered with `Deprecation` (RFC 9745) and `Link: rel="successor-version"`
  headers, and removed in 0.3.0. SDK: `ctx.documents` (`DocumentsAPI`), with `ctx.files` and
  `FilesAPI` kept as aliases for one release; `ctx.tools.record(...)` keeps its name, because
  the method names the action and the route names the resource.
- **Unchanged.** The `MEMORY__` environment prefix, the compose service and image names, the
  directory and repository names (the owner renames those). The OpenTelemetry
  `service.name` (`SERVICE_NAME`) is `trellis-memory`.
- **Observability split** (recorded here because it is why the trace id must be W3C):
  Langfuse traces every agent, Datadog receives every service's spans over OTLP, and the two
  are joined on `traceparent` and `X-Request-ID`. The exporters land in later phases.

## Removed in 0.2.0, without a window
The `universal_memory` package; the nested `{"error": {...}}` envelope on the wire (a 0.2
SDK still reads a 0.1 server's, a 0.1 SDK does not read a 0.2 server's); the path-derived
operation ids; `memory-service` as the value of `/version.service`; `X-Trace-ID` as a
request header (send `traceparent`, or `X-Correlation-ID` for an opaque id). Deploy the
service before the SDK: a 0.2 SDK speaks `POST /v1/documents` and `/v1/tools/invocations`,
which a 0.1 service does not serve.

## Deprecation window
Removed in 0.3.0: the `X-Memory-*` request headers, the `X-Memory-LLM-Tokens` response
header, `ScopeBody.trace_id`, `POST /v1/files`, `POST /v1/tools/record` (removed early, in the overhaul),
`MemoryContext.files` (which warns), `FilesAPI`, and the SDK's reading of the nested `error`
envelope.

## Consequences
- Every trellis package imports `trellis.memory` and speaks these headers and errors; there
  is nothing older to support, and no second error shape to test.
- A client that sent an opaque value in `X-Trace-ID` is answered with a fresh W3C id; the
  opaque value belongs in `X-Correlation-ID`, which is echoed when it is an id (a letter or digit,
  then letters, digits and `._:-`, at most 200 characters) and replaced otherwise, as `X-Request-
  ID` is.
- Historical documents keep the old names as the record of what they were: ADR 0020, the
  dated measurement notes (`docs/LATENCY-LAYERS-2026-09.md`, `docs/AUDIT-2026-09-23.md`) and
  the dated benchmark artifacts under `benchmark/results/`.
- OpenAPI documents the response headers (`components.headers`), the 413 and 429 every
  operation can answer, and a per-operation problem example.
- `docs/openapi.json` changes on every operation (ids, header parameters, error content
  type); generated clients are regenerated once.
