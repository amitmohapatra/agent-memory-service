# The API, area by area

One page per area: what it is for, a diagram of how it works, every route in it, and the SDK
call that makes each one. Looking for *which* call fits a job rather than what a route does?
Start with [Which API for which scenario](../USAGE.md). The generated contract — every schema, every error, examples — is
[`../openapi.json`](../openapi.json) (`make openapi`), and `/docs` on a running service is the
same thing, browsable. These pages are the explanation; the contract is the authority.

| Page | The question it answers | Routes |
| --- | --- | --- |
| [memory.md](memory.md) | how does something get remembered, and what is held about this scope? | `/v1/messages` (events), `/v1/memories` (incl. `/supersede`, `/restore`), `/v1/graph/*`, `/v1/jobs/{id}` |
| [context.md](context.md) | what goes into the prompt for this turn — and did the answer follow from it? | `/v1/context`, `/v1/recall`, `/v1/verify`, `/v1/threads/{id}`, `/v1/messages` |
| [agent-tools.md](agent-tools.md) | what can an agent pull from memory itself, and what does that teach the push? | `/v1/agent-tools` |
| [profile.md](profile.md) | what does every prompt start from: pinned blocks and the thread's summary? | `/v1/profile`, `/v1/threads/{id}` (its `summary`) |
| [documents.md](documents.md) | how does a file become retrievable knowledge with page-level provenance? | `/v1/documents` |
| [tools.md](tools.md) | which tool, which plan, which arguments — and what may run unasked? | `/v1/tools/*` |
| [feedback.md](feedback.md) | how is a judgement on a run (and its answer), a memory, a tool call or a procedure recorded, and what does it change — and who reviews a vote before it counts? | `/v1/feedback` (incl. `?review=pending`), `/v1/feedback/{id}/approve`, `/v1/feedback/{id}/dismiss` |
| [tenancy.md](tenancy.md) | who may see what: workspaces, keys, model keys, and the read audit | `/v1/workspaces/*`, `/v1/keys`, `/v1/keys/self`, `/v1/keys/{id}`, `/v1/model-key`, `/v1/model-key/policy`, `/v1/model-key/usage`, `/v1/agents/model-key`, `/v1/reads` |
| [admin.md](admin.md) | onboarding a tenant, and is the service healthy? | `/v1/admin/tenants`, `/health/live`, `/health/ready`, `/version`, `/metrics` |

## What every call shares

**Identity comes from the credential and the trusted headers, never from the body.** A key
names its tenant; `X-Trellis-Tenant`, `X-Trellis-Workspace` and `X-Trellis-User` narrow the
scope within it (one value each: a header sent twice with different values is refused). `X-API-Key` carries the key — a `mk_…` model key
may also arrive as a Bearer token. A request body's `scope` may carry **lineage** (thread,
session, turn, work, task, agent, agent run, parent run); if it also names a tenant, workspace or
user that disagrees with the trusted header, the request is refused rather than resolved in
either direction. The SDK sends both halves from `MemoryClient.bind(**scope)`.

| Header | What it is |
| --- | --- |
| `X-API-Key` | the credential; it fixes the tenant |
| `X-Trellis-Tenant` / `-Workspace` / `-User` | the scope, within what the credential permits |
| `X-Request-ID` | the caller's request id, echoed back and joined to traces and logs |
| `traceparent` | W3C trace context; the service continues the caller's trace |
| `X-Trellis-LLM-Tokens` | **response** header: what this request spent on model calls, if any |

**Writes are acknowledged, then processed.** A `2xx` on a write means the record *and* its
processing job are committed in one transaction — not that the result is retrievable yet. Poll
`GET /v1/jobs/{job_id}` (a `202` names the first job in `Location`), or the document's own status,
rather than reading immediately. A `201` names the created resource in `Location`.

**Every write takes `Idempotency-Key`.** A retry with the same key and body is the first response
again — status, body, `Location` — with `Idempotent-Replayed: true`: a second `DELETE` is the first
`204`, not a `404`; a retried profile edit is not a `409`. The same key with another body is a `409`.
The read-only POSTs (`/v1/recall`, `/v1/context`, `/v1/verify`, `/v1/tools/hints`) take no key.

**Reads are audience-filtered before search runs**, not after. A memory or chunk is retrievable
only by a principal in its audience, and the filter is applied in the store. That is why a
WORKSPACE-visible write needs a workspace row and a member ([tenancy.md](tenancy.md)).

**Errors are RFC 9457 problem documents** (`application/problem+json`) with a `type`, a `title`,
a `status`, a `detail` and, where a field is at fault, `errors`. The SDK raises them as
`ValidationError`, `AuthenticationError`, `AuthorizationError`, `NotFoundError`,
`ConflictError`, `RateLimitedError` and friends from `trellis.memory.errors`. `retryable: true`
means the same request may pass later, and a retryable `429`, `503` or `504` says when in
`Retry-After` (seconds): `503 DEPENDENCY_UNAVAILABLE` is a store that is down (PostgreSQL
unreachable or its pool exhausted, Qdrant or OpenFGA away), `504 TIMEOUT` a database statement
stopped at its budget. A `detail` never quotes a driver's or a server's message. Insufficient
evidence is not an error: a context answers `evidence_status` ([context.md](context.md)).

**Pagination is a cursor**, not an offset: every list route takes `cursor` and `limit` and sends
`Link: <…>; rel="next"` exactly when a next page exists; an envelope body (`{"…": [...]}`) also
carries `next_cursor`, a bare-array body only the header. The SDK exposes both `list(...)` (one
page) and `page(...)` / `iter_*` (the cursor).

**Polled reads validate.** `GET /v1/agent-tools` and `GET /v1/tools` send an `ETag`; a request
whose `If-None-Match` names it is answered `304` without a body.

## The SDK in four lines

```python
from trellis.memory import MemoryClient

memory = MemoryClient("http://localhost:8080", api_key="dev-key")
ctx = memory.bind(tenant_id="acme", user_id="u1", thread_id=thread_id, session_id=s, turn_id=t)
bundle = await ctx.context("what did we decide about the refund?")
```

`ctx` carries the scope for every call under it. The per-turn verbs are on it directly:
`context`, `remember`, `update`, `forget`, `search`, `history` (the transcript: `add`,
`thread`, `update`), `feedback`, `record_tool`, `tool_hints`, `agent_tools`,
`call_agent_tool`, `profile`, `verify`. Everything else is under
`ctx.advanced`: `documents`, `graph`, `tools` (catalog, approval suggestions),
`model_keys`, `memories` (inventory), `job(id)`, and the client's `tenant` and `admin`.
Tenant administration is also `memory.administer(tenant_id)` (keys, workspaces, model keys,
reads) and platform onboarding `memory.admin`.
