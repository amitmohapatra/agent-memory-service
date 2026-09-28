# The API, area by area

One page per area: what it is for, a diagram of how it works, every route in it, and the SDK
call that makes each one. The generated contract — every schema, every error, examples — is
[`../openapi.json`](../openapi.json) (`make openapi`), and `/docs` on a running service is the
same thing, browsable. These pages are the explanation; the contract is the authority.

| Page | The question it answers | Routes |
| --- | --- | --- |
| [memory.md](memory.md) | how does something get remembered, and what is held about this scope? | `/v1/observations`, `/v1/memories`, `/v1/graph/query`, `/v1/jobs/{id}` |
| [context.md](context.md) | what goes into the prompt for this turn — and did the answer follow from it? | `/v1/context`, `/v1/recall`, `/v1/verify`, `/v1/threads`, `/v1/messages`, `/v1/briefs` |
| [documents.md](documents.md) | how does a file become retrievable knowledge with page-level provenance? | `/v1/documents` |
| [tools.md](tools.md) | which tool worked for this task, in what order? | `/v1/tools/*`, `/v1/runs/{id}/outcome` |
| [feedback.md](feedback.md) | how is a judgement on a run, an answer or a memory recorded, and what does it change? | `/v1/feedback` |
| [webhooks.md](webhooks.md) | how does another service hear that something was remembered? | `/v1/webhooks/*` |
| [tenancy.md](tenancy.md) | who may see what: workspaces, groups, keys, model keys, and the read audit | `/v1/workspaces/*`, `/v1/groups/*`, `/v1/keys`, `/v1/model-key`, `/v1/agents/model-key`, `/v1/reads` |
| [admin.md](admin.md) | onboarding a tenant, and is the service healthy? | `/v1/admin/tenants`, `/health/live`, `/health/ready`, `/version`, `/metrics` |

## What every call shares

**Identity comes from the credential and the trusted headers, never from the body.** A key
names its tenant; `X-Trellis-Tenant`, `X-Trellis-Workspace` and `X-Trellis-User` narrow the
scope within it (their pre-Trellis spellings `X-Memory-*` are read as aliases until 0.3.0, and
sending both with different values is refused). `X-API-Key` carries the key — a `mk_…` model key
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
`GET /v1/jobs/{job_id}`, or the document's own status, rather than reading immediately.

**Reads are audience-filtered before search runs**, not after. A memory or chunk is retrievable
only by a principal in its audience, and the filter is applied in the store. That is why a
WORKSPACE-visible write needs a workspace row and a member ([tenancy.md](tenancy.md)).

**Errors are RFC 9457 problem documents** (`application/problem+json`) with a `type`, a `title`,
a `status`, a `detail` and, where a field is at fault, `errors`. The SDK raises them as
`ValidationFailed`, `AuthorizationFailed`, `NotFoundError`, `ConflictError`,
`InsufficientEvidence` and friends from `trellis.memory.errors`.

**Pagination is a cursor**, not an offset: a list route answers `{"…": [...], "next_cursor": …}`
and the SDK exposes both `list(...)` (one page) and `page(...)` / `iter_*` (the cursor).

## The SDK in four lines

```python
from trellis.memory import MemoryClient

memory = MemoryClient("http://localhost:8080", api_key="dev-key")
ctx = memory.bind(tenant_id="acme", user_id="u1", thread_id=thread_id, session_id=s, turn_id=t)
bundle = await ctx.context("what did we decide about the refund?")
```

`ctx` carries the scope for every call under it: `ctx.chat`, `ctx.documents`, `ctx.graph`,
`ctx.tools`, `ctx.runs`, `ctx.feedback`, `ctx.briefs`, plus `context`, `recall`, `observe`,
`remember`, `memories`, `forget` and `verify` directly. Tenant administration is
`memory.administer(tenant_id)` (keys, workspaces, groups, webhooks, model keys, reads) and
platform onboarding is `memory.admin`.
