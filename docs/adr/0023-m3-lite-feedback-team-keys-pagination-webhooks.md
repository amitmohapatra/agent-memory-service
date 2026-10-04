# ADR 0023: M3-lite — feedback, team model keys, cursor pagination, outbound webhooks

Status: accepted (2026-09-28). Amends ADR 0012 (agent credentials), ADR 0021 (tenancy)
and ADR 0022 (API conventions). Design of record: the platform design doc, phase 2.
Partly superseded (2026-10, final overhaul, migration 0021): outbound webhooks, workspace
model keys and the ANSWER/BRIEF feedback targets are removed; notifications live in agent-runs,
whose SDK verifies them (`trellis.runs.webhooks.verify_signature`).

## Context

The harness (phase 3) needs four things from the Memory Service that did not exist: a place
to store what people and judges think of what an agent did, and to learn from it; model keys
at team level so a harness passes only its own service key; cursor pagination on every list
it pages through; and outbound webhooks so a disconnected client learns that memory changed
or a verdict was projected. Contracts 0.3.0 fixed the `Feedback` wire shape; this service
accepts it as is and does not import the contracts package, so it stays usable on its own.

## Decisions

### Feedback is stored apart from memory and projected off the request path

`POST /v1/feedback` takes the contracts record. The record only claims who it is from: its
`tenant_id`, `workspace_id` and `user_id` must agree with the trusted headers (422 when they
do not), the missing ones are filled from the headers, and provenance is the request's:
`trace_id` and `created_at` in the body are accepted for the contracts shape and replaced by
the request's trace and the service's clock. A `memory` target must be readable by the caller
(404/403 exactly as `GET /v1/memories/{id}`); `reject`, `correct` and `edit` also need the
owner, the user an agent acts for, or a tenant admin, the rule that governs forgetting, so a
colleague who can read a memory cannot rewrite it. The client's `feedback_id` makes a retry
return the stored record (200) instead of a duplicate.

The projector runs as the `feedback.project` job, enqueued in the same transaction as the
row, so a verdict is never lost and never slows the request that carried it. On a memory
that is CURRENT: `confirm` and `approve` reinforce it (count +1, confidence +0.1 capped);
`reject` retracts it (temporal status RETRACTED, index entry removed); `correct` and `edit`
write a corrected memory that supersedes the old one through the existing revision chain
(the correction is the text, or `{content: ...}`; the new memory keeps the old memory's
scope, visibility keys and evidence, plus an evidence reference to the feedback, and none of
its lifecycle state: no inherited TTL, derived slot or admission trail). The job locks the
record and the memory for its transaction, so a re-dispatched job or two verdicts on one
memory apply in order and never fork the revision chain. Any other
target kind is recorded and listed, with projection `none` and the reason; phase 3 consumes
those. What the projector did is written back as `projection` and every projection bumps
the memory revisions, so a cached bundle that showed the old memory stops being served.

Visibility: a record about a memory is visible to whoever can read that memory; a record
about anything else is visible to its author, to the workspace it was given in, and to a
tenant admin. `GET /v1/feedback?target_kind&target_id` pages newest first.

### Model keys resolve agent → workspace → tenant

`ModelIdentity` carries the call's workspace. Resolution reads the principal's own row, then
`workspace:<id>`, then `tenant`, and uses the first that exists. A revocation tombstone at the
first level hit refuses the call: a revoked agent never silently borrows the team's or the
operator's key. `confirm` re-resolves after the call and requires the same row and revision,
so a key registered, rotated or revoked mid-call fails that call closed instead of mixing
keys. Tenant admins manage the team levels at `PUT/GET/DELETE /v1/workspaces/{id}/model-key`
and `/v1/model-key`; the agent route is unchanged; no secret is ever returned. The service
is `ModelCredentials` (was `AgentCredentials`), registered as `model_credentials`.

### One pagination convention

Every list route takes `cursor` and `limit`, answers `Link: <url>; rel="next"` (RFC 8288)
exactly when a next page exists, and envelope bodies also carry `next_cursor`. The cursor is
base64url JSON of the keyset the repository ordered by; a malformed or foreign cursor is a
422 `VALIDATION` problem, never a database error; it is not signed because it only names a
position and every query is scoped by the caller's tenant. Repositories fetch `limit + 1`
rows: the extra row is the proof of a next page, and the cursor names the position after the
last row returned, so a row written between two requests is never skipped or duplicated.
Memories page by `(created_at, memory_id)`, a keyset that never moves (a memory reinforced
during a walk is not skipped), and a page holds `limit` *visible* memories however many
hidden rows lie between them; the repository fills a page past lapsed derived rows, so fewer
than `limit` rows means the scope is exhausted. The older parameters (`after`, `before`,
`before_sequence`) keep working; a cursor wins when both are sent.

### Outbound webhooks ride the transactional outbox

A subscription names a URL, the events it wants and optionally a workspace; the HMAC secret
is generated here, shown once, and stored encrypted with the credential envelope keys, bound
to the subscription id. Publishing an event inside the transaction that made it true writes
one `webhooks.fanout` job when the tenant has any enabled subscription (one indexed existence
check per publish, nothing more). Fan-out writes one delivery row and one `webhooks.deliver`
job per matching subscription. Delivery POSTs the event as JSON with `X-Trellis-Event`,
`X-Trellis-Delivery`, `X-Request-ID`, `traceparent` and `X-Trellis-Signature:
t=<unix seconds>,v1=<hex hmac-sha256 of "t.body">`, times out after 10 s, reads at most 64 KiB of
the answer, is retried with exponential backoff up to six attempts, then marked DEAD; a
subscription is disabled after twenty consecutive dead deliveries and forgiven when
re-enabled; delivery rows are purged after thirty days (the tuning is `WebhookTuning` in
`config/constants.py`: the same for every deployment, so not a setting). A target that has
become unreachable by policy (it now resolves to a private address) or unsignable (the
envelope no longer opens the secret) is DEAD at once, not retried. Events: `memory.created` (from the
ingestion pipeline), `memory.superseded` and `memory.retracted` (from the feedback projector;
consolidation and forgetting do not yet publish), `feedback.received`, `feedback.projected`,
`webhook.test`. A delivery body carries identities and verdicts, never memory content. A workspace-scoped subscription
only receives events of its workspace; a tenant never receives another tenant's.

The service performs the request, so targets are checked as a server-side request forgery
would be: https only, no credentials or fragment, never `localhost`, `.local`, `.internal`, loopback, link-local (including the cloud
metadata address), private, multicast or unspecified addresses, including IPv4-mapped IPv6,
checked at subscription time and again against every address the host resolves to at every
delivery; the connection is then pinned to that vetted address, with the original host in
`Host` and SNI, so a record that changes between check and connect (DNS rebinding) cannot
steer the request onto the local network. Shared address space (100.64/10) counts as private. The one setting, `webhooks.allow_local_targets`, permits http and local addresses
for docker-compose development and is refused in deployed environments. The SDK shipped `trellis.memory.webhooks.verify_signature` for receivers; it went with the webhooks (see Status), and the memory SDK ships no webhook code. Run notifications are signed by agent-runs, and a receiver verifies them with `trellis.runs.webhooks.verify_signature` (pip `trellis-runs`, the agent-runs SDK).

## Consequences

- SDK 0.2.1 (0.3.0 stays the alias-removal release of ADR 0022): `ctx.feedback.submit/get/list_for/page_for`, `admin.webhooks.*`,
  `tenant.model_key_status/set_model_key/revoke_model_key`, the same on workspaces,
  `cursor` on every list (`list()` returns a page as a list, `page()` returns `items` with
  `next_cursor`), `ctx.memories_page` and `ctx.iter_memories`.
- Migration 0015 adds `feedback`, `webhook_subscriptions` and `webhook_deliveries`; no
  existing row changes.
- Feedback on tool calls and runs is stored but not yet learned from (phase 3 consumes it
  for tool memory); Langfuse scores from feedback are the harness's job (phase 6).
- The workspace level of a model key is chosen by the trusted workspace header; a tenant
  whose teams must not name each other's workspace binds each team's API key to its
  workspace (keys carry `workspace_id`), which the credential scope enforces.
- A delivery body is the event as the service serialises it; corrections are not in it (the
  receiver reads the record through the API), so a webhook endpoint sees ids and verdicts,
  never memory content.
