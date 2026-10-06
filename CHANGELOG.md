# Changelog

What changed in the service (`trellis-memory-service`) and its SDK (`trellis-memory`), newest
first. The service and the SDK carry separate versions ([versioning.md](docs/versioning.md));
each entry says which one moved. Decisions behind each change are in the
[ADRs](docs/adr/README.md).

## Unreleased

### Added
- Learned skills: the tenant's active procedures are offered to its administrator as draft
  Agent Skills (`GET /v1/tools/skill-drafts`); publishing one writes a `SKILL.md` where agents
  load skills from (`SKILLS_DIR`, else the Bifrost gateway's skills repository) as its next
  version, and dismissing one stops it being offered until its steps change. Migration
  `0025_procedure_skill`; settings `SKILLS_DIR`, `BIFROST_ADMIN_TOKEN`
  ([learned skills](docs/api/tools.md#learned-skills), ADR 0033). SDK:
  `ctx.advanced.tools.skill_drafts()`, `publish_skill()`, `dismiss_skill()`.
- Search past conversations by message: `kinds=["message"]` (`POST /v1/recall`, the
  `memory_search` agent tool, `ctx.search`) reads this conversation's messages and then the
  user's earlier conversations', this conversation's first among equal matches; each item
  carries its `thread_id`. Another user's threads are never read; without a user, this thread
  only ([past conversations](docs/guide/04-retrieval.md#past-conversations)).

### Fixed
- A document uploaded into a thread that did not exist yet reached `READY` but was never
  returned by `search` or `context`: its THREAD audience named a thread nobody had been
  granted. The upload now creates the thread for the uploader, as a first message does
  ([documents](docs/api/documents.md)); examples 04 and 06 no longer send a message first.

### Changed
- The service requires `bifrost-sdk>=0.3`: model calls send the gateway's deny-all MCP scope,
  which 0.3.0 introduced.

### Documentation
- A Start-here README; the high-level design with the five Trellis repositories
  ([ARCHITECTURE.md](docs/ARCHITECTURE.md)); a sequence diagram per key flow
  ([flows.md](docs/flows.md)); a [configuration reference](docs/configuration.md) with every
  setting, kept in step with the code by a test; [troubleshooting](docs/troubleshooting.md),
  [versioning](docs/versioning.md) and this changelog.
- Numbered [examples](examples/README.md), simple to complex, that run offline against an
  in-process service: `make examples`, run in CI.
- `make docs-check`: every relative link and anchor resolves, and every snippet calls SDK
  methods and keywords that exist. Run in CI.
- Dated reports moved out of the user docs to `benchmark/reports/`, design records to
  `design/`; superseded handoffs, plans and logs deleted (git history keeps them). The model
  uses page merged into [chapter 8](docs/guide/08-models.md#the-twelve-uses).
- ADR headers carry their amendments (0004, 0009, 0021, 0022, 0023); ADR 0017 is closed as
  superseded by 0024-0026.
- `.env.example` lists `MEMORY__RETAIL_CALENDAR`, the one setting it was missing.

## 2026-10-04: SDK `trellis-memory` 0.4.0 (service unchanged at 0.3.0)

### Changed
- Lean context, tool-hint and recall responses (ADR 0029): the prompt form of `/v1/context`
  carries the rendered text and the tools that fit, `{name, confidence}`; the full form is
  structured data only; every number is in 0..1; empty keys are absent. Recall items no
  longer carry a citation string. SDK 0.4.0 models follow.
- The API surface (ADR 0030): database outages are a retryable `503`/`504` with
  `Retry-After`; every `413` is `PAYLOAD_TOO_LARGE`; every write honours `Idempotency-Key`;
  `201`/`202` carry `Location`; one cursor convention for every list; conditional GETs on the
  tool catalog and the agent tools; every schema field described.
- The runtime (ADR 0031): cache freshness by the audience a write changes; overload bounds and
  request deadlines; one connection budget per pod and PgBouncer support; readiness decided by
  PostgreSQL and the process; the job worker's lifecycle and metrics; Qdrant layout from
  settings; the tenant registry's invalidation channel.
- SDK resilience and dev defaults (ADR 0032): `Retry-After`, full-jitter backoff, retries of
  the read-only POSTs, status-classed errors, a circuit breaker, `MEMORY_URL` and
  `TRELLIS_API_KEY` defaults; `may_act_as` checks agents too; the admission gate behind a
  tenant switch, off by default; the forget cascade moves readers' revisions; a development
  tenant and a development envelope key in `dev` and `test`.
- A vote waits for the tenant administrator before it changes anything (ADR 0028).
- A conversation turn can be restated at ingest and searched with the turn, opt-in per tenant
  (ADR 0027); memories are ranked by a general fusion and four rules (ADR 0026); late
  interaction over two keys per memory (ADR 0025); a retailer's fiscal calendar
  (`MEMORY__RETAIL_CALENDAR`).

## 0.3.0: 2026-09-30 (service and SDK)

### Removed
- The one-release aliases 0.2.0 kept: `POST /v1/files` and the `X-Memory-*` header spellings
  (`POST /v1/tools/record` had gone before), and the SDK's `ctx.files`. One route per
  operation, one spelling per header.
- `query_decomposition` as a model use: a model call on the read path took 3.2-12.2 s against
  a 300 ms budget.

### Changed
- The SDK's surface: the per-turn verbs are on the bound context (`context`, `remember`,
  `update`, `forget`, `search`, `history`, `feedback`, `record_tool`, `tool_hints`, `verify`,
  ...), the conversation is `ctx.history` (`add`, `thread`, `update`, `delete`) where it was
  `chat.*`, and everything else is under `ctx.advanced`.
- The multilingual runtime (ADR 0024): every query and record in every language is encoded
  in a multilingual dense space beside the English one, with a multilingual grounding model.

## 0.2.0: 2026-09-28 (service and SDK)

### Changed
- Trellis names and API conventions (ADR 0022): the SDK is `trellis.memory` (was
  `universal_memory`); the scope headers are `X-Trellis-Tenant`, `-Workspace` and `-User`;
  errors are RFC 9457 problem documents; W3C `traceparent` is continued and `X-Trace-ID`
  returned; operation ids are `<tag>.<function>`. The old spellings were kept as aliases for
  one release.
- Tenants, API keys and workspaces are the service's own (ADR 0021); feedback, cursor
  pagination and agent model keys (ADR 0023).
