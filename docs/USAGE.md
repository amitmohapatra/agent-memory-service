# Which API for which scenario

For an engineer wiring an agent to trellis-memory. Each section says **when** to use a part of
the API, the **endpoint(s)**, a short **SDK** example (`pip install -e sdk/python`, package
`trellis.memory`) and the **gotchas**. The route-by-route reference is
[`api/README.md`](api/README.md); the contract is [`openapi.json`](openapi.json).

Every SDK snippet below runs inside an `async` function and assumes:

```python
from trellis.memory import MemoryClient

memory = MemoryClient()  # MEMORY_URL and TRELLIS_API_KEY; or MemoryClient(url, api_key=SERVICE_KEY)
ctx = memory.bind(user_id="u1", thread_id="chat-42")  # the key names the tenant
```

A development key (`dev-key` on the local stack) names the development tenant, `default`
(`MEMORY__AUTHENTICATION__TRUSTED_DEV_TENANT`): with it no tenant is needed either, and
`bind(tenant_id=...)` picks another. The client retries what cannot duplicate anything, honours
`Retry-After`, and fails fast with `CircuitOpenError` (a retryable
`DependencyUnavailableError`) while the service is down — see
[the SDK README](../sdk/python/README.md#retries-and-the-circuit-breaker).

---

## 1. Before the first call

**When:** once per tenant, before any agent traffic.

| Step | Endpoint | SDK |
| --- | --- | --- |
| Onboard the tenant (platform bootstrap key) | `POST /v1/admin/tenants` → the first **admin key**, shown once | `MemoryClient(url, api_key=BOOTSTRAP).admin.create_tenant("Acme", tenant_id="acme")` |
| Issue a **service key** for the harness/agent (admin key) | `POST /v1/keys` | `admin.tenant.keys.issue("service", "support-bot", workspace_id=…, may_act_as=["*"])` |
| Create a team and admit members (admin key) | `POST /v1/workspaces`, `PUT /v1/workspaces/{workspace_id}/members/{principal_ref}` (`user:…` or `agent:…`) | `admin.tenant.workspaces.create("Finance", workspace_id="finance")`, `.set_member("finance", "user:u1", role="member")` |
| Check who a key is | `GET /v1/keys/self` | `await memory.tenant.keys.whoami()` → `key_id, tenant_id, principal, role, may_act_as` |

```python
platform = MemoryClient(url, api_key=BOOTSTRAP_ADMIN_KEY)
created = await platform.admin.create_tenant(
    "Acme", tenant_id="acme", idempotency_key="onboard-acme"
)
admin = MemoryClient(url, api_key=created.admin_key.token)  # store the token now
await admin.tenant.workspaces.create("Finance", workspace_id="finance")
await admin.tenant.workspaces.set_member("finance", "user:u1")
service = await admin.tenant.keys.issue("service", "support-bot")
print((await MemoryClient(url, api_key=service.token).tenant.keys.whoami()).role)  # "service"
```

- **Admin key vs service key.** `admin` administers one tenant (keys, workspaces, model keys and
  policy, the read audit, the feedback review queue); `service` reads and writes memory for the
  users and agents it may act for. Give an agent a service key, never the admin key.
- The tenant comes from the key; a `X-Trellis-Tenant` header that disagrees is a `403`.
- `may_act_as` restricts a key to the listed principals: the request's user must be listed as
  `user:<id>` and its `agent_id` as `agent:<id>`, or it is a `403`; a restricted key naming
  neither acts only as itself.
- A secret is shown once; with `idempotency_key` a retry replays the record with `token=None`.
- A `WORKSPACE`-visible write needs the workspace row and a membership first, or it answers
  `Workspace not found`. Workspace roles are `admin`, `member` (read + write) and `viewer`
  (read only). A deleted workspace id is never reusable.

## 2. Pick a mode: push, pull or manual

| Mode | When | Endpoint | SDK |
| --- | --- | --- | --- |
| **Push** — context injected every turn | default for any chat agent: the agent sees the profile, thread summary, recent messages and relevant memories/documents without deciding to look | `POST /v1/context` | `await ctx.context(question)` → `.rendered` into the system prompt |
| **Pull** — the agent's own tools | the agent needs to search further, store or correct mid-run (ReAct, function calling) | `GET /v1/agent-tools`, `POST /v1/agent-tools/{name}` | `ctx.agent_tools()`, `ctx.call_agent_tool(name, args)` ([§9](#9-agent-tools-for-function-calling-agents)) |
| **Manual** — your code calls verbs | pipelines, back-office jobs, importers, tests | the routes below | `remember`, `update`, `forget`, `search`, `history`, `feedback`, `record_tool`, `tool_hints`, `profile`, `verify`, `ctx.advanced.*` |

They combine: push every turn *and* hand the agent the pull tools (that is what the harness
does). What the agent keeps pulling for a kind of request is pre-included in later pushes
(prefetch learning).

## 3. Writing: turns, facts or documents

| You have… | Use | Endpoint | SDK |
| --- | --- | --- | --- |
| a conversation turn, or something that happened | messages: the service extracts what is worth keeping, asynchronously | `POST /v1/messages` (202) | `await ctx.history.add([("USER", text), ("ASSISTANT", reply)])`; `("EVENT", "User cancelled the Pro plan")` |
| a fact you *know* (a form field, a CRM value, a setting) | a memory stored verbatim, now | `POST /v1/memories` | `await ctx.remember("Prefers metric units", memory_type="PREFERENCE", visibility="USER")` |
| a file (PDF, DOCX, HTML, …) | a document: parsed, chunked, indexed with page provenance | `POST /v1/documents`, then `GET /v1/documents/{id}` or `GET /v1/jobs/{id}` | `doc = await ctx.advanced.documents.add("fy26.pdf", title="FY26")`; `await ctx.advanced.documents.wait_ready(doc.document_id)` |

```python
[ack] = await ctx.history.add([("EVENT", "Castor Supply raised lead time to 12 days.")])
for job_id in ack.job_ids:
    print((await ctx.advanced.job(job_id)).status)  # PENDING ... SUCCEEDED
```

- A `2xx` means *durable*, not *retrievable*: messages and documents are processed by jobs;
  poll `GET /v1/jobs/{job_id}` in scripts and tests, never inside a turn.
- `remember` deduplicates on content per owner and scope (`deduplicated=True`); messages
  deduplicate on the idempotency key, or on `source_system` + `source_message_id` when you set
  them (how the harness re-sends a transcript safely).
- `history.*` needs a `thread_id` (or an agent run, whose id names the thread); `remember`,
  `search` and `context` need only a tenant.
- `remember` with no `visibility` takes the scope's own; documents default to the thread, else
  the user.

## 4. Correcting: supersede, forget, restore

| Scenario | Endpoint | SDK |
| --- | --- | --- |
| The fact changed (keep the history) | `POST /v1/memories/{id}/supersede` | `await ctx.update(memory_id, "Prefers imperial units", reason="user corrected it")` |
| It must stop being retrieved | `DELETE /v1/memories/{id}` | `await ctx.forget(memory_id)` |
| Automatic forgetting archived something still needed | `POST /v1/memories/{id}/restore` | `await ctx.advanced.memories.restore(memory_id)` |

- `update` closes the old version (`SUPERSEDED`, still readable with `search(as_of=…)`); an
  already superseded memory answers `409`. `update`/`forget` accept a bundle handle (`"m3"`)
  with `bundle_id=`.
- Update, forget and restore need the memory's owner, the user an agent acts for, or a tenant
  admin.
- A conversation turn is also kept verbatim beside the facts read out of it (so retrieval
  finds what no rule extracted). Forgetting a fact removes the fact, not the sentence: the
  verbatim turn (predicate `said`, listed by `GET /v1/memories`) stays searchable until you
  forget it too. Updating a fact likewise keeps the old turn as what was said then; only
  the current fact is served as the fact.
- `restore` only undoes **archiving** (the idle-and-low-score sweep). A forgotten memory —
  `DELETE`, the `memory_forget` tool, or the tenant's `retention_days` sweep — stays forgotten
  (`404`).

## 5. Reading: which read

| Question | Use | Endpoint | SDK |
| --- | --- | --- | --- |
| "What should the model see for this turn?" | the context: ranked, deduplicated, token-budgeted, with an evidence status | `POST /v1/context` | `await ctx.context(q, token_budget=2000)`; `format="full"` for the parts |
| "Give me ranked items for my own prompt/UI" | recall | `POST /v1/recall` | `await ctx.search(q, kinds=["memory", "chunk"], limit=20)` |
| "What was true / known on date X?" | recall with time travel | `POST /v1/recall` | `await ctx.search(q, as_of=dt)` / `known_at=dt`; `time_from`/`time_to` filter by when observed |
| "Who/what is X, and how is it related?" | the knowledge graph | `GET /v1/graph/entities`, `GET /v1/graph/entities/{id}` | `[e] = await ctx.advanced.graph.entities("acme", limit=1)`; `await ctx.advanced.graph.entity(e.entity_id, depth=2)` |
| "Show the conversation" | thread history | `GET /v1/threads/{id}/messages`, `GET /v1/threads/{id}` | `await ctx.history(limit=50)`; `(await ctx.history.thread()).summary` |
| "Search earlier conversations" | recall over episodes / this thread | `POST /v1/recall` | `await ctx.search(q, kinds=["episode"])` or `kinds=["message"]` |
| "What do we hold about this user?" (audit, unranked) | the inventory | `GET /v1/memories` | `await ctx.advanced.memories.page(limit=100)` |

- `context` returns `rendered`, `bundle_id`, `evidence_status` and `token_estimate` by
  default; items are cited by handle (`[m1]`, `[d2]`). `window=False` when your framework keeps
  its own history.
- Memories are ranked by fixed weighted reciprocal-rank fusion of BM25, two dense spaces and
  ColBERT arms, then session/speaker/time/period rules (ADR 0026) — nothing to tune per call.
- No read calls a model unless the tenant policy's `read_assist` is on and a key can pay
  ([§11](#11-model-keys-policy-and-spend)); there is no per-request switch.

## 6. Profile blocks and standing questions

**When:** facts every prompt should start from (persona, preferences, team conventions), and a
question whose answer should stay current without anyone asking it.

| Endpoint | SDK |
| --- | --- |
| `GET /v1/profile` | `await ctx.profile()` |
| `PATCH /v1/profile/{block}` | `await ctx.profile.edit("agent.persona", "Be brief.")`; `ctx.profile.edit("user", new, old=old)` |

```python
await ctx.profile.edit(
    "user.suppliers", source_query="Which suppliers does this team buy from, and on what terms?"
)
```

- Blocks are `user`, `agent`, `workspace`, optionally `.<name>`; at most 4,000 characters.
  They are pinned into every pushed context.
- `old` must still be in the block or the edit is a `409` (`ConflictError`): read and retry.
- A `source_query` is answered by a background job when set and hourly after; reading it never
  calls a model. `source_query=None` removes the question and keeps the last answer.
- The service maintains the `user` block from USER/PREFERENCE memories; your edits are kept.

## 7. Evidence: the status and `/v1/verify`

**When:** answers that must be grounded (finance, policy, support macros).

| Endpoint | SDK |
| --- | --- |
| `POST /v1/context` (`evidence_status`) | `prompt = await ctx.context(q)` |
| `POST /v1/verify` | `report = await ctx.verify(answer, bundle_id=prompt.bundle_id)` |

```python
prompt = await ctx.context("What were FY26 restructuring savings?")
if prompt.evidence_status == "INSUFFICIENT":
    return "I don't have that in my sources."
answer = await my_llm(prompt.rendered)
report = await ctx.verify(answer, bundle_id=prompt.bundle_id)
print(
    report.supported, report.unsupported, report.contradicted, report.per_claim_hallucination_rate
)
```

- There is no `require_evidence` flag and nothing raises: check `evidence_status`
  (`COMPLETE`, `INCOMPLETE`, `INSUFFICIENT`) yourself.
- `verify` needs the `bundle_id` of the context the answer was given, in the same scope, within
  30 minutes (`404` otherwise).
- With a run (`run_id=`, or the scope's `agent_run_id`) the verdict is recorded as the
  service's own judge feedback on the run (`report.feedback_id`), applied without review.
- `representative=False` on the report means the NLI is a stand-in: do not trust the numbers.

## 8. Tool memory

**When:** your agent calls tools and you want it to stop picking the wrong one twice.

| Scenario | Endpoint | SDK |
| --- | --- | --- |
| Describe your tools (once, idempotent) | `PUT /v1/tools/catalog` | `await ctx.advanced.tools.put_catalog([{"name": "erp-create_po", "description": "…", "side_effects": "write"}])` |
| Read the catalog with statistics | `GET /v1/tools?names=…` | `await ctx.advanced.tools.catalog(names=[…])` |
| Record each call | `POST /v1/tools/invocations` | `await ctx.record_tool("erp-get_stock", {"sku": "A4"}, output={…}, task=question, step=0)` |
| Label the run | `POST /v1/feedback` | `await ctx.feedback("run", ctx.scope.agent_run_id, "confirm", source="system")` |
| Which tool, plan, arguments | `POST /v1/tools/hints` (or inline: `context(q, tools=[…])`) | `hints = await ctx.tool_hints(task, available=names)` |
| Approval rules the decisions support | `GET /v1/tools/approval-suggestions` | `await ctx.advanced.tools.approval_suggestions(tool=…)` |
| Adopt one | `POST /v1/tools/approval-suggestions/{id}/accept` | `await ctx.advanced.tools.accept_suggestion(s.id)` |

- The service never executes a tool. `task` is the question in words (it becomes the pattern
  procedures are keyed on), not an id.
- A procedure needs ≥ 2 supporting runs and ≥ 60 % success; an unlabelled run counts as a weak
  success only after a day without failed calls.
- Recording is idempotent on run + step + tool + arguments; `visibility` defaults to
  `PRIVATE` (share with `AGENT_GROUP`/`WORKSPACE`/`TENANT` to learn across agents).
- A suggestion can be accepted only by the agent it was learned from (else `404`); `409` when
  the decisions no longer support it, or when it would auto-approve an `irreversible` tool.

## 9. Agent tools for function-calling agents

**When:** the model itself should search, remember, correct or edit the profile mid-run.

| Endpoint | SDK |
| --- | --- |
| `GET /v1/agent-tools` | `tools = await ctx.agent_tools()` → `name`, `description`, `input_schema` |
| `POST /v1/agent-tools/{name}` | `await ctx.call_agent_tool("memory_search", {"query": "refund policy", "k": 5})` |

The six tools: `memory_search`, `memory_remember`, `memory_update`, `memory_forget`,
`profile_edit`, `tool_search`. Pass `toolbox=[your tool names]` to `call_agent_tool` for
`tool_search` to choose among them.

- Hand the schemas to your model as-is; return the `result` to it as the tool output.
- `memory_update`/`memory_forget` accept a context handle (`m3`); ownership rules still apply.
- Every call is logged as a pull and feeds prefetch learning; do not record these calls again
  with `record_tool`.
- Bad arguments are a `422` naming `args.<field>`; an unknown tool `404`.

## 10. Feedback: what applies now and what waits

| Scenario | Endpoint | SDK |
| --- | --- | --- |
| Record a verdict | `POST /v1/feedback` | `await ctx.feedback("memory", mem_id, "correct", correction="…")` |
| Read one / a target's history | `GET /v1/feedback/{id}`, `GET /v1/feedback?target_kind=…&target_id=…` | `ctx.feedback.get(id)`, `ctx.feedback.page_for("run", run_id)` |
| The review queue (tenant admin key) | `GET /v1/feedback?review=pending` | `await admin_ctx.feedback.pending(limit=50)` |
| Decide a vote (tenant admin key) | `POST /v1/feedback/{id}/approve`, `/dismiss` | `admin_ctx.feedback.approve(id, note=…)`, `.dismiss(id, note=…)` |

**Applied immediately:** the service's own grounding judge (`/v1/verify`); a run reporting its
own status (`source="system"`, target = the calling `agent_run_id`, citing no memories); an
owner's `reject`/`correct`/`edit` of their own memory; any `tool_call` verdict; anything sent
with the tenant admin key or by a tenant admin user in person.

**Waits for review** (`review.state == "pending"`, changes nothing): a `confirm`/`approve` of a
memory by anyone else, a person's or client's verdict on a run, any procedure verdict — and a
client-claimed `source="judge"`.

```python
admin_ctx = MemoryClient(url, api_key=TENANT_ADMIN_KEY).bind()
page = await admin_ctx.feedback.pending(limit=50)
for vote in page.items:
    print(vote.target_kind, vote.verdict, vote.source, vote.author_record)  # author's track record
    await admin_ctx.feedback.approve(vote.feedback_id, note="checked")  # or .dismiss(...)
```

- Approve/dismiss a vote that is not pending: `409`. A dismissed vote is kept for statistics.
- Nobody working the queue means votes never count.
- `reject`/`correct`/`edit` of a memory requires its owner or a tenant admin (`403` otherwise).
- A `POST` answers before the projector runs: `projection` is null until then.

## 11. Model keys, policy and spend

**Who pays** (most specific row wins):

| Level | Endpoint | SDK |
| --- | --- | --- |
| The agent (a request whose scope names the `agent_id`) | `PUT`/`GET`/`DELETE /v1/agents/model-key` | `ctx.advanced.model_keys.set(vk)`, `.status()`, `.revoke()` |
| The tenant (admin key) | `PUT`/`GET`/`DELETE /v1/model-key` | `admin.tenant.set_model_key(vk)`, `.model_key_status()`, `.revoke_model_key()` |
| The operator (only while no row exists) | `BIFROST_VIRTUAL_KEY` env, gateway `BIFROST_URL` | — |

**What it may be spent on** — the tenant policy, three fields only: `uses`, `read_assist`,
`models`:

```python
await admin.tenant.set_model_policy(
    ["contextual_extraction", "summaries", "grounding_judge", "memory_restatement"],
    read_assist=False,  # reads stay deterministic
    models={"memory_restatement": "openai/gpt-4.1-mini"},  # per-use model through Bifrost
)
print(await admin.tenant.model_policy())  # GET /v1/model-key/policy
for day in (await admin.tenant.model_usage()).days:  # GET /v1/model-key/usage
    print(day.day, day.use, day.tokens, day.calls)
```

- No policy row = every use **except** `memory_restatement` (opt-in), reads assisted. A stored
  policy runs exactly the uses it names: list `memory_restatement` to opt in.
- A revoked key refuses; it never falls through to the tenant's or the operator's.
- There is no workspace-level key or policy, no per-request `use_llm`, and no
  `MEMORY__MODELS__*` variable; model defaults are code constants (`auto` via the gateway).
- Changing a policy or key invalidates cached model-assisted reads.
- Each use, its tier and its fallback: [`LLM-USES.md`](LLM-USES.md).

## 12. Tenancy operations

| Scenario | Endpoint | SDK |
| --- | --- | --- |
| Rotate a key | `POST /v1/keys` (new), deploy it, `DELETE /v1/keys/{key_id}` (old) | `keys.issue(...)`, `keys.revoke(old_id)` |
| Narrow whom a key may act for | `PATCH /v1/keys/{key_id}` | `admin.tenant.keys.update(key_id, may_act_as=["agent:support-bot"])` |
| Retention / rate limit / suspend / admission gate (platform key) | `PATCH /v1/admin/tenants/{id}` | `platform.admin.update_tenant("acme", retention_days=365, rate_limit_per_minute=600)`; `status="suspended"`; `admission_gate=True` (off by default: score extracted candidates and store only the admitted ones) |
| Who read what | `GET /v1/reads` | `await admin.tenant.reads(limit=50)` |

- Revocation and `may_act_as` changes apply on the key's next request, on every instance.
- Retention forgets canonical memories older than `retention_days` (daily); conversation rows
  and documents are not covered yet.
- The rate limit is a one-minute window (service default 6,000 + 200 burst, constants); a
  breach is `429` with `Retry-After`. It fails open if the cache is down.
- The read audit stores `query_hash`, never the query text; newest first, cursor paged.

## 13. Backfills and operations

| Scenario | Command |
| --- | --- |
| A tenant just opted into `memory_restatement`: restate turns stored before | `python -m memory_service.tools.restate --tenant <id> [--limit N] [--force]` |
| Index lost, encoder changed, or a key-layout change (ADR 0026's `mk3`) | `make reindex` (`REINDEX_ARGS="--drop"` for a full rebuild; `make reindex-image` inside the runtime image) |

- `restate` binds each turn to its owner, so their key pays and their policy decides; `--force`
  also redoes turns that already have a restatement.
- PostgreSQL is the source of truth; Qdrant is rebuilt from it.

## 14. Through the harness

Sections 1 to 13 are the SDK on its own: it plugs into any framework (LangGraph, OpenAI
Agents, the Claude Agent SDK) or plain code, and the
[SDK README](../sdk/python/README.md#use-it-on-its-own-or-with-the-harness) shows one turn.
If your agent runs under [`agent-harness`](https://github.com/amitmohapatra/agent-harness)
(`trellis-harness`, see its `docs/memory.md`), `h.wrap(agent)` with `MEMORY_URL` set does
the above for you:

- **Push:** `/v1/context` for the run's question with a 2,000-token budget, `window=False`
  when the framework keeps its own history, and the run's tool names when it has 5 or more
  (so the `tools` that come back, each with a confidence, narrow what the model is offered).
- **Pull:** adds the six agent tools to the run's tools.
- **Records:** the transcript (`history.add`, named by `source_system`/`source_message_id`),
  every non-memory tool call (`record_tool`), and interrupt decisions as feedback.
- **Outcome:** `system` feedback on the run (`SUCCESS` → `confirm`, `ERROR` → `reject`).
- **Grounding:** on a 10 % sample of runs, `/v1/verify` with the run's `bundle_id`; the score
  goes on the run's trace.
- **Model key:** registers `BIFROST_VIRTUAL_KEY` as each agent's model key.
- **Documents:** `h.add_document(file, user=…, thread=None)` uploads into the user's (or
  thread's) scope and waits until it is indexed; the next context cites it.
- **People's verdicts:** `h.feedback(run_id, verdict)` returns the stored record, pending
  until the tenant administrator approves it (§10).

## 15. Cheat sheet

| Scenario | Endpoint | SDK call |
| --- | --- | --- |
| Who is this key? | `GET /v1/keys/self` | `memory.tenant.keys.whoami()` |
| Context for this turn | `POST /v1/context` | `ctx.context(q)` |
| Ranked items | `POST /v1/recall` | `ctx.search(q, kinds=…)` |
| Append turns / an event | `POST /v1/messages` | `ctx.history.add([...])` |
| Read the thread | `GET /v1/threads/{id}/messages` | `ctx.history(limit=…)` |
| State a fact | `POST /v1/memories` | `ctx.remember(text, memory_type=…)` |
| Change a fact | `POST /v1/memories/{id}/supersede` | `ctx.update(id, text, reason=…)` |
| Forget a fact | `DELETE /v1/memories/{id}` | `ctx.forget(id)` |
| Un-archive a fact | `POST /v1/memories/{id}/restore` | `ctx.advanced.memories.restore(id)` |
| List what is held | `GET /v1/memories` | `ctx.advanced.memories.page()` |
| Upload a document | `POST /v1/documents` | `ctx.advanced.documents.add(file)` |
| Wait for processing | `GET /v1/jobs/{id}`, `GET /v1/documents/{id}` | `ctx.advanced.job(id)`, `documents.wait_ready(id)` |
| Graph lookup | `GET /v1/graph/entities[/{id}]` | `ctx.advanced.graph.entities(q)`, `.entity(id, depth=…)` |
| Pinned profile | `GET /v1/profile`, `PATCH /v1/profile/{block}` | `ctx.profile()`, `ctx.profile.edit(...)` |
| Check an answer | `POST /v1/verify` | `ctx.verify(answer, bundle_id=…)` |
| Tool catalog | `PUT /v1/tools/catalog` | `ctx.advanced.tools.put_catalog([...])` |
| Record a tool call | `POST /v1/tools/invocations` | `ctx.record_tool(tool, args, …)` |
| Tool hints | `POST /v1/tools/hints` | `ctx.tool_hints(task, available=…)` |
| Approval suggestions | `GET …/approval-suggestions`, `POST …/{id}/accept` | `ctx.advanced.tools.approval_suggestions()`, `.accept_suggestion(id)` |
| Agent tools | `GET /v1/agent-tools`, `POST /v1/agent-tools/{name}` | `ctx.agent_tools()`, `ctx.call_agent_tool(name, args)` |
| A verdict | `POST /v1/feedback` | `ctx.feedback(kind, id, verdict, …)` |
| Review votes (admin) | `GET /v1/feedback?review=pending`, `POST …/approve`, `…/dismiss` | `ctx.feedback.pending()`, `.approve(id)`, `.dismiss(id)` |
| Agent model key | `PUT /v1/agents/model-key` | `ctx.advanced.model_keys.set(vk)` |
| Tenant model key (admin) | `PUT /v1/model-key` | `client.tenant.set_model_key(vk)` |
| Model policy (admin) | `PUT /v1/model-key/policy` | `client.tenant.set_model_policy(uses, read_assist=…, models=…)` |
| Spend (admin) | `GET /v1/model-key/usage` | `client.tenant.model_usage()` |
| Keys (admin) | `POST`/`GET /v1/keys`, `PATCH`/`DELETE /v1/keys/{id}` | `client.tenant.keys.issue/list/update/revoke` |
| Workspaces (admin) | `/v1/workspaces[/{id}/members/{p}]` | `client.tenant.workspaces.create/set_member/remove_member` |
| Read audit (admin) | `GET /v1/reads` | `client.tenant.reads()` |
| Onboard / retention / quota (platform) | `/v1/admin/tenants[/{id}]` | `client.admin.create_tenant(...)`, `.update_tenant(id, …)` |
