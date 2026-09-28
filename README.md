# trellis-memory

**Durable, scope-aware memory for AI agents.** One service your agents talk to so they
remember what happened, what the user prefers, what the documents say, and which tools
actually worked — with hard guarantees about what is never lost and who may never see what.

Works with any framework (LangGraph today, plain Python anywhere) because the core knows
nothing about your agent library.

```python
from trellis.memory import MemoryClient

memory = MemoryClient("http://localhost:8080", api_key="dev-key")

ctx = memory.bind(
    tenant_id="acme",
    user_id="u1",
    thread_id=thread_id,
    session_id=session_id,
    turn_id=turn_id,
)

await ctx.chat.user("I'm in Berlin and I prefer short answers.")
bundle = await ctx.context("draft a reply about the Q3 numbers")  # everything relevant
answer = await my_agent.run(bundle.rendered)  # your agent, your model
await ctx.chat.assistant(answer)
```

Next turn — in a different session, a week later — the agent already knows the timezone and
the preference. You did not write retrieval code, a vector store, or a summariser.

---

## Contents

| | |
|---|---|
| [What you get](#what-you-get) | the five kinds of memory, and the guarantees |
| [Install and run](#install-and-run) | 5 minutes to a running service |
| [Use it in one agent](#use-it-in-one-agent) | chat, facts, documents, context bundles |
| [Use it with multiple agents](#use-it-with-multiple-agents) | who sees what, hand-offs, sharing |
| [Tool memory](#tool-memory) | stop calling the wrong tool twice |
| [Grounding](#grounding-did-the-answer-actually-follow-from-the-evidence) | verify an answer against its evidence |
| [Framework integrations](#framework-integrations) | LangGraph, and the shape of the rest |
| [Configuration](#configuration) | the knobs that matter |
| [How it works](#how-it-works) | architecture in one screen |
| [Status](#status-read-this-before-you-trust-a-number) | what is proven and what is not |

---

## What you get

Five kinds of memory, one API:

| Kind | What it holds | Example |
|---|---|---|
| **Conversation** | threads, sessions, turns, messages — archived and replayable | "what did we decide last Tuesday?" |
| **Semantic** | durable facts and preferences extracted from what was said | "prefers concise answers", "timezone Europe/Berlin" |
| **Document** | uploaded files parsed into a hierarchy with page-level provenance | "what does the FY26 report say about EBITDA?" |
| **Knowledge graph** | typed entities and relations with validity windows | "who approved the acquisition, and when?" |
| **Tool** | which tool worked for which task, and in what order | "which API do I call to reprice a quote?" |

And four guarantees the service is built around:

- **Acknowledged data is never lost.** A `2xx` comes back only after the record *and* its
  processing job are committed to PostgreSQL in one transaction.
- **Nothing leaks across a boundary.** Tenant, user, agent and run isolation is enforced as a
  store-side filter *before* search runs, not as a filter on results.
- **Answers cite their evidence.** Retrieval verifies it has the evidence groups an answer
  needs, and says `INSUFFICIENT_EVIDENCE` rather than guessing.
- **It runs without an LLM.** Extraction, the knowledge graph, summaries and tool learning are
  deterministic. A model is optional and only sharpens specific decisions.

---

## Install and run

Requires Docker. (Python 3.12 only if you want to work on the service itself.)

```bash
git clone <repo> memory-service && cd memory-service
docker compose up -d
```

That is the whole thing. Bringing the stack up downloads the model weights into `./models`
(~1 GB, once, git-ignored) and applies both database schemas before the API and the worker
start, so a fresh clone reaches a working service with no further steps. Allow a few minutes
the first time — most of it is the download and the image build. Afterwards the weights are
recognised as present and skipped.

To work on the service rather than just run it:

```bash
make setup          # uv venv + dependencies
make models         # the same weights, without Docker
make dev-up         # the same stack
make migrate        # the same schemas
```

The service reads its weights from local directories and never downloads at run time, so a
missing model is a startup error rather than a silent fall back to something weaker. (The test
suite and `examples/serve.py` do have a deterministic stand-in, which is why they run
without weights — see below.)

The API is on **http://localhost:8080** — interactive docs at `/docs`, health at
`/health/ready`. The dev API key is `dev-key`.

```bash
pip install -e sdk/python        # the trellis-memory SDK
```

### What actually runs

`make models` fetches the frozen set into `models/`, each in a directory named after the
model. The set is `FROZEN_MODELS` in `src/memory_service/config/constants.py`, and the
download catalogue is derived from it, so the code and the weights cannot drift apart:

| Role | Frozen | Size | Why |
|---|---|---|---|
| Embedding | `ibm-granite/granite-embedding-small-english-r2` (384-dim) | 94 MB | lowest query p95 of every candidate benchmarked, at the smallest useful dimension |
| Sparse | BM25 (client term frequencies, Qdrant server-side IDF) | — | no weights |
| Reranker | none | — | `cross-encoder/ms-marco-MiniLM-L6-v2` measured significantly *worse* on SciFact (nDCG 79.3 vs 84.5, p = 0.012) at 21x the latency |
| Grounding NLI | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` | 371 MB | claim-support classifier for `/v1/verify` |

The benchmark challengers (Granite R2 base, Granite reranker, SPLADE, GLiNER2, the former
reranker) are listed in `benchmark/challengers.txt`; `make models-all` fetches them, and only
`make bench-embedding` / `make bench-reranker` ever load one. **No Chinese-origin model or
derivative runs anywhere in the stack** - not as a default, a challenger, an operator setting
or a gateway model. The rule is `src/memory_service/domain/provenance.py`, and
`tests/unit/test_model_provenance.py` asserts it on every surface that names a model.

**These defaults were chosen by measurement, on CPU.** `make bench-embedding` runs every
candidate through the real pipeline and the golden set; the numbers below are from
`benchmark/results/embedding.json` (p95 over 18 golden queries, 4-core container):

| Candidate | dim | Recall@20 | EGR | query p95 | index |
|---|---|---|---|---|---|
| `granite-embedding-small-english-r2` | 384 | 1.00 | 1.00 | **204 ms** | 59 s |
| `granite-embedding-english-r2` | 768 | 1.00 | 1.00 | 2,020 ms | 369 s |

Rerankers, scoring 20 candidates: `ms-marco-MiniLM-L6-v2` 1,399 ms ·
`granite-embedding-reranker-english-r2` 11,978 ms. (Rows for models since excluded by
provenance remain in `benchmark/results/embedding.json`; they are evidence, not candidates.)

Three things that table is actually telling you:

- **Every candidate scores a perfect 1.00.** That is not evidence they are equally good — it
  means the golden set (18 questions over 2 documents) is too easy to separate them. Quality
  here is *undiscriminated*, not *equal*, and the set needs harder questions before it can
  rank encoders.
- **Bigger is not better under a latency budget.** The 768-dim Granite and the 1024-dim
  challengers since excluded bought no measurable recall here and cost 10-26x their smaller
  siblings.
- **The 300 ms recall budget does not survive real models on this hardware.** End-to-end
  recall p95 was 3.5-4.1 s for *every* candidate. Those budgets were set against the
  deterministic stand-in and need re-justifying against a deployed instance — see
  [Status](#status-read-this-before-you-trust-a-number).

The model set is frozen in `src/memory_service/config/constants.py` (`FROZEN_MODELS`): a swap
is a code change, reviewed like one, never an env edit. Changing the embedding changes the
vector space, so **re-index after a swap** (`make reindex`); the model, backend, ONNX graph and
dimension are part of the collection name, so old and new vectors can never silently mix.

**Where the stand-in applies.** The test suite, `examples/serve.py` and a host-side `make gates`
fall back to a deterministic hash embedding when `models/` is absent, so they exercise the
plumbing without a download. The stand-in is an `Overrides` field on the container, never a
setting; `serve.py` prints which mode it is in, and any benchmark produced that way is labelled
`representative: false`. Never read a retrieval number that carries that flag.

---

## Use it in one agent

### Bind a scope, once

Every call happens inside a scope. Bind it once per request and forget about it:

```python
from trellis.memory import MemoryClient

memory = MemoryClient("http://localhost:8080", api_key="dev-key")

ctx = memory.bind(
    tenant_id=request.tenant_id,  # hard isolation boundary
    workspace_id=request.workspace,  # a project or app
    user_id=request.user_id,  # who this is for
    thread_id=request.thread_id,  # the conversation
    session_id=request.session_id,  # a continuous stretch of it
    turn_id=request.turn_id,  # this exchange
)
```

`tenant_id` is the wall nothing crosses; everything else narrows visibility further.

Thread, session and turn are **your** identifiers — pass the ones your app already has, and
the service creates them on first use. They are required for `chat.*` (a message has to belong
to a turn); for `remember`, `observe`, `recall` and `context` a tenant is enough. If you have
no conversation to attach to, `await ctx.chat.create()` gives you a thread to start from.

### Record the conversation

```python
await ctx.chat.user("Revenue was EUR 412 million in FY26.")
await ctx.chat.assistant("Noted — that's up 4% year on year.")
```

That is all. In the background the service stores the messages, archives them immutably,
extracts durable facts, and indexes everything for retrieval. Your request already returned.

### Ask for context instead of doing retrieval

```python
bundle = await ctx.context("how did revenue develop?")
answer = await my_llm(bundle.rendered)
```

`bundle.rendered` is a token-budgeted, ready-to-prompt block containing the recent
conversation, the relevant memories, document passages, and graph facts — deduplicated and
ordered. Inspect the parts if you want them separately:

```python
bundle.conversation  # recent turns plus a rolling summary
bundle.memories  # durable facts and preferences
bundle.knowledge  # document passages, each with document, page and evidence
bundle.graph_facts  # entity relations
bundle.evidence  # what was found, what was missing, and the status
```

Need just the search results? `await ctx.recall("...")` returns ranked items.

Reads default to `use_llm=False`, independently of ingestion's model settings. Pass
`use_llm=True` to permit the configured read helpers. Agent-owned virtual keys and persistent
standing questions/pages are exposed through `ctx.set_model_key(...)` and `ctx.briefs`;
see the [SDK examples](sdk/python/README.md) and
[capability/validation handoff](docs/AGENT-CAPABILITIES-HANDOFF-20260927.md).

### State a fact directly

When your app knows something rather than inferring it from chat:

```python
await ctx.remember("Prefers metric units", memory_type="PREFERENCE")
await ctx.observe("User cancelled the Pro plan", kind="EVENT")
```

`remember` stores a fact you assert. `observe` hands the service a raw event and lets it
decide what, if anything, is worth keeping.

### Add documents

```python
doc = await ctx.documents.add(open("fy26.pdf", "rb"), title="FY26 annual report")
await ctx.documents.wait_ready(doc.document_id)
```

The file is parsed into sections, tables and footnotes with page numbers preserved. After
that it answers questions through the same `ctx.context(...)` call — passages come back with
the document and page they came from, so you can cite them.

### Require evidence

For answers that must be grounded:

```python
bundle = await ctx.context("what were FY26 restructuring savings?", require_evidence=True)
if bundle.evidence.status == "INSUFFICIENT_EVIDENCE":
    return "I don't have enough in my sources to answer that."
```

The service escalates its own search first (broader retrieval, the graph, structured lookup)
and only reports insufficiency when it genuinely cannot assemble the evidence.

---

## Use it with multiple agents

The hard part of multi-agent memory is not storage — it is **who is allowed to see what**.
A researcher agent's scratch notes should not leak into the user's history; a hand-off to a
sub-agent should flow *down* but not *sideways*; a shared finding should be shared on purpose.

### Each agent run gets its own memory scope

```python
researcher = ctx.agent("researcher", agent_group_id="analysis-crew")  # a new run, isolated
writer = ctx.agent("writer", agent_group_id="analysis-crew")  # a sibling run

await researcher.remember("Source A contradicts source B", visibility="RUN")
```

The `agent_group_id` names the crew these runs belong to. It costs nothing while the notes
stay `RUN`-scoped, and it is what makes the explicit sharing below possible: an
`AGENT_GROUP` memory needs a group to be shared *with*, and the service rejects the write
if the run does not belong to one.

`visibility="RUN"` (the default for agent notes) means: this run, and any run it spawns.
Not the user, not sibling agents, not the next run of the same agent.

### Hand-offs flow down, never up or sideways

```python
child = researcher.agent("fact-checker")  # spawned by the researcher
# child sees the researcher's RUN-scoped notes — that is the hand-off
# writer (a sibling) still sees nothing of either
```

### Sharing is explicit

```python
await researcher.remember(  # the run must belong to an agent group
    "FY26 revenue is EUR 412m, confirmed in two sources",
    memory_type="SHARED",
    visibility="AGENT_GROUP",  # now the whole crew can use it
)
```

The visibility ladder, narrowest first:

| Visibility | Who can read it |
|---|---|
| `PRIVATE` | only the agent or user that wrote it |
| `RUN` | this agent run and the runs it spawns (hand-off) |
| `AGENT_GROUP` | a named crew of cooperating agents |
| `USER` | the user, across all their threads |
| `THREAD` | everyone in this conversation |
| `WORKSPACE` | the members and viewers of a workspace - a team (see below) |
| `TENANT` | the whole tenant |

### What the user sees

A user never sees agent chatter unless it was explicitly shared to them. This is enforced in
the database query, not by filtering afterwards — the isolation suite asserts zero leaks
across tenant, user, agent and run boundaries.

### When two agents disagree

If two agents record contradicting facts in a shared scope, the service does **not** silently
pick one. Both are kept and linked as contradicting, with their evidence, so a human or a
later signal can resolve it. Corroboration works the same way in reverse: when several agents
independently record the same fact, its confidence rises and the contributors are recorded.

---

## Use it with multiple teams

One deployment serves several teams, each with its own customers. Switch authentication to
`api_key` and give the deployment one secret — the platform key — and everything else is
done through the API (executed end to end by `tests/agent/test_platform_lifecycle.py`):

```python
platform = MemoryClient(url, api_key=BOOTSTRAP_ADMIN_KEY)
acme = await platform.admin.create_tenant("Acme", tenant_id="acme")   # admin key, shown once

admin = MemoryClient(url, api_key=acme.admin_key.token)
await admin.tenant.workspaces.create("Finance", workspace_id="finance")  # a team
await admin.tenant.workspaces.set_member("finance", "user:u1")
await admin.tenant.groups.create("Analysts", group_id="analysts")
await admin.tenant.groups.add_user("analysts", "u2")
await admin.tenant.workspaces.set_member("finance", "group:analysts")   # a group at once
service = await admin.tenant.keys.issue("service", "finance-harness")   # what the harness holds

harness = MemoryClient(url, api_key=service.token)                      # no tenant_id anywhere
u1 = harness.bind(user_id="u1", workspace_id="finance")
await u1.remember("Forecast review is Tuesdays at 10:00.", visibility="WORKSPACE")
```

What that buys, and what is enforced rather than promised:

- **The tenant comes from the key.** A key names its tenant (and optionally one workspace);
  a header may agree with it and may not contradict it. Another tenant's key naming `acme`
  gets 403 — the "any tenant by changing a header" hole of shared deployments is closed.
- **`WORKSPACE` is a team's shared audience.** Members and viewers read it; members write
  it; the rest of the tenant does neither. Documents ingested and threads opened inside a
  team are its members' work: only members may create them, members read the team's
  documents, and a workspace admin may read and write its threads. A request inside a
  workspace reads that team; one naming no workspace reads every team the caller is in —
  the same rule threads follow. A workspace id that names no team stays what it was before
  teams existed: a label that grants nothing.
- **Removal is immediate.** Removing a member (or a user from a group, or a key) takes
  effect on the next request, not when a cache expires. The author of a memory keeps it.
  Suspending a tenant (`PATCH /v1/admin/tenants/{id}`) stops every one of its keys the same
  way, with 403 rather than 401; resuming restores them. Deleting a workspace revokes the
  keys bound to it; deleting a group removes it from every workspace.
- **Secrets are shown once, and retries are safe.** Onboarding and key issuance honour
  `Idempotency-Key` (`idempotency_key=` in the SDK): a retried request returns the same record
  with `Idempotent-Replayed: true` and `token: null`, never a second copy of the secret.
  Without it, every call is a new key. Deleted workspace and group ids are never reused
  (409), so audit entries keep their meaning.
- **Retention, quota, audit.** Per tenant: `retention_days` (a daily sweep forgets
  canonical memories through the same soft delete a `DELETE` uses), `rate_limit_per_minute`
  (429 with `Retry-After`), and `GET /v1/reads` — who read which records, under which scope.

Roles are `admin` (manages one tenant's keys, workspaces and groups; always tenant-wide)
and `service` (acts for its users and agents; may be pinned to one workspace). The platform
key onboards tenants and never reads or writes memory, but it administers every tenant —
it can issue a tenant's keys — so it is root: at least 32 characters in deployed
environments, and unset once onboarding is done. ADR 0021 has the design.

## Tool memory

Agents rediscover the same chain of calls every run, and the same failure every time. Tool
memory records what happened and mines the chains that worked.

**The service never executes your tools.** It records what your agent did, and learns from
runs you label successful.

### Record what your agent did

```python
await ctx.tools.record(
    "pricing.lookup_price",
    args={"sku": "SKU-22", "region": "EMEA"},
    output={"price": 1200, "currency": "EUR", "quote_id": "Q-1183"},
    task="update quote Q-1183 with EMEA price for SKU-22",
    latency_ms=42,
)
```

Recording is idempotent on (run, step, tool, arguments), so retries never double-count.

`task` is the important argument, and it is the **question**, not an identifier. The service
normalises it into a typed-placeholder pattern — `"update quote {entity} with {entity} price
for {entity}"` — so two phrasings of the same request mine the same trajectory. Pass an
opaque id and nothing will ever match.

### Read back what worked

```python
plan = await ctx.tools.plan(task, available_tools=tools)
# plan.steps      the tool sequence, in order
# plan.support    how many successful runs back it
# plan.success_rate
```

Each step carries an argument template whose bindings point at earlier steps' outputs, the
preconditions those bindings imply, and the failure modes observed after that step. A plan is
`valid` only if every tool exists and every binding resolves against a real run — nothing is
invented.

### What this deliberately does not do

There is no `suggest`, no `next`, no tool registry and no output cache. Modern models plan
tool use better than a support count can, every agent framework already owns a tool
catalogue, and caching tool output replays stale results — `stock_level(SKU-1)` returning
yesterday's number is the exact failure the rest of this service exists to prevent. What the
model *cannot* know is what worked here before, which is the one thing this keeps.

### The loop in one call

```python
result = await ctx.tools.execute(
    ToolCall(tool="pricing.lookup_price", args={"sku": "SKU-22"}, task=task),
    executor=my_tool_runner,  # your function, or a call out to an MCP gateway
)
```

Your executor, then an idempotent record. Failures are recorded with their error class, so
the next run gets the correction.

### Tell it whether the run worked

```python
await ctx.runs.outcome(run_id, success=True)
```

Only successful runs turn into procedures. This is the single most valuable signal you can
give tool memory — without it, the service waits and treats an old, error-free, uncorrected
run as a weak positive.

---

## Grounding: did the answer actually follow from the evidence?

```python
report = await ctx.verify(answer, bundle=bundle)
report.per_claim_hallucination_rate  # 0.0 when every claim is supported
for claim in report.claims:
    claim.verdict  # supported | unsupported | contradicted | borderline
```

The cascade is deterministic first and expensive last: split the answer into claims, check
citation spans against the evidence, score entailment with a local NLI model, ask the LLM
only about the genuinely borderline band, and scan the retrieved-but-unused passages for
contradictions.

---

## Using it from a framework

The SDK is framework-neutral, and deliberately so: bind a scope, call `context()` before
your agent thinks and `chat.assistant()` after, and everything in this README works from
LangGraph, CrewAI, Google ADK, an MCP server or plain code.

There is no LangGraph adapter in this repository, and that is the design. A memory service
that ships adapters knows the names of its consumers — the dependency points the wrong way,
and the service image ends up carrying framework packages it never imports. Framework
adapters belong in the layer that drives the framework: in this platform that is
[`agent-harness`](https://github.com/amitmohapatra/agent-harness), whose
`universal-agent-harness-langgraph` package already wraps LangGraph nodes and threads a
`MemoryClient` through them.

### Plain HTTP

Every capability is a documented REST route (`/docs`, or `docs/openapi.json`). The SDK is a
convenience, not a requirement.

---

## Configuration

Everything is environment variables; copy `.env.example` to `.env`.
The ones that actually matter:

```bash
# Backing services
MEMORY__DATABASE__URL=postgresql+psycopg://memory:memory@localhost:5432/memory
MEMORY__SEARCH__QDRANT_URL=http://localhost:6333
MEMORY__CACHE__URL=redis://localhost:6379/0
MEMORY__AUTHORIZATION__OPENFGA_API_URL=http://localhost:8081

# The models are not settings: `make models` puts the frozen set in ./models (git-ignored)
# and the service finds it there, or under /models in the image.

# Automatic assistance requires an agent/operator key. False prohibits generation.
MEMORY__MODELS__LLM__ENABLED=auto
```

### About the LLM

The default `enabled=auto` mode activates bounded ingestion assistance when the acting
agent has a registered virtual key. No per-agent model/use configuration is needed: the
service queries the gateway's authenticated model catalogue and selects a recognized
eligible text model. Opaque aliases are not guessed. Catalogue discovery is cached per
owner/key revision for five minutes and performs no generation. Reads remain model-free
unless `use_llm=True`; `enabled=false` prohibits generation even for registered agents.
Explicit `enabled=true` configuration below remains supported for operator-selected uses.


The service runs without one. Native model calls go through
[Bifrost](https://github.com/maximhq/bifrost), an external gateway you run yourself. The
operator key comes from deployment secrets; agent-owned virtual keys are encrypted in the
native database. Provider keys stay in Bifrost. Agent requests exclude MCP clients/tools.
The pinned Hindsight SDK provides extraction preview for eligible non-agent ingestion;
that server owns its model configuration. Agent extraction stays on the Bifrost path because
the SDK cannot carry a per-request model virtual key. Source storage and authorization stay
native. See the [integration boundary](docs/HINDSIGHT-CAPABILITY-STATUS-20260927.md).

There is deliberately **no gateway service in `docker-compose.yml`**. Starting one from this
repository's own compose file would put provider keys inside the application's deployment,
which is the coupling the gateway exists to remove. `deploy/bifrost/` holds an example config
and nothing that runs.

You choose, per capability, where a model is allowed to help:

```bash
MEMORY__MODELS__LLM__ENABLED=true
MEMORY__MODELS__LLM__BASE_URL=https://<your-gateway>/v1
MEMORY__MODELS__LLM__MODEL=anthropic/claude-sonnet-5              # complex judgement
MEMORY__MODELS__LLM__FAST_MODEL=anthropic/claude-haiku-4-5        # cheap classification
MEMORY__MODELS__LLM__USES=["conflict_adjudication","summaries"]
```

**Sizing `max_tokens` for a reasoning model.** The output budget is spent on reasoning before
any text is produced, so a budget that looks generous can return an empty answer. Measured
against `gemini-3.6-flash`: answering "Reply with exactly: OK" consumed 57 reasoning tokens,
so `max_tokens=16` produced no content at all. The adapter now raises instead of handing back
an empty string, and names the cause. Start at `MEMORY__MODELS__LLM__MAX_TOKENS=2048` for a
reasoning model.

**Rate limits.** A `429` is retried against the delay the gateway asks for rather than the
exponential backoff, because a per-minute quota is not something a 1.5-second retry schedule
can wait out. The delay is read from the `Retry-After` header *and* from the response body,
because some providers only put it there — Gemini answers "Please retry in 59.18s" in prose.
A `429` also never opens the circuit breaker: backpressure is the gateway working, and
counting it turns "slow down" into "stop".

**How much of this is configuration.** Five of these variables are this service's own policy
— whether a model may be consulted at all, which capabilities may consult it, and which of
the two models each gets. The rest (`BASE_URL`, `API_KEY`, `MAX_TOKENS`, `TIMEOUT_SECONDS`,
`MAX_RETRIES`, `RETRY_BACKOFF_SECONDS`, `CIRCUIT_FAILURE_THRESHOLD`, `CIRCUIT_OPEN_SECONDS`)
are passed straight to the shared [`bifrost-sdk`](https://github.com/amitmohapatra/bifrost-sdk)
client, which owns the transport, the retries, the rate-limit parsing and the breaker — the
same client the agent harness uses, so neither service can learn a lesson the other misses.

Available uses: `contextual_extraction`, `ambiguous_extraction`, `ambiguous_worthiness`, `relation_extraction`,
`entity_resolution`, `conflict_adjudication`, `summaries`, `reflection`, `query_expansion`,
`chunk_context`, `grounding_judge`, `briefs`.

Each use has its own gate. Ambiguous/contextual extraction consults the model only for
eligible inputs; assisted brief refresh generates only when evidence or synthesis
configuration changes. Native paths remain available. An explicitly assisted brief fails
refresh if no valid cited model result is available; it does not silently substitute native
text. Model-free operation is a supported mode, not a claim of equal answer accuracy.

### Feedback, team model keys, pagination and webhooks (ADR 0023)

**Feedback.** `POST /v1/feedback` takes the `trellis.contracts.Feedback` record (target kind
`run | answer | memory | tool_call | brief | procedure`, verdict `confirm | reject | correct |
approve | edit`, source `human | judge | interrupt`). Identity fields come from the trusted
headers; a body that disagrees is refused. A memory target must be readable; `reject`,
`correct` and `edit` also need its owner or a tenant admin. The record is stored and
projected asynchronously: on a memory, `confirm`/`approve` reinforce it, `reject` retracts
it, `correct`/`edit` write a corrected memory that supersedes it; the outcome is written
back as `projection`. Read it
back with `GET /v1/feedback/{id}` or list a target's feedback with
`GET /v1/feedback?target_kind=memory&target_id=mem_...`.

```python
record = await ctx.feedback.submit("memory", memory.memory_id, "correct",
                                   correction="The renewal is in March, not May.")
page = await ctx.feedback.page_for("memory", memory.memory_id)   # .items, .next_cursor
```

**Team model keys.** A model call resolves the most specific Bifrost key that exists: the
agent's own, then the workspace's, then the tenant's, then the operator's. Tenant admins set
the team levels with `PUT /v1/workspaces/{id}/model-key` and `PUT /v1/model-key`
(`admin.workspaces.set_model_key(...)`, `admin.set_model_key(...)`); a revoked key at any
level refuses instead of borrowing the next one.

**Pagination.** Every list route takes `cursor` and `limit` and answers
`Link: <...>; rel="next"` when a next page exists (envelope bodies also carry
`next_cursor`). In the SDK every `list()` returns one page as a list and its `page()`
sibling returns `items` with `next_cursor`: `await ctx.memories_page()`,
`async for m in ctx.iter_memories(): ...`, `await admin.keys.page(cursor=...)`.

**Webhooks.** Tenant admins subscribe a public https URL to `memory.created`,
`memory.superseded` and `memory.retracted` (the last two from the feedback projector),
`feedback.received`, `feedback.projected` (and `webhook.test`) with `POST /v1/webhooks`;
the signing secret is shown once and a delivery carries ids and verdicts, never content. Deliveries are
signed (`X-Trellis-Signature: t=<unix>,v1=<hmac-sha256>`), retried with backoff, and listed
under `GET /v1/webhooks/{id}/deliveries`. Receivers verify with
`trellis.memory.webhooks.verify_signature(secret, header, raw_body)`.

## Headers, tracing and errors

The SDK sends these; a client without the SDK sends them itself (`docs/openapi.json` names
them on every operation, ADR 0022 explains them):

| Header | Purpose |
|---|---|
| `X-API-Key` (or `Authorization: Bearer`) | The calling service's credential. |
| `X-Trellis-Tenant`, `X-Trellis-Workspace`, `X-Trellis-User` | Who the request acts for. In `api_key` mode the tenant comes from the key. The pre-0.2 spellings `X-Memory-*` are read until 0.3.0; a request carrying both spellings, or either more than once, with different values is refused before the credential is read, so a gateway that stamps these headers must strip both spellings. |
| `traceparent` | W3C Trace Context. Sent when the agent is tracing; the response carries the trace the request ran under, and `X-Trace-ID` repeats its 32-hex trace id. |
| `X-Request-ID` | One id per call, kept across the SDK's retries; echoed when it is an id (a letter or digit, then letters, digits and `._:-`, at most 200 characters), else replaced. |
| `X-Correlation-ID` | An opaque id of yours (a letter or digit, then letters, digits and `._:-`, at most 200 characters), echoed. A `correlation_id` in the body scope wins over the header. `bind(trace_id=...)` with a non-W3C value lands here. |
| `X-Trace-ID` | Response: the 32-hex trace id the request ran under, the same one `traceparent` carries. |
| `Idempotency-Key` | Makes a write safe to retry. The SDK derives one for messages, observations, documents, and deletes of memories and threads; other writes take an explicit `idempotency_key`. |
| `X-Trellis-LLM-Tokens` | Response: LLM tokens the request spent, when it spent any (also sent as `X-Memory-LLM-Tokens` until 0.3.0). |

Every error is an RFC 9457 problem (`application/problem+json`):

```json
{"type": "urn:trellis:problem:scope-denied", "title": "Outside the caller's scope",
 "status": 403, "detail": "thread thr_01J... is outside the caller's scope",
 "instance": "/v1/recall", "code": "SCOPE_DENIED", "retryable": false,
 "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736", "request_id": "req_01J...", "details": {}}
```

The SDK raises one exception class per `code` (`AuthorizationError`, `NotFoundError`,
`RateLimitedError`, ...) with `trace_id` and `request_id` on it, `TimeoutError` or
`DependencyUnavailableError` (status 0) when no response came, and retries `retryable`
errors on reads and idempotent writes; a connection that never opened is retried for any
call, a timeout only when the write is idempotent.

---

## How it works

```
your agent ──► trellis-memory SDK ──► trellis-memory service (FastAPI)
                                              │
                            ┌─────────────────┼──────────────────┐
                         domain            modules            ports
                      (contracts)      (conversation,      (Protocols)
                                        retrieval, memory,      │
                                        graph, tools, ...)      ▼
                                                             adapters
   PostgreSQL · Qdrant · Dragonfly · OpenFGA · Procrastinate · GCS · Docling · Bifrost
```

Hexagonal, and enforced: the core never imports a provider SDK (checked by a test, not a
convention). Swapping Qdrant for something else is an adapter, not a refactor.

**Where data lives.** PostgreSQL is the only source of truth. Qdrant is a rebuildable index
(`make reindex` reconstructs it). Dragonfly is a cache that is never load-bearing. Raw
conversation and files are archived immutably to object storage with checksums.

**How a write is durable.** The payload, its metadata and its processing job commit in one
transaction (a transactional outbox). Only then do you get a `2xx`. A worker that dies
mid-job is detected and its work requeued; replay is safe because every write is idempotent.

**How retrieval works.** Authorized scope → exact lookup → a rules-based router → BM25 +
dense retrieval fused with RRF → bounded cross-encoder rerank → context expansion over the
document graph → evidence verification → abstain if still insufficient.

More: [ARCHITECTURE.md](docs/ARCHITECTURE.md) · design decisions in [docs/adr/](docs/adr/).

---

## Status: read this before you trust a number

This service is **not production-ready**, and the reason is specific.

Everything was built and gated in a sandbox with **no model weights, no real Qdrant /
Dragonfly / OpenFGA servers, and no LLM**. The logic gates are real and passing:

| Gate | Status |
|---|---|
| Acknowledged data loss | **0** over a chaos run with killed workers and outages |
| Unauthorized retrieval | **0** across tenant, user, agent and run boundaries |
| Knowledge-graph fact recall / false facts | **1.00 / 0** |
| False-merge rate | **0.00** |
| Tool memory (suggestion, next-step, plan validity) | **1.00 / 1.00 / 1.00**, zero violations |

But every **retrieval-quality** number was measured with a deterministic hash embedding
standing in for a real model, and every latency number in-process without a network hop.
Those figures bound the service's own logic; they say nothing about production performance.

Making them real means running `make validate` with the real weights in `models/`, the real
servers, and a network hop — everything needed is in the repository and that work is in
progress. Until then, treat retrieval quality and latency as **unmeasured**.

**Also in progress:** the framework-side tool hooks — the tool-memory service, API, SDK and
gate are done; the adapter-side wrappers that call them live in the consuming framework and
are not. The grounding cascade runs with a deterministic lexical
stand-in for the NLI model on this machine, so its verdicts are labelled
`representative: false` until the DeBERTa weights are loaded.
[docs/FINAL_REPORT.md](docs/FINAL_REPORT.md) is the honest per-gate account.

---

## Development

```bash
make lint typecheck        # ruff + pyright
make unit                  # fast, no external services
make integration e2e       # needs the dev stack
make security-test         # isolation gates (release blocking)
make eval                  # retrieval / memory / KG / tool gates
make validate              # everything, then the release-gate evaluator
```

`make validate` fails the build if any hard gate lacks evidence, and flags evidence produced
with stand-in providers as non-representative. Gates are never relaxed to make a build pass.

See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for the workflow and conventions.
