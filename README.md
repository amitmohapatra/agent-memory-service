# trellis-memory

**Durable, scope-aware memory for AI agents.** One service your agents talk to so they
remember what happened, what the user prefers, what the documents say, and which tools
actually worked — with hard guarantees about what is never lost and who may never see what.

Works with any framework (LangGraph today, plain Python anywhere) because the core knows
nothing about your agent library.

```python
from trellis.memory import MemoryClient

# MEMORY_URL and TRELLIS_API_KEY name the service and the key; the local stack is the
# default address, and its development key acts in the tenant "default"
memory = MemoryClient(api_key="dev-key")

ctx = memory.bind(
    user_id="u1",  # tenant_id="acme" names another tenant; an issued key names its own
    # your own id: the service creates the thread on first use (and its session and turns,
    # unless you pass session_id / turn_id of your own)
    thread_id="chat-42",
)

await ctx.history.add([("USER", "I'm in Berlin and I prefer short answers.")])
bundle = await ctx.context("draft a reply about the Q3 numbers")  # everything relevant
answer = await my_agent.run(bundle.rendered)  # your agent, your model
await ctx.history.add([("ASSISTANT", answer)])
```

Next turn — in a different session, a week later — the agent already knows the timezone and
the preference. You did not write retrieval code, a vector store, or a summariser.

That is verified rather than illustrative: those calls were run against a service on
`localhost:8080` with the dev key while this documentation was written — `chat.user`, then
`context` (a 1,313-character rendered bundle over 9 memories and 1 document passage, evidence
status `INSUFFICIENT` for that query), then `chat.assistant`, `remember`, `recall`, `history` and
`memories`. The service was running its stand-in providers (`embedding=hash-embedding`,
`nli` non-representative, `llm=disabled`, per `GET /version`), so that run
says the wire and the scope rules work and says **nothing** about retrieval quality — for quality,
read [`docs/MEASUREMENTS.md`](docs/MEASUREMENTS.md) and
[`docs/FINAL_REPORT.md`](docs/FINAL_REPORT.md).

**Integrating an agent?** [`docs/USAGE.md`](docs/USAGE.md) — *Which API for which scenario* — is the
decision guide: push vs pull, which write and which read fits which job, corrections, feedback
review, model keys and policy, with the SDK call and the gotchas for each.

**On a fresh local service there is nothing to set up for memory**: the development key acts in
the tenant `default` (`MEMORY__AUTHENTICATION__TRUSTED_DEV_TENANT`), which is also what
`GET /v1/keys/self` reports, so a harness started with only `MEMORY_URL` and `TRELLIS_API_KEY`
needs no tenant either. Workspaces are different: to write anything WORKSPACE-visible, onboard
the tenant and the team first — `POST /v1/admin/tenants`, then `POST /v1/workspaces` and a
member. See [`docs/api/tenancy.md`](docs/api/tenancy.md); skipping it is why a first script gets
`Workspace not found`.

---

## Contents

| | |
|---|---|
| [What you get](#what-you-get) | the five kinds of memory, and the guarantees |
| [Three ways an agent uses it](#three-ways-an-agent-uses-it) | push, pull, or your own calls |
| [Install and run](#install-and-run) | 5 minutes to a running service |
| [Use it in one agent](#use-it-in-one-agent) | chat, facts, documents, context bundles |
| [Use it with multiple agents](#use-it-with-multiple-agents) | who sees what, hand-offs, sharing |
| [Tool memory](#tool-memory) | stop calling the wrong tool twice |
| [What it learns](#what-it-learns) | feedback, procedures, prefetch, summaries |
| [Any language](#any-language) | language on write, the model where the rules cannot read |
| [Grounding](#grounding-did-the-answer-actually-follow-from-the-evidence) | verify an answer against its evidence |
| [Using it from a framework](#using-it-from-a-framework) | the harness, or plain HTTP |
| [Configuration](#configuration) | the knobs that matter |
| [How it works](#how-it-works) | architecture in one screen |
| [Status](#status-read-this-before-you-trust-a-number) | what is proven and what is not |
| [**Which API for which scenario**](docs/USAGE.md) | the integration guide: which endpoint and SDK call fits each job |
| [**The API, area by area**](docs/api/README.md) | every route, a diagram each, and the SDK call that makes it |
| [The decision records](docs/adr/README.md) | every ADR, one line and a status each |

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

The API is on **http://localhost:8080** — interactive docs at `/docs`, liveness at
`/health/live` and readiness at `/health/ready`. The dev API key is `dev-key`; it acts in the
tenant `default` unless a request names another (`X-Trellis-Tenant`, or `bind(tenant_id=...)`).

`/health/ready` is the one to wire to a load balancer: `200` when every mandatory store answered
(`ready`) *or* only an optional provider is down (`degraded`), and **`503`** when a mandatory one
is (`not_ready`). Mandatory: `postgres`, `task_queue`, `qdrant`, `blob`, `openfga`. Optional:
`cache`, `llm`. `/version` says which provider is actually *running* per port and lists anything
that fell back under `degraded`. All of it, with the response bodies:
[`docs/api/admin.md`](docs/api/admin.md).

```bash
pip install -e sdk/python        # the trellis-memory SDK
```

### What actually runs

`make models` fetches the frozen set into `models/`, each in a directory named after the
model. The set is `FROZEN_MODELS` in `src/memory_service/config/constants.py`, and the
download catalogue is derived from it, so the code and the weights cannot drift apart:

| Role | Frozen (`config/constants.py::FROZEN_MODELS`) | License | Why |
|---|---|---|---|
| Embedding, English | `ibm-granite/granite-embedding-small-english-r2` (384-dim) | Apache-2.0 | lowest query p95 of every candidate benchmarked, at the smallest useful dimension; encodes Latin-script queries |
| Embedding, multilingual | `hotchpotch/bekko-embedding-v1-a8m` (384-dim) | MIT | every script, every query (ADR 0024) |
| Late interaction (ColBERT) | `mixedbread-ai/mxbai-edge-colbert-v0-32m` (64-dim per token) | Apache-2.0 | the late-interaction arms over two keys in the memory ranking (ADR 0025, 0026) |
| Sparse | BM25 (client term frequencies, Qdrant server-side IDF) | — | no weights |
| Reranker | none (removed) | — | `cross-encoder/ms-marco-MiniLM-L6-v2` measured significantly *worse* on SciFact (nDCG 79.3 vs 84.5, p = 0.012) at 21x the latency, and no reranker beat the fused order on LoCoMo; the offline scorer lives in `benchmark/cross_encoder.py` |
| Grounding NLI | `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` | MIT | multilingual claim-support classifier for `/v1/verify` |

The benchmark challengers (Granite R2 base, Granite reranker, SPLADE, GLiNER2, the former
reranker) are listed in `benchmark/challengers.txt`; `make models-all` fetches them, and only
`make bench-embedding` / `make bench-rerank-offline` ever load one. **No Chinese-origin model or
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

## Three ways an agent uses it

| mode | who decides what the model sees | the calls |
|---|---|---|
| **auto** (push) | the service, every turn | `ctx.context(task, tools=...)` → `rendered` goes into the prompt: the pinned profile, the thread's durable summary and the messages after it, the procedure learned for the task, tool hints, and the memories, passages and graph facts that clear the relevance floor, within `token_budget`. The agent harness does this for you with `memory="read_write"`. |
| **react** (pull) | the agent, mid-run | `ctx.agent_tools()` lists six tools (`memory_search` — `kinds` includes `message` for the transcript —, `memory_remember`, `memory_update`, `memory_forget`, `profile_edit`, `tool_search`) with JSON schemas; `ctx.call_agent_tool(name, args)` runs one. Every call is logged as a pull, and what the agent keeps pulling for a kind of request is what the push starts including (prefetch learning). |
| **manual** | your code | the verbs: `remember`, `update`, `forget`, `search`, `history`, `feedback`, `record_tool`, `tool_hints`, `agent_tools`, `call_agent_tool`, `profile`; everything else (documents, graph, admin, model keys) under `ctx.advanced`. |

The three mix: a harness pushes context and also hands the agent the pull tools.

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
    session_id=request.session_id,  # optional: a continuous stretch of it
    turn_id=request.turn_id,  # optional: this exchange
)
```

`tenant_id` is the wall nothing crosses; everything else narrows visibility further.

Thread, session and turn are **your** identifiers — pass the ones your app already has, and
the service creates them on first use. `history.*` needs a thread (or a run id, whose thread it names); without a session a message
joins the thread's own session, and without a turn a user message opens the thread's next turn
and a reply joins it. For `remember`, `search` and `context` a tenant is enough. To title a thread before its
first message, `await ctx.history.update(title=...)`.

### Record the conversation

```python
await ctx.history.add(
    [
        ("USER", "Revenue was EUR 412 million in FY26."),
        ("ASSISTANT", "Noted — that's up 4% year on year."),
    ]
)
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
ordered, with nothing shown twice. Ask for `format="full"` for the same content as structured
data (no `rendered` copy of it), to build your own prompt:

```python
bundle = await ctx.context("how did revenue develop?", format="full")
bundle.conversation, bundle.thread_summary  # recent messages and the rolling summary
bundle.memories  # durable facts and preferences: id, text, relevance 0..1, date, sources
bundle.knowledge  # document passages: id, text, relevance, document, page, section
bundle.graph_facts  # entity relations: subject, predicate, object, relevance
bundle.evidence_status, bundle.missing_evidence  # COMPLETE / INCOMPLETE / INSUFFICIENT
```

Need just the search results? `await ctx.search("...")` returns ranked items.

A read consults the model only when the tenant's model policy allows it (`read_assist`, on by
default, and only once a key can pay); a request cannot override the policy. Agent-owned virtual keys are exposed through `ctx.advanced.model_keys.set(...)`; a
standing question is a profile block with a `source_query`;
see the [SDK examples](sdk/python/README.md) and
[capability/validation handoff](docs/history/AGENT-CAPABILITIES-HANDOFF-20260927.md).

### State a fact directly

When your app knows something rather than inferring it from chat:

```python
fact = await ctx.remember("Prefers metric units", memory_type="PREFERENCE")
await ctx.update(fact.memory_id, "Prefers imperial units", reason="user corrected it")
await ctx.history.add([("EVENT", "User cancelled the Pro plan")])
```

`remember` stores what you assert verbatim, as one memory, before it returns (`POST
/v1/memories`; the same content in the same scope is the same memory). `update` replaces it
with a new version and closes the old one, which stays readable in a temporal view. An `EVENT`
message hands the service raw evidence and lets it decide, asynchronously, what is worth keeping.

### Add documents

```python
doc = await ctx.advanced.documents.add(open("fy26.pdf", "rb"), title="FY26 annual report")
await ctx.advanced.documents.wait_ready(doc.document_id)
```

The file is parsed into sections, tables and footnotes with page numbers preserved. After
that it answers questions through the same `ctx.context(...)` call — passages come back with
the document and page they came from, so you can cite them.

### Require evidence

For answers that must be grounded, check the evidence status the context comes with:

```python
bundle = await ctx.context("what were FY26 restructuring savings?")
if bundle.evidence_status == "INSUFFICIENT":
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
acme = await platform.admin.create_tenant("Acme", tenant_id="acme")  # admin key, shown once

admin = MemoryClient(url, api_key=acme.admin_key.token)
await admin.tenant.workspaces.create("Finance", workspace_id="finance")  # a team
await admin.tenant.workspaces.set_member("finance", "user:u1")
await admin.tenant.workspaces.set_member("finance", "user:u2")
service = await admin.tenant.keys.issue("service", "finance-harness")  # what the harness holds

harness = MemoryClient(url, api_key=service.token)  # no tenant_id anywhere
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
- **Removal is immediate.** Removing a member (or a key) takes
  effect on the next request, not when a cache expires. The author of a memory keeps it.
  Suspending a tenant (`PATCH /v1/admin/tenants/{id}`) stops every one of its keys the same
  way, with 403 rather than 401; resuming restores them. Deleting a workspace revokes the
  keys bound to it.
- **Secrets are shown once, and retries are safe.** Onboarding and key issuance honour
  `Idempotency-Key` (`idempotency_key=` in the SDK): a retried request returns the same record
  with `Idempotent-Replayed: true` and `token: null`, never a second copy of the secret.
  Without it, every call is a new key. Deleted workspace ids are never reused
  (409), so audit entries keep their meaning.
- **Retention, quota, audit.** Per tenant: `retention_days` (a daily sweep forgets
  canonical memories through the same soft delete a `DELETE` uses), `rate_limit_per_minute`
  (429 with `Retry-After`), and `GET /v1/reads` — who read which records, under which scope.

Roles are `admin` (manages one tenant's keys, workspaces, model keys and policy, and the
feedback review queue; always tenant-wide)
and `service` (acts for its users and agents; may be pinned to one workspace). The platform
key onboards tenants and never reads or writes memory, but it administers every tenant —
it can issue a tenant's keys — so it is root: at least 32 characters in deployed
environments, and unset once onboarding is done. ADR 0021 has the design.

## Tool memory

Agents call tools. Tool memory keeps a catalog of what each tool is and does, records what
your agent called, learns what worked, and answers "which tool, which plan, which arguments"
in one call.

**The service never executes your tools.** It records what your agent did, and learns from
runs you label successful.

### Tell it what your tools are

```python
await ctx.advanced.tools.put_catalog(
    [
        {
            "name": "pricing-lookup_price",
            "description": "List price for a SKU in a region",
            "side_effects": "read",
        },
        {
            "name": "crm-update_quote",
            "description": "Write a price onto a quote",
            "side_effects": "write",
            "argument_entity_types": {"customer": "ORG"},
        },
    ]
)
```

`side_effects` (read, write, irreversible) is what a harness decides approval and code mode
by; `argument_entity_types` lets hints fill an argument from the knowledge graph.

### Record what your agent did

```python
await ctx.record_tool(
    "pricing-lookup_price",
    args={"sku": "SKU-22", "region": "EMEA"},
    output={"price": 1200, "currency": "EUR", "quote_id": "Q-1183"},
    task="update quote Q-1183 with EMEA price for SKU-22",
    latency_ms=42,
)
# only successful runs teach a procedure
await ctx.feedback("run", ctx.scope.agent_run_id, "confirm", source="system")
```

Recording is idempotent on (run, step, tool, arguments), so retries never double-count.
`task` is the **question**, not an identifier: it is normalised into a typed-placeholder
pattern (`update quote {id} with {entity} price for {id}`), so two phrasings of the same request
learn together.

### Ask what to call

```python
hints = await ctx.tool_hints(task, available=["pricing-lookup_price", "crm-update_quote"])
# hints.tools   best first, each with confidence (0..1), success_rate, the arguments
#               found (run, graph, profile or task) and the required ones missing
# hints.next    the plan's next step for this run, else the best tool
# hints.plan    the learned procedure: its steps (tool names), success_rate, runs
```

A background job stores one procedure per task pattern once at least two labelled runs
support it; with the tenant's model it also writes a title and a strategy from what
succeeded and what failed. `ctx.context(query, tools=[...])` carries the same
hints inline. Approvals and rejections of tool calls (feedback) become suggested approval
rules (`ctx.advanced.tools.approval_suggestions()`) that nothing applies automatically. See
[docs/api/tools.md](docs/api/tools.md).

---

## What it learns

Background jobs, never on a request's path, each with a deterministic version that runs
without a model and a model version that runs when the tenant's key and policy allow it
([`docs/LLM-USES.md`](docs/LLM-USES.md)):

- **Feedback.** Verdicts on answers, memories, tool calls and runs adjust the confidence of the
  memories an answer cited, label run outcomes and feed tool statistics; a memory's standing
  moves its ranking within a bounded ±15%. A vote from a person or an agent counts only once
  the tenant admin approves it in the review queue (ADR 0028).
- **Procedures.** Tool runs with outcomes are mined into one stored procedure per task pattern,
  admitted at ≥ 2 supporting runs and ≥ 60% success, updated by delta; `tool_hints` and the
  push offer it as a plan, the next step's arguments filled from the earlier steps' outputs.
- **Approval suggestions.** Approve / reject / edit decisions per tool and argument shape become
  suggested rules after 5 decisions (`GET /v1/tools/approval-suggestions`); never applied by
  the service.
- **Prefetch.** A memory the agent pulled for a kind of request at least 3 times and used at
  least half the time is included in the next push for that kind of request (at most 5).
- **Thread summaries and the profile.** Every 20 messages a thread's durable summary is rolled
  forward (so the summary plus the 20-message window always cover the thread); the `user`
  profile block is kept from USER and PREFERENCE memories.
- **Reflection and connections.** Periodic, cited insights over a principal's memories and typed
  links between memories (supersedes / contradicts / relates).

## Any language

Every observation, memory and chunk carries its language (`lang`, detected without a model).
The extraction rules and the query router's cue patterns are English; a message in another
language is kept verbatim (retrievable through the multilingual dense space every query
searches) and, when a key can pay, read by the model into typed facts in its own language,
with the knowledge graph's entities and relations read the same way. A question in another
language is never routed by English cues. Tested with Japanese, Hindi, German, Spanish and
Chinese fixtures; see [`docs/MULTILINGUAL-RUNTIME.md`](docs/MULTILINGUAL-RUNTIME.md) for the
encoders.

---

## Grounding: did the answer actually follow from the evidence?

```python
report = await ctx.verify(answer, bundle_id=bundle.bundle_id)
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
your agent thinks and `history.add([("ASSISTANT", answer)])` after, and everything in this README works from
LangGraph, CrewAI, Google ADK, an MCP server or plain code.

There is no LangGraph adapter in this repository, and that is the design. A memory service
that ships adapters knows the names of its consumers — the dependency points the wrong way,
and the service image ends up carrying framework packages it never imports. Framework
adapters belong in the layer that drives the framework: in this platform that is
[`agent-harness`](https://github.com/amitmohapatra/agent-harness) (`trellis-harness`), which
attaches to a LangGraph graph, an OpenAI Agents agent, Claude Agent SDK options or a plain
callable, pushes `/v1/context` into the run and adds the pull tools (`memory="read_write"`).

### Plain HTTP

Every capability is a documented REST route (`/docs`, or `docs/openapi.json`). The SDK is a
convenience, not a requirement.

---

## Configuration

Everything deployment-specific is an environment variable; copy `.env.example` to `.env`.
The ones that actually matter:

```bash
# Backing services
MEMORY__DATABASE__URL=postgresql+psycopg://memory:memory@localhost:5432/memory
MEMORY__SEARCH__QDRANT_URL=http://localhost:6333
MEMORY__CACHE__URL=redis://localhost:6379/0
MEMORY__AUTHORIZATION__OPENFGA_API_URL=http://localhost:8081

# The model gateway and the operator's key on it (the platform's names, unprefixed).
# Unset BIFROST_URL: no model call is ever made.
BIFROST_URL=https://<your-gateway>/v1
BIFROST_VIRTUAL_KEY=            # optional: pays for tenants without a key of their own

# Envelope keys that encrypt the agent and tenant model keys registered through the API
# (required in staging and prod; dev and test derive an unprotected development key)
# MEMORY__AGENT_CREDENTIALS__ACTIVE_KEY_ID=v1
# MEMORY__AGENT_CREDENTIALS__ENCRYPTION_KEYS={"v1":"<base64-encoded-32-byte-key>"}

# The models are not settings: `make models` puts the frozen set in ./models (git-ignored)
# and the service finds it there, or under /models in the image.
```

Only deployment facts are settings (`src/memory_service/config/settings.py`). Models, budgets,
timeouts, retries and thresholds are constants in `config/constants.py` — a change there is a
code change, reviewed like one. There are **no** `MEMORY__MODELS__*` variables.

### About the LLM

The service runs without one, and a model is used only where a key can pay for it and the
tenant's policy allows it:

* **Who pays.** The acting agent's registered virtual key (`PUT /v1/agents/model-key`), else
  its tenant's (`PUT /v1/model-key`), else — only while neither level has a row — the
  operator's `BIFROST_VIRTUAL_KEY`. A revoked key refuses instead of borrowing the next one.
* **What it may be spent on.** The tenant's model policy (`PUT /v1/model-key/policy`, the
  tenant admin key) has exactly three fields: `uses` (which uses may run), `read_assist`
  (whether reads consult the model; a request cannot override it) and `models` (the gateway
  model per use). With no policy row the default is every use **except** the opt-in
  `memory_restatement`, reads assisted, the service's model per use.
* **Which model.** By default `LLMTuning.model` / `LLMTuning.fast_model` (constants, both
  `auto`): the service queries the gateway's authenticated model catalogue and selects a
  recognised eligible text model; opaque aliases are not guessed. A tenant pins a model per
  use with its policy's `models` map.
* **What it cost.** `GET /v1/model-key/usage` (tokens and calls per day and use) and
  `memory_llm_tokens_total{tenant,use,direction}`; a request reports its own spend in
  `X-Trellis-LLM-Tokens`.

See [docs/api/tenancy.md](docs/api/tenancy.md) and [docs/LLM-USES.md](docs/LLM-USES.md).

Native model calls go through [Bifrost](https://github.com/maximhq/bifrost), an external
gateway you run yourself. The operator key comes from deployment secrets; agent- and
tenant-owned virtual keys are encrypted in the native database. Provider keys stay in Bifrost.
Memory-service calls exclude MCP clients/tools.
The pinned Hindsight SDK (the optional `[hindsight]` extra) provides extraction preview for
eligible non-agent ingestion;
that server owns its model configuration. Agent extraction stays on the Bifrost path because
the SDK cannot carry a per-request model virtual key. Source storage and authorization stay
native. See the [integration boundary](docs/history/HINDSIGHT-CAPABILITY-STATUS-20260927.md).

There is deliberately **no gateway service in `docker-compose.yml`**. Starting one from this
repository's own compose file would put provider keys inside the application's deployment,
which is the coupling the gateway exists to remove. `deploy/bifrost/` holds an example config
and nothing that runs.

**Sizing `max_tokens` for a reasoning model.** The output budget is spent on reasoning before
any text is produced, so a budget that looks generous can return an empty answer. Measured
against `gemini-3.6-flash`: answering "Reply with exactly: OK" consumed 57 reasoning tokens,
so `max_tokens=16` produced no content at all. The adapter now raises instead of handing back
an empty string, and names the cause. The ceiling is the constant `LLMTuning.max_tokens`
(1024).

**Rate limits.** A `429` is retried against the delay the gateway asks for rather than the
exponential backoff, because a per-minute quota is not something a 1.5-second retry schedule
can wait out. The delay is read from the `Retry-After` header *and* from the response body,
because some providers only put it there — Gemini answers "Please retry in 59.18s" in prose.
A `429` also never opens the circuit breaker: backpressure is the gateway working, and
counting it turns "slow down" into "stop".

**How much of this is configuration.** Two variables: where the gateway is (`BIFROST_URL`)
and the operator's key on it (`BIFROST_VIRTUAL_KEY`). Which uses run and which model each
calls is the tenant's policy, through the API. The transport tuning — `max_tokens`,
`timeout_seconds`, `max_retries` (`constants.LLM`) and `retry_backoff_seconds`,
`circuit_failure_threshold`, `circuit_open_seconds` (`constants.LLM_TRANSPORT`) — is passed
straight to the shared [`bifrost-sdk`](https://github.com/amitmohapatra/bifrost-sdk) client,
which owns the transport, the retries, the rate-limit parsing and the breaker — the same client
the agent harness uses, so neither service can learn a lesson the other misses.

Available uses: `contextual_extraction`, `relation_extraction`,
`entity_resolution`, `conflict_adjudication`, `summaries`, `reflection`, `memory_connections`,
`query_expansion`, `chunk_context`, `memory_restatement` (opt-in, ADR 0027),
`grounding_judge`, `procedure_abstraction`
(each described in [`docs/LLM-USES.md`](docs/LLM-USES.md): when it runs, its tier, its
fallback).

Each use has its own gate. Contextual extraction consults the model only for
eligible inputs. Native paths remain available. Model-free operation is a supported mode, not a claim of equal answer accuracy.

### Feedback, model keys and pagination (ADR 0023, ADR 0028)

**Feedback.** `POST /v1/feedback` takes the `trellis.contracts.Feedback` record (target kind
`run | memory | tool_call | procedure`, verdict `confirm | reject | correct |
approve | edit`, source `human | judge | interrupt | system`). Identity fields come from the
trusted headers; a body that disagrees is refused. A memory target must be readable; `reject`,
`correct` and `edit` also need its owner or a tenant admin. The record is stored and
projected asynchronously: on a memory, `confirm`/`approve` reinforce it, `reject` retracts
it, `correct`/`edit` write a corrected memory that supersedes it; the outcome is written
back as `projection`. Read it
back with `GET /v1/feedback/{id}` or list a target's feedback with
`GET /v1/feedback?target_kind=memory&target_id=mem_...`.

**A vote waits for review (ADR 0028).** A verdict that would change what was learned on a
person's or an agent's word alone — a `confirm`/`approve` of a memory, a verdict on a run or a
procedure — is stored with `review.state = "pending"` and changes nothing until the tenant's
admin key approves it. Applied as they arrive: the service's own grounding judge
(`/v1/verify`), a run reporting its own status (`source="system"`, the target is the calling
`agent_run_id`, citing no memories), an owner's `reject`/`correct`/`edit` of their own memory,
any `tool_call` verdict, and anything sent with the tenant admin key or by a tenant admin user
in person. A client that sends `source="judge"` is not the judge and waits like anyone else.

```python
record = await ctx.feedback(
    "memory", memory.memory_id, "correct", correction="The renewal is in March, not May."
)  # the owner's correction: applied as it arrives (record.review is None)
page = await ctx.feedback.page_for("memory", memory.memory_id)  # .items, .next_cursor

admin = MemoryClient(url, api_key=tenant_admin_key).bind()
for vote in (await admin.feedback.pending()).items:  # GET /v1/feedback/pending
    await admin.feedback.approve(vote.feedback_id, note="checked")  # or .dismiss(...)
```

**Model keys.** A model call resolves the most specific Bifrost key that exists: the
agent's own (`PUT /v1/agents/model-key`), then the tenant's (`PUT /v1/model-key`,
`client.tenant.set_model_key(...)`), then the operator's (`BIFROST_VIRTUAL_KEY`); a
revoked key refuses instead of borrowing the next one. There is no workspace level.

**Pagination.** Every list route takes `cursor` and `limit` and answers
`Link: <...>; rel="next"` when a next page exists (envelope bodies also carry
`next_cursor`). In the SDK every `list()` returns one page as a list and its `page()`
sibling returns `items` with `next_cursor`: `await ctx.advanced.memories.page()`,
`async for m in ctx.advanced.memories.iter(): ...`, `await client.tenant.keys.page(cursor=...)`.

**Notifications.** The memory service sends none: run notifications (paused, escalated,
finished) are tenant webhook subscriptions in agent-runs.

## Headers, tracing and errors

The SDK sends these; a client without the SDK sends them itself (`docs/openapi.json` names
them on every operation, ADR 0022 explains them):

| Header | Purpose |
|---|---|
| `X-API-Key` (or `Authorization: Bearer`) | The calling service's credential. |
| `X-Trellis-Tenant`, `X-Trellis-Workspace`, `X-Trellis-User` | Who the request acts for. In `api_key` mode the tenant comes from the key. A request carrying one of them more than once with different values is refused before the credential is read. |
| `traceparent` | W3C Trace Context. Sent when the agent is tracing; the response carries the trace the request ran under, and `X-Trace-ID` repeats its 32-hex trace id. |
| `X-Request-ID` | One id per call, kept across the SDK's retries; echoed when it is an id (a letter or digit, then letters, digits and `._:-`, at most 200 characters), else replaced. |
| `X-Correlation-ID` | An opaque id of yours (a letter or digit, then letters, digits and `._:-`, at most 200 characters), echoed. A `correlation_id` in the body scope wins over the header. `bind(trace_id=...)` with a non-W3C value lands here. |
| `X-Trace-ID` | Response: the 32-hex trace id the request ran under, the same one `traceparent` carries. |
| `Idempotency-Key` | Makes a write safe to retry. The SDK derives one for messages, observations, documents, and deletes of memories and threads; other writes take an explicit `idempotency_key`. |
| `X-Trellis-LLM-Tokens` | Response: LLM tokens the request spent, when it spent any |

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

**How retrieval works.** Authorized scope → exact lookup → a rules-based router (English cues;
any other language is routed as a general question) → hybrid search, the graph traversal
running underneath in one budgeted statement → context expansion over the document graph →
evidence verification → abstain if still insufficient. The push then packs, within the token
budget, what clears the encoder's relevance floor.

Memories are ranked by a general fusion nothing was fitted to (ADR 0026,
`modules/retrieval/memory_ranking.py`): weighted reciprocal-rank fusion (`K = 10`) of BM25 and
two dense spaces over two keys — the memory alone, and the memory read with the turn it
answers — plus ColBERT late-interaction arms (weight 6 on the memory's own key, 2 on its context
key); then four rules: **session** (each memory gains 0.3 × the best fused score among the top
50 in its session, the day it was observed), **speaker** (a question naming a person lifts that
person's memories), **time** (a "when" question lifts memories that name a time) and
**period** (a question naming a period lifts memories in it). No learned fusion and no reranker:
a cross-encoder rerank was measured and rejected. Document chunks are fused by the store's RRF.

More: [ARCHITECTURE.md](docs/ARCHITECTURE.md) · design decisions in [docs/adr/](docs/adr/).

---

## Status: read this before you trust a number

Pre-1.0: the API changed in place in 0.2.0 and again in 0.3.0 (the aliases removed), and it
has no external users yet.

What is measured, with the real encoders, PostgreSQL and Qdrant, on a 2015 4-core laptop
(no AVX2) shared with other workloads - every number and its caveats is in
[`docs/MEASUREMENTS.md`](docs/MEASUREMENTS.md):

| | |
|---|---|
| `/v1/context` p50 / p95, model off | ~0.3 s / ~0.4-0.6 s on that laptop; the 300 ms p95 target is set for an 8 vCPU VM and has not been measured there |
| context packing | 10 off-topic questions: 30.8 → 1.5 memories and 76 → 23 KB per response with the relevance floor; every evidence memory of 135 LoCoMo questions still packed |
| LoCoMo source recall, SciFact nDCG@10, XQuAD R@10 | [`docs/history/PHASE7-RESULTS-2026-09-28.md`](docs/history/PHASE7-RESULTS-2026-09-28.md), [`docs/history/PHASE9-RESULTS-2026-09-29.md`](docs/history/PHASE9-RESULTS-2026-09-29.md) |
| acknowledged data loss, unauthorized retrieval | 0 and 0 in the failure and security suites |

Not measured: generated-answer accuracy with the current write path, and anything at the
target VM's scale.

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
