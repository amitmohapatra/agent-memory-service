# Memory Service

**Durable, scope-aware memory for AI agents.** One service your agents talk to so they
remember what happened, what the user prefers, what the documents say, and which tools
actually worked — with hard guarantees about what is never lost and who may never see what.

Works with any framework (LangGraph today, plain Python anywhere) because the core knows
nothing about your agent library.

```python
from universal_memory import MemoryClient

memory = MemoryClient("http://localhost:8080", api_key="dev-key")

ctx = memory.bind(
    tenant_id="acme", user_id="u1",
    thread_id=thread_id, session_id=session_id, turn_id=turn_id,
)

await ctx.chat.user("I'm in Berlin and I prefer short answers.")
bundle = await ctx.context("draft a reply about the Q3 numbers")   # everything relevant
answer = await my_agent.run(bundle.rendered)                       # your agent, your model
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

Requires Docker and Python 3.12.

```bash
git clone <repo> memory-service && cd memory-service
make setup          # uv venv + dependencies
make dev-up         # PostgreSQL, Qdrant, Dragonfly, OpenFGA, api, worker
make migrate
```

The API is on **http://localhost:8080** — interactive docs at `/docs`, health at
`/health/ready`. The dev API key is `dev-key`.

```bash
pip install -e sdk/python        # the universal-memory SDK
```

Local model weights (embeddings, reranker) go in `models/` and are read from there; without
them the service still runs, using a deterministic stand-in that is fine for development but
**not** representative of retrieval quality. See [Status](#status-read-this-before-you-trust-a-number).

---

## Use it in one agent

### Bind a scope, once

Every call happens inside a scope. Bind it once per request and forget about it:

```python
from universal_memory import MemoryClient

memory = MemoryClient("http://localhost:8080", api_key="dev-key")

ctx = memory.bind(
    tenant_id=request.tenant_id,     # hard isolation boundary
    workspace_id=request.workspace,  # a project or app
    user_id=request.user_id,         # who this is for
    thread_id=request.thread_id,     # the conversation
    session_id=request.session_id,   # a continuous stretch of it
    turn_id=request.turn_id,         # this exchange
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
bundle.conversation   # recent turns plus a rolling summary
bundle.memories       # durable facts and preferences
bundle.knowledge      # document passages, each with document, page and evidence
bundle.graph_facts    # entity relations
bundle.evidence       # what was found, what was missing, and the status
```

Need just the search results? `await ctx.recall("...")` returns ranked items.

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
doc = await ctx.files.add(open("fy26.pdf", "rb"), title="FY26 annual report")
await ctx.files.wait_ready(doc.document_id)
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
researcher = ctx.agent("researcher")        # a new run, isolated by default
writer     = ctx.agent("writer")            # a sibling run — cannot see the researcher's notes

await researcher.remember("Source A contradicts source B", visibility="RUN")
```

`visibility="RUN"` (the default for agent notes) means: this run, and any run it spawns.
Not the user, not sibling agents, not the next run of the same agent.

### Hand-offs flow down, never up or sideways

```python
child = researcher.agent("fact-checker")     # spawned by the researcher
# child sees the researcher's RUN-scoped notes — that is the hand-off
# writer (a sibling) still sees nothing of either
```

### Sharing is explicit

```python
await researcher.remember(
    "FY26 revenue is EUR 412m, confirmed in two sources",
    memory_type="SHARED",
    visibility="AGENT_GROUP",     # now the whole crew can use it
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
| `WORKSPACE` / `TENANT` | the project / the whole tenant |

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

## Tool memory

Agents forget which tool worked, call the same expensive endpoint twice in one run, and
rediscover the same failure every time. Tool memory fixes that.

**The service never executes your tools.** It records what happened, caches what is safe to
cache, learns the chains that worked, and advises.

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

### Ask what to call

```python
tools = [{"name": "pricing.lookup_price"}, {"name": "crm.update_quote"}]

for s in await ctx.tools.suggest("update quote Q-9 with APAC price for SKU-3",
                                 available_tools=tools):
    print(s.render())
# pricing.lookup_price(sku='SKU-3', region='APAC')  [confidence 0.94]
```

Only tools you declare in `available_tools` are ever suggested — nothing is invented.

### Ask what comes next, mid-task

```python
nxt = await ctx.tools.next(
    task,
    trajectory_so_far=[{"tool": "pricing.lookup_price", "status": "ok",
                        "output_fields": {"quote_id": "Q-9", "price": 1200}}],
    available_tools=tools,
)
nxt.suggestions[0].tool                          # 'crm.update_quote'
nxt.suggestions[0].argument_template["quote_id"] # 'Q-9'  ← bound from the previous output
nxt.stop                                         # True when the chain is finished
```

The service learned that `quote_id` flows from the first tool's output into the second tool's
arguments, by observing it happen — not by being told.

### Get the whole plan up front

```python
plan = await ctx.tools.plan(task, available_tools=tools)
print(plan.render())        # the ordered chain with bindings and known failure modes
plan.render_script()        # Starlark, for Bifrost code mode
```

### Don't call the same thing twice

```python
hit = await ctx.tools.lookup("pricing.lookup_price", {"sku": "SKU-22"})
if hit.cached:
    print(hit.output_fields, f"({hit.age_seconds:.0f}s old)")
```

A cached result is served **only** for a tool registered as deterministic, cacheable and
free of write side effects — and only inside the scope it was cached for. Everything else is
a miss by design, because replaying a call that writes would hide it. Defaults are
conservative: an unregistered tool is never cached. Widen it deliberately:

```python
await ctx.tools.register(
    "pricing.lookup_price",
    policy={"deterministic": True, "cacheable": True,
            "cache_ttl_seconds": 900, "cache_scope": "thread",
            "side_effects": "read", "redact": ["auth.token"]},
)
```

`redact` paths never reach storage.

### The whole loop in one call

```python
result = await ctx.tools.execute(
    ToolCall(tool="pricing.lookup_price", args={"sku": "SKU-22"}, task=task),
    executor=my_tool_runner,        # your function, or a call out to an MCP gateway
)
```

Cache lookup → your executor on a miss → idempotent record. Failures are recorded with their
error class, so the next run gets the correction.

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
report.per_claim_hallucination_rate      # 0.0 when every claim is supported
for claim in report.claims:
    claim.verdict    # supported | unsupported | contradicted | borderline
```

The cascade is deterministic first and expensive last: split the answer into claims, check
citation spans against the evidence, score entailment with a local NLI model, ask the LLM
only about the genuinely borderline band, and scan the retrieved-but-unused passages for
contradictions.

---

## Framework integrations

### LangGraph (available today)

```python
from universal_memory_langgraph import LangGraphMemory

memory = LangGraphMemory(client, tenant_id="acme", user_id="u1", agent_group_id="crew")

graph.add_node("research", memory.wrap(research_node, agent="researcher", recall="question"))
graph.add_node("answer",   memory.wrap(answer_node,   recall="question"))
```

`recall` names the state key to build the context bundle from (or a callable returning the
query). LangGraph's `thread_id` becomes the memory thread.

Wrapping a node gives it an evidence-gated context bundle in `state["memory"]` before it
runs, records its messages and observations after, and makes subgraphs into agent runs with
proper lineage — so the hand-off rules above hold automatically. Recording is idempotent
across checkpoint retries.

### Any other framework

The SDK is framework-neutral: bind a scope, call `context()` before your agent thinks and
`chat.assistant()` after, and everything in this README works. Adapters for Google ADK and
CrewAI, and an MCP server exposing memory as tools, are **in progress** — see
[Status](#status-read-this-before-you-trust-a-number).

### Plain HTTP

Every capability is a documented REST route (`/docs`, or `docs/openapi.json`). The SDK is a
convenience, not a requirement.

---

## Configuration

Everything is environment variables (or `config/memory.yaml`); copy `.env.example` to `.env`.
The ones that actually matter:

```bash
# Backing services
MEMORY__DATABASE__URL=postgresql+psycopg://memory:memory@localhost:5432/memory
MEMORY__SEARCH__QDRANT_URL=http://localhost:6333
MEMORY__CACHE__URL=redis://localhost:6379/0
MEMORY__AUTHORIZATION__OPENFGA_API_URL=http://localhost:8081

# Local CPU models (weights live in ./models, git-ignored)
MEMORY__MODELS__EMBEDDING__MODEL_PATH=./models/granite-embedding-small-english-r2
MEMORY__MODELS__RERANKER__MODEL_PATH=./models/ms-marco-MiniLM-L6-v2

# Optional LLM — off by default, and only ever through a Bifrost gateway
MEMORY__MODELS__LLM__ENABLED=false
```

### About the LLM

The service runs fully without one. When you enable it, **every call goes through
[Bifrost](https://github.com/maximhq/bifrost)**, an external gateway you run yourself. The
service holds only a Bifrost virtual key (in a git-ignored `secrets.env`); your provider keys
stay in Bifrost. No provider SDK is importable anywhere in the codebase — a lint rule and an
architecture test enforce it.

You choose, per capability, where a model is allowed to help:

```bash
MEMORY__MODELS__LLM__ENABLED=true
MEMORY__MODELS__LLM__MODEL=anthropic/claude-sonnet-5              # complex judgement
MEMORY__MODELS__LLM__FAST_MODEL=anthropic/claude-haiku-4-5        # cheap classification
MEMORY__MODELS__LLM__USES=["conflict_adjudication","summaries"]
```

Available uses: `ambiguous_extraction`, `ambiguous_worthiness`, `relation_extraction`,
`entity_resolution`, `conflict_adjudication`, `summaries`, `reflection`, `query_expansion`,
`chunk_context`, `tool_reflection`, `grounding_judge`.

Each one is consulted **only** when the deterministic path signals genuine ambiguity, and any
failure — gateway down, bad output, timeout — falls back to the deterministic result. Turning
them all off changes quality, never correctness.

---

## How it works

```
your agent ──► universal-memory SDK ──► Memory Service (FastAPI)
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

More: [ARCHITECTURE.md](ARCHITECTURE.md) · design decisions in [docs/adr/](docs/adr/).

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

**Also in progress:** Google ADK and CrewAI adapters, the MCP server, and the framework-side
tool hooks (the tool-memory service, API, SDK and gate are done; the LangGraph tool wrapper
and ADK/CrewAI callbacks are not). The grounding cascade runs with a deterministic lexical
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

See [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow and conventions.
