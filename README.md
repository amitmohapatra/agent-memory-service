# trellis-memory

**Durable, scope-aware memory for AI agents.** One service your agents talk to, so they remember
what happened, what the user prefers, what the documents say and which tools worked, with hard
guarantees about what is never lost and who may never see what. It works with any framework,
because the core knows nothing about your agent library. It runs without an LLM; a model, through
a gateway you run, only sharpens specific decisions.

This repository holds the service and its Python SDK, `trellis-memory` (`trellis.memory`, in
[`sdk/python`](sdk/python/README.md)). It is one of the five Trellis repositories
([where it fits](#where-this-fits-two-ways-to-use-trellis)).

## Start here

| You want to | Do this |
|---|---|
| see it work in a minute, with no server | `make setup` then `make examples`: [eleven examples](examples/README.md), offline, against an in-process service (they need only PostgreSQL) |
| run the service | `docker compose up -d` ([Install and run](#install-and-run)) |
| give your agent memory | the [quickstart](#quickstart) below, then [Which API for which scenario](docs/USAGE.md) |
| understand how it works | [Architecture](docs/ARCHITECTURE.md) and the [key flows](docs/flows.md), then [the guide](docs/README.md#the-guide) |
| deploy and configure it | [Configuration reference](docs/configuration.md), [chapter 11 · Operations](docs/guide/11-operations.md) |
| fix something | [Troubleshooting and FAQ](docs/troubleshooting.md) |

## Quickstart

```python
# example: examples/01_quickstart_context.py
from trellis.memory import MemoryClient

# MEMORY_URL and TRELLIS_API_KEY name the service and the key. The local stack is the default
# address, and its development key acts in the tenant "default".
memory = MemoryClient(api_key="dev-key")

# your own id: the service creates the thread on first use
ctx = memory.bind(user_id="u1", thread_id="chat-42")

await ctx.history.add([("USER", "I'm in Berlin and I prefer short answers.")])
bundle = await ctx.context("draft a reply about the Q3 numbers")  # everything relevant
answer = await my_agent.run(bundle.rendered)  # your agent, your model
await ctx.history.add([("ASSISTANT", answer)])
```

Next turn, in a different session a week later, the agent already knows the city and the
preference. You wrote no retrieval code, no vector store and no summariser.
`bundle.evidence_status` says whether the service found what an answer needs (`COMPLETE`,
`INCOMPLETE` or `INSUFFICIENT`); nothing raises on `INSUFFICIENT`, so check it and say you do
not know. [`examples/01_quickstart_context.py`](examples/01_quickstart_context.py) is this
snippet, run.

## Install and run

Requires Docker (and Python 3.12 with [uv](https://docs.astral.sh/uv/) to work on the service).

```bash
docker compose up -d          # PostgreSQL, Qdrant, Dragonfly, OpenFGA, the API and the worker
pip install -e sdk/python     # the trellis-memory SDK
```

The first start downloads the model weights into `./models` (about 1 GB, once, git-ignored)
and applies both database schemas before the API and the worker start, so a fresh clone
reaches a working service with no further steps. The API is on **http://localhost:8080**:
interactive docs at `/docs`, liveness at `/health/live`, readiness at `/health/ready` (the one
to wire to a load balancer). The development key is `dev-key`; it acts in the tenant `default`
unless a request names another (`bind(tenant_id=...)`). There is deliberately no model gateway
in the stack: without `BIFROST_URL` the service runs complete, with no model call
([chapter 8](docs/guide/08-models.md)).

To work on the service rather than run it:

```bash
make setup        # uv venv + every extra + dev tools (also regenerates vendor/bifrost-sdk)
make models       # the frozen model set, without Docker
make dev-up       # the stack
make migrate      # both schemas
```

Before you write a WORKSPACE-visible memory, onboard the tenant and the team: a first script
that skips it gets `Workspace not found` ([troubleshooting](docs/troubleshooting.md)).

## Examples

Numbered from simple to complex. Each runs **offline** (an in-process service, the test
suite's stand-ins, no network, no model; PostgreSQL is the one dependency) and asserts what it
shows; `make examples` runs them all and CI runs `make examples` on every push.

| # | Example | Shows |
|---|---|---|
| 01 | [quickstart_context](examples/01_quickstart_context.py) | bind a scope, record a turn, get the context; an agent run and its outcome |
| 02 | [conversation_history](examples/02_conversation_history.py) | threads and turns, internal messages, events, idempotent replays, isolation |
| 03 | [remember_search_forget](examples/03_remember_search_forget.py) | learned and stated memories, supersede, recall, forget |
| 04 | [documents](examples/04_documents.py) | upload, wait until ready, cited passages, evidence status |
| 05 | [agents_runs_and_sharing](examples/05_agents_runs_and_sharing.py) | run-private memory, hand-offs to child runs, crew sharing |
| 06 | [verify_grounding](examples/06_verify_grounding.py) | claim-by-claim verification of an answer, recorded as the judge's verdict |
| 07 | [tool_memory_and_hints](examples/07_tool_memory_and_hints.py) | the catalog, recorded calls, a learned plan, arguments bound from earlier steps |
| 08 | [feedback_and_review](examples/08_feedback_and_review.py) | votes that wait for the tenant admin, what applies at once |
| 09 | [admin_onboarding_and_keys](examples/09_admin_onboarding_and_keys.py) | onboard a tenant, scoped keys, a workspace, revocation |
| 10 | [background_jobs_and_summary](examples/10_background_jobs_and_summary.py) | the jobs a write queues, and the rolling thread summary |
| 99 | [full_tour](examples/99_full_tour.py) | every SDK method and every route, as one checklist |

## Where this fits: two ways to use Trellis

Trellis is five repositories, each owning one concern: [agent-harness](https://github.com/amitmohapatra/agent-harness)
runs agents of any framework, **this service** is their memory,
[agent-runs](https://github.com/amitmohapatra/agent-runs) holds durable runs, the inbox and
schedules, [agent-contracts](https://github.com/amitmohapatra/agent-contracts) the shared record
types, and [bifrost-sdk](https://github.com/amitmohapatra/bifrost-sdk) the client for the model
gateway. [ARCHITECTURE.md](docs/ARCHITECTURE.md#in-the-platform-the-five-trellis-repositories)
draws them and says what each owns; [versioning.md](docs/versioning.md) says which versions
work together.

Each block works both ways:

- **Way 1, wrapped.** `h = Harness(); agent = h.wrap(my_agent)`. With `MEMORY_URL` and
  `TRELLIS_API_KEY` set, the harness binds each run's scope and makes every call below: it
  pushes `context()` into the framework's input, adds the six pull tools, records the transcript
  and every tool call, sends the run's outcome and each approval as feedback, and verifies a
  sample of answers. You write no memory code ([USAGE §14](docs/USAGE.md#14-through-the-harness)).
- **Way 2, pluggable.** Your framework runs the agent, untouched, and you call the SDK: read
  the context into the prompt, record the turn, send feedback when you know how the run went.

```python
# example: examples/01_quickstart_context.py
from trellis.memory import MemoryClient

memory = MemoryClient()  # MEMORY_URL and TRELLIS_API_KEY from the environment
run = memory.bind(user_id="u1", thread_id="thr_1").agent("support")

pushed = await run.context(question, window=False)  # window=False: your framework keeps the history
answer = await my_agent.run(system=pushed.rendered, user=question)
await run.history.add([("USER", question), ("ASSISTANT", answer)])
await run.feedback("run", run.scope.agent_run_id, "confirm")  # or "reject"
```

Choose Way 1 for memory on every run with nothing to write, including the parts that are easy
to forget. Choose Way 2 when your framework owns the prompt and the loop, when the caller is
not an agent (an import job, a profile page), or when you want only a part: the push, the pull
tools (`agent_tools()`, `call_agent_tool`) or the tool hints.

## What you get

| Kind of memory | What it holds | Example question |
|---|---|---|
| Conversation | threads, sessions, turns, messages, archived and replayable, with a rolling summary | "what did we decide last Tuesday?" |
| Semantic | facts and preferences learned from what was said, with validity windows | "prefers concise answers" |
| Document | uploads parsed into a hierarchy with page-level provenance | "what does the FY26 report say about EBITDA?" |
| Knowledge graph | typed entities and relations, current and as of a date | "who approved the acquisition, and when?" |
| Tool | which tool worked for which task, in what order, with which arguments | "which API do I call to reprice a quote?" |

The guarantees it is built around: **acknowledged data is never lost** (a `2xx` comes back only
after the record and its processing job are committed in one transaction); **nothing leaks
across a boundary** (tenant, user, agent and run isolation is a store-side filter applied
before search); **answers cite their evidence** and the service says `INSUFFICIENT` rather
than guess; **it runs without an LLM**. Every language is stored and retrieved; the extraction
rules are English and a model reads the rest when a key can pay
([MULTILINGUAL-RUNTIME.md](docs/MULTILINGUAL-RUNTIME.md)).

## Documentation

| | |
|---|---|
| [docs/README.md](docs/README.md) | the index: three ways in, by why you are here |
| [Which API for which scenario](docs/USAGE.md) | the integration guide: which call fits each job, with the gotchas |
| [Architecture](docs/ARCHITECTURE.md) · [Key flows](docs/flows.md) | the high-level design, and a sequence diagram per flow |
| [API reference](docs/api/README.md) · [`openapi.json`](docs/openapi.json) | every route and its SDK call, one page per area; the generated contract (also `/docs` on a running service) |
| [Configuration](docs/configuration.md) | every setting, its default, an example, and what is automatic |
| [Troubleshooting and FAQ](docs/troubleshooting.md) | the errors a first integration meets, and what to do |
| [Versioning and compatibility](docs/versioning.md) · [CHANGELOG](CHANGELOG.md) | what is stable, what changed, and which versions work together |
| [The guide](docs/README.md#the-guide) | twelve chapters, from why a memory service to its release gates |
| [Decision records](docs/adr/README.md) | every ADR, one line and a status each |

## Status: read this before you trust a number

Pre-1.0: the API changed in place in 0.2.0 and again in 0.3.0 (the one-release aliases
removed), and it has no external users yet ([versioning.md](docs/versioning.md)). What is
measured, with the real encoders, PostgreSQL and Qdrant, on a 4-core laptop shared with other
work, is in [`docs/MEASUREMENTS.md`](docs/MEASUREMENTS.md) with every caveat:

| | |
|---|---|
| `/v1/context` p50 / p95, model off | about 0.3 s / 0.4-0.6 s on that laptop; the 300 ms p95 target is set for an 8 vCPU VM and has not been measured there |
| context packing | 10 off-topic questions: 30.8 → 1.5 memories and 76 → 23 KB per response with the relevance floor; every evidence memory of 135 LoCoMo questions still packed |
| LoCoMo source recall, SciFact nDCG@10, XQuAD R@10 | [PHASE7](benchmark/reports/PHASE7-RESULTS-2026-09-28.md), [PHASE9](benchmark/reports/PHASE9-RESULTS-2026-09-29.md) and [PHASE11](benchmark/reports/PHASE11-RESULTS-2026-10-01.md) results |
| acknowledged data loss, unauthorized retrieval | 0 and 0 in the failure and security suites |

Not measured: generated-answer accuracy with the current write path, and anything at the
target VM's scale. The offline examples and the test suite run on deterministic stand-ins
(a hash embedding, a lexical NLI): they prove the wire and the scope rules, never retrieval
quality.

## Development

```bash
make lint typecheck   # ruff, ruff format --check, pyright
make unit             # fast, no external services
make integration e2e  # need PostgreSQL and Redis (make dev-up)
make examples         # every example, offline
make docs-check       # every link and anchor, every snippet in the docs
make validate         # everything, then the release-gate evaluator
```

[CONTRIBUTING.md](docs/CONTRIBUTING.md) has the workflow and the rules that do not bend;
[chapter 12](docs/guide/12-testing-and-gates.md) what CI runs and why.
