# Documentation

Three ways in. Pick the one that matches why you are here.

### I want to use it

Start with the [README](../README.md): what the service is, a working agent in a dozen lines,
how to run it, and the [examples](../examples/README.md), which run offline in a minute. Then
[**Which API for which scenario**](USAGE.md), the decision guide for an engineer integrating an
agent: which endpoint and SDK call fits each job, with the gotchas. Keep the
[API reference](api/README.md), the [configuration reference](configuration.md) and
[troubleshooting](troubleshooting.md) at hand.

### I want to understand it

[Architecture](ARCHITECTURE.md) is the high-level design on one page, with the service's place
among the five Trellis repositories; [the key flows](flows.md) are a sequence diagram each.
[The guide](#the-guide) is written to be read front to back: chapter 1 assumes you have never
seen a memory service, chapter 12 that you are about to operate one.

### I want to check a claim

[Reference](#reference): the measurements, the decision records, the generated API contract.
Nothing in the guide claims a number that is not in one of those files.

---

## Using it

| Page | What it answers |
|---|---|
| [README](../README.md) | start here: what it is, the quickstart, install and run |
| [examples/](../examples/README.md) | eleven runnable programs, simple to complex, offline (`make examples`) |
| [USAGE.md](USAGE.md) | which API for which scenario: push or pull, which write and which read, corrections, feedback, model keys |
| [api/](api/README.md) | the API reference: every route and its SDK call, one page per area, each with a diagram |
| [openapi.json](openapi.json) | the generated HTTP contract: every route, schema and error, with examples (`make openapi`; `/docs` on a running service) |
| [configuration.md](configuration.md) | every setting: its default, a working example, and whether it is automatic |
| [troubleshooting.md](troubleshooting.md) | the errors and surprises a first integration meets, and what to do |
| [versioning.md](versioning.md) | what is stable, how versions change, which Trellis versions work together |
| [CHANGELOG](../CHANGELOG.md) | what changed, release by release |

The API reference, area by area:

| Area | Page |
|---|---|
| observations, memories, the knowledge graph, jobs | [api/memory.md](api/memory.md) |
| context bundles, recall, verify, conversation | [api/context.md](api/context.md) |
| what an agent pulls itself, and what that teaches the push | [api/agent-tools.md](api/agent-tools.md) |
| pinned profile blocks and the thread summary | [api/profile.md](api/profile.md) |
| documents into retrievable knowledge | [api/documents.md](api/documents.md) |
| tool memory and procedures | [api/tools.md](api/tools.md) |
| feedback, the review queue, and what it changes | [api/feedback.md](api/feedback.md) |
| workspaces, keys, model keys and policy, the read audit | [api/tenancy.md](api/tenancy.md) |
| onboarding a tenant; health, readiness and `/version` | [api/admin.md](api/admin.md) |

## The guide

| # | Chapter | What you get |
|---|---|---|
| 1 | [Why a memory service](guide/01-why-a-memory-service.md) | the problem, and why a vector database is not the answer |
| 2 | [Concepts](guide/02-concepts.md) | memories, observations, scopes, principals: the vocabulary |
| 3 | [Time](guide/03-time.md) | how a fact stops being true without being deleted |
| 4 | [Retrieval](guide/04-retrieval.md) | four retrievers, one ranking, and why there is no reranker |
| 5 | [The knowledge graph](guide/05-knowledge-graph.md) | the questions vector search cannot answer |
| 6 | [Trust](guide/06-trust.md) | grounding, contradiction, and memory poisoning |
| 7 | [Authorization](guide/07-authorization.md) | seven visibility levels, and who an agent really is |
| 8 | [Models](guide/08-models.md) | which models, where, why; every model use; running without an LLM |
| 9 | [API and SDK](api/README.md) | every endpoint and its SDK call, side by side, one page per area |
| 10 | [Architecture](guide/10-architecture.md) | ports, adapters, and the path a write and a read take, step by step |
| 11 | [Operations](guide/11-operations.md) | deploying, what to set, workers and jobs, hardware, observability |
| 12 | [Testing and gates](guide/12-testing-and-gates.md) | how the claims in this documentation are kept true |

Each chapter cites the code (`src/memory_service/…` paths, given relative to the package) and
the ADRs it describes, and every figure names its source in [`MEASUREMENTS.md`](MEASUREMENTS.md),
an ADR's evidence or `benchmark/results/`; where something is designed but not built, the
chapter says so. Two deployment pages go with chapter 11:
[deploy/database.md](deploy/database.md) (the connection budget, PgBouncer, online migrations)
and [deploy/search.md](deploy/search.md) (Qdrant shards, replicas and the tenant index).

## Reference

- [`ARCHITECTURE.md`](ARCHITECTURE.md): the high-level design, the five repositories, the data
  placement, durability, retrieval, caching and runtime diagrams. Chapter 10 is the explained
  version; [`flows.md`](flows.md) has the sequence diagrams.
- [`adr/README.md`](adr/README.md): the index of the architecture decision records, one line
  and a status each, in the order they were taken. A decision that is later reversed or
  narrowed is amended rather than rewritten, and its header says by which ADR.
- [`MEASUREMENTS.md`](MEASUREMENTS.md): the raw measurements behind the defaults, each
  traceable to a file in `benchmark/results/`.
- [`MULTILINGUAL-RUNTIME.md`](MULTILINGUAL-RUNTIME.md): how a query and a record in any
  language are encoded, indexed and verified.
- [`benchmark/reports/`](../benchmark/reports/README.md): the dated measurement reports the
  ADRs cite as evidence. Each describes the code as it was on its date.
- [`CONTRIBUTING.md`](CONTRIBUTING.md): how to make a change, and the rules that do not bend.

---

## The rule these documents follow

Nothing here claims a measurement that does not exist in `benchmark/results/`, and any number
produced with a stand-in provider is labelled as such. If a document and a gate artifact
disagree, the artifact wins. `make docs-check` keeps every link resolving and every snippet
calling the SDK that exists; `make examples` runs the examples.
