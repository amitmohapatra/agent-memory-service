# Documentation

Three ways in. Pick the one that matches why you are here.

### I want to use it

Start with the [README](../README.md) — install, a working agent in twenty lines, and what
the service needs to run. Then [the guide](#the-guide), in order.

### I want to understand it

[The guide](#the-guide) is written to be read front to back. Chapter 1 assumes you have
never seen a memory service; chapter 12 assumes you are about to operate one. Every chapter
stands alone if you already know the rest.

### I want to check a claim

[Reference](#reference) — the measurements, the decision records, the generated API
contract. Nothing in the guide claims a number that is not in one of those files.

---

## The guide

| # | Chapter | What you get |
|---|---|---|
| 1 | [Why a memory service](guide/01-why-a-memory-service.md) | the problem, and why a vector database is not the answer |
| 2 | Concepts *(not written yet)* | memories, observations, scopes, principals — the vocabulary |
| 3 | Time *(not written yet)* | how a fact stops being true without being deleted |
| 4 | Retrieval *(not written yet)* | four retrievers, one ranking, and why reranking is off |
| 5 | The knowledge graph *(not written yet)* | the questions vector search cannot answer |
| 6 | Trust *(not written yet)* | grounding, contradiction, and memory poisoning |
| 7 | Authorization *(not written yet)* | ten visibility levels, and who an agent really is |
| 8 | Models *(not written yet)* | which models, where, why — and running without an LLM |
| 9 | API and SDK *(not written yet)* | every endpoint and its SDK call, side by side |
| 10 | Architecture *(not written yet)* | ports, adapters, and the path a write takes |
| 11 | Operations *(not written yet)* | deploying, configuring, and what hardware it needs |
| 12 | Testing and gates *(not written yet)* | how the claims in this documentation are kept true |

Only chapter 1 is written. The rest is the planned outline, listed so you can see where
the guide is going — the chapters are unlinked rather than linked-and-missing on purpose.

## Reference

**Describes what exists**

- [`openapi.json`](openapi.json) — the generated HTTP contract: every route, schema and
  error, with examples. Regenerate with `make openapi`.
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — the one-page structural summary. Chapter 10 is the
  explained version.
- [`adr/`](adr/) — architecture decision records, numbered in the order taken. Each states
  the context, the decision, and what it cost. When a decision is later reversed or never
  carried out, the ADR is amended rather than rewritten, so the record stays honest.
- [`MEASUREMENTS.md`](MEASUREMENTS.md) — the raw measurements behind the defaults, each
  traceable to a file in `benchmark/results/`.
- [`CAPABILITY_COVERAGE.md`](CAPABILITY_COVERAGE.md) — which capability provides what, and
  the argument for the seven retrieval flags that were removed rather than left off.
  `tests/eval/test_capability_coverage.py` is the executable form of it.
- [`FINAL_REPORT.md`](FINAL_REPORT.md) — the per-gate account of what has been measured, in
  what environment, and which numbers are **not** yet representative. Read this before
  quoting any performance figure.
- [`TOOL_MEMORY.md`](TOOL_MEMORY.md) — the tool-memory design.
- [`DATA_PLACEMENT_REVIEW.md`](DATA_PLACEMENT_REVIEW.md) — which store holds what, and why.
- [`PRODUCT_DECISIONS.md`](PRODUCT_DECISIONS.md) — product-level choices and their evidence.

**History**

- [`MILESTONES.md`](MILESTONES.md) — the build log, M0 to M13. Archaeology, not a
  description of the current API.

**Planned or partly built** — specifications written before the work. Where they disagree
with the code, the code is right; each carries a status line saying how much is built.

- [`TARGET_STACK.md`](TARGET_STACK.md) — the chosen models and components, the evidence for
  each, and the changes still to make.

**Contributing**

- [`CONTRIBUTING.md`](CONTRIBUTING.md) — how to work on this, and the rules that do not bend.

---

## The rule these documents follow

Nothing here claims a measurement that does not exist in `benchmark/results/`, and any
number produced with a stand-in provider is labelled as such. If a document and a gate
artifact disagree, the artifact wins.
