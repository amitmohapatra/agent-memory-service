# Documentation map

Start with the [README](../README.md) — it covers installing, using the service from one
agent, multi-agent visibility, tool memory and configuration, and every example in it is
executed by a test.

This folder is mostly for people working **on** the service rather than with it. Each
document says up front whether it describes something that exists or something planned.

## If you want to…

| … then read |  |
|---|---|
| **use the service** | [README](../README.md) — the only document you need |
| **call the HTTP API directly** | [openapi.json](openapi.json), or `/docs` on a running service |
| **understand how it is built** | [ARCHITECTURE.md](ARCHITECTURE.md) |
| **know why something is built that way** | [adr/](adr/) — one decision per file, with its trade-offs |
| **contribute** | [CONTRIBUTING.md](CONTRIBUTING.md) |
| **judge whether to trust a benchmark number** | [FINAL_REPORT.md](FINAL_REPORT.md) |

## What each document is

**Reference — describes what exists**

- [`openapi.json`](openapi.json) — the generated HTTP contract. Every route, schema and error,
  with examples. Regenerate with `make openapi`.
- [`adr/`](adr/) — architecture decision records, numbered in the order they were taken. Each
  states the context, the decision and the consequences. `0017` and `0018` are the most recent
  (real-component validation, and tool memory).
- [`FINAL_REPORT.md`](FINAL_REPORT.md) — the per-gate account of what has been measured, in
  what environment, and which numbers are **not** yet representative. Read this before
  quoting any performance or retrieval figure.
- [`MEASUREMENTS.md`](MEASUREMENTS.md) — the raw measurements behind the decisions, each
  traceable to a file in `benchmark/results/`.
- [`CAPABILITY_COVERAGE.md`](CAPABILITY_COVERAGE.md) — which capability provides what, and
  the argument for the seven retrieval flags that were removed rather than left off.
  `tests/eval/test_capability_coverage.py` is the executable form of it.
- [`PRODUCT_DECISIONS.md`](PRODUCT_DECISIONS.md) — product-level choices and the evidence
  for each.
- [`DATA_PLACEMENT_REVIEW.md`](DATA_PLACEMENT_REVIEW.md) — which store holds what, and why
  each piece of data lives where it does.

**History — how the service got here**

- [`MILESTONES.md`](MILESTONES.md) — the build log, milestone by milestone (M0–M13). Useful
  for archaeology; not a description of the current API.

**Design — describes what is planned or partly built**

These are specifications written before the work. Where they disagree with the code, the code
is right. Each has a status line at the top saying how much of it is built.

- [`TARGET_STACK.md`](TARGET_STACK.md) — the chosen models and components, the evidence behind
  each choice, and 25 numbered changes to make. Partly implemented.
- [`INTEGRATIONS_PLAN.md`](INTEGRATIONS_PLAN.md) — framework adapters: LangGraph (built),
  Google ADK, CrewAI and an MCP server (not yet).
- [`TOOL_MEMORY.md`](TOOL_MEMORY.md) — the tool-memory design. The service, API, SDK and gate
  are built; the framework adapters described in §30.5 and §30.8 are not.

## The rule these documents follow

Nothing here claims a measurement that does not exist in `benchmark/results/`, and any number
produced with a stand-in provider is labelled as such. If a document and a gate artifact
disagree, the artifact wins.
