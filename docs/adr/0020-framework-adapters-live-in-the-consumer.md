# ADR 0020: Framework adapters do not live in the memory service

**Status:** accepted · **Date:** 2026-09-22

## Context
The repository carried `integrations/langgraph`, `integrations/mcp`, `integrations/adk` and
`integrations/crewai` — four packages that adapt the memory SDK to somebody else's agent
framework — plus `universal_memory.integrations`, 573 lines of shared adapter core inside the
SDK that existed only to serve them.

Two of the four were never written: `adk` and `crewai` contained `"""Package placeholder."""`
and nothing else, behind a pyproject pinning `google-adk>=2.9` / `crewai>=1.6`, a uv.lock, a
workspace exclusion explaining their dependency conflicts, ruff per-file ignores, isort
entries, coverage sources and 553 MB of resolved virtualenv on disk.

The two that were written had no consumer. A grep across every sibling repository found zero
imports of `universal_memory_langgraph` or `universal_memory_mcp` outside this repository.
`agent-harness` — the only thing that consumes the memory service at all, via a path
dependency on `sdk/python` — ships its **own** LangGraph adapter,
`universal-agent-harness-langgraph`, whose `lineage.py` parses the same LangGraph
`checkpoint_ns` format with the same `Segment` / `Lineage` / `subgraphs` / `task` / `path` /
`lineage_from_config` symbols, and whose node wrapper already threads a `MemoryClient`
through. Two implementations of one mapping, in two repositories, and the live one was not
this one.

`deploy/Dockerfile` copied the whole `integrations` tree into the runtime image and chmodded
it, so the memory server shipped LangGraph, MCP and CrewAI adapter packages it never imports.

## Decision
- **The service knows its SDK and nothing above it.** A memory service that ships a LangGraph
  package depends on its own consumer; the dependency arrow points the wrong way, and it
  shows up as framework names in this repository's dependency graph, its lint configuration
  and its container image.
- **Framework adapters live in the layer that drives the framework.** In this platform that
  is `agent-harness`, which already has `integrations/langgraph` as a workspace member with
  tests, an example and an extra.
- **The SDK stays framework-neutral and stays here.** `sdk/python` is the public surface;
  `universal_memory.integrations` went with the adapters, because it was adapter-support code
  — never re-exported from `universal_memory/__init__.py`, absent from the SDK README, and
  covered by no SDK test.
- **Plans are not documentation.** `docs/INTEGRATIONS_PLAN.md` specified an ADK adapter, a
  CrewAI adapter and a cross-adapter conformance suite (`tests/integrations/`, an empty
  directory that still had a ruff rule). Specifications for work that will not be done here
  are deleted, not carried.

## Consequences
- 2,465 lines of Python and four packages leave the repository. Nothing that runs imports
  them; `ruff check` and `ruff format --check` pass unchanged.
- The runtime image no longer carries framework packages.
- ADR 0014 is superseded on location only: its reasoning — wrap nodes, do not implement
  `BaseCheckpointSaver` or `BaseStore` — remains the guidance for whoever maintains the
  adapter in `agent-harness`.
- The MCP server is the one real loss: 504 lines exposing memory as MCP tools, untested and
  unreferenced, but a genuine product surface rather than a duplicate. It is recoverable from
  git history and belongs in its own package beside the other SDK consumers, not here.
- `TID251` still bans `langgraph` under `src/`, now with a message that points at the harness
  instead of at a directory that no longer exists.
