# Framework integrations — LangGraph, Google ADK, CrewAI, MCP

> **Status: LangGraph is built and tested. Google ADK, CrewAI, the MCP server and the
> conformance suite are specified here but not finished** — the ADK and CrewAI packages are
> scaffolds and the MCP server is partial. Use the LangGraph adapter, or the
> framework-neutral SDK, which works anywhere.

Companion to `TARGET_STACK.md` (changes 26–29). The Python SDK (`universal-memory`) is
framework-agnostic; each framework gets a thin adapter that maps its own session/agent model
onto the service's tenant / workspace / user / agent-group / agent-run / thread scopes. No
framework package is ever imported by the core service (ruff `banned-api`, like `langgraph`).

## What exists today

- **LangGraph** — `integrations/langgraph` (`universal_memory_langgraph.LangGraphMemory`):
  wrap any node with `memory.wrap(node, recall=, observe=, agent=, require_evidence=,
  record_messages=)`; `thread_id` → memory thread; subgraphs become agent runs with lineage
  (parent/child), RUN-scoped hand-off memory, explicit sharing to the agent group,
  idempotent recording across retries, evidence-gated context in `state["memory"]`.
  Verified by `examples/langgraph_crew/app.py` and the adapter test suite.

## 26. Google ADK adapter — `integrations/adk` (`universal_memory_adk`)

ADK's contract is `BaseMemoryService` with `add_session_to_memory(session)`,
`add_events_to_memory(session, events)`, `add_memory(entries)` and `search_memory(app_name,
user_id, query)`; agents use it through the built-in `load_memory` / `preload_memory` tools
and callbacks, and a single service is passed to `Runner(memory_service=...)`.

- `MemoryServiceAdapter(BaseMemoryService)`: `app_name` → workspace, `user_id` → user,
  `session.id` → thread, agent name → agent id (agent runs created per invocation with
  lineage for sub-agents), `add_session_to_memory` / `add_events_to_memory` → idempotent
  `observe` of the new events only (keys from session id + event id), `add_memory` →
  `remember`, `search_memory` → `/v1/context` (evidence-gated) rendered as `MemoryEntry`
  items with source, page and evidence status in metadata.
- Optional `UniversalMemoryTool` set (`recall`, `remember`, `forget`, `graph_query`,
  `verify`) for agents that want explicit control beyond `load_memory`.
- Example: `examples/adk_assistant/` (an ADK agent with sub-agents, same FY26 fixture,
  same checks as the LangGraph crew); contract tests against ADK's in-memory session
  service; documented scope mapping table.

## 27. CrewAI adapter — `integrations/crewai` (`universal_memory_crewai`)

CrewAI takes third-party memory through `Crew(external_memory=ExternalMemory(storage=...))`
where the storage implements `save()`, `search()` and `reset()`; scoping is by a stable
identifier passed to the storage.

- `UniversalMemoryStorage(Storage)`: crew id → agent group, agent role → agent id (one run
  per kickoff, tasks as child runs), `save(value, metadata, agent)` → `observe` with hints
  (task outputs as AGENT memories, explicit `share=True` metadata → AGENT_GROUP),
  `search(query, limit, score_threshold)` → `recall` with evidence status in metadata,
  `reset()` → forget-by-scope (soft: invalidation, never canonical deletion).
- `MemoryTools` for agents (`RecallTool`, `RememberTool`, `GraphQueryTool`, `VerifyTool`).
- Example: `examples/crewai_research/`; contract tests with a stub crew.

## 28. MCP server — `integrations/mcp` (`universal-memory-mcp`)

One MCP server over the SDK so any MCP client (Claude, Cursor, OpenAI Agents SDK, AutoGen,
Semantic Kernel, custom agents) gets the same tools without a framework adapter:
`memory.recall`, `memory.context`, `memory.observe`, `memory.remember`, `memory.forget`,
`memory.graph_query`, `memory.verify`, `memory.files.add`, `memory.threads.*`.
Scopes come from the server configuration plus per-call arguments validated against the
authorized scope; auth is the same service API key / JWT; every tool result carries evidence
status and sources. Ships as `uvx universal-memory-mcp` and as a Docker image.

## 29. Conformance suite — `tests/integrations/`

One scenario run through every adapter (LangGraph, ADK, CrewAI, MCP): user turn recorded,
agent hand-off visible to child run only, explicit share visible to the group, user never
sees agent notes, preference honoured on the next turn, evidence-gated answer, idempotent
retries. Any adapter that cannot pass the same scenario is not released.

## Scope mapping (all adapters)

| Framework concept | Service scope |
|---|---|
| app / project / crew | workspace (tenant fixed by the API key) |
| user id | user (`principal:`) |
| session / thread id | thread (session/turn derived) |
| agent / sub-agent / crew member | agent id → agent run with lineage |
| team / crew | agent group (explicit sharing only) |
