# ADR 0014: The LangGraph adapter wraps nodes; it does not become a checkpointer or store

**Status:** superseded by ADR 0020 (layering), 2026-09-22 · **Date:** 2026-09-15

> **Superseded — the decision below was right, the location was not.** Wrapping nodes rather
> than implementing `BaseCheckpointSaver` / `BaseStore` still holds, and the lineage mapping
> it describes is still how a LangGraph execution becomes agent runs. What changed is where
> the adapter lives: `integrations/langgraph` has been removed from this repository, because
> a memory service that ships a LangGraph package depends on its own consumer. Framework
> adapters live in `agent-harness`, which already carries `universal-agent-harness-langgraph`
> with the same `checkpoint_ns` lineage parser. Kept as the record of *why* the wrapper shape
> was chosen, for whoever maintains that adapter.

## Context
LangGraph applications need the Memory Service at three points: before a node runs (what
does the user/agent know?), after it runs (what happened?), and across subgraphs (which
agent is acting, for whom, under which parent run?). LangGraph also offers two extension
seams of its own — `BaseCheckpointSaver` (graph state) and `BaseStore` (key/value long-term
memory) — that look like natural homes for an integration.

## Decision
- **Node wrapper, not framework plumbing.** `LangGraphMemory.wrap(node, recall=,
  observe=, agent=)` returns an async node that resolves the `MemoryContext`, records the
  pending human turn, fetches a bounded `ContextBundle`, runs the node, records the new
  messages and submits observations. Graph state, checkpoints and streaming are untouched.
- **Not a checkpointer.** Graph state is execution state, not memory; a checkpointer
  backed by the Memory Service would couple graph replay to memory availability and blur
  the "canonical store = PostgreSQL, memory = derived intelligence" boundary (ADR 0001).
- **Not a `BaseStore`.** LangGraph's store is exact-key key/value with namespace listing;
  the service is observation-based (extract, classify, consolidate, supersede). A bridge
  would have to fake exact-key semantics (search-then-filter) or add a bespoke key API to
  the service for one framework. Applications that want raw key/value keep LangGraph's own
  store; the adapter gives them memory intelligence beside it.
- **Lineage comes from the checkpoint namespace.** `checkpoint_ns` segments are
  `node:task_id`; task ids are deterministic per (checkpoint, step, node). Every segment
  but the executing node's is a subgraph invocation → an agent run (`lg-<task_id>`) with
  the enclosing segment as parent; root-graph nodes act as the user. This makes the M11
  semantics (hand-off flows down one hop, never up or sideways) fall out of the graph
  structure with no configuration, and makes run ids stable across retries.
- **Idempotency keys are derived, never random.** (thread, namespace path, node, step,
  content) → key; a superstep replayed from a checkpoint reuses the same keys, and the
  service's idempotency layer (ADR 0004) turns the replay into an acknowledgement.
- **Writes never fail silently.** Recall may be best-effort (`strict=False`); recording
  and observing always raise so LangGraph's retry policy sees the failure. This keeps the
  "acknowledged data loss = 0" property end to end.
- **Async only.** The SDK is async; wrapped nodes are coroutines, so graphs run with
  `ainvoke`/`astream`. Sync node functions are still supported inside the wrapper.

## Consequences
- The package depends on the SDK only; LangGraph is imported lazily and only for reading
  the running config and injecting `runtime`/`store`/`writer` when a node asks for them.
- Turn ids default to the pending human message id (messages convention) or the
  superstep; apps with their own session/turn model pass `configurable["memory"]`.
- Tested against real LangGraph graphs (checkpointer, nested subgraphs, `RetryPolicy`)
  and the real service in-process: `integrations/langgraph/tests/`.
