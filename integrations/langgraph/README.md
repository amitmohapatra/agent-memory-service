# universal-memory-langgraph

Optional LangGraph integration for `universal-memory`. Maps LangGraph `thread_id` and
runtime context to a `MemoryContext`, fetches a ContextBundle before configured nodes,
captures observations after nodes, propagates child agent run ids, generates stable
idempotency keys across checkpoint retries, and supports subgraphs.

LangGraph is never a dependency of the Memory Service core.
