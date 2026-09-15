# universal-memory-langgraph

Optional LangGraph integration for `universal-memory`. LangGraph is never a dependency of
the Memory Service core; this package depends on the SDK and imports LangGraph lazily.

```python
from universal_memory import MemoryClient
from universal_memory_langgraph import LangGraphMemory

memory = LangGraphMemory(MemoryClient(url, api_key=key), tenant_id="acme", user_id="u1")


async def answer(state):
    bundle = state["memory"]  # ContextBundle: conversation window, memories, evidence
    ...  # prompt with bundle.rendered
    return {"messages": [("assistant", reply)]}


graph.add_node("answer", memory.wrap(answer, recall=lambda s: s["messages"][-1].content))
await graph.ainvoke(
    {"messages": [("user", "My timezone is Europe/Berlin.")]},
    {"configurable": {"thread_id": "chat-42"}},
)
```

What `wrap` does around a node, every time it runs:

1. resolves the `MemoryContext` from the LangGraph config — `thread_id` becomes the memory
   thread, the checkpoint namespace becomes the agent lineage (see below), and
   `configurable["memory"] = {...}` can set or override any scope field (user, workspace,
   session, turn, work...). The context is bound for the node's duration, so tools can call
   `universal_memory.current_context()`;
2. records the pending human turn and, after the node, the new messages it returned
   (`messages` convention: human → user turn, ai → assistant, tool/system → internal, never
   in the visible history);
3. fetches a bounded `ContextBundle` for the `recall` query (a state key or a function) and
   hands it to the node under `inject` (default `"memory"`); `require_evidence=True` raises
   `InsufficientEvidence` instead of letting the node answer from nothing;
4. submits observations from `observe` (a result key or `fn(state, result)` returning text,
   a list, or `{"content", "kind", "hints", "metadata"}`).

Every write carries a deterministic idempotency key built from the thread, the checkpoint
namespace (whose task ids are derived from checkpoint + step + node), the node and the
content — a superstep retried from a checkpoint records nothing twice.

## Subgraphs are agent runs

`checkpoint_ns` is `node:task|node:task|...`, one segment per graph level. Every segment
except the executing node's is a subgraph invocation and maps to an agent run: `agent_id`
= the node name, `agent_run_id` = its task id, parent = the enclosing segment. Root-graph
nodes act as the user; a node in the `research` subgraph nested in the `crew` subgraph acts
as agent `research`, child of the `crew` run. Working memory written by a run is
`RUN`-visible (ADR 0013): the run, the subgraphs it invokes, and the same agent later can
read it; the user, sibling runs and other agents cannot. `wrap(node, agent="planner")`
turns a single node into its own run under the enclosing one. Share on purpose with
`observe=lambda s, r: {"content": ..., "hints": {"memory_type": "SHARED",
"visibility": "AGENT_GROUP"}}`.

## Failure policy

Write failures (messages, observations) always propagate — LangGraph's retry policy and
checkpoints handle them, and nothing is lost silently. Read failures propagate unless
`strict=False`, in which case the node runs with `state["memory"] = None`. Graphs must
run with `ainvoke`/`astream` (the SDK is async); sync node functions are fine.

## Ids

LangGraph ids are coerced to the service's id alphabet (`[A-Za-z0-9._:-]`); `session_id`
defaults to one session per thread and `turn_id` to the pending human message's id (or the
superstep). Applications with real sessions and turns pass them in
`configurable["memory"]`.
