# Examples

Everything here runs against a **live server over HTTP** — nothing is mocked — so the
examples double as an end-to-end acceptance run of the service, the Python SDK and the
LangGraph adapter.

```bash
./examples/run_server.sh &                    # API on :8080, API key "dev-key"
uv run python examples/sdk_tour.py            # every SDK method and API route, checklist
uv run python examples/langgraph_crew/app.py  # a LangGraph research crew on top of memory
make examples                                 # both, against a server you started
```

`run_server.sh` needs PostgreSQL (migrations are applied) and Redis/Dragonfly. It runs a
single process with jobs executed inline because, without a Qdrant server, the search index
lives in the API process (qdrant-client local mode). With the compose stack (`make dev-up`)
set `MEMORY__TASKS__PROVIDER=procrastinate`, run `uv run memory-worker`, and everything
else stays the same. Models: the hash embedding + lexical reranker are used unless
`MEMORY__MODELS__*` point at local weights.

## `sdk_tour.py` — the SDK, method by method

Fourteen checks, each an assertion against the live service:

| Area | What is exercised |
|---|---|
| Operations | `health()`, `alive()`, `version()` |
| Conversation | `chat.create` (idempotent), `chat.user/assistant/internal`, `chat.history(include_internal)`, `chat.message(id)`, `chat.thread()`, `chat.delete_thread()`; scope isolation (other user → 403, other tenant → not found) |
| Files | `files.add` (path / bytes, title, visibility), `files.wait_ready`, `files.document`, content dedup, `job()` |
| Retrieval | `recall` (citations, pages incl. definition + footnote, table rows), `context` (COMPLETE evidence, cache hit, token budget, `require_evidence` → `InsufficientEvidence`) |
| Memory | `observe` (idempotent replay), `remember` (PREFERENCE / SEMANTIC / EPHEMERAL), `memories()`, `get_memory`, supersede + reinforce + temporal history, `forget` everywhere (list, recall, get) |
| Agents | `agent()` run lineage (child reads parent's RUN memory; user, sibling and stranger do not), `AGENT_GROUP` sharing, corroboration (`contributors`), cross-agent conflict kept and flagged in the evidence report |
| Graph | `graph.query` by entity / free text / alias (`ARR`), 2-hop questions, fact attributes (period, change, previous value), counterfactual kept apart, `as_of` temporal view |
| Ergonomics | `async with ctx` + `current_context()`, `derive()` |

## `langgraph_crew/app.py` — a crew of agents with memory

A root graph (`intake → crew → answer`), a `crew` subgraph (`plan → research → review`)
and a nested `research` subgraph (`gather → summarise`). Every node is wrapped with
`LangGraphMemory.wrap`, which:

- records the user's turn and the assistant's reply in the memory thread,
- hands each node a `ContextBundle` (`state["memory"]`) recalled for the question,
- runs subgraph nodes as **agent runs** derived from the checkpoint namespace: the crew's
  plan is RUN-scoped hand-off context that the research run (its child) reads and the user
  never sees; research findings are shared with the agent group explicitly,
- submits observations with idempotency keys that survive checkpoint retries.

The example has no LLM: the `answer` node composes from the verified bundle's graph facts
and shows how a stored preference ("I prefer concise answers") changes later replies. Drop
an LLM call into `answer` with `state["memory"].rendered` as context to make it talk.
