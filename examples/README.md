# Examples

Everything here runs against a **live server over HTTP** — nothing is mocked — so the
examples double as an end-to-end acceptance run of the service and the Python SDK.

```bash
./examples/run_server.sh &                    # API on :8080, API key "dev-key"
uv run python examples/sdk_tour.py            # every SDK method and API route, checklist
make examples                                 # the same, against a server you started
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
