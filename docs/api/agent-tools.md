# Agent tools: the memory an agent pulls for itself

`/v1/context` pushes what a turn probably needs. Agent tools let the agent pull the rest itself
(ReAct): a fixed set of memory tools with JSON input schemas, called in the caller's scope like
every other route. The harness adds them to a wrapped agent's tool list (Way 1); any other
framework lists them with `ctx.agent_tools()` and offers them to its model (Way 2).

## Routes

| Route | Purpose | SDK |
| --- | --- | --- |
| `GET /v1/agent-tools` | the tools, each `{name, description, input_schema}`; `ETag` + `Cache-Control: private, max-age=300`, `If-None-Match` → `304` | `ctx.agent_tools()` |
| `POST /v1/agent-tools/{name}` `{"args": {...}}` | call one; answers `{"result": ...}` | `ctx.call_agent_tool(name, args)` |

## The tools (the final set)

| Tool | Arguments | Result |
| --- | --- | --- |
| `memory_search` | `query`, `kinds?` (memory, chunk, summary, episode, message — what was said here, then in this user's earlier conversations), `time_from?`, `time_to?` (applied before ranking), `k?` (≤ 20) | `[{id, kind, text, observed_on, …}]` (the `/v1/recall` item) |
| `memory_remember` | `content`, `kind` (the 8 primary kinds), `scope` (user, agent, run, thread, group, workspace) | `{id, deduplicated}` |
| `memory_update` | `id` (or its bundle handle, `m3`), `content` | `{id, supersedes}` |
| `memory_forget` | `id` (or its bundle handle) | `{id, forgotten}`; memories derived from it are retracted too ([memory.md](memory.md#forgetting-takes-what-was-derived-from-it)) |
| `profile_edit` | `block`, `old` (empty replaces the block), `new` | `{block, text, version}`; 409 when `old` is gone |
| `tool_search` | `task` | `{tools: [{name, confidence, success_rate?, next?, args?, missing?}], plan?}` (the `/v1/tools/hints` answer) |

Descriptions and schemas are language-neutral: queries and content may be in any language.
Bad arguments answer 422 naming the field (`args.<field>`); an unknown tool 404. Writes follow
the rules of their routes (a memory is updated or forgotten by its owner, a workspace block by a
member).

## Pulls and prefetch learning

Every call is logged as a **pull**: the request's pattern (the typed-placeholder form of the
query or task), the tool, its arguments and the ids it returned. A returned id is **used** when
the run later cites it (a supported claim of `POST /v1/verify`, the cited memories of a
confirmed answer verdict) or acts on it (`memory_update`, `memory_forget`). Every five minutes a
job folds pulls older than ten minutes into per (principal, pattern, item) counts. `/v1/context`
pre-includes (one indexed read) up to five items that were pulled at least three times for the
same pattern and used in at least half of those pulls, re-checked against the caller's audience.
