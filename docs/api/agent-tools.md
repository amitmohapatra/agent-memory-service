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
