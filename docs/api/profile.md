# Profile blocks and thread summaries: what every prompt starts from

Two things are in every pushed context before any retrieval: the **pinned profile blocks** of
the caller's user, agent and workspace, and the **durable summary** of the thread. Both are
kept by the service; neither costs a model call on the request path.

## Routes

| Route | Purpose | SDK |
| --- | --- | --- |
| `GET /v1/profile` | the blocks of this user, agent and workspace | `await ctx.profile()` |
| `PUT /v1/profile/{block}` | replace a block's text | `ctx.profile.set(block, text)` |
| `PATCH /v1/profile/{block}` | replace `old` with `new` once; 409 when `old` is not there | `ctx.profile.edit(block, old, new)` |
| `GET /v1/threads/{thread_id}/summary` | the thread's durable summary (404 until it has one) | `await ctx.summary()` (None until then) |

## Profile blocks

A block is named `user`, `agent` or `workspace`, optionally followed by `.<name>`
(`user.preferences`, `agent.persona`). The level decides whose block it is, from the caller's
own scope: `user` is the bound user's, `agent` is this agent acting for this user (the agent
principal, never the bare agent id), `workspace` is the bound workspace's (writing it needs
membership). So a caller can only ever read and write the blocks of its own user, agent and
workspace. A block holds at most 4,000 characters; each write bumps its `version` and the
revision the pushed context depends on, so a cached bundle with the old text is not served.

`PATCH` is how an agent edits memory in place (the `profile_edit` agent tool): a stale edit
(`old` is no longer in the text) answers 409, and the agent reads the block again.

The service keeps the **`user`** block from the user's own USER and PREFERENCE memories (the
`profile.refresh` job, queued whenever one is indexed): with the tenant's model (use
`summaries`) it writes a concise profile, one `name: value` line per fact; without one, a line
per memory. Once a person or an agent has edited the block, the job only appends facts learned
after the edit, so an edit is never lost or undone. `name: value` lines are also what tool
hints fill an argument of the same name from.

## Thread summaries

Every 20 messages (`SUMMARY_EVERY`) the `summary.refresh` job folds the messages after the last
summary into it and stores a new version with `covers_to_sequence`: abstractive with the
tenant's model (use `summaries`, paid by whoever wrote the message that crossed the mark),
extractive otherwise (a line per message: role and first sentence, the oldest lines dropped
past 2,000 characters). `/v1/context` carries the latest summary and only the messages after
it.
