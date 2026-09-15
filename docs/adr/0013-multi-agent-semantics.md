# ADR 0013: Multi-agent memory semantics

**Status:** accepted · **Date:** 2026-09-15

## Context
Several agents act for the same user inside one thread, spawn each other (planner → writer
→ formatter), and report findings into shared scopes. The spec requires: private vs shared
memory, hand-off context that flows to the agents a run delegates to, no agent chatter
polluting the user's chat history or the user's own memories, corroboration across agents
without duplication, and conflicts that are surfaced rather than silently resolved — all
without loosening the zero-leak isolation gates.

## Decision
- **A run is a visibility boundary.** `Visibility.RUN` (new) is the default for an agent's
  working memory (`AGENT`, `TOOL`, `WORKING` types) whenever the context carries an
  `agent_run_id`; without a run id it stays `PRIVATE`. RUN keys are
  `run:<tenant>/<run_id>` plus `principal:<tenant>/<owner>`. A reader's scope carries its own
  run id and its parent's (`AuthorizedScope.run_ids`), so hand-off context flows **down**
  exactly one level per hop: a child reads its parent's notes, a grandchild reads its parent's
  but not the grandparent's, siblings read nothing of each other, and the human never reads
  an agent's working notes upwards. The writing agent keeps its own notes across later runs
  via the principal key. The security oracle states the rule independently
  (`obj.run ∈ reader.runs or reader.principal == obj.owner`) and the property-based and
  matrix isolation suites include the run dimension.
- **Sharing is explicit.** An agent shares by hint (`memory_type=SHARED` and/or
  `visibility=AGENT_GROUP|THREAD|WORK|...`); nothing is shared by accident. The scope anchor
  (`AGENT_GROUP` level) and the visibility keys are derived from the hint and the context.
- **Corroboration is counted, not duplicated.** The same finding from a second agent in a
  shared scope reinforces the existing memory: `reinforcement_count` grows, confidence rises
  by 0.15, and the second principal is recorded in `system_metadata.contributors`; the
  owner stays the first writer. Retrieval payloads carry `contributors`.
- **Conflicts between principals are kept, not overwritten.** A different value for a
  single-valued slot (manager, timezone, name, ...) written by *another* principal in a
  shared scope yields `CONTRADICT`: both memories stay CURRENT, linked through
  `temporal.contradicts`, indexed with `contradicts`, and the evidence report adds a
  "conflicting memories" note so the caller (or a human) decides. A principal correcting
  **its own** finding still supersedes it — consolidation matches the writer's own memories
  first — and an explicit replacement signal ("actually … now …") from another principal is
  honoured as a supersede, because it is an intentional correction rather than a second
  opinion.
- **No chat pollution.** Agent messages are `INTERNAL` and never appear in the visible
  history or the conversation window of the user's context bundle (ADR 0005). First-person
  facts extracted from an agent-authored observation ("my timezone is UTC") are re-typed
  from `USER`/`PREFERENCE` to `AGENT` before classification, so an agent's self-description
  becomes the agent's run memory and never a USER memory of the human it acts for.
- **SDK.** `ctx.agent(id)` mints a fresh run id when none is given and records the parent
  run, so lineage is correct by default; `ctx.agent(...).agent(...)` nests it.

## Consequences
- Isolation gates remain 0 leaks with the new visibility; caching keys include the run
  lineage (`scope_fingerprint` covers `agent_run_id` and `parent_agent_run_id`).
- Retrieval over `kinds=("memory",)` from a child run returns the parent's hand-off
  context ranked with everything else the run may see; there is no separate "hand-off"
  API to keep in sync.
- Existing deployments that relied on `PRIVATE` for agent memories with a run id now get
  `RUN`; the difference is only that direct child runs can read them.

Evidence: `tests/integration/test_multi_agent.py`, `tests/security/test_isolation.py`,
`tests/security/test_retrieval_isolation.py` (run dimension), `tests/unit/test_memory_native.py`.
