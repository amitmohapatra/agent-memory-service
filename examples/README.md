# Examples

Runnable programs, numbered from simple to complex. Each one asserts what it shows, so a run
that exits 0 is a check that the service still does it.

```bash
make examples                                    # all of them, offline (CI runs this)
uv run python examples/04_documents.py           # one of them
```

## Offline by default

An example builds the service **in its own process** (`create_app`) and the SDK talks to it
through `httpx.ASGITransport`: no server, no port, no network, no model. The stand-ins are the
test suite's (`Overrides` in [`_support.py`](_support.py)): in-process search, cache,
authorization and blob store, jobs run inline, the hash embedding and the lexical NLI. The
calls, the scope rules and the durability are the real ones; retrieval quality under the
stand-ins means nothing.

**The one dependency is PostgreSQL**, the source of truth, which has no stand-in: a
PostgreSQL 16 on `localhost:5432` with the user and password `memory` (`make dev-up` starts
one), or `MEMORY_EXAMPLES_DATABASE_URL`. The examples create a database of their own,
`memory_examples`, migrate it, and empty it at the start of every run; a URL whose database
name does not end in `examples` is refused, so they can never empty yours.

## Against a running service

```bash
uv run python examples/_serve.py &               # one process, jobs inline, http://localhost:8080
EXAMPLES_LIVE=1 uv run python examples/03_remember_search_forget.py
```

With `EXAMPLES_LIVE=1` an example uses `MEMORY_URL` and `TRELLIS_API_KEY` (default
`http://localhost:8080` and `dev-key`), so it runs against `_serve.py`, the compose stack or a
deployment. Examples 08 and 09 onboard tenants, so live they also need the operator's key in
`TRELLIS_BOOTSTRAP_KEY`, and skip without it. `_serve.py` needs PostgreSQL and Redis, and uses the real encoders when
`./models` holds them (`make models`) and says which it runs.

## The examples

| # | File | What it shows | Flow |
|---|---|---|---|
| 01 | [01_quickstart_context.py](01_quickstart_context.py) | bind a scope, record a turn, get the context of the next one; the same turn as an agent run, with its outcome as feedback (the README's two snippets) | [context build](../docs/flows.md#2-context-build) |
| 02 | [02_conversation_history.py](02_conversation_history.py) | a thread created on first use, turns, an agent's internal note, an `EVENT`, an idempotent replay, another user refused, a soft delete | [ingest](../docs/flows.md#1-ingest-history-add) |
| 03 | [03_remember_search_forget.py](03_remember_search_forget.py) | a fact learned from a message, facts stated with `remember`, a fact superseded and never served again, `search`, `forget` | [search and recall](../docs/flows.md#3-search-and-recall) |
| 04 | [04_documents.py](04_documents.py) | upload a report, wait for `READY`, deduplication, passages with their pages, `COMPLETE` and `INSUFFICIENT` evidence | [context build](../docs/flows.md#2-context-build) |
| 05 | [05_agents_runs_and_sharing.py](05_agents_runs_and_sharing.py) | a `RUN` note seen by its run and the run it spawned, not by a sibling or the user; explicit `AGENT_GROUP` sharing | [search and recall](../docs/flows.md#3-search-and-recall) |
| 06 | [06_verify_grounding.py](06_verify_grounding.py) | an answer verified claim by claim against its bundle (`supported`, `contradicted`), recorded as the judge's verdict on the run | [verify](../docs/flows.md#5-verify-and-grounding) |
| 07 | [07_tool_memory_and_hints.py](07_tool_memory_and_hints.py) | publish the catalog, record three successful runs, get the learned plan, and the next step's argument bound from the previous step's output | [tool hints](../docs/flows.md#6-tool-catalog-and-hints) |
| 08 | [08_feedback_and_review.py](08_feedback_and_review.py) | `api_key` mode: onboard, issue a service key, a user's vote waits in the review queue, the admin approves, the memory is reinforced; a run's own status applies at once | [feedback and review](../docs/flows.md#4-feedback-and-the-review-queue) |
| 09 | [09_admin_onboarding_and_keys.py](09_admin_onboarding_and_keys.py) | the bootstrap key onboards a tenant, the admin key issues a key scoped with `may_act_as`, creates a workspace and a member, and revokes the key | [api/tenancy.md](../docs/api/tenancy.md) |
| 10 | [10_background_jobs_and_summary.py](10_background_jobs_and_summary.py) | the jobs a write queues and their status; the 20th message rolls the thread's durable summary forward | [background jobs](../docs/flows.md#7-compaction-and-background-jobs) |
| 99 | [99_full_tour.py](99_full_tour.py) | every SDK method and every API route as one 14-step checklist: operations, conversation, documents, retrieval, memory, agents, the knowledge graph | all |

`_support.py` is the shared setup, `_serve.py` the single-process server for live runs. Which
call fits which job, beyond these: [docs/USAGE.md](../docs/USAGE.md).
