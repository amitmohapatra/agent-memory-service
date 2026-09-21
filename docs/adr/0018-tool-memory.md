# ADR 0018: Tool memory — record, cache, learn, advise; never execute

**Status:** accepted · **Date:** 2026-09-15

## Context
Agents forget which tool worked for which task, call the same expensive endpoint twice inside
one run, and rediscover the same failure on every retry. `docs/TOOL_MEMORY.md` specifies the
memory for that. The open questions were where the boundary sits, what may be replayed from a
cache, how a chain is learned without a model inventing it, and how any of it is ranked.

## Decisions

**The service never executes a tool.** It registers descriptors, records invocations, serves a
cached output when the policy allows, mines chains, and advises. Execution stays in the agent
framework or behind Bifrost's MCP gateway. This keeps the service free of tool credentials and
side effects, and means the same records serve a LangGraph `ToolNode`, an ADK callback and a
Bifrost gateway call identically.

**Conservative policy defaults, widened only by an admin.** An unregistered or per-call-declared
tool is `deterministic=false`, `cacheable=false`, `side_effects=unknown`. A declaration made by
an agent in a request never widens a stored policy; only `POST /v1/tools` (since removed) under tenant admin
does. The failure mode this avoids is an agent talking itself into a cache hit for a tool that
writes.

**Replay requires deterministic ∧ cacheable ∧ side effects in {none, read}.** Serving a cached
answer suppresses a call the agent would otherwise make, so the bar is all three, not one. Every
hit carries `cached=true` and `age_seconds` so a caller can decide to call anyway.

**The cache key is (tenant, scope anchor, tool version, args_hash), and the hash covers the full
arguments — redacted ones included.** Only the redacted copy is persisted. Hashing the redacted
form instead would collapse two callers with different credentials onto one entry and serve one
caller's output to the other; the hash is one-way, so the secret is not recoverable from it. A
scope whose anchor is absent in the current context is never written or read, rather than being
silently widened to the tenant.

**Records are idempotent on (run, step, tool, args_hash).** A replayed graph step re-reads its
row. Without this, retries inflate the very statistics the advice is computed from, and a flaky
tool would look more popular the more it failed.

**Visibility is the memory rule, not a new one.** Tool records get their audience keys from the
same helper canonical memories use, filtered store-side before any ranking. An agent's tool
chatter reaches a user or a group only when explicitly shared; a hand-off is visible to the
child run and no further.

**Chains are mined, never invented.** A `feeds` edge exists when a value in step *i*'s output
reappears in step *j*'s arguments, and it keeps both field paths — that pair *is* the data-flow
binding a plan later uses. `followed_by` records adjacency. Values shorter than three characters
are ignored, otherwise every `1` and `true` in a payload looks like data flow.

**A procedure is the longest chain most successful runs follow, not the best-supported one.**
Support alone is the wrong objective: every prefix of a sequence is itself a prefix, so a
one-step path always has at least as much support as the chain it begins, and maximising support
truncates every procedure to its first tool. The rule is coverage first, length second: keep the
paths at least `MIN_PATH_COVERAGE` (0.5) of successful runs follow, with a floor of two runs so
one accident is never a procedure, then take the longest. Consecutive calls to the same tool
collapse to one step, because a retry is an adjustment inside a step, not an extra step.
*This was found by the gate*: the first implementation maximised support and scored 0.90 on
next-step hit rate because a single run with a retry truncated a three-step chain to two; the
corrected rule scores 1.00.

**Mem^p update rules.** Validation: only a run labelled successful contributes a procedure.
Adjustment: a failed step is corrected in place, the correction recorded on the edge, never
duplicated into a competing procedure. Decay: a procedure unused for 60 days, or whose success
rate falls below 0.5, is archived rather than deleted.

**Outcome labelling.** A run is successful when it is explicitly labelled through
`POST /v1/runs/{id}/outcome`, or — as a weak positive only — when it is older than
`weak_positive_after_hours` (24h), had no failing call and nobody corrected it. Unlabelled and
recent means *unknown*, not *successful*.

**Ranking formula**, deterministic and explainable:

```
score = 0.5 · procedure_match + 0.3 · (success_rate · min(1, invocations/5)) + 0.2 · recency
        − 0.4 if the tool failed on a recent call in this scope
recency = 0.5 ^ (age_days / 14)
procedure_match = 1.0 first step · 0.8 later step · 0.0 not in the procedure
```

The `min(1, invocations/5)` damping is why one lucky call does not outrank a long record.
`next` additionally drops candidates whose preconditions cannot be met from the trajectory's own
outputs, and sets `stop=true` when the matched path has no successor.

**Suggestions are closed over `available_tools`.** Nothing is ever named that the caller did not
declare, in `suggest`, `next` or `plan` — so a plan that needs a tool this agent lacks is
returned as invalid with the reason, rather than as an unusable chain.

**Bifrost paths.** Explicit execution is the recommended path: the app gets tool calls from the
model, and `client.tools.execute(call, executor)` does lookup → the caller's executor → record,
so records, chains and suggestions work identically whether the executor is a local function or
a POST to Bifrost's `/v1/mcp/tool/execute`. In agent mode Bifrost executes tools itself and the
app never sees the calls, so records must come from Bifrost's side and cache lookups are not
possible; that limitation is documented rather than worked around. LLM involvement is confined
to optional procedure abstraction behind the `tool_reflection` flag, accepted only when every
step exists in the registry and every binding resolves against a real trajectory.

## Consequences
- `tools`, `tool_invocations` and `run_outcomes` (migration `0007_tool_memory`) are new
  canonical tables; everything derived is rebuilt from them, so only they must be durable.
- A new hard gate (`benchmark/results/tool_gate.json`, `tests/eval/test_tool_gate.py`) with
  thresholds: suggestion hit rate ≥ 0.95, next-step hit rate ≥ 0.90 on held-out trajectories,
  plan validity = 1.00, and zero cache, isolation or undeclared-tool violations.
- Cold start returns empty advice and the agent behaves exactly as it would without memory.
  Nothing is guessed.

## Status of the adapters
The API, SDK (`client.tools.*`, `client.runs.outcome`) and gate are in place. The framework
adapters that call them — the LangGraph tool wrapper and Bifrost execute node, ADK tool
callbacks, the CrewAI wrapper and the MCP server's tool verbs — and the cross-adapter
conformance suite are tracked separately and are not yet complete; see the report in the pull
request for exactly what is and is not built.
