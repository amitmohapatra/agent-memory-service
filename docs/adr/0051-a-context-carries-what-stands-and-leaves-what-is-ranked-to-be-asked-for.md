# ADR 0051: A caller that searches on demand gets a context with what stands, and what is ranked is left to be asked for

Date: 2026-11-27. Status: accepted; not measured against the pushed context yet (see
Measurements).

## Context

A context holds two kinds of thing.

- **What stands for the person whatever is asked:** their profile, the rules and
  preferences that apply, the conversation and its summary, the learned skills and the tool
  guidance.
- **What retrieval ranks for this one request:** memories, earlier values, graph facts,
  document passages and their summaries, what the assistant answered in earlier
  conversations.

The second kind is chosen by being *near* the request (ADR 0048). In the hand-counted run
that ADR opens with, 1% of what was attached was acted on and 71% was about something else
or worked against the request. A model check removes what does not help where a model can be
reached on the read; it costs a call per context, and it judges before the agent has done
anything.

An agent with a search tool can make that judgement later and better: after reading the
request, in its own words, as often as the task needs. That is how the tools the team
compared against work - an always-loaded instruction file and a memory read on demand; a
core memory in the prompt and an archive behind a search call. Published measurements point
the same way for long histories: on LongMemEval, answers from the evidence sessions alone
scored 0.870 against 0.606 with the full history in the prompt.

The service offered no such split. A caller took the whole context, or built none and lost
the profile, the rules and the preferences with it.

## Decision

`/v1/context` takes `recall` (default `true`: the context as it was).

- **`recall: false` carries what stands** - the profile blocks, the
  standing rules, preferences and what the person said of themselves that apply to the
  request, the conversation window and the thread's summary, the learned skills, the tool
  hints - **and leaves out what is ranked** (`modules/context/demand.py`): `memories`,
  `history`, `graph_facts`, `knowledge`, `summaries`, `earlier_chats`.
- **Everything left out is one search away.** `/v1/recall` and the `memory_search` agent
  tool return these as kinds (`memory`, `chunk`, `summary`, `episode`, `message`, `fact`),
  by meaning or by time (`order`).
- **The evidence status is kept.** It says what memory holds for the request, not what the
  context carries: `INSUFFICIENT` tells the caller there is nothing to search for, anything
  else that a search is worth making. The memories found join the evidence the bundle does
  not carry (`unused`), which `/v1/verify` still checks an answer against.
- **The same build.** The standing sections read the request's vectors and the tool hints
  read what the ranking found, so retrieval still runs; what changes is what is carried.
  The diagnostics count what was left, by section (`recall`).
- **The context check (ADR 0048) reads what is carried.** With nothing ranked in the
  context, it is shown the standing lines and the learned skills alone.
- **Cached apart.** The choice is part of the bundle's key: a context built one way never
  answers for the other.
- **One request field, no setting.** The choice is the caller's, per call: an agent that
  has the search tool makes it; a caller without one keeps the default.

## What it does not change

- The default. A caller that does not send `recall` gets the context it always got.
- The standing sections' own selection (ADR 0041): a rule still comes when it applies to the
  request.
- Documents named by `document_ids`: they restrict document knowledge, and with
  `recall: false` none is carried. Their passages are a search (`kinds: ["chunk"]`).
- The build's cost on the service. Retrieval runs either way; this is about what a prompt
  holds, not about the read's latency.

## Measurements

None against the pushed context yet. `tests/unit/test_context_demand.py` and
`tests/integration/test_context_on_demand.py` prove what is carried and left, that the
status and the checked evidence are kept, that the two forms are cached apart and that the
left-out memory is found by a search. What decides whether an agent should use it is
answer quality with a real model: LoCoMo's J through an agent, pushed against on demand, by
category - multi-hop is where a search the model must think to make is most at risk.

## Consequences

- An agent that searches sees an old memory only if its model decides to search. What must
  hold on every request belongs in what stands: a rule the person gave, or a block a person
  writes.
- Agents that answer from uploaded documents rely on passages being attached; for them the
  default remains the right choice until a measurement says otherwise.
- The prompt of an on-demand agent has the same shape on every request: the same sections,
  in the same order, with nothing that depends on how near something else happened to be.
