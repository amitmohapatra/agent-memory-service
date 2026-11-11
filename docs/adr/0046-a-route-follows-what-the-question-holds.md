# ADR 0046: A route follows what the question holds, not how it is worded

Date: 2026-11-11. Status: accepted. Amends ADR 0008 (pipeline order: the deterministic
`QueryRouter`).

## Context

The router decided a question's route with English wording patterns. Three of them stood for
no store of their own: `_TEMPORAL` ("when did", "since", "last quarter"), `_ENTITY` ("who
owns", "does X include") and `_DECISION` ("why did we decide", "rationale") each gave a label
(`TEMPORAL`, `ENTITY_RELATION`, `DECISION`) and the graph, which a question with none of those
words already gets as `GENERAL_SEMANTIC`. So the same need was searched differently by its
wording. "When did we ship the exporter?" was `TEMPORAL` (graph, one hop, the semantic bundle
cache off); "The exporter's ship date?" and the same question in German were
`GENERAL_SEMANTIC` (graph, three hops, cache on). One thing only the `TEMPORAL` label did:
the graph answered as of a date in the question, and only then. "Who worked for Acme on
2026-02-01?" got the relation that held that day; "Acme staff, 1 February 2026" and the German
wording got nothing, because the replaced relation is not current.

## Decision

A route says which stores a question searches, and is decided by what the question holds.

- `_TEMPORAL`, `_ENTITY` and `_DECISION` are removed, with their branches. Such a question is
  `GENERAL_SEMANTIC`: every dense space, BM25, and the graph by entity name, three hops.
- The graph answers as of a date the question names, for every question: `GraphStage` reads
  the date with `parse_date` (an ISO date, an English month with a year, a quarter with a
  year) instead of asking the route whether the question was worded as a time question. A
  question that names no date gets the current relations, as before.
- `_GLOBAL` (an overview request) and `_MULTI_HOP` / `_MULTI_HOP_NAMED` (several facts to
  gather) stay. Each is the only way to a search the general route does not run, and
  switching that search on for every question is not free (below).
- Not wording patterns, unchanged: identifiers, the references to this conversation, to the
  asker and to a passage, the "asks about the past" cue (ADR 0041) and the earlier-chat cues
  (ADR 0044).
- The three labels stay in the published `QueryType` enumeration until the next contract
  version and are no longer returned.

## Measurements

Private stack, 2026-11-11; "off" is the pattern switched off on main's code, everything else
equal. The pull request that carries this ADR has the full per-pattern table. The suite's
failures on both sides are tests that need a local stack this sandbox does not have.

| Check | main | the three off, date read from every question |
|---|---|---|
| Repository suite (unit, SDK, integration, contract, e2e, agent, security, eval, failure) | 4,676 of 4,679 pass | 4,681 of 4,683 pass (4 new tests) |
| Golden retrieval gate (`benchmark.retrieval`, 948 points): recall@20 / evidence-group recall, all and critical | 1.00 / 1.00 | 1.00 / 1.00 |
| The as-of question in three wordings (English cue, no cue, German) that get the relation that held then | 1 of 3 | 3 of 3 |
| LoCoMo, beliefs on, 1,986 questions: gold-turn recall; questions with all gold in the context | 90.64%; 1,813 | 90.64%; 1,813 (no gold turn gained or lost) |
| Held-out set, 131 questions: gold-turn recall; all gold in the context | 96.43%; 125 | 96.43%; 125 |
| Answers to the 11 questions whose context differed, 3 samples each | 26 of 33 | 26 of 33 |
| Harness history cases through the memory API (22 cases, 45 scored turns), run twice | 0.976, 0.976 | 0.976, 0.976 (with the multi-hop patterns off as well) |

What stays, and why:

- **`_GLOBAL`.** With it off, an overview question no longer gets the document and section
  summaries (`test_global_summary_and_the_durable_thread_summary` fails). Searching the
  summaries for every question instead is not free: on the golden gate recall@20 falls from
  1.00 to 0.949, evidence-group recall from 1.00 to 0.833, and cold context p95 rises from
  294 ms to 809 ms.
- **`_MULTI_HOP`, `_MULTI_HOP_NAMED`.** They switch on the per-person search. Given time to
  finish (5 s instead of the shipped 200 ms, which this host misses), on the 250 LoCoMo
  questions that match them it puts all gold turns in the context for 208 questions against
  205 without it (485 against 480 of 560 gold turns), at 895 ms against 596 ms median. Running
  it for every question adds about 200 ms at the median and lowers the questions complete in
  the first ten results from 231 to 215 on two conversations.

## Consequences

- The same need gets the same search in any wording and any language; the tests assert it for
  three needs in three wordings each, and the as-of question in three wordings.
- A question that names a date reads the graph as of that date even when it has no time cue.
  A relation that first held after the named date is not among its graph facts, as was
  already the case for a question with a time cue.
- A question that had a time, entity or decision cue now shares the semantic bundle cache
  of `GENERAL_SEMANTIC` questions (same question modulo polite framing, same revisions) and
  traverses three hops where a time or decision cue gave one.
- With `semantic_graph` off (not the default), the graph is read for identifiers and
  several-facts questions only; before, a question with an English entity, time or decision
  cue still read it.
- `query_type` (returned with `debug`) no longer says `TEMPORAL`, `ENTITY_RELATION` or
  `DECISION`, and `diagnostics.signals` has no `temporal`, `entity` or `decision` key. No
  endpoint, field or SDK call changed.
- The golden set's expected labels for the eight questions that were `TEMPORAL` or
  `ENTITY_RELATION` are `GENERAL_SEMANTIC`; their expected evidence is unchanged.
