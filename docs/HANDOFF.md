# Handoff — agent-memory-service, 27 September 2026

Two agents worked this repository concurrently: Codex built, this one measured. That split is
deliberate and worth keeping — whoever builds a fix should not be the one who decides it worked.

## Run it

```bash
make dev-up          # postgres + qdrant + the gateway
make migrate
make unit            # 736 passed, 1 skipped on a clean clone
make verify-quick    # lint, typecheck, unit + contract, no Docker
```

`.env` is gitignored. Copy `.env.example` and edit it. **If `make unit` fails with six
environment-guard errors, your `.env` is the cause** — those tests assert the shipped defaults
and any local override breaks them. A clean clone is green.

Benchmarks need the models: `make models`. Then `make bench-locomo` (~45 min, no model credits).

## What is true about accuracy

    72.92%   judged answer correctness, full LoCoMo, our strict ruler, flash reader
    81.17%   the SAME answers under Mem0's published rules
    92.50%   what Mem0 publishes

    complete_evidence_in_candidates@50   0.7188      every gold turn retrieved
    multi_hop complete@50                0.3333      the defect
    p99                                  300.8 ms    at load_per_core 0.43

The 81.17% is not a separate system. It is our answers scored by Mem0's own criteria — one item
of a list, dates within 14 days, durations within 50%, same referent — applied mechanically to
the predictions already on disk. Most of the published gap is the scoreboard.

## What was measured, and what it bought

| change | delta | |
|---|---|---|
| judge budget un-gated | **+3.6** | the only real accuracy gain; it was a scoring bug |
| fix D — fusion constant + tie ordering | +0.07 | |
| fix A — source lineage | +0.07 | |
| fix B — companion cap | +0.13 | |
| fix C — graph returns memories | +0.07 | hydrates 603 questions, none carrying gold evidence |
| head widening 10 → 30 | +0.55 | |
| GROUP_BY_SOURCE | 0.00 | provably inert |
| cross-encoder reranker | worse | 6x latency, every metric down |
| consolidation / reflection | negative | 73 wins, 78 losses |

Five isolated arms, own database and Qdrant namespace each. **`multi_hop complete@50` reads
0.3369 in every single arm, identical to four decimals.** Retrieval plumbing was not the
constraint.

## Where the remaining loss actually is

Of 416 wrong answers, in three non-overlapping buckets:

    226  FRAGMENTATION  evidence in the bundle, split across memories       ceiling +14.7
    109  READER         complete evidence retrieved, wrong answer anyway    ceiling  +7.1
     81  ABSENT         never retrieved at any depth                        ceiling  +5.3

All four retrieval fixes worked on the smallest bucket. The 109 reader failures break down as
33 date arithmetic, 30 wrongly abstained with the evidence in context, 9 partial lists — so
roughly 63 are prompt-addressable, about 4.1 points, and **nothing has been built for them that
has been measured**.

126 questions (8.2%) are already answered correctly *without* complete evidence, so retrieval
gains on those convert to exactly zero.

## Merged, and why

`91a2a46` merges fixes A–D. They land because they are **correct**, not because they are fast:
`rrf_k=60` had never reached the store, the graph arm returned zero memory candidates while
`Relation.memory_id` was written and read by nothing, and a memory's "source turn" was its own
record id. Graph-derived memories pass the same `visibility.allows()` key intersection as a
directly retrieved one — verified, not assumed.

## NOT merged, deliberately

| branch | why it is held |
|---|---|
| `codex/retrieval-stack-20260927` (`a121dc7`) | wider retrieval programme, unmeasured |
| `codex/write-path-stack-20260927` (`f988c26`) | adds migrations 0009–0011 and an LLM call per message at ingest; **changes the corpus**, which is the input to every number |
| `codex/evaluation-archive-20260927` (`f842c58`) | evaluation source and artifact inventory |

Five Tier 2/3 builds (reflection wiring, temporal rendering, typed relations, entity
canonicalisation, source reconstruction) were built and adversarially reviewed. **None merged.**
The reflection build renders *fabricated dates* into the prompt — `observed_at` is the ingest
clock, so lines appear stamped with today's date between 2023 turns, on exactly the axis where
33 of the 109 reader failures live. Its BELIEF is a 7-value sliding window, not an aggregate:
12 preferences in, the last 7 kept. The reviewers' expectations for the other four range from
0.05 points to "currently unmeasurable with the instrument shipped".

## Instrument fixes that matter more than they look

Four numbers were published and retracted today, each because a metric was read as evidence
before checking it could respond to what changed:

    +12.63 → +0.55      evidence_in_head imported its head size from the constant under test
    +8.6   → 0.00       evidence_reconstructed grouped by an id that was the memory's own
    9.6-26ms → 0.8ms    a dedup cost inherited from an audit and never re-measured
    p99 484.6 → 300.8   load contamination; we were at target and did not know

So: **every rank metric is now also reported at a fixed ladder (10/20/30/50)** the renderer
cannot influence; `complete_in_candidates` is scored at a fixed depth rather than at whatever
the bundle returned; and **host load is recorded in every artifact**.

The rule: *no benchmark delta is accepted unless the metric's implementation is shown to be
sensitive to the exact behaviour that commit changed.*

## Known blockers

1. **Sustained LLM runs fail.** DeepSeek is out of credit; OpenRouter answers single calls but
   429s under sustained load and trips the adapter circuit breaker — a judged rejudge died at
   800 of 1,986. Every judged arm and the whole oracle-reader experiment needs ~2,000 paced
   calls. This is the binding blocker on the reader work, which is where the headroom is.
2. **Both fragmentation ceilings are token-overlap artifacts** (+13.2 and +21.7, and they
   disagree). A metric keyed on gold source-turn IDs is needed before anyone builds against
   that bucket again.
3. **OpenAI structured output** needs every property in `required` plus
   `additionalProperties: false`. Fixed in `JUDGE_SCHEMA`; the same trap applies anywhere else
   a schema is sent.

## If you pick this up

Read `docs/STATE-2026-09-25.md` — 30 sections, including the retractions with their mechanisms,
the factor checklist, and what every experiment showed. Then `docs/CODEX-STACK-HANDOFF-2026-09-27.md`
for the unmerged stacks and their database isolation requirements.

The next thing worth doing is **not** another retrieval fix. It is the 143 questions that abstain
with the evidence in front of them, and a working LLM path to measure the attempt.
