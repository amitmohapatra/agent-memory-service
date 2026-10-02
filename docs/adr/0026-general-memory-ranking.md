# ADR 0026: A general memory ranking instead of a learned one

Date: 2026-10-02. Status: accepted. Supersedes decision 3 of ADR 0025 (the learned ranking);
the late-interaction arm and the two keys stay.

## Context

ADR 0025 ranked memories by a logistic regression over twenty-six features, fitted on
LoCoMo. Scored leave-one-conversation-out it read 0.841 recall@10, and 0.829 through the
service. The question that number cannot answer is whether the coefficients describe
conversations or describe LoCoMo. It was tested directly, offline, on LoCoMo and a 100-question
stratified sample of LongMemEval-S (seed 7), each corpus scored by coefficients fitted only on
the other:

| ranking | fitted on | LoCoMo recall@10 | LongMemEval recall@10 |
|---|---|---|---|
| learned, 26 features | LoCoMo | 0.841 (its own corpus) | **0.852** |
| learned, 26 features | LongMemEval | **0.677** | 0.902 (its own corpus) |
| equal-weight reciprocal-rank fusion, nothing fitted | - | 0.742 | 0.868 |

On the corpus it had not seen, the learned ranking lost to plain fusion both ways. It had
learned each corpus's shape (how big a session is, how evidence clusters, who names whom),
not what makes a memory relevant. A deployment's own conversations are a third corpus.

## Decision

The memories are ranked by `modules/retrieval/memory_ranking.py`:

1. **Weighted reciprocal-rank fusion** of the seven arms ADR 0025 reads, `1 / (K + rank)`,
   `K = 10`, every arm at weight one except the late-interaction arm at `LATE = 6`.
2. **Session**: every memory gains `SESSION = 0.2` times the best fused score among the
   fused top 50 in its session (the day it was observed on).
3. **Speaker**: a question naming a person lifts that person's memories by one rank-1 unit.
4. **Time**: a "when" question lifts memories that name a time by one rank-1 unit.

**How the values were chosen.** Each knob was chosen on one corpus and scored on the other,
and kept only where both agreed: both corpora independently chose the late arm at six, and
`K` at ten; the session rule helps both. The speaker and time rules cannot be tested on
LongMemEval (its turns are "user" and "assistant", and it asks few "when" questions), so they
are round values, not tuned on LoCoMo. A rule that helped one corpus and cost the other (the
neighbour lift ADR 0025 used) is not shipped. Offline, per rule, LoCoMo / LongMemEval:
base 0.769 / 0.883; + session 0.782 / 0.888; + speaker 0.778 / 0.883; + time 0.772 / 0.883;
+ neighbour lift 0.777 / 0.874 (dropped).

## Evidence

Through the service, LoCoMo, all ten conversations, 1,536 answerable questions, the corpus
re-used from Phase 11 (`benchmark/results/phase12/locomo_general.summary.json`): recall@10 **0.778** (single-hop
0.853, temporal 0.843, multi-hop 0.562, open-domain 0.537), @20 0.847, @50 0.907, @100 0.935;
context build p50 151 ms, p95 336 ms (192 / 345 with the learned ranking).

That is lower than the 0.829 the learned ranking showed on the corpus it was fitted on, and
it is the number that carries to a corpus nobody fitted it on.

## Consequences

- No reindex: the arms and the payload are unchanged; `preceding_source_id` is no longer
  read at query time (the projection drops it) but is still written.
- The coefficients, the two first stages and `memory_pool_k` are gone; there is nothing to
  refit.
- Offline, two more keys read the same way add to it without fitting: the late-interaction
  arm over the memory's context key, and a window key (the turn read with the turns either
  side). They need new vectors and a reindex and are measured before they ship.
- LoCoMo's remaining misses are multi-hop and open-domain questions whose evidence is spread
  over sessions. No ranking of single turns joins them; facts written at ingest can.
