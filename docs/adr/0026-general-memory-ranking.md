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

### ColBERT over the context key (mk3)

The late-interaction arm over the memory's context key too, fused at 2, with the session rule
at 0.3 (both chosen on LongMemEval; offline LoCoMo 0.790 -> 0.807, LongMemEval 0.888 ->
0.898). Through the service after a reindex (7,787 memories, no failures;
`benchmark/results/phase12/locomo_mk3.summary.json`): LoCoMo recall@10 **0.800** (single-hop
0.888, temporal 0.852, multi-hop 0.571, open-domain 0.511), @20 0.855, @50 0.910; context
build p50 130 ms, p95 279 ms on an otherwise idle host. Its cost, measured on the same host:

| | measured |
|---|---|
| store time per memory search (`search_arms`, 200 LoCoMo questions, with / without the arm) | p50 71.7 / 66.7 ms, p95 91.3 / 82.1 ms: **+5 ms p50, +9 ms p95** |
| query encoding | none: the query's token vectors are already computed for the first late arm |
| ingest, one more late encode per memory (399 context keys) | **30 ms CPU**, 17 ms wall per memory |
| disk, the `colbert_ctx` vector storage | 169.9 MB for 7,787 memories: **~22 KB per memory** (half precision, allocation included) |

A third key (the turn read with the turns either side) was measured and is not shipped: its
late arm lifted LoCoMo (0.807 -> 0.820 at x4) and cost LongMemEval (0.898 -> 0.864); at the
only weight that cost LongMemEval nothing (x1) it read +0.004.

## Consequences

- The context key's late vectors change the key layout to `mk3`: a reindex (`make reindex`).
  `preceding_source_id` is no longer read at query time (the projection drops it) but is
  still written.
- The coefficients, the two first stages and `memory_pool_k` are gone; there is nothing to
  refit.
- LoCoMo's remaining misses are multi-hop and open-domain questions whose evidence is spread
  over sessions. No ranking of single turns joins them; facts written at ingest can.
