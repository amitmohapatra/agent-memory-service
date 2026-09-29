# Phase 9 results, 2026-09-29

Four measurement items and three code items. **No product default changed**: every arm ran
against a flag or a fitted file, and the three retrieval items each failed their gate, could
not be measured, or proved to be a no-op. The two hazards the arms exposed are fixed, and so
are three service defects the agent API suite found. Every number names its artifact under
`benchmark/results/phase9/`.

Host: 4 cores, 8 GB, load 13 to 40 throughout. That matters for one gate, and it is said where
it does.

## Item 1 - fitted fusion weights (D6 step 2): quality up, gate not met

Fitted over Phase 7's rank dumps: bm25 2.0, dense_en 0.5, dense_ml 2.0, at k=1. Every one of
the top ten weightings down-weights the English specialist, which is the same story XQuAD told
from the other direction.

Three arms of 1,986 questions, no failures, corpus reused. Paired on the same questions:

| comparison | n | recall@10 | multi_hop@10 |
|---|---|---|---|
| halving the candidate depth alone | 1525 | +0.0030 | +0.0041 |
| the fitted weights, at the same depth | 1525 | +0.0143 | +0.0176 |
| both, against the Phase 7 baseline | 1525 | +0.0173 | +0.0217 |

Better on 55 questions, worse on 20, for +7.8 ms p50 and +181 ms p99. Halving the depth is
free.

**Why it is not promotable.** The gate asks for evidence_all_recall at or above 0.99 at the
halved depth, and 0.99 is unreachable by the instrument the gate names: the ID-keyed
all-gold-present measure tops out at 0.7587 at depth 50 on this corpus, against a 0.7521
baseline. The gate cites a different metric from the one it was written for, so no change can
meet it. The p99 half is **unmeasured rather than failed**: both Phase 9 halved arms are
equally slower than Phase 7's, which points at the host, not at the weighting.

The weights are a fitted file, not a default. Adopting them needs a gate naming a metric it
can reach.

Artifacts: rrf_weight_fit_ensemble.json, locomo_source_weighted_halved.summary.json,
locomo_source_equal_halved.summary.json.

## Item 2 - entity routing (D6 step 3): a no-op, since fixed

query_entities anchors 1,442 of 1,540 answerable questions (93.6%) and 256 of 282 multi-hop
ones, on 254 distinct names. Every one of the 25 most-asked anchors matched **zero** points as
the query spells it; 15 matched hundreds the moment the name was scoped, john against
user:john being 0 against 1,153.

The write path indexed a memory's subject as stored, scoped; the query path reads names out of
a question and can only produce the bare form. The arm confirmed it end to end: multi_hop@10
moved -0.0002, complete@50 was unchanged, and 1,515 of 1,524 questions were untouched.
Turning the flag on bought a per-query prefetch with a provably empty result set.

Fixed after the phase in 3ced4b1: both forms are indexed, because the scheme is not part of the
name and the render path already stripped it. **The flag is still off.** Whether the prefetch
earns its query now has to be measured, which it never could be before.

Artifacts: anchor_reach.json, locomo_source_entity_prefetch.summary.json.

## Item 3 - dated aggregates (D6 step 4): not measurable in this phase

The corpus holds **0 BELIEF and 0 ENTITY_SUMMARY rows out of 7,785**. Consolidation is off by
default and was off in both Phase 7 arms, so nothing was ever minted. That also explains a
detail nobody had questioned: the Phase 7 artifact's lineage summary is byte-identical to its
summary.

With consolidation on, this corpus would mint 198 beliefs over slots holding 6,788 memories,
the widest gathering 334 memories under one predicate - not the handful of dated values the
design assumes. The target bucket is real: 353 questions, 243 of them multi-hop, 276 still
incomplete at depth 10. Measuring it needs a consolidation re-ingest on its own database plus
a judged arm, because a belief's evidence is memory-typed and the aggregate render changes
text rather than bundle membership.

Artifact: belief_reach.json.

## Item 5 - does the cross-encoder rank better? No, and the question is closed

All 282 multi-hop questions, 14,100 pairs, ettin-reranker-17m, reordering the bundle each
question already returned:

| order | recall@10 | all gold @10 | recall@50 |
|---|---|---|---|
| fused | 0.3887 | 0.1809 | 0.6219 |
| reranked | 0.2512 | 0.0816 | 0.6219 |
| oracle | 0.6219 | 0.3688 | 0.6219 |

35% worse at depth 10. It is **not a broken instrument**: gold carriers score above the rest
on 89% of questions, with a mean separation of 0.19. It ranks, and it ranks worse than the
fusion does. The oracle row says 23 points are genuinely available at depth 10 to something
that ranks well, and this teacher gives back 14 of them.

So: **no reranker, and no distillation.** The competitive review's "missing a cross-encoder"
line is answered by a measurement instead of an assumption. The older claim that reranking
hurts quality, which rested on 2 conversations and 16 multi-hop questions, now has 282
questions and a teacher verified to be working behind it. Boundaries: one teacher, one
bucket, reordering only.

Artifact: rerank_offline_ettin17m_multihop.json.

## Items 6, 7 and 4 - the code half

**Item 6, all 72 operations through the SDK.** Coverage is recorded rather than declared: each
test claims the operations it drives and a collection-time gate asserts every operation is
claimed, proven to bite by deleting one claim. Three service defects came out of it, all
fixed:

* A service-derived Idempotency-Key was compared against the whole request body, so an adapter
  reconnecting to a thread it had already created was answered 409 for a header it never sent,
  on routes documenting the opposite. A client's key now compares the body; a derived key
  compares what it was derived from.
* The e2e and agent truncation list was missing five tables with no foreign key to tenants, so
  a cascading truncate never reached them and webhook subscriptions, feedback, briefs and agent
  credentials leaked between tests. A tenant with one subscription listed six.
* The metrics endpoint was unreachable through the SDK, whose transport decoded JSON
  unconditionally.

**Item 7, typed connections** and **item 4, question decomposition** (behind the existing
use_llm flag, never a default) are built and unit-tested, and **unmeasured**. Their arms need a
model-enabled run, and the ID-keyed source harness hard-disables the LLM by design, so they
need the judged path or a new one. That is the first thing a next phase should measure.

## Two hazards the arms exposed, both fixed

* One Qdrant read exceeding the 5 s deadline killed a whole 1,986-question arm on its first
  question: a deadline is neither a lost connection nor an empty response, so the store's retry
  did not apply. The harness now records the failed question and continues, with the count in
  the artifact so a degraded run cannot be published as a clean one. It earned its place
  immediately: the entity arm lost one question and completed 1,985.
* Reusing a corpus silently became a destructive re-ingest whenever the ledger key differed.
  Now refused, naming the field that differs, with an explicit opt-in.

## Budget

current_usage on the governing key: **0.006426 USD before, 0.006426 USD after. Nothing spent.**
Every arm ran with no paid LLM calls; none reached the gateway. The phase was allowed 3 USD.

## What a next phase should carry

1. Arms for items 7 and 4, which need a judged path the source harness cannot provide.
2. A gate for D6 step 2 naming a metric it can reach, then a decision on the fitted weights:
   +0.0143 recall@10 and +0.0176 multi_hop for +8 ms p50 is a real gain going unused.
3. The entity prefetch, now that its two halves can meet.
4. Consolidation on its own corpus, the only way item 3 becomes measurable.
5. The 23 points the oracle says are available at depth 10 to something that ranks better than
   RRF and is not this cross-encoder.
