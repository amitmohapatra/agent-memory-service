# Phase 11 results, 2026-10-01: late interaction, two keys, and a learned memory ranking

The change is ADR 0025: a ColBERT arm (`mxbai-edge-colbert-v0-32m`) on every collection,
every memory indexed under two keys (its own text, and its text read after the turn it
answers), and the memories ranked by a logistic regression over twenty-six features read
from what the seven arms return. No cross-encoder; CPU only.

Every number below is through the service on the real models (`representative: true`),
on a 4 vCPU 2.1 GHz Xeon with two ORT threads per model (`WEB_CONCURRENCY=2`, the share
three workers get on the 8 vCPU target).

## Conversational memory: LoCoMo

`benchmark.native_source_retrieval`, all ten conversations, 1,986 questions, 0 failures.
Artifact: `benchmark/results/phase11/locomo_source.summary.json`.

| bucket (n) | @10 | @20 | @50 | @100 |
|---|---|---|---|---|
| all answerable (1536) | **0.829** | 0.864 | 0.902 | 0.931 |
| single-hop (841) | 0.911 | 0.936 | 0.959 | 0.972 |
| temporal (321) | 0.879 | 0.907 | 0.932 | 0.959 |
| multi-hop (282) | 0.608 | 0.683 | 0.781 | 0.849 |
| open-domain (92) | 0.582 | 0.608 | 0.654 | 0.704 |

Phase 10's service run read 0.731 at @10 (with a stand-in English encoder; the shipped
ensemble's offline equivalent is 0.724). Sequential context-build latency: p50 192 ms,
p95 345 ms, p99 412 ms.

The coefficients were fitted on LoCoMo; leave-one-conversation-out (no conversation scored
by a model that saw it) the same ranking reads 0.841 offline, so the in-sample service
number is not flattered by the fit.

**The cross-encoders that are not shipped.** The same ranking with two cross-encoders as
further features (`mmarco-mMiniLMv2-L12-H384-v1`, `mxbai-rerank-xsmall-v1`, the top 20 of
each first stage) was run through the service over six conversations before it was dropped:
0.822 at @10 against 0.835 for the shipped ranking over the same 883 questions; profiled,
a context build took p50 909 ms with them (680 ms of it reranking) against 192 ms. Offline they lifted a rank-only model from 0.815 to 0.842; the
arms' own scores, sessions and speakers lift it to 0.841 without them.

## Documents: SciFact

`benchmark.runtime_retrieval --suite scifact`, 5,183 abstracts, 300 queries.
Artifact: `benchmark/results/phase11/runtime_scifact.json`.

| | nDCG@10 | recall@10 |
|---|---|---|
| Phase 7 (equal weights, no late arm) | 0.7533 | 0.8926 |
| fitted weights, no late arm | 0.7464 | 0.8786 |
| **fitted weights + late arm at 2.0** | **0.7564** | **0.8919** |
| M2 gate | 0.7557 | 0.8926 |

nDCG@10 passes the gate; recall@10 misses it by 0.0007 (a fifth of one query's evidence).
The late arm adds 0.010 / 0.013 over the same fusion without it. Query: encode p50 14 ms,
search p50 16 ms.

Until this phase the harness wrote its records by hand with dense and BM25 vectors only and
searched without the late vectors, so the first two runs measured no late arm at all; both
fixes are in `benchmark/runtime_retrieval.py`.

## Multilingual: XQuAD, 12 languages

`benchmark.runtime_retrieval --suite xquad`, 14,280 questions.
Artifact: `benchmark/results/phase11/runtime_xquad.json`. Same-language recall@10:

| ar | de | el | en | es | hi | ro | ru | th | tr | vi | zh | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.988 | 0.998 | 0.993 | 1.000 | 0.998 | 0.992 | 0.998 | 0.999 | 0.995 | 0.992 | 1.000 | 0.998 | **0.996** |

Gate 0.98: passes. The English-trained late arm costs nothing here (0.9963 without it).

## Throughput

In-process open-loop load on the context builder (Poisson arrivals, LoCoMo questions salted
so the bundle cache always misses). Artifact: `benchmark/results/phase11/load_in_process.json`.

| | sustained | p50 | p95 | CPU-s per request (service + Qdrant + PostgreSQL) |
|---|---|---|---|---|
| one process | 8.9 rps | 365 ms | 814 ms | 0.196 |
| two processes, 4 vCPU | 14.4 rps, 0 errors | ~400 ms | ~950 ms | ~0.19 |

A memory search with the cross-encoders cost 2.4 CPU-seconds; without them 0.17 in the
service process. On the 8 vCPU target that is a CPU ceiling near 40 rps and three workers at
~9 rps each: 20 rps is within both, at about half the CPU. A run on the 8 vCPU box itself is
what confirms the p95 there.

What it took: the arms return ids and scores only and each point's payload is read once
(~350 points rather than ~800 hits a query - payload conversion from protobuf was most of a
search's Python time), and the Qdrant client no longer walks every query vector looking for
inference objects it is never sent.

## LongMemEval: does a ranking fitted on LoCoMo transfer?

`benchmark.longmemeval_retrieval`, LongMemEval-S (cleaned), a stratified sample of 18
answerable questions (seed 7), each haystack (~50 sessions, ~500 turns) ingested from an
empty store. Same coefficients as LoCoMo, unchanged. Artifact:
`benchmark/results/phase11/longmemeval_retrieval.json`.

| type (n) | turn @10 | turn @20 | session @10 | session @20 |
|---|---|---|---|---|
| all (18) | **0.843** | **0.926** | **0.944** | **1.000** |
| knowledge-update (3) | 1.000 | 1.000 | 1.000 | 1.000 |
| temporal-reasoning (5) | 1.000 | 1.000 | 1.000 | 1.000 |
| single-session-user (2) | 1.000 | 1.000 | 1.000 | 1.000 |
| single-session-assistant (2) | 1.000 | 1.000 | 1.000 | 1.000 |
| single-session-preference (1) | 0.000 | 1.000 | 1.000 | 1.000 |
| multi-session (5) | 0.633 | 0.733 | 0.800 | 1.000 |

Every question's evidence sessions are in the first twenty memories; the miss is turns of
multi-session questions, whose evidence is spread over several sessions. Context build p50
131 ms, p95 171 ms. Eighteen questions is a sample, not an estimate to the point: the
harness and `make bench-longmemeval-retrieval` run all 470 where the hardware allows.
