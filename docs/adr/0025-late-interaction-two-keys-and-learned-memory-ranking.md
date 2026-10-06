# ADR 0025: Late interaction, two keys per memory, and a learned memory ranking

Date: 2026-10-01. Status: accepted. Supersedes the `colbert` part of ADR 0012. Decision 3
(the learned ranking) is superseded by ADR 0026: its coefficients did not transfer between corpora.
Results: `benchmark/reports/PHASE11-RESULTS-2026-10-01.md`.

## Context

After ADR 0024 the memories were ranked by one weighted RRF over BM25 and two dense spaces,
each memory indexed once with the turn it answers prepended. Measured on LoCoMo's 1,986
answerable questions, offline on the same components: recall@10 0.724 (single-hop 0.871,
temporal 0.750, multi-hop 0.385, open-domain 0.328). The research round behind this ADR
tried every CPU-feasible, non-Chinese component and combination it could find (dense
challengers, learned sparse, two ColBERT checkpoints, five cross-encoders, blends,
cascades, query-side expansions, learned fusion), each one first on a screen, then on all
ten conversations, scored leave-one-conversation-out wherever anything was fitted.

ADR 0012 removed two of those components. One reason no longer holds and one still does:

- **ColBERT** was removed because its weights could not be downloaded in that environment,
  not because it was measured. mixedbread's `mxbai-edge-colbert-v0-32m` (Apache-2.0, 32M,
  64-dim tokens) has an official ONNX export.
- **The reranker** was measured worse as a *replacement* for the fusion. Used as features
  of a learned fusion, two cross-encoders helped this time - and cost 2.4 CPU-seconds a
  query, which a CPU-only service sized for 20 requests a second cannot pay (below).

## Decisions

1. **A late-interaction arm on every collection.** `colbert` is a Qdrant multi-vector
   (MaxSim, half precision, on disk, no HNSW graph) written for every chunk, summary,
   episode and memory. At query time it *rescores the union of the other arms'
   candidates* (a nested prefetch), so its cost is bounded by the arms' depth and not by the
   tenant's size. The encoder is the publisher's FP32 ONNX graph, tokenised as the
   checkpoint's `onnx_config.json` says PyLate does; it scores within 0.011 of PyLate with
   identical top-10s. The int8 graph changed every top-10 checked and is not shipped.
   Documents fuse it at weight 2.0 (`RetrievalSettings.hybrid_weights`).
2. **Two keys per memory.** The memory's own text (`memory_index_text`) and the same text
   read after the turn it answers (`memory_context_text`), each with its own dense vectors
   per space and its own BM25 vector (`dense_en_ctx`, `dense_ml_ctx`, `bm25_ctx`). Indexing
   only the second key lifted single-hop questions and cost multi-hop and temporal ones the
   turn's own words; both keys keep both. A memory with no preceding turn has equal keys and
   is encoded once.
3. **The memories are ranked by a learned fusion** (`modules/retrieval/learned_fusion.py`):
   all seven arms read unfused in one `query_batch_points`; two weighted-RRF first stages
   with a neighbour lift; and a logistic regression over the top 30 of each, from
   twenty-six features read off what the arms returned - each arm's rank and own score,
   both first stages' ranks, the neighbouring turns' scores, how many arms put the
   candidate in their top ten, its session's best score and share of the head (a session
   is the day it was said on), whether the question names the person it is about, and
   whether a "when" question meets a candidate that names a time. The coefficients are
   constants in the module, fitted on LoCoMo with the standardisation folded in.
4. **What goes.** The fixed neighbour lift over the fused pool (`adjacent_turn_weight`) is a
   feature of the learned fusion now and is removed. The generic hybrid search remains for
   documents, summaries, episodes and the per-person memory searches of multi-hop questions.
5. **No switch.** The arm, the keys and the ranking are the service; the hermetic suite
   runs the same path with a stand-in late encoder (`HashLateInteraction`) that loads
   nothing, labelled non-representative as the hash embedding is.
6. **No cross-encoder.** The service is CPU-only and sized for 20 requests a second; see
   Evidence. Nothing in it reranks.

## Evidence

Offline, all of LoCoMo, leave-one-conversation-out (no conversation scored by a model that
saw it), recall@10:

| | all | single | temporal | multi-hop | open |
|---|---|---|---|---|---|
| hybrid search before this ADR | 0.724 | 0.871 | 0.750 | 0.385 | 0.328 |
| ranks only (14 features) + two cross-encoders | 0.842 | 0.935 | 0.886 | 0.618 | 0.518 |
| ranks only, no cross-encoder | 0.815 | 0.912 | 0.877 | 0.564 | 0.492 |
| **26 features, no cross-encoder (shipped)** | **0.841** | 0.931 | 0.884 | 0.600 | 0.603 |

CPU per memory search, measured in-process on a 4 vCPU 2.1 GHz Xeon (encoders, fusion;
Qdrant and PostgreSQL excluded): 2.4 s with the two cross-encoders (mmarco-mMiniLMv2 and
mxbai-rerank-xsmall over ~26 pairs), 0.17 s without. At 20 requests a second that is ~48
cores against ~3.4: the cross-encoders' 2.6 points were the whole of the CPU budget, and
the arms' own scores, sessions and speakers recovered them.

SciFact (300 queries, 5,183 abstracts), offline: nDCG@10 0.746 -> 0.759, recall@10
0.872 -> 0.883 with the late arm at 2.0. Rerankers were measured on SciFact too: too slow on
512-token abstracts, and the multilingual one lowered nDCG.

Through the service: `benchmark/reports/PHASE11-RESULTS-2026-10-01.md` (LoCoMo, SciFact, XQuAD,
LongMemEval, latency).

## Consequences

- **Reindex.** The collection fingerprint gains the late encoder (hashed: the composite is
  stored in a 200-character column) and the key layout `mk2`; every tenant is re-indexed
  into new collections (`make reindex`). The late vectors are the largest thing a point
  holds: ~5 KB for a memory, ~65 KB for a 512-token chunk at half precision, on disk.
- **Latency and throughput.** A memory search is the query's three encodes (two dense,
  one late) and one batched round trip to the store; the learned score is arithmetic over
  ~45 candidates. `memory_pool_k` sets the scored head; the coefficients are fitted at its
  value. Numbers: `benchmark/reports/PHASE11-RESULTS-2026-10-01.md`.
- **The coefficients are LoCoMo's.** LongMemEval is scored with them unchanged, as the test
  of whether they transfer; a corpus whose questions differ in kind may want a refit, which
  is a script over a feature dump and a constant change.
- **Ingest cost.** Each memory is encoded under two keys by two dense encoders and once by
  ColBERT; a conversational turn costs roughly three times what it did.
