# Phase 10 results, 2026-10-01: a turn indexed with the turn it answers

One change (`449c580`), two halves, both language-agnostic and model-free:

* **Ingest** (`MemoryIntelligenceSettings.index_preceding_turn`): a verbatim turn's index
  text is prefixed with the message said just before it in the same conversation (same
  tenant, workspace and thread, within 6 hours). A reply rarely restates its question, and a
  bare question is never a memory of its own, so the question's words were in no index.
* **Query** (`RetrievalSettings.adjacent_turn_weight = 0.25`): a memory candidate gains a
  quarter of the best fused score of the turn before and the turn after its own, among the
  candidates already pooled. No extra query.

## Measured

`benchmark.native_source_retrieval`, all 10 conversations, 1,986 questions, 0 failures in
either arm, paired on the same 1,525 answerable questions. Base is `4fa614b`.
Artifact: `benchmark/results/phase10/locomo_source_preceding_turn.summary.json`.

| bucket (n) | @10 | @20 | @50 | @100 |
|---|---|---|---|---|
| all answerable (1525) | 0.660 -> **0.731** | 0.747 -> 0.796 | 0.825 -> 0.866 | 0.879 -> 0.907 |
| single-hop (830) | 0.733 -> **0.871** | 0.821 -> 0.914 | 0.888 -> 0.951 | 0.934 -> 0.968 |
| multi-hop (282) | 0.439 -> 0.424 | 0.548 -> 0.547 | 0.678 -> 0.690 | 0.756 -> 0.795 |
| temporal (321) | 0.747 -> 0.742 | 0.809 -> 0.806 | 0.869 -> 0.886 | 0.911 -> 0.923 |
| open-domain (92) | 0.374 -> 0.369 | 0.473 -> 0.466 | 0.558 -> 0.573 | 0.640 -> 0.654 |

At @10, 200 questions better and 95 worse. Latency p50/p95/p99: 76/135/163 ms ->
77/140/166 ms.

**Boundaries.** Hugging Face is unreachable from the session that ran this, so the English
dense slot held a stand-in 384-d encoder and the multilingual arm was off
(`BENCH_DENSE=english`); `representative: false`. BM25 and the change itself are real. The
base arm reads 0.660 against the shipped ensemble's 0.666, so the stand-in sits at the same
level, but the gain with the shipped ensemble needs a re-run (`make bench-locomo-source`).

The whole gain is single-hop. Multi-hop, temporal and open-domain move by -0.015 to -0.005
at @10 and gain at @50/@100: their evidence is reached, then outranked inside the top ten by
single-turn matches the neighbour lift raised.

## Tried offline and dropped

A turn-level simulator (BM25 with this tokenizer, one dense encoder, weighted RRF; its
baseline 0.664) screened these first: resolving English relative dates into the index text
(+0.010, English-only), z-score fusion in place of RRF (-0.004 to +0.017 by weight, below
RRF with the change), a per-session cap in the top ten (hurts), a boost for the speaker a
question names (+0.001), and neighbour weight 0.5 (gives back half the lift).

## Toward 0.80

recall@50 is 0.866, so the evidence is mostly pooled and ranking is the gap. What the
memory systems reviewed (Hindsight, Zep, Mem0, Supermemory) do about it is model work:
facts extracted by an LLM at ingest, and a cross-encoder over the fused head blended with
the fusion score rather than replacing it. Both need an arm this session could not run.
