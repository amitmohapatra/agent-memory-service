# ADR 0012: Advanced retrieval strategies are benchmark-gated extra retrievers

**Status:** superseded by measurement, 2026-09-20 · **Date:** 2026-09-15

> **Superseded.** Seven of the strategies this ADR introduced — `minicoil`, `colbert`,
> `pageindex`, `raptor`, `graph_ppr`, `graphrag_global`, `late_chunking` — have been removed
> from the codebase, along with `retrieval.rerank` being defaulted off.
>
> The gating mechanism this ADR describes was sound and is why removal was cheap: every one
> was additive and off by default, so nothing depended on them. What it could not settle was
> whether any of them added a capability the baseline lacked. `tests/eval/test_capability_coverage.py`
> answers that per strategy, and `docs/MEASUREMENTS.md` records the reranker ablation that
> decided `rerank` (significantly *worse*, p = 0.012, at 21x the latency).
>
> `splade` survives as the one genuinely uncovered capability. The rest of this document is
> kept as the record of why they were built.

## Decision
- **Every advanced strategy is off by default and additive.** A strategy is an *extra
  retriever* whose hit list is fused with the baseline hybrid list by reciprocal-rank fusion
  (`RetrievalEngine.retrievers`), or a ranking mode inside an existing stage
  (`graph_ppr`). Turning one on can add evidence; it cannot remove the baseline's, and the
  isolation guarantees are unchanged because every extra retriever queries the store with
  the same visibility filter.
- **Model-free strategies ship working:**
  `pageindex` (route the question to the best sections by their hierarchical summaries, then
  search only the chunks beneath them — a deterministic PageIndex over the document's own
  tree), `raptor` (summary nodes compete with chunks in the fused list — multi-level
  retrieval over the hierarchy instead of an LLM-built cluster tree), `graph_ppr`
  (personalised PageRank on the query's graph neighbourhood, seeded at the resolved
  entities, ranking facts and evidence by centrality).
- **Model-backed strategies ship as adapters that refuse to degrade:** `splade` /
  `minicoil` (fastembed learned sparse encoders replacing BM25, server-side IDF off),
  `colbert` (fastembed late-interaction multivectors in a Qdrant `late` field with MaxSim;
  every point in the collection carries the multivector), `late_chunking` (embed the whole
  document once and mean-pool each chunk's token span; overflow falls back to per-chunk).
  Missing weights raise `DependencyUnavailable` at startup — a deployment never runs a
  silently degraded configuration — and the harness records the strategy as *skipped*.
- **`graphrag_global` is documented, not implemented:** community summaries need an LLM;
  document/section summaries (ADR 0011) already answer GLOBAL_SUMMARY questions.
- **The adoption gate is mechanical.** `benchmark/advanced.py` runs baseline and each
  strategy on the same corpus and reports a verdict: *adoptable* only when no critical gate
  (Recall@20, Evidence-Group Recall, evidence-complete rate) drops below the baseline and
  recall p95 stays within `budgets.recall_p95_ms`; otherwise *rejected* with the numbers, or
  *skipped* with the reason. Collections are named by the full fingerprint
  (dense | sparse | late), so switching encoders is a rebuild, never a mixed space.

## Evidence (sandbox, hash embedding — `representative: false`)
`benchmark/results/advanced_retrieval.json`: baseline R@20 1.00 / EGR 1.00 / complete 1.00,
p95 ≈ 56 ms; `pageindex`, `raptor`, `graph_ppr`, `pageindex+raptor` all *adoptable* (same
gates, p95 60–82 ms); `splade`, `minicoil`, `colbert`, `late_chunking` *skipped* (weights
not downloadable here); `graphrag_global` *skipped* (needs LLM).
`tests/integration/test_advanced_retrieval.py` exercises PageIndex routing (evidence pages
11/14/20 reachable through the tree), RAPTOR fusion (summaries tagged `raptor`), PPR
ranking, and the real Qdrant multivector path with a deterministic fake encoder (including
visibility inside MaxSim queries). `tests/contract/test_advanced_adapters.py` proves the
loud failure without weights and runs the real adapters when `MEMORY_MODELS_DIR` is set.

## Consequences
- The verdicts above say the model-free strategies are *safe*, not *better*: with the hash
  embedding the baseline already saturates the golden set. The decision to enable any of
  them in production must be re-run with Granite/MiniLM weights on the target corpus
  (`make bench-retrieval`, `python -m benchmark.advanced`).
- ColBERT storage multiplies vector volume by tokens per chunk; the `late` field is built
  with `m=0` (no HNSW) so it is used as a reranker-style retriever over filtered candidates.
