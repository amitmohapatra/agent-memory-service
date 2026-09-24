# Competitive teardown: Mnemis, Hindsight, mem0 vs us

Captured 2026-09-25. Every number here is from a primary source - a released results file,
a committed config, or code - not from a README headline or a marketing page. Where a number
is only a vendor claim it says so.

Competitor code cloned to scratchpad (not vendored): microsoft/Mnemis, vectorize-io/hindsight,
mem0ai/mem0.

## 1. The headline claim we were given, and what is actually true

Claim: "Mnemis and Hindsight get 89%+ WITHOUT an LLM, at 20-200ms."

All three parts are false for Mnemis, and the first two are unverifiable for Hindsight.

### Mnemis 93.9 on LoCoMo

From its own released metrics file
(`results/locomo/metrics_graphiti_..._ragtopk30_gtopk60_RAG_GRAPH.json`):

    accuracy                    0.8061   1601/1986    <- ALL categories
    accuracy_exculde_category5  0.9390   1446/1540    <- the published headline
    category_accuracies["5"]    0.3475    155/446     <- adversarial

The headline deletes the adversarial category, where it scores 34.8%. Its all-categories
number is 0.8061.

Same file, `config.models`:

    graph_model == answer_model == grader_model == "gpt-41-mini-shortco-2025-04-14-Bing"

The same model answers the question and grades its own answer.

### Mnemis latency - published, in the per-question results, never in the README

`results/locomo/results_*_ragtopk30_*.json` is JSONL, 1986 rows, with timings. At the
ragtopk30 config that produces 93.9:

    search_duration    p50 1.279 s   p95 7.363 s   max 10.134 s
    answer_duration    p50 5.056 s   p95 10.586 s
    grading_duration   p50 2.983 s

At ragtopk10 (lower accuracy): search p50 0.511 s, p95 3.302 s.

Retrieval alone is 1,279 ms p50 - 17x our p50 and 4x our p99.

### Mnemis calls an LLM at retrieval time

`global_selection/global_selector.py`. `global_selection()` walks the hierarchy top-down:

    for layer in range(max_layer, 0, -1):
        ...
        selected, shortcuts = await self.layer_selection(query, current_layer_categories)

and `layer_selection` does:

    response = await self.llm_client.generate_response(..., model_size=ModelSize.large)

One sequential large-model call per hierarchy layer, per query. Sub-300ms is arithmetically
impossible for this design. They instrument it (`time_stats['layer_selection']`) and publish
no figure.

### Hindsight

- Numbers (LongMemEval-S 94.6, LoCoMo10 92.0) are on a website with NO stated ruler, NO judge
  model, NO methodology, NO latency.
- The repo publishes no per-question artifacts.
- `recall` does NOT call an LLM (verified in hindsight-api-slim/hindsight_api/engine/search/).
  `retain` does, and `reflect` does.
- Architecture: RRF fusion over semantic + BM25 + graph + temporal, then a CROSS-ENCODER
  reranker (`engine/search/reranking.py`), then recency blending. A "slim" deployment ships a
  passthrough cross-encoder, i.e. they know it is the expensive part and let you disable it.

### mem0

Their README's own caveat:

    "Scores reflect Mem0's managed platform, which includes proprietary optimizations not
     available in the open-source SDK"

So 92.5 LoCoMo / 94.4 LongMemEval / 64.1 BEAM(1M) / 48.6 BEAM(10M) are NOT reproducible from
the open-source repo at all. Their published latency for those: 0.88 / 1.09 / 1.00 / 1.05 s.
Their 2025 paper is a different, older architecture scoring 66.88 with search p50 148 ms.

## 2. Our measured position

Two different runs, and they must not be quoted together:

| | depth 100 (`final_judged`) | depth 50 (`p99_shipped_clean`, shipping) |
|---|---|---|
| all-categories | 0.7993 | 0.7829 |
| answerable only | 0.7983 | 0.7725 |
| adversarial | 0.8028 | 0.8169 |
| evidence_ALL | 0.9785 | 0.9485 |
| p50 / p95 / p99 | 122.8 / 657.0 / 1147.2 ms | 76.0 / 139.2 / 307.8 ms |

Caveats we must state ourselves before anyone else does:
- 0.7993 blends two metrics: 233 answerable graded for correctness, 71 adversarial graded by
  abstention rate. Correctness-only is 0.7983.
- Both runs are 2 of 10 conversations = 304 of 1986 questions (15.1% of the standard
  denominator). The full-set run is the fix.
- The depth-100 p99 is contaminated: its `encode` p99 is 3x the depth-50 figure for a
  fixed-size matmul over a ~12-token query, which depth cannot affect. That is CPU contention
  on a 4-core box, not architecture. Re-measure on a quiet box before accepting the tradeoff.

## 3. Ruler: the largest single term in every comparison

Measured on OUR OWN stored answers from one run (`locomo_judged_v5`), three rulers:

    refined   0.6908      multi_hop 0.3256   open_domain 0.6154
    strict    0.8454      multi_hop 0.6047   open_domain 0.7692    <- ours
    lenient   0.8914      multi_hop 0.8372   open_domain 0.8462    <- mem0-style

20.1 points between rulers on UNCHANGED predictions. The gap to Hindsight's claim is 12.2.
The ruler is bigger than the gap.

Everyone else excludes adversarial. For us that is nearly free (0.7993 -> 0.7983) because
adversarial is one of our strongest categories. It is a ruler-integrity point, NOT a points
point, and must not be presented as one.

## 4. Head-to-head on the one comparison that is like-for-like

All categories, each system's own ruler, its own released artifact:

| | Mnemis | us (depth 100) |
|---|---|---|
| all-categories accuracy | 0.8061 | 0.7993 |
| adversarial | 0.3475 | 0.8028 |
| retrieval p50 | 1,279 ms | 122.8 ms |
| retrieval p95 | 7,363 ms | 657.0 ms |
| LLM at retrieval | yes, one per layer | no |
| questions | 1986 | 304 |

Confounds that all run AGAINST us and cannot be removed by arithmetic:
- they answer with GPT-4.1-mini / Gemini 3.1 Pro / gpt-5; we answer with deepseek-flash
- they feed the answerer 8.8k-36k tokens; we feed ~5-8k
- we score 15% of the denominator

## 5. What cannot be executed, and why

A true head-to-head requires running their systems. It is not possible here:
- Mnemis needs Azure `gpt-41-mini-shortco-...-Bing`, Neo4j and Graphiti.
- Hindsight needs Gemini 3.1 Pro.
- mem0's headline numbers are explicitly from a managed platform with proprietary
  optimisations absent from the open-source SDK - not reproducible at any price.
- BEAM(1M)/BEAM(10M): no local dataset; only mem0 publishes numbers on it.

What IS possible, and is the plan: run OUR system on the full public benchmarks with and
without the LLM, fully instrumented, and compare against their RELEASED artifacts with the
ruler normalised. Mnemis's artifact makes that genuinely like-for-like. Hindsight's and
mem0's do not exist, so their numbers stay labelled as vendor claims.

## 6. What we are missing

1. A cross-encoder reranker. Hindsight has one on the recall path. Our rejection of reranking
   was measured on SciFact - document RAG, web-trained MiniLM - and does not transfer to
   conversational memory. That rejection should not have been generalised.
2. A reflect / consolidation loop that forms connections after the fact.
3. A structural/global view for "what is this archive about" questions. Mnemis's idea is
   sound; its LLM-per-layer implementation is disqualified on latency. Our multi_hop 0.58 and
   open_domain 0.46 are exactly what it targets.

## 7. What we have that they do not

- No LLM on the read path at all, and a measured p99 of 307.8 ms at shipping depth.
- Adversarial handling: 0.80 against Mnemis's 0.35, on the category they delete.
- Multi-tenancy with a structurally enforced wall, verified under identifier collision.
- Published per-stage latency.
