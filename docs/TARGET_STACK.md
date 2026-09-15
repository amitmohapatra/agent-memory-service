# Target stack and required changes — best-in-class open source (September 2026)

Goal: a memory service that is best in class on every axis that matters — complex relations and
a temporal knowledge graph, short- and long-term memory, retrieval/RAG, grounding, evaluation,
latency and scale — built only from open-source components, with every model swappable later
through configuration. This document fixes the defaults we adopt now, states the evidence, and
lists exactly what has to be modified or added in the repository. It is the input for the
real-component validation run (`VALIDATION_PROMPT_v3.md`); anything below that loses its
benchmark there is removed, as that prompt requires.

Rules that do not change: LLM calls go only through Bifrost (an external gateway, not part of
the service); the hard release gates stay at their current thresholds; nothing is claimed
"state of the art" until the public benchmark tables in section 5 exist.

## 1. What the research says (summary of evidence)

- **Retrieval.** The largest controlled comparison on text-and-table documents (T2-RAGBench,
  23k queries, April 2026) found hybrid BM25 + dense fused with RRF and reranked by a
  cross-encoder to dominate every alternative; reranking alone added +12.1 points Recall@5,
  BM25 beat dense retrieval on its own, and HyDE / multi-query added nothing. Qdrant's own
  five-dataset study found default RRF beat the stronger single retriever on four of five.
  Our baseline is exactly this pipeline; everything fancier must beat it to stay.
- **Embeddings.** Granite Embedding English R2 (149M, 768-d, 8k context, Apache 2.0) has the
  best MTEB-v2 (59.5), LongEmbed (67.8) and multi-turn-RAG (57.6) scores in its size class and
  is the fastest of the models IBM compared; the 47M "small" variant trails by ~4 points at
  ~40% more throughput. English-only.
- **Rerankers.** ms-marco-MiniLM (our current default) sits at ~60% BEIR nDCG@10.
  BGE-Reranker-v2-M3 (568M) reaches ~71.5% at ~12 ms/pair and is the accepted
  quality-per-latency sweet spot; Qwen3-Reranker-0.6B is similar quality; 4B–8B rerankers
  reach 75–77% but need GPUs. The reranker is the cheapest quality upgrade in the system.
- **Knowledge-graph memory.** On multi-hop QA (HotPotQA, TwoWiki, MuSiQue) graph-structured
  memory beats chunk/vector memory by a wide margin (Cognee's own benchmark: Cognee 0.85,
  Graphiti 0.74, LightRAG 0.67, Mem0 0.54 correctness — vendor-run, competitors untuned).
  The 2026 literature converges on: explicit temporal validity windows on facts (Zep/Graphiti),
  multi-layer graphs (entity + temporal + causal; MAGMA, LiCoMemory), structured
  retain/recall/reflect with separate networks for world facts, experiences, entity summaries
  and beliefs (Hindsight: 83.6% LongMemEval with a 20B open model), admission gating before
  storage, and aggressive semantic compression (SimpleMem: 30× fewer tokens, +26 F1).
- **Model-based extraction without an LLM.** GLiNER2 / GLiNER-Relex do zero-shot joint
  entity + relation extraction with an encoder (DeBERTa-v3), run on CPU, and beat GPT-5-mini
  on document-level (DocRED 31.3 vs 18.6) and cross-domain relation extraction. This is the
  right middle layer between our lexicon rules and the LLM.
- **Grounding.** Production practice in 2026 is a cascade: deterministic citation-span
  validation on 100% of answers, a DeBERTa NLI claim-support classifier on 100% of claims,
  an LLM judge only for borderline claims, and a contradiction scan over unused retrieved
  chunks; the metric is per-claim hallucination rate, not an answer-level average (a 0.92
  answer score was shown to hide an 18% per-claim rate).
- **Memory benchmarks.** LoCoMo, LongMemEval and BEAM are what the field reports on (Mem0
  66.9% LoCoMo; Zep 71.2%, Hindsight 83.6%, Mastra ~94.9% LongMemEval). They do not measure
  isolation, lineage, durability, contradiction handling or cost — the things this service is
  built for — so we report both: public numbers for comparability, our gates for what matters.
- **Document parsing.** Docling (MIT, CPU-viable, TableFormer cell grids) is the right
  interactive parser; Marker 2 is faster and more accurate on scanned/OCR batches
  (olmOCR-Bench 76.0 vs Docling 50.3, but Docling 64.0 on born-digital) under a restricted
  weights licence; MinerU wins on mathematics under a commercial-licence threshold.
- **Open-weight LLMs** (served behind Bifrost by vLLM/Ollama): Qwen3 (4B–30B, Apache 2.0) is
  the default choice for extraction and reflection; gpt-oss-20B (Apache 2.0) for structured
  reasoning; Phi-4-mini (MIT) where only CPU exists.

## 2. The stack we adopt now (all swappable by configuration)

| Layer | Choice now | Why | Keep as switchable alternative |
|---|---|---|---|
| Embedding | `ibm-granite/granite-embedding-english-r2` (768-d), ONNX int8 runtime on CPU | best MTEB-v2 / LongEmbed in class, fastest, Apache 2.0, 8k context | `granite-embedding-small-english-r2` for a low-latency tier; `Qwen/Qwen3-Embedding-0.6B` if multilingual is ever needed |
| Sparse | BM25 (Qdrant native, no model) | no inference cost; beats dense alone on documents | miniCOIL, SPLADE-v3 — only if they win the gate |
| Reranker | `BAAI/bge-reranker-v2-m3` on candidate_k=20; `granite-embedding-reranker-english-r2` as CPU-light option; MiniLM-L6 only for the low-latency tier | +11 points BEIR over MiniLM at ~12 ms/pair | `Qwen/Qwen3-Reranker-0.6B`; 4B/8B when GPUs exist |
| Late interaction | `answerdotai/answerai-colbert-small-v1` as an in-Qdrant multivector rerank stage | cheap second-stage precision; validated only by gate | Jina-ColBERT-v2 |
| Vector store | Qdrant server: hybrid Query API (prefetch → RRF), multivector, scalar/binary quantization, on-disk payload | native hybrid + multivector + quantization in one engine | — |
| Graph store | PostgreSQL (existing tables) with temporal validity, plus a **causal/typed-predicate layer** and **entity-summary nodes** | multi-layer graphs beat single-layer; keeps one canonical DB | Apache AGE / FalkorDB if traversal depth ever exceeds SQL recursion budgets |
| Extraction | Three-tier: (1) lexicon + grammar (existing, deterministic), (2) **GLiNER2 / GLiNER-Relex** zero-shot NER+RE on CPU, (3) LLM via Bifrost for ambiguous spans only | model-based tier beats GPT-5-mini on document-level RE with no API; LLM only where needed | Graphiti / Cognee providers stay only if they win the KG gate |
| Memory architecture | Explicit layers: **working** (Dragonfly, TTL), **episodic** (messages + archive), **semantic** (facts/preferences with validity windows), **procedural**, plus **entity summaries** and **beliefs** networks (Hindsight) and **observational compression** per thread (observer/reflector) | matches the 2026 consensus architecture; ours already has 4 of 6 | Mem0 / LangMem only if they win the memory gate |
| Consolidation | Admission gating before storage (worthiness, novelty, confidence, recency) + reflection pass that refines older memories when a new one lands; invalidation edges instead of deletion | structured admission and refinement outperform post-hoc filtering | — |
| Grounding | Cascade: citation-span validation → DeBERTa NLI (`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`) per claim → Bifrost LLM judge for borderline → contradiction scan on unused chunks; per-claim hallucination rate reported in the evidence report | field best practice; deterministic first, LLM last | HHEM-2.1-open, MiniCheck as alternative classifiers |
| Parsing | Docling (MIT) for interactive ingestion; `builtin` fallback | CPU-viable, structured tables, clean licence | Marker 2 batch backend behind the parser port for scanned archives |
| LLM | Bifrost gateway → Qwen3-14B (or 30B-A3B) on vLLM; Qwen3-4B for cheap uses | open weights, Apache 2.0, strong structured output | any model Bifrost routes to; nothing in the service knows |
| Eval | DeepEval (LLM-judge via Bifrost, 5+ repeated runs), our golden gates, **BEIR subset, LongMemEval, LoCoMo, BEAM** | our gates for what matters, public suites for comparability | RAGAS optional |
| Infra | PostgreSQL 16, Qdrant, Dragonfly, OpenFGA, Procrastinate, OTel + Prometheus | unchanged; all real in the validation run | — |

## 3. What must be modified or added in the repository

Ordered by expected impact per unit of work. Each item is gated: it ships only if the
validation run shows it helps and keeps every critical gate at 1.00.

### Retrieval (`modules/retrieval`, `adapters/models`, `adapters/search`)
1. **Reranker default → bge-reranker-v2-m3**; add ONNX/int8 export for CPU; keep MiniLM as
   `low_latency` tier; benchmark candidate_k 15/20/25 with real weights.
2. **Embedding runtime → ONNX int8** (measure vs fp32 sentence-transformers); enable Qdrant
   scalar quantization with rescoring, and binary quantization as an experiment on the
   768-d vectors; on-disk payload already on.
3. **Two-tier serving**: `retrieval.tier = balanced | low_latency`, selectable per request
   (`/v1/recall`, `/v1/context`); low-latency tier = small-r2 + MiniLM + no late interaction.
4. **Late-interaction rerank inside Qdrant** (multivector prefetch → ColBERT rescoring) as a
   stage between fusion and the cross-encoder; keep only if it beats cross-encoder-only.
5. **Query understanding without an LLM**: keep the deterministic router, add a lightweight
   query classifier (GLiNER2 entity/intent labels) to route relation questions to the graph
   with higher precision; LLM query expansion via Bifrost stays behind a `uses` flag.

### Knowledge graph and relations (`modules/graph`, `adapters/graph`)
6. **GLiNER2 / GLiNER-Relex extraction tier** behind the existing `GraphEnrichment` port:
   zero-shot entity + relation extraction with the predicate vocabulary we already define
   (has_value, driven_by, excludes, approved_by, acquired, works_at, …), merged with the
   lexicon output through the existing alias-aware resolution; CPU only.
7. **Multi-layer graph**: tag every relation with a layer (`entity`, `temporal`, `causal`,
   `structural`); the graph stage traverses layers with different weights (causal and
   temporal edges first for "why/when" questions); expose `layer` in `/v1/graph/query`.
8. **Entity summary nodes**: a maintained one-paragraph summary per entity (deterministic
   from its facts; optionally refined via Bifrost) indexed as `kind=summary`, so entity
   questions answer in one hop.
9. **Invalidation instead of deletion**: `invalidated_by` edges with reasons; `as_of` and
   `valid_at` both supported; conflict adjudication records which fact won and why.
10. **Cross-document entity resolution**: alias tables become tenant-wide with confidence,
    embedding-similarity blocking + deterministic rules; no LLM unless flagged.

### Memory (`modules/memory`, `modules/context`)
11. **Admission gating** made explicit and measurable: worthiness, novelty (dedup already),
    confidence, expected utility → stored `admission` record per memory; the memory gate
    reports precision/recall of admission.
12. **Observational memory per thread**: an observer that compresses older turns into dated
    observations (deterministic extractive first, Bifrost-refined optionally) and a reflector
    that merges observations into semantic memory; replaces the plain rolling summary.
13. **Beliefs and entity-summary networks** in the memory model (`memory_type` values
    `BELIEF`, `ENTITY_SUMMARY`) with their own consolidation rules and visibility.
14. **Reflection pass**: when a new memory lands, re-evaluate the top-k related memories
    (supersede / strengthen / weaken); bounded, transactional, idempotent.
15. **Forgetting policy**: importance × recency × access decay drives eviction from working
    memory and archival of semantic memories; never deletes canonical records.

### Grounding (`modules/context`, new `modules/grounding`)
16. **Grounding cascade** in the context bundle and in a new `/v1/verify` endpoint: claim
    decomposition (deterministic sentence/clause split), citation-span validation against
    evidence items, DeBERTa NLI support score per claim, Bifrost judge for the borderline
    band, contradiction scan across retrieved-but-unused chunks; output per-claim verdicts and
    a per-claim hallucination rate in `EvidenceReport`. SDK: `client.verify(answer, bundle)`.
17. **LangGraph adapter**: `require_evidence` gains `verify_answer=True` to run the cascade on
    the node's output before it is recorded as an assistant message.

### Evaluation (`benchmark/`, `tests/eval`)
18. **Public benchmark harness** `benchmark/public.py`: BEIR subset (nfcorpus, scifact,
    fiqa) through the real engine; LongMemEval, LoCoMo and BEAM through the memory pipeline
    with answers generated via Bifrost and scored by the benchmarks' own graders; results
    tables per configuration (native, native+Bifrost uses, each provider).
19. **Judge variance control**: every LLM-judged metric runs ≥5 times; report mean ± sd; a
    gate never depends on a single judged run.
20. **Per-claim grounding metrics** and **cost per recall / per observation** added to the
    gate artifacts.

### Performance and scale (`api/`, `adapters/`, `deploy/`)
21. **Model serving isolation**: embeddings/reranker/NLI run in a separate `memory-models`
    process (same image, `MEMORY__MODELS__SERVING=inprocess|remote`) so API workers stay
    small and models scale independently; batch + queue with bounded latency.
22. **Caching**: embedding cache (exists), reranker score cache keyed by (query, doc revision),
    graph-traversal cache keyed by (tenant, entity set, as_of, revision).
23. **Qdrant tuning**: HNSW `m`/`ef` per collection size, quantization with rescoring, sharding
    by tenant hash for large tenants; index rebuild remains `tools.reindex`.
24. **PostgreSQL**: partial indexes for visibility keys, `graph_relations` partitioned by
    tenant hash above a threshold, prepared statements, connection pool sizing by worker.
25. **Budgets**: p95 targets per tier written into settings and enforced by the performance
    gate against the deployed stack (Locust), not in-process.

## 4. What stays as it is

Hexagonal layout and ports; PostgreSQL as the canonical store with the outbox; Procrastinate;
OpenFGA visibility with store-side filtering; the deterministic native extractor and router
as tier 1; Docling; the SDK and LangGraph adapter surfaces (extended, not changed); the
release gates and their thresholds.

## 5. Decision rule and reporting

A component or change survives if, with real weights and real services, it beats the baseline
it replaces on Recall@20, evidence-group recall, KG fact recall, per-claim grounding rate or
false-merge rate, within the p95 budget of its tier, and ties or better on every critical
gate. Every decision gets one row with numbers in `docs/adr/0017-real-component-validation.md`;
losers are deleted from code, config, docs and lock files.

## Sources

- T2-RAGBench retrieval strategy benchmark (arXiv 2604.01733): https://arxiv.org/html/2604.01733v1
- Qdrant hybrid search study: https://qdrant.tech/articles/hybrid-search/
- Granite Embedding English R2 model card: https://huggingface.co/ibm-granite/granite-embedding-english-r2
- Open-weight reranker ranking (May 2026): https://presenc.ai/research/best-open-weight-reranker-models-2026
- Cognee KG memory benchmarks (vendor-run): https://www.cognee.ai/blog/deep-dives/knowledge-graph-memory-benchmarks
- 2026 memory literature scan (Hindsight, MAGMA, SimpleMem, LiCoMemory, A-MEM): https://lin-guanguo.github.io/llm-memory-research/memory.literature-scan/
- Agent memory benchmark numbers and caveats: https://memnode.dev/articles/agent-memory-benchmarks-2026-real-numbers
- GLiNER-Relex (arXiv 2605.10108): https://arxiv.org/html/2605.10108v1
- RAG faithfulness cascade (2026): https://futureagi.com/blog/evaluating-rag-faithfulness-deep-dive-2026/
- PDF parsers compared (Docling / Marker / MinerU): https://builderai.tools/blog/pdf-parsing-for-rag-mineru-docling-marker-compared
- Open-weight LLMs for local deployment: https://huggingface.co/blog/daya-shankar/open-source-llm-models-to-run-locally
