# CPU multilingual selection, 28 September 2026

Status: measured shortlist, final promotion pending the queued application/ensemble/OCR
checks. This document is not a deployment or 90% answer-accuracy certificate.

## Constraints

- No Chinese-origin model checkpoints or derivatives based on them. Language support still
  includes Chinese. Historical Qwen/DeepSeek artifacts are not selected runtime models.
- CPU inference, target 8 vCPU / 16 GiB. The 20 requests/second target applies to interactive
  retrieval/context; PDF OCR and generative synthesis require separate throughput budgets.
- No paid model calls during this validation. Supplied agent Bifrost credentials are used
  through the native credential boundary; no MCP tools are sent or accepted.
- Test English LoCoMo and SciFact separately from multilingual paragraph retrieval and NLI.
  No conversion between source recall, ranking quality and generated-answer accuracy.

## Current measured choices

| Role | Decision supported by current evidence |
| --- | --- |
| Small multilingual encoder | Bekko a8m ONNX, Japanese maintainer and mmBERT/ModernBERT lineage. Strongest measured compact speed/quality balance; not a claim of universal best. |
| English specialist | Keep Granite English as baseline. Separate ranked-vector fusion with Bekko improves full SciFact; application integration awaits multilingual regression results. |
| Sparse retrieval | Existing Unicode BM25, no English-only SPLADE dependency. Language/script-aware token handling is functional coverage, not universal linguistic segmentation. |
| Reranking | Leave off. The measured cross-encoder hurt ranking and cost substantially more CPU. A small model's name alone is not evidence of a useful reranker. |
| Multilingual grounding | FP32 mDeBERTa is a viable isolated verification candidate. Quantized version failed quality; FP32 is too expensive to add indiscriminately to context reads. Current English default is not yet replaced. |
| OCR | Docling with local layout/table weights and Tesseract language/script packs; headless OpenCV, CPU thread bounds. Real 12-language image/PDF screen must pass before OCR is certified. |
| Local generative SLM | No eligible small generative model has demonstrated an improvement in this programme. Do not add an unmeasured decoder to the interactive CPU path. |
| User-provided LLM | Credential-gated automatic model discovery, native prompts, separate permission for read assistance. Ingestion and background briefs can use the key while reads remain model-free. Mocked contract tests are not evidence of an answer-quality gain. |
| Semantic caching | Existing scope/revision/policy-bound context cache. Preserve deletion, permission and credential invalidation; report warm and cold loads separately. |

## Measured numbers

Bekko a8m completed 1,200 scheduled encoder requests at 20 RPS with no rejected/failed
requests. p50/p99: 10.68/108.84 ms; CPU work: 0.013445 seconds/request; peak process RSS:
753.53 MiB. At 20 RPS this is about 0.27 host core-equivalents for the encoder. An 8-vCPU
budget capped at 70% utilization leaves about 5.33 cores for everything else. That is a
necessary budget estimate, not a measured whole-service capacity promise or CPU portability
model. SQL, authorization, Qdrant, workers, ingestion contention and external LLM quotas
still need a deployment load run.

Bekko a25m also completed 20 encoder RPS, but p99 was 1,046.88 ms and English SciFact was
worse. Its small multilingual recall gain does not justify replacing a8m in the fast path.

Full XQuAD, 14,280 questions / 12 languages / 240 paragraphs per language, a8m dense R@10:
98.83% same-language, 97.25% against the English corpus (including the English control).
These percentages are paragraph retrieval, not answers or evidence of all-language support.

Full SciFact, 5,183 documents / 300 queries, Qdrant component screen with cached vectors:

| Arm | nDCG@10 | Recall@10 | Search p50/p99 ms |
| --- | ---: | ---: | ---: |
| Granite English + BM25 | 0.7409 | 0.8912 | 16.10 / 54.77 |
| Granite English + Bekko a8m + BM25 | 0.7557 | 0.8926 | 20.14 / 97.89 |

This is a fixed equal-RRF hypothesis on public test data used during model selection, not
a held-out estimate. It uses three independent ranking lists; concatenating embeddings
would implement a different algorithm. Other-tenant/sibling sentinel records were excluded.
The screen's dedicated container was removed; it did not change production indexes.

## Product boundaries

Keep native source storage, run-tree authorization, provenance, forgetting and indexes as
the authority. Reuse Hindsight extraction preview where its SDK can represent the request;
agent VKs stay on native Bifrost because the pinned SDK cannot carry those credentials.
Native standing briefs provide model-free repeated reads; synthesis happens in background.
Continuous learning means source-backed updates with invalidation and scoped consolidation,
not unbounded ingestion-time generation or training weights on every conversation.

Hindsight's retain path uses LLM extraction. A model-free recall request over that corpus
is different from fully model-free ingestion plus retrieval. Compare equivalent modes.
See the capability inventory (a dated note since removed; see git history) for implemented,
partial and missing product features; the repository does not have complete Hindsight API
parity, a full reflect agent, webhooks, or an evaluated multi-run agent-task score.

## Evidence and outstanding checks

- `benchmark/results/cpu_capacity_bekko-a8m.json`
- `benchmark/results/cpu_capacity_bekko-a25m.json`
- `benchmark/results/multilingual_dense_bekko.json`
- `benchmark/results/scifact_qdrant_ensemble_screen.json`
- `benchmark/results/multilingual_nli_mdeberta_fp32.json`
- Fresh full LoCoMo source-recall arms, corrected real OCR, the 12-language equal-RRF fusion
  screen, and an uninterrupted full test run are queued; check completion artifacts before
  asserting success. No fresh reader answer-accuracy measurement exists.

Operational details and failed-run history: implementation handoff (a dated note since removed; see git history).
