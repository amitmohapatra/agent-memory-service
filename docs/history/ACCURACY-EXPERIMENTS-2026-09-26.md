# Accuracy experiments — 26 September 2026

The user permits a small latency increase for better accuracy. No replacement p99 SLO
was specified. These experiments keep the strict DeepSeek Flash reader/judge, preserve
adversarial evaluation, and do not claim a full-corpus 85% result.

## Changes

- `RetrievalSettings.memory_recall_k` provides bounded wider retrieval for memory-only
  ranked pools. Explicit caller limits, document selectors and mixed/document pools keep
  their original limit. Dense/sparse prefetch scales with the requested memory depth;
  authorization and current-memory filters still execute inside the search store.
  The setting defaults to **0 (off)**. The tested 100-memory profile also sets the existing
  `ContextSettings.memories_max=100`, retaining its 8000-token budget. This is an internal
  tuning/constructor setting, not a new API route or an environment variable.
- `hybrid_rrf_k` now controls the store's dense/sparse fusion. The existing `rrf_k=60`
  controls outer fusion with optional document strategies. They are different layers.
  The default inner constant is **1 using one-based ranks**, equivalent to the old native
  Qdrant zero-based constant 2, using the original `FusionQuery` wire request. The adapter
  translates custom constants by adding one, and tests compare
  native scores against client fusion for constants 0, 1 and 60. Do not silently change
  both fusion layers to 60: the measured inner-60 experiment reduced recall.
  A real-server diagnostic found identical overlapping scores between legacy and explicit
  k=2 requests but different cutoff-tie membership in one of three memory queries. Keeping
  the original default wire request avoids introducing that unnecessary change. Sorting
  returned hits stabilizes ties inside the pool, not server-side cutoff membership.
- `benchmark.locomo_probe` records alternating paired retrieval arms, actual settings,
  source-body audits, unique-question recall and separate repeated-request percentiles.
  It saves first-pass contexts for the existing resumable reader. Optional ingestion is
  restricted to a fresh tenant in the new isolated `memory_bench_accuracy` database;
  it never resets existing stores. Conversation 2 uses tenant `bench_accuracy_conv2`.
- `benchmark.locomo_oracle` prepares gold-evidence diagnostics without including gold
  answers in the reader context. It checks question/answer identity and reports missing
  annotations. Ingestion, graph audits and oracle preparation share the same turn/caption
  reconstruction helper. Gold contexts remain evaluation-only.

No landing reflection, broad LLM ingestion, query-time LLM, new reranker or conversation
hierarchy was enabled by this work. Those remain separate engineering work; their quality
cannot be inferred from these depth experiments.

## Retrieval measurements

All arms use real Granite ONNX, BM25, PostgreSQL and Qdrant. Cache/authz/NLI stand-ins are
the documented benchmark configuration. These are warm serial builder measurements with
ten warmups and bundle caching disabled, not production HTTP/load or generated-answer
latency. The configured token budget remains 8000 in each pair.

| Corpus / candidate | Answerable source recall, control → candidate | Multi-hop source recall | p50 ms | p99 ms |
|---|---:|---:|---:|---:|
| Conversation 1, depth 100 | 78.91 → 86.11% | 58.33 → 84.09% | 130 → 175 | 423 → 480 |
| Conversation 1, depth 150 | 78.91 → 90.43% | 58.33 → 88.64% | 121 → 211 | 379 → 517 |
| Conversation 2, depth 100 | 83.33 → 87.94% | 66.67 → 73.92% | 128 → 172 | 334 → 530 |
| Conversation 1, inner RRF 60, depth 50 | 78.91 → 75.21% | 58.33 → 49.24% | 115 → 120 | 422 → 389 |

Conversation 1 has 81 answerable questions, including 11 multi-hop, plus 24 adversarial;
each latency arm has 210 timed requests. Conversation 2 has 152 answerable, including 31
multi-hop, plus 41 adversarial; each arm has 386 timed requests. Repetition does not increase
the independent quality sample size. Source presence does not prove all answer details
survive extraction or packing. Tail estimates vary between runs.

The first depth probes changed the existing global depth and context cap as an exploratory
control. The separate conversation measured the new memory-only setting from an immutable
source staging copy. Every artifact records its source hash and actual settings; those
hashes differ from the later explicit-fusion implementation. No retrieval/index writes ran
during the paired timing passes. The isolated SQL database and tenant preserve existing
corpora; the Qdrant collection itself is shared, so global BM25 corpus statistics can change
when another tenant is indexed. Comparisons are paired within each fixed-index run.

## Strict reader measurements

All completed pairs use `deepseek/deepseek-flash` and the same strict answer/judge prompts.
All final reader artifacts have zero failures. One depth-100 development pair required a
whole-pair retry; the original failure remains in `codex_depth100_reader_first_pass.json`.
Different reader runs have different control answers because generation/judging is not
perfectly deterministic. Compare each candidate against its own paired control.

| Experiment | Answer accuracy, control → candidate | Multi-hop | Adversarial abstention |
|---|---:|---:|---:|
| Conversation 1, depth 100 | 81.48 → 83.95% | 54.55 → 63.64% | 91.67 → 83.33% |
| Conversation 1, depth 150 | 83.95 → 85.19% | 54.55 → 54.55% | 87.50 → 83.33% |
| Conversation 2, depth 100 | 80.26 → 82.24% | 58.06 → 67.74% | 87.80 → 87.80% |

The two depth-100 runs together cover 233 answerable and 65 adversarial questions, not the
entire LoCoMo corpus: answer accuracy is 188/233 → 193/233 (80.69 → 82.83%), multi-hop
24/42 → 28/42 (57.14 → 66.67%), and abstention 58/65 → 56/65 (89.23 → 86.15%).
The 85.19% depth-150 score is one conversation, with one net extra correct answer, unchanged
multi-hop accuracy and worse abstention. It is **not** an achieved overall 85% target.

The held-out answer improvement is seven wins and four losses across 152 questions;
its exploratory question-paired 95% bootstrap interval is -2.63 to +6.58 percentage
points. This small sample does not establish a general improvement. Even at depth 100,
only 17/31 held-out multi-hop questions contain every annotated source (54.84%, versus
15/31 at depth 50). Average source recall of 73.92% must not be confused with complete
multi-hop evidence coverage.

The separate annotated-evidence diagnostic covers all 43 multi-hop questions from the saved
conversations 0/1 contexts. It scores 21/43 (48.84%) with retrieved context versus 30/43
(69.77%) with annotated turns, with 14 wins and 5 losses. This is not an accuracy ceiling:
several annotations omit an item required by their own gold list, reference an adjacent
turn, or name the wrong speaker. These cases remain in the results; the evaluator and gold
answers were not rewritten to improve the score. It also uses historical saved retrieval,
not the final-source context profile.

## Decision and remaining work

Keep broader memory recall opt-in and preserve the current default ranking. The measured
answer gains are small and development abstention regressed; increasing depth alone does
not justify a default rollout. Inner fusion 60 failed the evidence-recall experiment and
was not promoted. A passing regression suite establishes implementation correctness, not
an 85% answer score or a production latency SLO.

The next accuracy work should address contextual fact extraction and useful cross-session
evidence representations, with literal source support and lifecycle-safe derived memories.
Do not simply activate existing landing reflection: the audience, invalidation and
concurrency blockers in `FEATURE-AUDIT-2026-09-25.md` remain. Evaluate any next change on
held-out conversations, including open-domain and adversarial questions, under the same
reader and token budget. Avoid tuning to the known question-name/annotation errors.

## Final-source verification

Conversation identifiers above are zero-based dataset indices. After the final adapter
change, a fresh paired replay on conversation 2 reproduced **all 193 control and all 193
candidate rendered contexts exactly**, including question, answer and reference-date
identity. The existing held-out reader judgments therefore apply to these final-code
inputs; no new reader calls are implied. See `codex_accuracy_final_parity.json`.

This final replay used one timed pass per arm: source recall remained 83.33 → 87.94%,
p50 was 126.6 → 170.7 ms, p95 253.9 → 336.6 ms and p99 337.7 → 519.1 ms. Neither arm
met the original 300 ms p99 target. These timings exclude generated answers and are not
an HTTP concurrency/load test.

Two final-code full SciFact runs (5,183 documents, 6,814 chunks, 300 questions) preserved
all 300 rankings against each other and the previous final baseline. Both scored
**nDCG@10 0.7436, recall@10 0.8731 and hit@10 0.8867**. Their p50/p95/p99 were
140.2/291.7/407.5 ms and 147.9/297.9/511.2 ms. `external_retrieval.json` now contains the
first of these final-source runs; repeat and intermediate measurements remain separately
available. SciFact retrieval scores do not validate document answer faithfulness or PDF
parsing, and hit rate must not be reported as recall.

The shared runtime/benchmark source SHA-256 is
`bae240896abe90e9e9b7a5cf4e4403f983c4659ea0338c15dc9ffc54a1d33114`.
The final artifacts are `codex_accuracy_release_heldout*.json`,
`codex_accuracy_release_rag_baseline*.json` and
`codex_accuracy_release_rag_comparison.json`.

## Final regression validation

**859 passed, 2 skipped, 0 failures/errors**, in one final-source run (693.85 seconds).
Coverage includes all unit tests, graph/context/retrieval/memory/access integration,
security, evaluation gates, retrieval end-to-end and search/OpenAPI contracts. The new
tests cover bounded memory depth, explicit caller limits, mixed/document pools, search
authorization, native/client fusion equivalence, source annotation integrity and unique
quality denominators in repeated latency trials.

The skipped lanes require real DeBERTa and Docling PDF dependencies unavailable on this
host; no claim is made that this run validates them. Ruff lint passes, all 390 Python files
checked for formatting pass, and Pyright reports 0 errors with 21 optional-dependency
warnings. `git diff --check` passes. See `codex_accuracy_validation.json` for per-case
results and `codex_accuracy_gate_results.json` for the five newly generated gate artifacts.
Historical gate files were restored after preserving fresh results separately. Temporary
benchmark containers were removed; benchmark databases and result artifacts were retained.

## Artifacts and reproduction

- `codex_oracle_contexts.json`, `codex_oracle_reader.json`
- `codex_depth100_probe.json`, `codex_depth100_reader.json`
- `codex_depth150_probe.json`, `codex_depth150_reader.json`
- `codex_depth100_heldout_probe.json`, `codex_depth100_heldout_reader.json`
- `codex_rrf60_probe.json`
- `codex_accuracy_exploratory_analysis.json` (question-paired bootstrap and exact discordance
  tests; exploratory, no multiple-comparison correction or claim of conversation independence)

Use the runtime image/model mounts and environment documented in the previous handoff:

```bash
# Existing conversation 1 / memory_bench_conv, read-only:
python -m benchmark.locomo_probe --conversation 1 --depth 100 --out depth100.json
python -m benchmark.locomo_probe --conversation 1 --depth 50 --rrf-k 60 --out fusion60.json

# Existing held-out tenant / memory_bench_accuracy; --ingest is only for a fresh tenant:
python -m benchmark.locomo_probe --conversation 2 --depth 100 --out heldout.json

# Reader only; enable only grounding_judge through the existing Bifrost settings:
python -m benchmark.locomo_reader benchmark/results/heldout_contexts.json \
  --control-field control_context --concurrency 4 --calls-per-minute 120 --out heldout_reader.json
```
