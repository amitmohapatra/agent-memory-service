# Actor/topic retrieval experiment — 26 September 2026

The user requested concrete intelligence improvements after the wider-memory configuration
reached 83.90% source recall but only 42.91% complete multi-hop source coverage. This
experiment addresses aggregate conversational questions by separating entity selection from
semantic topic retrieval. It does not generate facts or train model weights.

## Design fixed before holdout measurement

- Development conversations: dataset indices 0, 1, 2. Remaining indices 3–9 are reserved
  for validation of this particular change. They are not an unseen external dataset;
  their baseline aggregate scores were already known.
- Control: the newly promoted default, memory recall 100 and context cap 100, 8000 tokens.
- Candidate: same settings plus `memory_entity_search=True`. No ingestion/index changes.
- Only queries whose existing router identifies a multi-hop signal qualify. Explicit
  result limits, document selectors and mixed/document candidate pools bypass the feature.
- Recognize up to two canonical user/agent/person subjects from authorized candidates that
  also occur explicitly in the query. Remove those names from the secondary topic query;
  encode it once, then search each subject independently with the existing hybrid store.
- Tenant, visibility and current-memory filters remain inside every store request. A
  subject match is not an authorization grant. Every returned item retains its own evidence.
- Rank fusion weights the original question twice as heavily as each actor view. Original
  evidence about a person spoken by somebody else remains eligible.
- Additional work has a 200 ms timeout including embedding and all secondary searches.
  Timeout/provider failures preserve the primary results; client cancellation propagates.
  Two secondary searches maximum; no recursion, LLM, new storage or full-bank scan.
- The existing final memory count and token budget still apply. Configuration participates
  in the context-cache fingerprint. The feature defaults OFF pending measurement.

## Initial diagnosis

Saved development bundles averaged 102.3 memories / 100.0 distinct source IDs (conversation
0), 102.1 / 101.7 (1), and 102.1 / 102.3 (2; multi-source memories can exceed one source per
memory). Repeated source IDs are not the dominant loss. No source-diversity change was
implemented. Earlier source-turn promotion remains disabled; this experiment is different.

## Validation protocol

Use `benchmark.locomo_probe --entity-search --depth 100 --repeats 1`. The control is
explicitly memory depth 100, not the older depth-50 comparison. Each question runs both
arms with alternating order and no bundle cache. Source turns are audited first. Record
actual settings, source hash, rendered contexts, selected memory IDs, source recall,
complete-source coverage, per-stage timings, plans and fallback counts. This uses real
Granite ONNX, BM25, PostgreSQL and Qdrant with the established auth/cache/NLI stand-ins.
Timings are serial builder timings, not HTTP/load or generated-answer latency. Development
measurements may overlap small local test runs; use a quiet confirmation before claiming a
production latency improvement.

Answer validation still needs the existing DeepSeek Flash provider, which returned HTTP
402. Do not translate evidence-recall improvements into answer-accuracy claims. Reuse reader
judgments only for exactly identical contexts and inputs; changed contexts need new answers.

## Results

Full evaluation completed on all **1986 questions**, without ingestion changes or LLM calls.
Source recall covers 1536 answerable questions with annotations; four have none. All 446
adversarial questions remain in the recorded contexts, but their evidence-source coverage
must not be reported as generated-answer abstention.

| Metric | Current default | Actor/topic experiment |
|---|---:|---:|
| Overall source recall | 83.9047% | 84.2410% |
| Complete-source coverage | 77.5391% | 78.0599% |
| Multi-hop source recall | 68.3416% | 69.9924% |
| Complete multi-hop coverage | 42.9078% | 45.3901% |
| p50 | 168.0 ms | 167.4 ms |
| p95 | 353.8 ms | 418.9 ms |
| p99 | 526.6 ms | 605.3 ms |

The additional search planned 247 queries, completed on 227 and fell back on 20 timeouts.
228 rendered contexts differ between arms (graph deadline variation can also change
contexts). The new control reproduces the previous full source-recall scores exactly.
Latency differs between runs; use the paired 526.6 → 605.3 ms comparison here, rather than
subtracting a historical latency from a new run.

Development recall improved 86.4447 → 86.5970%. On validation conversations 3–9, overall
recall improved by **0.3974 percentage points**: 17 questions improved and one regressed.
The conversation-cluster bootstrap interval is +0.2342 to +0.5698 points (5000 draws,
seed 20260926). Validation multi-hop recall improved **1.9576 points**, with 15 questions
improved and one regressed; interval +1.0681 to +2.9938 points. These are exploratory
intervals on seven conversations from the same dataset, not an external benchmark.

Decision: keep this feature **experimental and OFF by default**. It supplies a measured
way to recover some missing evidence, but the overall gain is small, p99 exceeds the
accepted 551 ms reference, and no changed-context answer/abstention evaluation has run.
It is not evidence of 85% or 90% answer accuracy. The current default remains the previously
promoted 100-memory configuration.

The source audit independently reconstructs every source-recall score from selected memory
IDs, verifies indexed versus canonical evidence for 6712 memories, and checks all 5882 source turns against
body, speaker, date and ingestion order. See `codex_entity_topic_exact_sources.json`.

Measured runtime/benchmark hash:
`55e77464ef4cbca8a9717705acc2cdd3fa12435ae5540c20e6d6fa45a10bc4bc`.
The final implementation additionally bypasses exact-identifier routes. None of the 1986
questions has both an identifier and a multi-hop signal, so this guard cannot affect the
measured run. Reconstructing the source with only that guard removed reproduces the
measured hash exactly. Final hash:
`6e29dfbba464f53ade0c94ec013ed5571f5326199bff9446ee93eaa9eca923a5`.
See `codex_entity_topic_final_parity.json`; the measured engine is archived as `.py.txt`.

Exact-input reader reuse can retain **2873 arm judgments / 1355 complete pairs** from the
existing checkpoint, under the same Flash model and prompts. There are **696 distinct
uncached context evaluations** left after accounting for identical arm contexts. This is
eligibility, not a new answer score. The manifest is `codex_entity_topic_reader_reuse.json`.
Restore provider billing before evaluating changed contexts. Do not overwrite the earlier
full-reader checkpoint; this is a separate experiment.

Final validation: **814 passed, 0 failed, 0 skipped**, including all unit tests and relevant
API/context/cache/graph integration tests. Nineteen cases exercise the new feature. Ruff
lint/format pass on the changed Python files; Pyright reports zero errors/warnings in the
retrieval engine and query planner. One initially incomplete evidence fixture was corrected
and the full selected suite rerun. Results are in `codex_entity_topic_validation.json` and
`codex_entity_topic_tests.xml`. Development, validation and source-audit containers exited
successfully and were removed after their logs were saved; benchmark databases, original
reader checkpoint/container and result artifacts remain intact. No deployment was performed.

## Additional diagnostics

The current saved reader checkpoint has 1199 valid answerable pairs. With the wider
configuration, 836/937 (89.22%) are correct when all annotated sources are present, versus
83/260 (31.92%) when some are absent. Two questions have no annotations. These are
associations on an incomplete cohort, not proof that any retrieval change will reach 90%.
Among the 101 errors despite complete source IDs, 24 are multi-hop, 27 temporal, 20
open-domain and 30 single-hop. Source presence does not guarantee all relevant details
survived extraction or that the reader/judge handled them correctly. The diagnostic keeps
original gold answers and judgments, including questionable annotations; nothing was
rewritten to increase scores. See `codex_current_error_diagnostic.json`.

A development-only source-adjacency diagnostic also checked whether nearby turns could
fill gaps. With 12 seed sources and a two-turn radius on routed multi-hop questions, mean
source recall rose 86.44 → 87.40%, but multi-hop recall only 73.20 → 74.10%. This calculation
does not enforce runtime permissions, source availability, token packing or latency and
is not a feature result. It was not enabled. See `codex_neighbor_source_diagnostic.json`.
