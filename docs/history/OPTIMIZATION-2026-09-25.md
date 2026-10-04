# Retrieval optimization verification — 2026-09-25

## What the existing results establish

The latest full saved DeepSeek run is `benchmark/results/v2_judged_shipped.json`:
1,986 questions, shipped depth 50, `deepseek/deepseek-flash`, strict grading. Its reported
answer score is 0.7292 and adversarial abstention is 0.8543. Three answerable rows failed
judging. Restricting to the 1,537 successfully judged answerable rows gives **0.73064**.
Neither figure establishes 85% answer accuracy.

Among those successfully judged rows:

| Category | Questions | Wrong answers | Wrong answers that abstained |
|---|---:|---:|---:|
| Multi-hop | 279 | 136 | 27 |
| Open-domain | 96 | 56 | 37 |
| Single-hop | 841 | 146 | 76 |
| Temporal | 321 | 76 | 40 |

Retrieval coverage and reader behavior need separate measurements. Token overlap across
an entire rendered bundle cannot establish that the reader received an intact supporting turn. Conversely,
retrieving a turn's source ID cannot establish that its extracted proposition preserves
every detail required by the question. The benchmark now keeps these measurements separate.

## Implemented changes

* Preserve canonical `EvidenceRef` objects through the memory search payload projection
  and the exact-ID read path. Memory citations remain `memory_id:...`. Older indexes and
  working memories retain their record-pointer fallback. A bundle-cache format fingerprint
  prevents serving bundles constructed before this change.
* Group source siblings deterministically by `(source_type, source_id)`, rather than
  unordered source IDs. This activates the existing renderer's intended grouping after
  reindexing; it does not fetch missing memories or prove an accuracy improvement.
* Add an **off-by-default** source-turn promotion experiment. Before the candidate cut,
  a fact may be replaced by a containing verbatim turn already in the authorized search
  results. Both must have exactly one matching source reference, the parent must have
  `predicate="said"`, and literal containment must hold. Exact lookups, documents, legacy
  records, multiple-source memories and truncated parents do not qualify. This adds no
  model call or database round trip and leaves the retrieval/token budgets unchanged.
* Disable ONNX worker spinning between tasks. The encoder shares CPU with stores;
  the scheduling change does not change model weights, precision, tokenization or pooling.
  ONNX documents the CPU-versus-wakeup tradeoff in its
  [thread management guide](https://onnxruntime.ai/docs/performance/tune-performance/threading.html).

The existing graph arm still does not expand relations into memory candidates, and the
configured engine RRF constant still does not configure Qdrant's native fusion. These are
separate experiments, not changes smuggled into the source-turn comparison.

## Measurement protocol

The development corpus is the first two conversations, **304 questions**, not the full
1,986-question benchmark. No number measured on this subset is a full-set achievement.

`codex_baseline_shipped.json` records a fresh run of the original code. Its complete-evidence
proxy is 0.7532, with p50 135.9 ms / p95 542.8 ms / p99 922.7 ms. A type-checking process
overlapped part of the query phase, and the first query took 3.87 seconds. Its latency is
diagnostic only, not a clean baseline for attributing an optimization gain.

The source-turn trial saves a paired control with promotion disabled on the **same index,
same questions and same budgets**. It uses a separate bundle-cache fingerprint and restores
the configuration even on failure. Saved contexts allow the reader to be tested without
repeating ingestion. Reader/control call order alternates; identical contexts share one
prediction. Only pairs with two valid verdicts enter paired accuracy. Failures are counted
explicitly, never replaced with token overlap. The strict prompt and model are held fixed.
The reader runs at bounded concurrency four, paced to 120 logical model calls/minute.
Successful checkpoint pairs are reused on resume; source bytes, question identities,
prompts and model configuration are checked before reuse. The initial serial session's
30 completed rows were preserved when switching to this driver.

The latency probe runs after other validation finishes, using the already indexed last
conversation. Bundle caching is disabled, ten warmup queries are excluded, and authorization
scope caching remains warm. Spinning-on and spinning-off arms run **sequentially**. Saved
memory IDs allow ranking equivalence to be checked. This measures single-client retrieval,
not HTTP throughput, cold startup, answer generation or NLI verification.

New artifacts include a SHA-256 of Python sources, in addition to the commit, so an
uncommitted experiment is distinguishable from the original tree. Source-ID recall is an
exact provenance measurement; the existing lexical/rank measurements remain available.

## Reproduction

Use the project's runtime Docker image and the isolated `memory_bench_conv` database.
`make bench-locomo` supplies the normal benchmark environment:

```sh
make bench-locomo BENCH_EXTRA_ENV='-e BENCH_DEPTH=shipped -e BENCH_THREADED_INGEST=1' \
  LOCOMO_ARGS='--conversations 2 --source-turns --save-contexts --out codex_source_trial.json'

# Inside the same runtime environment, after the preceding run finishes:
python -m benchmark.locomo_latency --conversation 1 --repeats 3 --spinning --out spin_on.json
python -m benchmark.locomo_latency --conversation 1 --repeats 3 --out spin_off.json

# With the existing DeepSeek/Bifrost benchmark environment enabled:
python -m benchmark.locomo_reader benchmark/results/codex_source_trial.json \
  --control-field control_context --out codex_source_reader.json --concurrency 4
# Add --resume to reuse valid pairs from an interrupted or partially failed run.
```

Existing deployments need a normal, non-destructive memory reindex to populate `source_refs`.
The existing `memory_service.tools.reindex --tenant <tenant>` command upserts from canonical
PostgreSQL records; no new schema migration or collection deletion is required. Benchmark
ingestion exercises the new payload automatically. No production corpus was reindexed here.

## Verification results

### Retrieval experiment

`codex_source_trial.json` completed all 304 questions. Among the 231 answerable questions
with annotated evidence, complete-evidence recall was **174/231 (75.32%)** in the control
and **175/231 (75.76%)** with source-turn promotion. Multi-hop remained **20/43 (46.51%)**.
This small observed difference does not justify enabling the feature or claiming an answer
accuracy gain. The strict reader experiment is recorded separately.

Exact source-ID completeness in the candidate arm was **168/231 (72.73%)**, including
**16/43 (37.21%)** multi-hop questions. This asks whether every annotated source has a
retrieved memory pointing to it. It is neither answer accuracy nor proof that the full
supporting text survived extraction. The two answerable open-domain questions without
annotated evidence are excluded from these retrieval denominators but retained for judging.

### Strict paired DeepSeek reader

`codex_source_reader.json` completed **304 valid pairs with zero remaining call failures**.
Both arms use `deepseek/deepseek-flash` and the existing strict judge. One empty answer in
the first pass was treated as a failure, not an incorrect answer; the entire failed pair
was rerun while preserving the other 303 pairs. `codex_source_reader_first_pass.json`
retains that failure. No valid incorrect answers were selectively retried.

| Category | Questions | Promotion off | Promotion on |
|---|---:|---:|---:|
| Answerable overall | 233 | **189/233 (81.12%)** | **188/233 (80.69%)** |
| Multi-hop | 43 | 24/43 (55.81%) | 25/43 (58.14%) |
| Open-domain | 13 | 10/13 (76.92%) | 9/13 (69.23%) |
| Single-hop | 114 | 95/114 (83.33%) | 94/114 (82.46%) |
| Temporal | 63 | 60/63 (95.24%) | 60/63 (95.24%) |
| Adversarial abstention | 71 | 55/71 (77.46%) | 59/71 (83.10%) |

There were four answerable wins and five losses: **−0.43 percentage points**. A paired
question-level bootstrap (20,000 resamples, seed 20260925) gives a 95% interval of
**−3.00 to +2.15 points**; exact two-sided McNemar p=1.0. These statistics do not account
for clustering within just two conversations. Details are in `codex_comparison.json`.

**Decision: keep source-turn promotion off.** This trial supplies no reliable answer
accuracy gain. The control includes the provenance fix and source grouping, so this is
not an ablation of those changes against the original code. Its 81.12% on two conversations
must not be compared as an improvement over the historical 72.92% on all ten. **85% full-set
accuracy has not been achieved or established by this work.**

### Isolated scheduling comparison

All four runs used the same 472-memory index, 105 questions repeated three times, zero
bundle-cache hits, the same Python source digest and the same model. No other benchmark or
test suite ran during these query phases.

| Run | Spinning | p50 ms | p95 ms | p99 ms |
|---|---|---:|---:|---:|
| `codex_spin_on.json` | On | 108.0 | 269.5 | 384.7 |
| `codex_spin_off.json` | Off | 104.7 | 181.5 | 259.7 |
| `codex_spin_on_repeat.json` | On | 102.6 | 206.8 | 325.0 |
| `codex_spin_off_repeat.json` | Off | 99.1 | 183.0 | 231.0 |

Both optimized runs met 300 ms **on this warm, single-client, uncached retrieval workload**.
This is not a full-corpus, cold-start, HTTP or concurrent-load SLO certification.

`codex_scheduling_parity.json` compared real-model embeddings for 100 questions spread
across all ten conversations: **bit-identical vectors, maximum absolute difference 0.0**.
This supports retaining the scheduling change without a model-quality tradeoff.

Retrieval ordering itself is variable: only 47/315 ordered memory lists matched between
the first on/off runs, but only 61/315 matched between the two unchanged on runs. Candidate
set overlap was 99.85% and 99.88%, respectively. Do not mistake this existing ordering
variation for a measured embedding change, or a one-question retrieval delta for a robust
accuracy gain.

### Code checks

Ruff lint/format and Pyright passed (the host has 21 warnings for optional model/cloud
packages unavailable outside Docker). Exported OpenAPI is byte-identical.

Across the completed suite segments and targeted reruns, **1,052 distinct test cases passed**,
with 12 skips; real-model, Docker and Bifrost markers were excluded from the host suite.
The separate integration run passed 160 tests. `codex_validation.json` records each test's
latest result and the preceding failures; `codex_gate_results.json` preserves the new gate
reports without overwriting historical benchmark artifacts.

Two stale gate assumptions were corrected after reproducing the failures:

* The graph gate demanded one prose relation's ID even though traversal correctly selected
  a higher-confidence table relation. It now requires the complete golden subject,
  predicate, object, attributes, page and source document. Graph semantics are unchanged.
* The tool-learning gate marked training calls `RUN` while expecting an unrelated probe
  run to learn from them. Training is now `PRIVATE` to the agent, as in the integration
  tests. Cross-agent isolation and undeclared-tool assertions remain intact.

The new reader tests cover incomplete-pair exclusion, explicit failures, checkpoint
identity/model/context validation, resumption without duplicate calls, concurrency limits,
and restoration of the retrieval control's configuration after errors.

## What to test next for 85% accuracy

These are hypotheses requiring experiments, not projected accuracy gains:

1. Measure the strict DeepSeek reader on intact annotated source turns, then with realistic
   distractors. This separates missing evidence from reader errors before more retrieval
   changes. Keep gold-only contexts confined to evaluation; never use them in retrieval.
2. For missed multi-hop evidence, measure reachability from retrieved seeds. The current
   graph arm emits facts/chunks, not linked canonical memory candidates. If missing turns
   are reachable, test one bounded, authorization-filtered batch expansion into memories,
   with per-entity fan-out and a fixed context budget. Measure its extra round trip against
   the remaining latency budget before enabling it.
3. Verify native fusion behavior and per-arm rank stability before changing RRF weights or
   increasing depth. More candidates alone has not established an answer-quality gain.
4. Promote only changes that improve paired strict answer accuracy without sacrificing
   adversarial abstention, then confirm on all 1,986 questions and an untouched validation
   corpus. Run concurrent HTTP/load and cold-start tests separately for the latency SLO.

The 300 ms results here concern retrieval. DeepSeek answer generation takes seconds and
is outside that measurement; an end-to-end answer SLO would need a different budget.
