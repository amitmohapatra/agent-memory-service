# Paired consolidation answer diagnostic — 2026-09-26

The latest ingestion/consolidation changes have **no established overall accuracy gain**.
In a completed ten-question development diagnostic, both arms supplied all rubric-required
facts on nine questions. Strict manual grounding scored native 8/10 and assisted 9/10,
but the one-point difference depends on a debatable inferred reason. This is not a new
LoCoMo result or evidence that the service achieves 90% overall.

## Design and results

The native arm reused six canonical memories from genuine earlier live ingestion. The
assisted arm reused sixteen extracted memories plus five final GPT-4o factual reflections.
All six earlier speculative GPT-4o-mini reflections were excluded. Both snapshots were
imported into a fresh isolated database and indexed anew; ingestion was not called again.
This compares the combined extraction/consolidation configuration, not reflection alone.

Ten new questions cover software and gardening, including multi-hop, actor attribution,
negation, abstention and causal overreach. The source corpus had already been inspected;
this is synthetic development data, not an independent holdout. Questions and rubric were
saved before answer generation. The same reader prompt/model/output cap was used within
each pair, and gold answers were excluded from reader requests. Arm order alternated.

| Metric | Native ingestion | LLM-assisted ingestion/consolidation |
| --- | ---: | ---: |
| GPT-4o-mini required-fact completeness | 9/10 | 9/10 |
| GPT-4o-mini strict manual correctness | 8/10 | 9/10 |
| Gemini completed pairs, strict correctness | 2/2 | 2/2 |
| Context builder median, cache disabled | 76.03 ms | 98.74 ms |
| Context builder maximum observed | 188.45 ms | 216.46 ms |
| Mean rendered context characters | 612.6 | 2,606.9 |

The assistant manually graded answers with arm labels masked; grades were saved before
decoding the key. This was not an independent human or external model judge. Each reader
answer was generated once. Native's extra strict failure asserts a reason for Noor not
watering on the first evening, using a later soil observation. Assisted avoids that
inference. Whether to reject native's wording is debatable, so required-fact completeness
is reported separately. Both arms fail the delivery rubric because they omit Tuesday;
the frozen rubric requires Tuesday although the question does not explicitly ask when.
Do not silently loosen that rubric after observing the outputs.

Gemini completed only the first two pairs before HTTP 503 and then a free-tier HTTP 429
on one bounded resume. Remaining Gemini questions are unscored. GPT-4o-mini completed all
twenty answer requests successfully. Do not pool the two readers or claim a Gemini uplift.

## Latency and request accounting

Retrieval used real PostgreSQL, Qdrant and ONNX dense encoding with BM25, one warm-up and
three timed builds per question/arm (thirty per arm). Authorization and task queue were
in-process, NLI was lexical, and the corpus was tiny. These are builder timings, not HTTP
latency, a load test, production p99, or answer-generation latency.

GPT-4o-mini reader request medians were 1,459.57 ms native and 1,094.32 ms assisted;
Gemini's two-per-arm medians were 3,966.05 and 4,702.57 ms. These exclude rate-limit pacing.
Small sequential samples and stochastic generation do not establish a reader speedup.

This turn made 28 model requests: two route preflights, twenty paired mini answers,
and six Gemini requests. Twenty-six succeeded; one returned 503 and one returned 429.
Reported OpenRouter cost was $0.0042609, including preflight. Gemini cost was not reported.
There were no new ingestion or external grading calls. Benchmark jobs have stopped.

## Reproduction and handoff

- Harness: `benchmark/consolidation_accuracy.py`. It validates snapshot dependency
  revisions, refuses duplicate imports/results, and emits a gold-free reader manifest.
- Isolated database: `memory_consolidation_answer_eval`, migration 0009. Qdrant collection
  suffix: `__consolidation_answer_eval`. Original benchmark databases were not reset.
- Results share prefix `benchmark/results/codex_consolidation_answer_eval_`: `design.json`,
  `contexts.json`, `manifest.json`, `mini.json`, `gemini_manifest.json`, `gemini.json`,
  per-reader masked answers/keys/grades, `summary.json`, and `validation.json`.
- Snapshot inputs: `codex_consolidation_live.json` and
  `codex_consolidation_factual_live.json`. Their hashes are recorded in the contexts file.
- Run context preparation in the existing model-capable Docker image
  `memory-service-memory-api:latest`, mounting the repository at `/app`, setting
  `PYTHONPATH=/app/src:/app`, `CONSOLIDATION_DB_HOST=host.docker.internal`, and the Qdrant
  URL to `http://host.docker.internal:6333`, with `python -m benchmark.consolidation_accuracy`.
  Existing results are intentionally protected; do not delete them to repeat an experiment.
  Use a separately named experiment/database/collection for a new run.
- The host venv lacks the real ONNX runtime. Do not substitute hash embeddings and call
  that an equivalent rerun. Reader execution uses the existing Bifrost route and the
  resumable `benchmark.targeted_reader` harness; credentials remain outside result files.

Final focused validation: **19 passed**, Ruff clean, Pyright zero errors/warnings.
Tests cover snapshot membership/dependency validation and gold-free symmetric reader jobs,
plus the existing targeted/quota reader suites. Final test artifact:
`codex_consolidation_answer_eval_tests_final_verified.xml`. Earlier XMLs preserve fixture
failures that were corrected. Post-evaluation edits only initialize the warm-up bundle
explicitly and broaden the checkpoint type annotation to accept manifests; no answer or
grading result was regenerated. No application code was changed in this evaluation turn.

The original full retrieval artifact remains unchanged: source recall 83.90%, complete
source coverage 77.54%, and serial builder p99 550.56 ms. The earlier incomplete answer
evaluation remains 76.78% on 1,206 valid answerable questions. Neither number measures the
latest LLM-assisted ingestion on full LoCoMo. Original artifact hashes are in validation.

Next meaningful evidence requires a frozen, broader comparison using the same reader and
ruler on both native and assisted ingestion. Include unseen non-LoCoMo sources, actor and
temporal distractors, source updates/deletions, and longer corpora. Separate extraction and
reflection ablations to attribute any gain. A complete latest-LoCoMo accuracy claim still
requires reingestion and a full paired answer run; it was not performed here. Gemini quota
limited the cross-check, but did not prevent the completed mini diagnostic.
