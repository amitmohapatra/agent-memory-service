# Full LoCoMo evaluation — 26 September 2026

Status: full LLM-free retrieval and exact-source audit complete; reader blocked by provider
HTTP 402 (payment required). **1511 valid question pairs are saved; 475 remain incomplete
or untested.** No full-current-code answer accuracy result is available yet.

## Default promotion after measurement

The user accepted measured retrieval p99 **551 ms**. The working-tree defaults now match
the measured wider arm: `memory_recall_k=100`, `memories_max=100`, `token_budget=8000`.
Its source recall is **83.905%**. This is a configuration promotion backed by the saved
full evaluation, not a fresh latency measurement or a production deployment. The measured
source hash below predates this default change. API/load latency and generated-answer
latency are not included in the 550.6 ms figure.

All saved control/candidate results retain their original meaning. To replay the previous
control, explicitly set `memory_recall_k=0` and `memories_max=50`; the archived orchestrator
assumed the then-current defaults and must not be rerun unchanged. Reader resumption uses
the saved contexts, so the default promotion does not invalidate completed judgments or
require retrieval to run again.

## Provider block and recovery

The reader was interrupted after provider failures opened the client circuit breaker.
A separate small provider probe returned `LLMCallFailed: gateway returned 402`.
The checkpoint contains 1577 attempted rows, 1511 valid pairs and 120 failed arm results
(102 circuit/unavailable results, 17 `LLMCallFailed`, one `ValueError`). These are not
wire-call counts: identical-context errors can be copied, and circuit failures send no
model request. These failures
were not converted into scores. The interrupted checkpoint is preserved separately as
`codex_full_current_llm_interrupted.json`; status and error counts are in
`codex_full_current_status.json`. The partial cohort must not be presented as full LoCoMo
answer accuracy.

Thirteen complete pairs were reused from earlier experiments after exact input validation.
The initial reuse manifest is `codex_full_current_llm_reuse_initial.json`. The initial
reader process was interrupted before its final cache counters were written; resumed
process counters must not be presented as totals for the entire run.

After restoring provider billing/credit access, the existing container can resume:

```bash
docker start memory-full-current-reader
docker logs -f memory-full-current-reader
# Only after complete=true, paired_n=1986 and failures=0:
PYTHONPATH=src:. .venv/bin/python /tmp/analyze_full_locomo.py
```

The wrapper preserves valid checkpoint pairs and reattempts incomplete/untested ones.
It now cancels further work immediately on HTTP 402 and writes the checkpoint and available
per-process statistics. Six tests passed, including actual reader checkpoint preservation
on cancellation. Initial wrapper/helper/test versions are retained with `_initial`
filenames; current sources and validation are archived alongside them. Only evaluation
orchestration changed: the measured service source hash, prompts and model are unchanged.
If `/tmp` files are lost, reconstruct them from the corresponding `*.py.txt` artifacts
before starting the container. Do not re-ingest or repeat the retrieval phase.

## Full LLM-free results

All 1986 rows across ten conversations were evaluated, with the previous defaults as control
and memory recall 100 as candidate (now promoted). These are evidence metrics, not answer accuracy.

| Metric | Previous default | Wider recall (new default) |
|---|---:|---:|
| Exact annotated-source recall | 77.404% | 83.905% |
| Complete annotated-source coverage | 70.964% | 77.539% |
| Multi-hop source recall | 58.765% | 68.342% |
| Multi-hop complete-source coverage | 32.624% | 42.908% |
| Single-hop source recall | 84.324% | 90.091% |
| Temporal source recall | 83.048% | 87.902% |
| Open-domain source recall | 51.588% | 61.109% |
| Retrieval p50 | 121.2 ms | 162.9 ms |
| Retrieval p95 | 244.6 ms | 333.0 ms |
| Retrieval p99 | 439.7 ms | 550.6 ms |

Evidence recall covers 1536 answerable rows with annotations; four open-domain rows have
no evidence IDs. All 1540 answerable and 446 adversarial rows remain in the reader cohort.
Each latency arm has 1986 timed requests. Neither arm meets the original 300 ms p99 target.
LLM-free retrieval does not generate answers, so generated-answer accuracy and adversarial
answer abstention are not defined for this phase.

The exact-source audit verifies all 5882 turns against body, speaker, date and strict
ingestion order, and all 6703 selected memory IDs against indexed and canonical evidence.
Nine turns have ambiguous body matches; using exact identities changes none of the saved
recall measurements. Every ordered question/gold/category/reference-date row matches the
dataset, including duplicate question text, and aggregate metrics were independently
recomputed. See `codex_full_current_exact_sources.json` and
`codex_full_current_nollm_validation.json`. Source presence still does not prove that
extraction preserved every fact needed for the answer.

## Protocol and execution

The user requested full-dataset results with and without LLM use, reusing existing results
where valid. Saved historical full runs have no rendered contexts or current source digest,
so their reader judgments cannot be reused as current-code answers. The initial interpretation
is native LLM-free ingestion/retrieval, followed by DeepSeek Flash answer generation and the
existing strict judge over saved contexts. An optional clarification asks whether the user
also intends a separate LLM-assisted ingestion/extraction experiment.

Runtime/benchmark source SHA-256:
`bae240896abe90e9e9b7a5cf4e4403f983c4659ea0338c15dc9ffc54a1d33114`.
This is the source hash at measurement, before the default promotion. The orchestration script is saved
as `benchmark/results/codex_full_current_orchestrator.py.txt`, with its own hash in results.

The run composes `benchmark.locomo_probe` on all ten conversations. It compares current
then-shipped depth with memory recall 100 at the same 8000-token budget. All ingestion
finishes before timed queries. Each arm receives one alternating timed pass per question,
with ten warmups per conversation and bundle caching disabled. There are 1986 dataset rows,
including 1540 answerable and 446 adversarial. Question text is not a unique row identifier.

Native threaded ingestion uses tenants `bench_accuracy_conv0` through `bench_accuracy_conv9`
in `memory_bench_accuracy`. The existing, audited conversation 2 corpus is reused; other
empty tenants are ingested. Existing benchmark databases are not reset. The shared Qdrant
collection's BM25 corpus statistics change during ingestion, but the measured index is fixed
before any paired retrieval passes. Do not compare cross-run rankings as if indexing were
unchanged. Failed partial ingestion must be audited before reuse.

The no-LLM outputs measure annotated source recall/completeness. They do not constitute
answer accuracy or correct generated abstention. Retrieval timings use real ONNX/BM25,
PostgreSQL and Qdrant, with in-memory auth/cache and lexical NLI benchmark stand-ins; they
exclude HTTP/load and generated answers. Reader errors must remain errors, never overlap
fallback scores. Existing complete reader pairs may be reused only after exact context,
question, gold, category, reference-date, prompt and model identity checks.

The reader additionally memoizes identical judge requests: question, gold, generated answer
and ruler must all match, under the same model and prompt. Context is not an input to this
judge. Concurrent identical requests share one judgment; failures are never cached as
scores. The wrapper paces actual logical model calls at 120/minute with eight workers,
including answer generation. Cache hits do not consume a pacing slot. Initial isolated tests
passed for concurrent reuse, changed inputs and failed-call retries; seven fields in paired
context reuse were checked independently. See `codex_full_current_harness_validation.json`.
The wrapper/helper/test sources are archived as `codex_full_current_*py.txt` artifacts.

Evaluation orchestration note: the retrieval script had a final assertion
that assumes `(conversation, question text)` is unique. The dataset actually has 1974 such
keys across 1986 rows. That assertion failed *after* writing all complete results. The
original script/log are retained. The archived `codex_full_current_validation.py.txt`
validator passed: it compares every ordered row against
the dataset including question, gold, category and reference date and recomputes aggregate
metrics. Do not deduplicate rows to satisfy the incorrect postcondition. The reader wrapper
also performs the ordered row validation before any generation calls.

The additional offline exact-source audit passed; its script is archived in
`codex_full_current_source_audit.py.txt`. It verifies sequential observation order against
each turn's body, speaker and date, compares indexed and canonical memory evidence, and
reconstructs both exact-ID and body-matched recall from the saved memory selections.
It ran after retrieval timing ended and did not requery the reader or alter scored contexts.

Retrieval artifacts:

- `codex_full_current_ingestion.json`: per-conversation source audits.
- `codex_full_current_conv{0..9}.json`: complete per-conversation retrieval pairs.
- `codex_full_current_nollm.json`: complete full-corpus aggregate.

The retrieval container `memory-full-current-nollm` is stopped; its final incorrect
uniqueness assertion and successful replacement validation are documented above. The
`memory-full-source-audit` container exited successfully. Reader container
`memory-full-current-reader` is stopped and retained for resumption. Preserve saved artifacts
and remove only this run's temporary containers after successful completion. Keep underlying
services and benchmark databases.
