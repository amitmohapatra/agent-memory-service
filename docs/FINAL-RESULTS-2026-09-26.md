# Final experiment handoff — in progress

Read [the primary-source research and measurement contract](FINAL-RESEARCH-2026-09-26.md).
This file is a live handoff; pending measurements below are not claimed complete.

## Scope and immutable baseline

User authorized judicious use of a new $10 OpenRouter top-up. Initial target new spend
was <= $8, leaving $2 margin. A final bounded actor/topic answer validation adds at most
$0.65, with a cumulative ceiling of $9 and at least $1 margin; see protocol amendments. All credentials remain in Bifrost/ignored local environment files.
No deployed application restart, commit or destructive reset has occurred.

Original full native evidence remains `codex_full_current_nollm.json`: 1,986 questions,
source recall 83.9047%, complete-source coverage 77.5391% over 1,536 annotated answerable
questions; multi-hop 68.3416% / 42.9078%. Historical serial builder p99 550.56 ms is not
a new production/load p99. Original DeepSeek output covers 1,206 valid answerable
questions, 926 correct (76.7828%); do not call that a completed full answer evaluation.

## Completed at this checkpoint

- Frozen 369-turn conversation-1 experiment completed: 37 questions, identical GPT-4o
  reader, raw judge 34/37 native vs 32/37 assisted. No assisted gain. Separate manual
  completeness audit 31/37 vs 30/37. Artifacts: `codex_complex_consolidation_replay_*`,
  `codex_final_complex_manual_audit.json`, `codex_final_raw_source_audit.json`.
- Final provenance-dedup measurement completed on all 1,986 questions and audited all
  5,882 turns: source recall **83.8548%**, complete-source coverage **77.4740%**;
  multi-hop **68.2825% / 42.9078%**. Control was 83.9047% / 77.5391% overall.
  This is a small regression, not an accuracy improvement. Serial builder p50 **178.99 ms**,
  p95 **346.79 ms**, p99 **527.89 ms**. No query-time LLM. Control timings were captured
  separately under different contention; do not attribute the difference solely to code.
  Artifact: `codex_final_verified_retrieval.json` (final structural-subject normalization).
- Full SciFact validation: 5,183 documents, 300 queries, nDCG@10 **0.7436**,
  true recall@10 **0.8731**, hit rate@10 **0.8867**. Matches previous canonical quality.
  Serial query p50 **184.6 ms**, p95 **414.3 ms**, p99 **2,041.0 ms**, including cold first
  query; not HTTP/load latency. Artifact: `codex_final_scifact.json`.
- Final broad regression selection: **962 passed**, zero failures/skips. Additional derived
  date rendering selection: **35 passed**; backlog/clone harness tests: **4 passed**.
  Changed modules passed Ruff/Pyright; complexity thresholds unchanged. Earlier broad
  failures were repaired (old copied-algorithm oracle, missing provenance fixtures,
  complexity), with evidence in the retained first-run and verified XML files.
- Spending before the active full reader/consolidation stages: **$2.0570055**, from
  read-only Bifrost billing since 2026-09-26 12:05 UTC, plus conservative allowance for
  four cancelled in-flight requests without recorded cost. Recompute before each stage.

## Completed full native answer evaluation

Fixed GPT-4o-mini reader and judge, unchanged original strict ruler; zero unscored questions.
No LLM ingestion/retrieval in this arm. 1,974 exact distinct reader inputs cover all 1,986
questions. This is a new full model/judge configuration, not a comparable uplift against
the older partial DeepSeek run.

| Category | Correct / total | Automated accuracy |
|---|---:|---:|
| All questions | 1,361 / 1,986 | 68.5297% |
| Answerable | 1,107 / 1,540 | 71.8831% |
| Multi-hop | 160 / 282 | 56.7376% |
| Temporal | 186 / 321 | 57.9439% |
| Open-domain | 35 / 96 | 36.4583% |
| Single-hop | 726 / 841 | 86.3258% |
| Adversarial abstention | 254 / 446 | 56.9507% |

Artifact: `codex_final_native_answers_summary.json`. The 25-case fixed-seed manual
completeness sample is a separate diagnostic, not an overall estimate or independent judge.
It records both false positive/negative judging concerns without changing gold or labels.
The native pass is complete; pending experiment 1 below is retained as its reproduction
record. The higher source-coverage numbers are not answer accuracy or answer abstention.

## Completed full background-consolidation retrieval

All ten eligible backlogs drained before queries: **396 calls, $0.2447565 billed,
636 stored insights, zero eligible pending revisions**. Pre-send reservations totaled
$1.027443 of the $1.40 cap. Every question retrieved insights, averaging 21.33 per bundle;
all 1,986 rendered contexts changed and therefore require fresh candidate answers.

| Source metric (n=1,536 answerable with annotations) | Native | Consolidated |
|---|---:|---:|
| Source recall | 83.8548% | 84.0025% |
| Complete-source coverage | 77.4740% | 77.9297% |
| Multi-hop source recall | 68.2825% | 67.7449% |
| Complete multi-hop coverage | 42.9078% | 43.6170% |
| Temporal source recall | 87.9024% | 88.7072% |

Candidate serial builder p50 **214.94 ms**, p95 **370.96 ms**, p99 **483.95 ms**.
Control timings were captured separately; this is not a paired production latency result.
Direct original-source coverage does not grade the factual completeness of a synthesized
insight; the fixed-reader answer comparison remains the primary quality test. Its first watcher
was session **87766** (superseded); the balanced evaluator is session **69479**, running with exact native answer/judgment reuse and a stage
allowance of $3.5210051, based on $5.3789949 already billed plus $0.10 unknown-call margin.

## Current implementation

Provenance-aware dedup preserves source/speaker/time differences and fixes score updates
through discarded candidate aliases. It normalizes once and compares containment only
inside provenance groups. No new query-time model calls.

Reflection now preserves the source-audience intersection, includes bounded older related
facts/verbatim turns, constructs prompt lengths linearly and requests up to 2,048 output
tokens. Per-source revision receipts survive restarts, successful empty responses, updates
and late old-worker receipts. Native landing's raw-turn exclusion is preserved.
Migrations 0010 (recent-update index) and 0011 (reflection receipts) are required before
running the updated worker. See research limits: anti-join discovery and concurrent paid
consultations are not yet a fair leased work queue. Existing private reflections are not
automatically made shared.

Local ignored `.env` output cap was increased to 2,048. After validation, local main
`memory` database was upgraded from 0009 to 0011; its memory row count was unchanged.
The API/worker/image was not restarted and no paid production background job was launched.
`codex_final_local_migration.json` records the before/after schema and counts; experiments
continue to use isolated copies. An updated application/worker process is still required
for runtime rollout.

## Pending experiments and processes

1. Native full answer evaluation is COMPLETE, fixed GPT-4o-mini reader/judge, unchanged
   original strict ruler, 90 RPM, **$3.00 new stage ceiling**. Prefix/log
   `codex_final_native_answers`, exec session 73709. Current contexts occupy both arm
   slots in `codex_final_native_contexts.json`, producing one answer per distinct input.
   The older-control comparison (`codex_final_verified_answers`, session 58991) was
   deliberately stopped to avoid paying for an obsolete dedup variant. All successful
   outputs remain available and are reused by exact input; partial older-arm scores must
   not be presented as a completed comparison. `codex_final_protocol_amendments.json`
   records this cost-saving change. Final native-versus-consolidated coverage stays 1,986.
2. Full SciFact completed in follow-up pipeline session 32096. Its isolated database is
   `memory_final_docs`; original document database and index are preserved.
3. Full background consolidation and retrieval are COMPLETE in Docker `final-reflection` (same pipeline).
   It copies `memory_final_accuracy` into `memory_final_reflection`, migrates the copy,
   and clones all 8,026 original memory vectors to a suffixed collection, preserving
   initial BM25 corpus statistics. Repeated bounded worker passes **drain every eligible
   source revision across all ten tenants before any evaluation question**. Maximum
   500 wire requests / **$1.40 conservative pre-send reservation**, mini at 8 RPM to
   remain below the combined 100 RPM account limit. No question/gold enters ingestion.
   Prefix `codex_final_reflection`; output `_contexts.json`, log `.log`, budget `_budget.json`.
   Successful empty consultations receive durable receipts; failures remain pending.
4. After both finish, evaluate reflected contexts with the same reader/judge/ruler and
   `full_answer_eval --reuse-prefix .../codex_final_native_answers`. Reuse exact control
   answers and judgments, retaining artifact references. The revised evaluator is ACTIVE in session 69479 under `codex_final_reflected_balanced`.
   Determine its remaining allowance from actual cumulative billing, keeping at least $1 margin and $0.10 allowance for interrupted calls.
5. Canonical external artifact refreshed only after matching the frozen source digest;
   previous artifact preserved. External RAG gate: **3 passed**. Full Ruff check passed;
   **416 Python files** already formatted; changed modules Pyright: zero errors/warnings.
   Cross-thread real-database lifecycle selection: **3 passed**, including source deletion
   hiding a stale derived vector hit. Final paired outcomes still pending.

6. HTTP `/context` validation is queued in session 25314, log
   `codex_final_http_watch.log`. Runs after consolidation retrieval completes, on loopback
   port 8088 in two sequential temporary API containers. Same 200 evenly spaced queries
   per native/consolidated arm at concurrency 1 and 4; ten warmups, bundle cache disabled.
   Real ONNX/Postgres/Qdrant, trusted-dev authentication and the benchmark authorization/
   cache/task seams. Read-only endpoint requests except normal access bookkeeping. No LLM.
   This is a local closed-loop TCP probe, not a production OpenFGA/load SLA. Frozen server
   and client scripts are `.py.txt` artifacts so the measured source digest stays intact.

7. Existing actor/topic retrieval option: retrieval COMPLETE; original session 80598 was replaced
   by answer watcher 72833 (`codex_final_actor_answers_watch.log`). Original pipeline log
   `codex_final_actor_pipeline.log`, after both HTTP probes to avoid model/DB contention.
   Full 1,986 current-native contexts with only `memory_entity_search=True`; native
   contexts are the frozen control. No ingestion/model calls for retrieval. Then, after
   reflected answer evaluation finishes, the same mini reader/judge scores changed inputs,
   reusing native results. At most $0.65 new stage spend and $9 cumulative spending including
   earlier stages, with $0.10 held for interrupted calls. Original source-recall evidence is
   in `ENTITY-TOPIC-RETRIEVAL-2026-09-26.md`. No default is changed before answer validation.

A derived context timestamp is now explicitly labeled “summary created”; it is not an
asserted historical event date. Native evaluation banks contain zero derived rows, so this
rendering correction did not change their frozen reader inputs. Repeated question text in
LoCoMo can have different labels: all full comparison keys are conversation + original
question ordinal, not text. Original dataset and gold remain unchanged.

## Reproduction constraints

Use `.venv/bin/python` for host harnesses. Real ONNX evaluation uses Docker image
`memory-service-memory-api:latest`, `/app` bind mount, `PYTHONPATH=/app/src:/app`,
`BENCH_SEARCH=qdrant`, `BENCH_EMBEDDING=frozen`, `HF_HUB_OFFLINE=1`,
`OMP_NUM_THREADS=2`, `OPENBLAS_NUM_THREADS=1`. Services: Postgres host 5432,
Qdrant 6333/6334, Bifrost 8091; inside Docker use `host.docker.internal`.
Reflection additionally needs `CONSOLIDATION_DB_HOST=host.docker.internal` and its
isolated database URL. Never run evaluation cleanup against the original databases.

Do not infer answer accuracy from source recall, use partial denominators as whole-LoCoMo,
score provider failures wrong, alter gold to improve a score, or call this research
LoCoMo-independent validation. Do not claim 90% until the actual full result supports it.

## Latest scheduler amendment

The first reflected reader was interrupted after discovering a work-assignment issue:
reused control answers occupied two entire shards, leaving only two paid workers active.
`full_answer_eval` now stably partitions pending jobs before reused jobs and accepts several
immutable reuse prefixes. **16 focused tests passed**. Inputs, models, gold, output caps and
ruler are unchanged. All successful outputs remain reusable. The canonical reflected answer
prefix is now **`codex_final_reflected_balanced`**; its budget comes from current cumulative
billing. `codex_final_reflected_answers_summary.json` will be only a completion pointer for
the existing actor watcher, not a score file. Read the balanced summary directly.

This benchmark-only change makes the broad source digest differ from the completed SciFact
artifact. Re-run final SciFact/gate after the last source change; do not rewrite its recorded
provenance. API/runtime source files remain unchanged during these measurements.

## HTTP findings and budget continuity

The local TCP probe completed 200 queries per arm/concurrency, with no failed requests.
These are separate sequential runs, not randomized paired load/SLA evidence.

| HTTP arm | Concurrent clients | p50 ms | p95 ms | p99 ms |
|---|---:|---:|---:|---:|
| Native | 1 | 253.08 | 1420.02 | 3091.15 |
| Native | 4 | 822.02 | 1720.90 | 2852.89 |
| Consolidated | 1 | 268.82 | 590.73 | 1048.06 |
| Consolidated | 4 | 898.64 | 1269.50 | 1454.14 |

The 528/484-ms serial builder p99 values must not be presented as HTTP p99. A subsequent
selected-case diagnostic ran 80 host TCP and 80 in-container TCP requests, toggling gzip
and identity encoding. It found tails in encoding/search as well as additional API/host
time; it does not establish a single bottleneck or prove that disabling compression helps.
Artifacts: `codex_final_http_*` and `codex_final_http_profile_{host,container}.json`.
All temporary HTTP containers were stopped.

Read-only gateway log queries began returning `database disk image is malformed`, both
from the host and a Docker reader. No repair or write to gateway data was attempted.
Cost continuity is independently checked in `codex_final_cost_reconciliation.json`: paid
response receipts before the balanced stage sum to $6.16172325 versus the last readable
gateway total $6.16321365 (difference $0.0014904). Keep the higher snapshot, add new
balanced and actor response costs, and retain $0.10 for interrupted requests. This avoids
resetting the allowance or relying on a damaged log read. The actor stage uses 20-job
batches so conservative preflight reservations do not unnecessarily strand the last calls;
**19 scheduler/budget tests passed**. Cumulative ceiling remains $9.
