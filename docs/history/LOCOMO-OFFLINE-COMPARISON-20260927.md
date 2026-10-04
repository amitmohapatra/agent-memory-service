# Full LoCoMo source retrieval comparison — September 27, 2026

All three arms processed **1,986 questions across all 10 conversations**, with **zero paid
model calls**. Source coverage is scored on **1,536 annotated answerable questions**.
Four answerable questions lack source annotations; the 446 adversarial questions are not
counted as source-recall accuracy. No fresh answer-generation or judge results were produced.

## Results

Percentages below are direct annotated-source coverage. `Complete` means all annotated
source IDs occur within the first K returned memory items. It does not mean those fragments
contain a complete answer. Graph and derived items consume ranking positions normally.

| Arm | Recall@10 | Complete@10 | Recall@50 | Complete@50 | Recall@100 | Complete@100 |
|---|---:|---:|---:|---:|---:|---:|
| Write-path base `f988c26`, English Granite | 60.35% | 54.36% | 77.26% | 70.77% | 84.22% | 78.06% |
| Integration changes, English Granite | 60.27% | 54.30% | 77.42% | 70.96% | 84.08% | 78.06% |
| Integration changes, E5-small int8 | **64.38%** | **58.53%** | **79.70%** | **73.18%** | **85.61%** | **79.23%** |

The code changes with the same English encoder produced little retrieval-quality change.
E5 improved the aggregate and multi-hop results, but it regressed on open-domain questions.
**85.61% source recall@100 is not 85% answer accuracy, and neither is a 90% result.**

| Category | Questions | Granite recall@50 | E5 recall@50 | Granite complete@50 | E5 complete@50 |
|---|---:|---:|---:|---:|---:|
| Single-hop | 841 | 84.21% | **87.16%** | 82.76% | **85.85%** |
| Temporal | 321 | 84.09% | **84.61%** | 82.24% | **82.55%** |
| Multi-hop | 282 | 58.04% | **61.92%** | 32.27% | **35.82%** |
| Open-domain | 92 | **51.52%** | 48.84% | **42.39%** | 39.13% |

E5 versus updated Granite at complete@50: **70 wins, 36 losses** (net 34/1,536).
At complete@10: 97 wins, 32 losses. Question-level paired statistics are in the comparison
JSON; conversations cluster questions, so those p-values are descriptive rather than a
claim of independent held-out product validation.

## Latency

| Arm | p50 | p95 | p99 |
|---|---:|---:|---:|
| Base Granite | 115.08 ms | 256.54 ms | 367.62 ms |
| Updated Granite | 118.22 ms | 211.50 ms | 327.78 ms |
| Updated E5 | 112.97 ms | 305.24 ms | 489.84 ms |

These are **sequential context-builder measurements**, including PostgreSQL and the
isolated network Qdrant, excluding HTTP middleware/serialization and production load.
Authorization and cache use in-process adapters; grounding uses lexical NLI. Other tests
and processes ran on the host at different times, so this is not a controlled p99 speedup
claim. One E5 graph traversal logged a 150 ms budget expiry. Model/graph deadlines can affect
returned evidence under host contention. Do not advertise a 300 ms production p99 from this.

## Attribution and corpus

Each arm re-ingested the full conversations into its own baseline/current benchmark state;
no shared benchmark database was migrated. Consolidation was disabled in all three arms.
The new agent credential and brief modules were developed in a second worktree and are
not part of these recorded retrieval source manifests.

| Inventory | Base Granite | Updated Granite | Updated E5 |
|---|---:|---:|---:|
| Observed turns | 5,882 | 5,882 | 5,882 |
| Canonical memories across conversations | 7,057 | 7,147 | 7,147 |
| Directly represented turns | 5,588 | 5,588 | 5,588 |
| Questions with every gold source represented somewhere | 1,521 | 1,521 | 1,521 |

The two updated encoder arms have equal aggregate memory counts, but two conversations
exchange one canonical record each. The encoder is used during ingestion as well as recall;
this is a pipeline comparison, not a strictly fixed-corpus reranking ablation. Source-ID
inventory alone does not certify full source-content preservation or answerability.

## Reproduction and files

Artifacts are in `ams-hindsight-integration/benchmark/results/`:

- `locomo_source_baseline.json`
- `locomo_source_changed_granite.json`
- `locomo_source_changed_e5.json`
- `locomo_source_comparison_20260927.json`

Each full run records imported source-file hashes, model profile, dataset SHA-256, category
metrics, per-question ranking/source IDs and timings. The generic git provenance reflects
the harness checkout; the baseline's imported-source manifest identifies the older code.
Use that manifest when attributing the baseline.

Harness: `benchmark/native_source_retrieval.py`. The run environment and exact commands are
in the ignored `.bench_data/hindsight-integration/serial-benchmarks.sh` in that checkout.
The baseline imports `ams-codex-write/src`; current arms import the frozen integration source.
Qdrant uses ports 16333/16334. The databases are `memory_hi_perf_base_20260927` and
`memory_hi_perf_changed_20260927`. Do not point reset/ingest commands at another database.

See `MULTILINGUAL-CPU-EVALUATION-20260927.md` for the separate 12-language paragraph task.
Full SciFact encoder/fusion screening is a different experiment and is reported separately.


## Residual source-coverage diagnosis

The saved E5 candidates distinguish ranking depth from absent source records. No model
calls or answer judging were used. Buckets are exclusive and preserve the original
1,536-question denominator, including three questions referencing nonexistent source IDs.

| Outcome | All annotated answerable | Multi-hop |
|---|---:|---:|
| Every gold source present by rank 50 | 1,124 | 101 |
| Every gold source first complete by rank 100 | 93 | 30 |
| Gold sources represented in corpus, still incomplete at rank 100 | 304 | 147 |
| An observed source lacks a canonical representation | 12 | 2 |
| Gold annotation names a turn absent from the dataset | 3 | 2 |
| Total | 1,536 | 282 |

This points to candidate retrieval/ranking as a remaining problem: wider output alone
recovers 30 multi-hop questions, but 147 still lack at least one annotated source even
though the source is represented in the corpus. It does not prove that extraction is
complete: a record carrying a source ID can retain only part of that turn. It also does
not show that a reader would answer correctly once every source is present.

The failures include entity attributes, lists of activities across sessions and comparison
questions. General entity/topic coverage and evidence-set retrieval are plausible next
experiments; no question-specific production rules were added from these examples.
The three invalid annotations stay in the score, avoiding an after-the-fact denominator
improvement. Exact input/dataset hashes and category counts are recorded in
`benchmark/results/locomo_source_failure_buckets_20260927.json`.
