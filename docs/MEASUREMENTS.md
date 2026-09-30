# Measurements

Every number here was produced by a command in this repository against real models and a real
corpus, on one machine, on the date given. Where a measurement is not trustworthy it says so
and is not used to decide anything. Where I previously reported a number that turned out to be
an artifact, the retraction is recorded rather than deleted — the retractions are the most
useful part of this file, because each one is a way this harness can lie again.

**Hardware.** 2015 Intel i5, no AVX2, Docker VM limited to 5.8 GB. Every latency and
throughput figure is specific to it. Quality figures (recall, nDCG) are not.

---

## 1. Retractions

| reported | actual status | cause |
|---|---|---|
| LoCoMo recall 0.52% | withdrawn | harness matched verbatim dialogue turns against memories the pipeline had rewritten |
| LoCoMo abstention 0.0 | withdrawn | `evidence_status` read off `RetrievalResult` instead of `ContextBundle` |
| LoCoMo recall 4.35% | withdrawn | scorer required the whole gold answer as a contiguous substring |
| SciFact nDCG@10 0.012 | withdrawn | 93% of the corpus was never indexed before scoring |
| "ingest is O(n²)" | withdrawn | inferred from coarse sampling across a model load; the measured curve is flat |
| every retrieval score before 2026-09-20 | withdrawn | benchmarks truncated PostgreSQL but never the vector store, so each run competed against the previous runs' orphaned vectors |

Four of the five were the measuring instrument, not the system. Each is now guarded:

- scoring functions are pinned by `tests/unit/test_benchmark_locomo_scoring.py`
- `external_retrieval` refuses to emit a score if the corpus is not fully indexed
- `external_retrieval` refuses to emit a score if no candidate maps back to the corpus
- both harnesses print progress, so a slow run is distinguishable from a hung one
- per-question records are persisted, so a score can be re-derived without re-running
- each benchmark owns its own database, so one cannot truncate another's corpus (a chained
  LoCoMo run destroyed a SciFact index that had cost forty minutes to build; `--reuse-index`
  caught it by refusing to score an empty store)
- one shared `reset_store()` clears *both* stores, so a run cannot inherit another's vectors

## 2. Ingest throughput — 2026-09-20

`benchmark/data/scifact.json`, real granite embeddings, real Qdrant server, 150 documents per
batch, measured per batch to test for superlinear cost.

| batch | documents | drain | per document |
|---|---|---|---|
| 1 | 150 | 736 s | 4.91 s |
| 2 | 300 | 708 s | 4.72 s |
| 3 | 450 | 608 s | 4.05 s |
| 4 | 600 | 611 s | 4.07 s |

**Flat.** Cost per document does not grow with corpus size; there is no scaling defect in the
ingest path. It is simply expensive on this hardware — a full 5,183-document corpus needs
about six hours here. Each document pays a dense pass (granite, 30M params) *and* a sparse
pass (SPLADE, BERT-sized, ~110M params) plus chunking, entity extraction and graph building.

Consequence for benchmarking: corpus size must be chosen to fit the time available, and the
harness must refuse to score a corpus it did not finish indexing. It now does.

## 3. Reranker ablation, conversational memory — 2026-09-20

LoCoMo, 2 conversations, 60 questions each drawn as a seeded per-category sample, real models,
`make bench-locomo` with and without `--off rerank`.

| | rerank ON | rerank OFF |
|---|---|---|
| answer recall @10 | 18.5% | 16.3% |
| evidence recall | 41.3% | 41.3% |
| strict substring recall | 4.35% | 4.35% |
| query p50 | 1686 ms | 529 ms |
| query p95 | 3684 ms | 821 ms |

> **Superseded — see §3c.** This ran against a vector store that was never cleared between
> runs, and the two arms carried different amounts of leftover data. The numbers below are
> kept for the record, not for deciding anything.

Paired over the 92 answerable questions: both correct 14, both wrong 74, rerank-only 3,
no-rerank-only 1. **Exact McNemar two-sided p = 0.625.** Wilson 95% intervals overlap almost
entirely: 18.5% [11.9, 27.6] against 16.3% [10.1, 25.2].

**Identical evidence recall is the informative part.** The reranker is reordering the same
retrieved sources, not surfacing better ones, on this path.

Not concluded from this: anything about document RAG. LoCoMo exercises conversational memory,
where candidates are short rewritten memories. Cross-encoder rerankers are published on
passage retrieval, which is a different shape of problem.

## 3b. Reranker ablation, document RAG — 2026-09-20

BeIR/SciFact, 600-document subset (sized to the measured 4 s/document), 40 queries whose
relevant documents fall inside the subset, real granite embeddings, real Qdrant server.

| | rerank ON | rerank OFF |
|---|---|---|
| nDCG@10 | 76.64% | **80.54%** |
| recall@10 | 92.50% | 92.50% |
| query p50 | 10,868 ms | 895 ms |
| query p95 | 13,206 ms | 6,179 ms |

nDCG@10 = 0.766 with reranking is consistent with published SciFact figures for a hybrid
dense+sparse stack, which is the first evidence in this file that the retrieval path is
sound — the earlier 0.012 was a missing corpus, nothing more.

**Identical recall with different nDCG means the reranker only reorders inside the top ten,
and on this corpus it reorders it worse.** Costing 12× the query latency to do so.

Not yet established: whether a 3.9-point nDCG difference over 40 queries is signal. The
harness now persists per-query nDCG so the arms can be paired and tested; that re-run is
pending. Treat the direction as suggestive and the magnitude as unmeasured.

`comparable_to_published` is false by the harness's own flag: a 600-document subset has
fewer distractors than the full 5,183, so absolute numbers are inflated. The *comparison*
between arms is unaffected, because both arms see the same corpus and the same index.

## 3c. The contamination that invalidated everything above it — 2026-09-20

Each harness began with `TRUNCATE` over the SQL tables and stopped there. The vector store is
a separate server, and a SQL truncation does not touch it. So every run left its vectors
behind, and the next run retrieved against them.

The scale, measured when the fix first ran: a tenant that should have held ~780 chunks held
**4,288 knowledge vectors and 512 memory vectors**. Orphaned entries from earlier runs took
top-ten slots that could never map back to the corpus being scored.

The same benchmark, same code, same 600-document corpus:

| store | nDCG@10 | recall@10 |
|---|---|---|
| clean-ish | 76.64% | 92.50% |
| after accumulation | 35.82% | 42.50% |

Both figures came from runs I would have reported. Neither is a property of the retrieval
system. This is why `unmapped_candidates` is now recorded separately from the chunk-to-
document deduplication it used to be conflated with: 242 unmapped candidates was the signal
that something other than retrieval was wrong.

**This invalidates the LoCoMo reranker ablation too**, and not only its absolute numbers: the
two arms ran back to back, so the second arm carried the first arm's leftovers as well as
everything before it. The arms were not comparable. It is being re-run.

## 3d. Open: an intermittent `Illegal instruction` — 2026-09-20

Three `make bench-external` runs died with SIGILL (exit 132). A later run of the same command
on the same corpus completed. It is intermittent, and it has only ever happened here.

**What is established:**

| | |
|---|---|
| host CPU | Intel i5-5257U (Broadwell) — **has AVX2** |
| Docker VM `/proc/cpuinfo` | `avx bmi1 bmi2 fma sse4_1 sse4_2` — **no avx2** |
| Docker VM CPUID leaf 7 EBX | `0x001c0389`, AVX2 bit clear — **agrees with the kernel** |
| torch 2.14.0+cpu | `CPU capability: DEFAULT` — correctly avoiding AVX2 kernels |
| memory at crash | flat 867 MiB of a 5.8 GiB limit — not OOM |

**What that rules out.** The obvious story — a library reading AVX2 from CPUID while the
kernel masks it, then emitting instructions the VM cannot run — does not hold: both sources
agree there is no AVX2, and torch is behaving accordingly. "This box has no AVX2" is true of
the *guest*; the host has it, and Docker Desktop is not passing it through.

**What remains.** Either a wheel compiled with unconditional AVX2 and no runtime dispatch
(onnxruntime 1.30.0 is present, via fastembed, and does its own dispatch), or memory
corruption in a native extension — an intermittent SIGILL is a classic symptom of the latter,
because a corrupted function pointer lands in data and gets executed.

**Whether it needs fixing.** Not as a product change, on this evidence. It has occurred only
inside a virtualised CPU with AVX2 withheld, which is not a configuration any deployment
target has — AVX2 has been standard on server parts since 2013. The cheap insurance is
already in place: `PYTHONFAULTHANDLER=1` on all four benchmark targets, so the next
occurrence prints a Python traceback naming the frame instead of one line of shell output.
If that traceback lands in a native extension rather than an ISA fault, this becomes a real
bug and the assessment changes.

## 3e. The reranker, decided — 2026-09-20

BeIR/SciFact, **1,000 documents, 70 paired queries**, clean vector store, one shared index,
real models. This is the run that matters, because at 300 documents recall@10 was 100% on
both arms — the first stage saturated and the reranker had no headroom to show anything. At
1,000 documents it no longer saturates, which is precisely the regime a reranker exists for.

| | rerank ON | rerank OFF |
|---|---|---|
| recall@10 | 97.14% | **98.57%** |
| nDCG@10 | 79.33% | **84.51%** |
| query p50 | 11,151 ms | 533 ms |
| query p95 | 14,783 ms | 1,472 ms |

Paired, per query:

- queries the reranker **rescued** that the first stage missed: **0**
- queries the reranker **lost**: **1**
- nDCG better on 4, worse on 16, identical on 50
- **exact sign test p = 0.012**
- mean nDCG delta (ON − OFF) = **−0.0518**, 95% CI **[−0.0917, −0.0120]** — excludes zero

So it is *significantly worse*, not merely not better, and it never once did the thing it is
for. `retrieval.rerank` now defaults to **false**.

**Why, probably.** `ms-marco-MiniLM-L6-v2` is trained on web passage ranking. Reordering
scientific claim–evidence pairs is a different task, and a cross-encoder applied out of
domain can confidently reorder correct results into the wrong order. This is a finding about
*this default*, not about reranking as an idea — an in-domain reranker may well earn its
place, and the flag is still there to turn on.

**And the cost settles it either way.** 11.2 s per query against 0.53 s: 20 RPS needs ~161
cores with it and ~8 without. A component at 21x the latency has to buy something.

## 4. Degenerate input — 2026-09-20

`make bench-degenerate`: 18 queries and 11 observations that are empty, malformed, hostile or
in a script the tokeniser did not handle, against a small real corpus.

| | before | after |
|---|---|---|
| queries behaving as intended | 7/18 | 15/18 |
| unhandled exceptions | 1 | 0 |
| junk admitted to the store | 0 | 0 |

Three defects found, all fixed, all now covered by tests:

1. **An empty query returned `COMPLETE` with ten memories attached.** `overlaps()` treated an
   empty content-term set as vacuously overlapping everything.
2. **The abstention gate was ASCII-only** (`[a-z][a-z0-9\-]+`), so every Japanese, Chinese,
   Korean, Cyrillic, Greek and Arabic query produced zero terms and could never abstain.
   Fixing (1) alone would have flipped those to *always* abstain — strictly worse. CJK now
   uses character bigrams, as in Lucene's CJKAnalyzer.
3. **A NUL byte in an agent's tool result returned HTTP 500.** The document path had been
   sanitised since an uploaded file did the same thing; the observation and message paths
   never were.

Still open, deliberately recorded as failing: a prompt-injection string is answered rather
than declined because it shares one incidental word ("previous") with the corpus. A lexical
overlap gate cannot catch this; entailment can.

Also open: a 2,000-character garbage query costs 20 seconds. There is no query length cap.

## 5. Gateway behaviour with a reasoning model — 2026-09-20

Found by pointing the service at a real gateway with `gemini-3.6-flash` on a free-tier key.

- **Reasoning consumes the output budget before any text is produced.** "Reply with exactly:
  OK" burned 57 reasoning tokens; `max_tokens=16` returned HTTP 200 with an empty string and
  `finish_reason=length`. The adapter returned that as a completion, so an empty answer
  reached every call site as if the model had meant it. It now raises and names the cause.
- **A free key allows about six calls before `429`.** The adapter treats 429 as retryable but
  backed off 0.5 s then 1 s — three attempts inside 1.5 seconds against a per-minute quota.
  It now honours `Retry-After` when the gateway sends one.

## 5b. What a real rate limit did to a real run — 2026-09-20

The first judged LoCoMo run produced a complete-looking result document: recall 19.7%,
abstention 0.0, `abstention_measurable: true`. **Every one of its 79 judge calls had failed.**
The scores were the token-overlap fallback, and the run said nothing about it.

Three separate defects lined up to produce that:

1. **A provider counts wire requests, not logical calls.** The free tier allows 20 requests
   per minute. The harness paced 10 *questions* per minute — but each question is two calls,
   and `max_retries=2` makes each call up to three requests. A run "paced at half the limit"
   was sending up to 60 requests a minute.
2. **A rate limit tripped the circuit breaker.** 17 rate limits opened it, and the next 62
   calls failed instantly having sent nothing at all. A breaker exists so a *broken* gateway
   costs one timeout instead of one per request; a gateway answering 429 is working, and
   saying so. Rate limits no longer count toward it — in both repositories.
3. **The retry delay was not in `Retry-After`.** Gemini puts it in the error body ("Please
   retry in 59.18s"). A client that honours the header — which both of these now do — sees
   nothing there and falls back to a backoff measured in milliseconds against a window
   measured in a minute.

And the reporting defect that made it dangerous rather than merely annoying: the result
claimed the adversarial category had been measured. A judged run now counts its own failures
and says so in `caveats`, and `abstention_measurable` is false unless the judge actually
answered.

## 5c. What the Phase-2 gate has to know before it reads /metrics — 2026-09-23

Not a measurement: a correction to how the measurements of the next phase must be taken.
Recorded here because it would otherwise be discovered as a number that does not add up.

**The registry is per process, and the API now runs three of them.** `prometheus_client`
keeps counters in the memory of the process that incremented them. Three uvicorn workers
share one listening socket, so a scrape of `/metrics` is answered by whichever worker
accepts that connection. Every series on that endpoint is therefore one worker's share —
roughly a third of the traffic, from a different worker each time, so a counter read across
two scrapes can go *down*. The roadmap's own verification commands read exactly these:

- Phase 0 step 9: `curl -s localhost:8080/metrics | grep memory_stage_seconds_bucket | grep stage="encode"`
- Phase 2 step 7: `curl -s localhost:8080/metrics | grep http_requests_total | grep 429`

Both are one-third samples as the service ships. **The gate must scrape with one worker**
(`WEB_CONCURRENCY=1`, which now genuinely sets the worker count) **or run the client in
multiprocess mode** (`PROMETHEUS_MULTIPROC_DIR` + `MultiProcessCollector`, which is not
wired). A throughput or CPU number taken from a three-worker scrape is not wrong by a
constant factor, so it cannot be scaled back up afterwards.

Until multiprocess collection is wired, `/metrics` says this itself: it exposes
`memory_api_workers` as the divisor and prefixes the exposition with a `# SCOPE:` comment
block naming the pid and the worker count.

**A known tail contributor, for whoever reads the p99.** An idempotent Qdrant read that hits
a connection-level failure is retried once after a jittered pause of 50–100 ms
(`_RETRY_PAUSE_SECONDS`). The worst case for such a read is one timeout plus 100 ms plus the
second attempt, against a 300 ms budget. It is rare — four occurrences in 304 judged
queries through Docker's host gateway — but it is in the tail the gate measures, and
`memory_search_read_retries_total` counts every occurrence, per operation.

## 6. What is not measured yet

- Document-RAG reranker ablation (running; 600-document SciFact subset, one shared index)
- LoCoMo scored the way LoCoMo scores it — generate, then grade with a model. Only this makes
  the adversarial category meaningful: the abstention decision that matters lives downstream
  of generation, so with `llm.enabled=false` category 5 cannot be scored at all, and a 0.0
  there is a statement about the setup rather than the system.

## 7. The encoder's runtime: torch, ONNX fp32 and ONNX int8 — 2026-09-23

The encoder is the floor of every retrieval request, so Phase 2 assumed an int8 ONNX graph
and budgeted around it. Nothing had measured that. This is the first artifact that has.

```bash
# inside the runtime image; ./models mounted read-write only for the export
python -m memory_service.tools.download_models --export-onnx /models/granite-embedding-small-english-r2
python -m memory_service.tools.download_models --measure-onnx /models/granite-embedding-small-english-r2 \
  --threads 2 --out benchmark/results/encoder_runtime_devbox.json
```

Forty single-query encodes after five warm-ups, one query at a time, `intra_op_num_threads`
and `torch.set_num_threads` both 2, on the 4-core no-AVX2 Docker VM with nothing else
running. `benchmark/results/encoder_runtime_devbox.json` and, forty seconds later,
`_repeat.json`.

| runner | p50 ms | mean ms | p95 ms | ×torch (mean) | min cosine vs torch |
|---|---|---|---|---|---|
| torch fp32 (sentence-transformers) | 80.6 / 80.7 | 83.1 / 81.6 | 128.1 / 111.3 | 1.00 | — |
| ONNX fp32 (`onnx/model.onnx`) | 27.3 / 19.5 | 30.0 / 24.2 | 51.5 / 52.0 | 2.77 / 3.37 | 1.000000 |
| ONNX int8 (`onnx/model_qint8.onnx`) | 36.6 / 23.1 | 38.2 / 26.0 | 54.6 / 50.9 | 2.18 / 3.14 | 0.966638 |

Both figures are the two runs. Only the ratios travel: this box has no AVX2, and every
absolute here is a statement about it. The earlier reading of the same command — torch 142
ms mean, ONNX fp32 54.9 — was taken while the unit suite was running on the same cores and
is not used for anything; it is kept here only as the reminder that a 1.7× swing is what a
busy box does to this measurement.

Two things fall out that the roadmap did not expect.

**int8 is not faster than fp32 here — it is slower, in both runs.** Dynamic quantisation
pays for itself through VNNI/AVX2 integer kernels, and this CPU has neither, so the int8
matmuls run on a fallback path and the extra dequantisation is a net loss. The int8 graph
may still be the right one on the 8 vCPU target VM, where the instructions exist; on the
evidence available it is not the obvious choice, and "int8 encoder" cannot be written into
an image bake until a VM number exists. What *is* established is that ONNX beats torch by
about 3× at p50 on identical hardware, and that this holds for the fp32 graph, which costs
no accuracy at all.

**int8 costs more accuracy than the usual hand-wave.** Over the fifty fixed texts the worst
ONNX-int8 vector sits at cosine 0.9667 from its torch counterpart, not the 0.99+ that
quantisation write-ups quote. fp32 is 1.000000 across all fifty. A 0.967 vector is a
different vector space in the only sense retrieval cares about — neighbour ordering — and
that is why the graph file is part of the embedding fingerprint: the two cannot share a
collection, and switching between them is a reindex, not a restart.

The frozen default stays `runtime="torch"` in this branch. Switching it is one constant in
`config/constants.py`, and the branch that switches it should be the one holding the
target-VM measurement, not this one.

`--export-onnx` traces the ModernBERT checkpoint with the TorchScript exporter at opset 17
with eager attention; the dynamo exporter and SDPA attention are tried in that order if it
fails, and if none traces, nothing is written. The fp32 graph is 191 MB, the int8 graph 48.

## 8. The read path, the relevance floor and the kept flags — 2026-09-30 (overhaul pass 3)

**Environment.** The same 2015 i5 (4 cores in the Docker VM, no AVX2), shared with three
other agents' stacks the whole time (VM load average 3-45 during these runs; each artifact
records its own). Real encoders (granite `dense_en` + bekko `dense_ml`, ONNX), BM25, real
PostgreSQL, the isolated Qdrant server, Dragonfly for the latency runs, in-process
authorization and lexical NLI, the model off. Every run is in-process ASGI (no network hop).
On this box one query's two encodes take ~100 ms and its two hybrid searches ~110 ms before
anything this pass touched runs, so the 300 ms p95 target (set for an 8 vCPU VM) is not
reachable here and is not claimed; what is claimed is the stage split.

### 8.1 `/v1/context` and `/v1/recall`, model off

`make bench-context-latency` (`benchmark/context_latency.py`): 300 turns of LoCoMo
conversation 0 recorded through `/v1/messages`, 2 salted copies of each golden document, an
8-tool catalog; 60 requests per series, every one a cache miss.

| run | recall p50 / p95 | context p50 / p95 | context+tools p50 / p95 | cached p50 / p95 | context bytes p50 |
|---|---|---|---|---|---|
| `before` (pass-2 tree, c337b91) | 222 / 441 | 279 / 482 | 344 / 509 | 24 / 113 | 99,971 |
| `floor0` (this pass, floor off) | 259 / 347 | 326 / 624 | 378 / 629 | 28 / 62 | 84,946 |
| `after` (this pass, floor 0.20) | 262 / 374 | 324 / 407 | 354 / 467 | 25 / 52 | 74,712 |
| `final` (the committed tree, flags decided) | 247 / 397 | 314 / 495 | 356 / 544 | 26 / 300 | 74,128 |

Milliseconds. `final` also packs the ten off-topic questions into 2.1 memories and 26 KB. The
runs are 20-40 minutes apart on a box whose load moved underneath them:
the stages this pass did not touch moved as much as the totals (encode 94 → 110 ms mean,
search 107 → 116), so the end-to-end rows do not isolate a change. The stages do:

| stage (mean ms) | before | after | why |
|---|---|---|---|
| graph traversal | 13.0 | 9.9 | one statement instead of up to nine round trips (8.3) |
| window (read after retrieval) | 36.8 | - | the recent messages are read under retrieval |
| similarity (new, after retrieval) | - | 17.6 | the floor's one read per collection |
| tools section | 41.0 | 37.7 | procedures and catalog search concurrently |

Net on the critical path after retrieval: 50 ms (scope + window) → 29 ms (scope +
similarity).

**Where the rest goes.** Timed against the isolated Qdrant directly, one hybrid query over
the memory collection with its full payload returns 221 KB and takes 34-320 ms end to end;
the same query returning ids only is 29 KB and 11-18 ms. The payload - text, source refs,
attributes of 200 candidates, of which the context packs ~33 - is the next hot spot: a
two-phase read (ranking fields first, the payload of the kept cut second) is the change,
not made in this pass.

### 8.2 The relevance floor

A fusion score only orders, so a context filled its budget with whatever ranked next. The
`floor0` run replays floors over one floor-free pass: every packed memory of the 135 LoCoMo
questions whose evidence is among the 300 turns, labelled by whether it is an evidence turn,
plus ten questions nothing in the corpus answers (`OFF_TOPIC`).

| floor (bekko cosine) | evidence memories kept | other memories kept | questions keeping evidence | off-topic memories packed |
|---|---|---|---|---|
| 0.00 | 100% | 100% | 100% | 30.8 |
| 0.15 | 100% | 98.8% | 100% | 6.3 |
| **0.20** | **100%** | **97.7%** | **100%** | **0.9** |
| 0.25 | 98.5% | 96.4% | 98.2% | 0.1 |
| 0.30 | 98.5% | 94.1% | 98.2% | 0.0 |
| 0.40 | 90.8% | 77.4% | 92.0% | 0.0 |

The encoder's cosines for on-topic but irrelevant memories sit well above 0.2, so the floor
does not trim a relevant question's context much (32.9 → 32.2 memories); what it removes is
the context of a question the store cannot answer: 30.8 → 1.5 memories and 76 KB → 23 KB per
response in the `after` run. 0.20 is shipped (`DenseModel.relevance_floor` of `dense_ml`:
a cosine is only comparable within one encoder, so the floor is the encoder's; the hash
stand-in has none). On the full LoCoMo source harness (8.4, control) with the floor on,
recall@50/@100 is 0.8110 / 0.8704 against 0.8113 / 0.8685 for the phase-7 ensemble artifact.

### 8.3 The graph stage: faster and not decided by a client timer

Three tests were flaky because graph facts vanished under load. Instrumented: the
traversal's SQL executed in < 1 ms server-side, but the stage made up to nine sequential
round trips, and a client-side 150 ms `wait_for` measured the client's own scheduling as
much as the graph - on a loaded box a traversal whose statements took 90 ms lost its facts
after a "150 ms" wait that lasted 370-590 ms. Now: one statement (chained per-hop CTEs,
prepared on first use; its Python construction alone had cost 34 ms per query before it was
built once per shape), on autocommit connections, and the budget is the connection's
`statement_timeout`, so PostgreSQL stops a slow walk and nothing is parked on the pool. With
the production budget the three tests then passed 6/6 consecutive runs at VM load 4-8; with
six CPU hogs added (load 26+) the server itself passed 150 ms and they failed, which is the
budget's job, so the hermetic suite runs the walk to the database's own statement timeout
(`tests/conftest.py:UNHURRIED_GRAPH`) and the budget is tested on the server with a 1 ms
budget against `pg_sleep` - after which they passed 2/2 at load 68-83. With the
production budget the control arm of 8.4 still saw 159 of 1,986 traversals stopped at
150 ms on this box - the budget doing its job, recorded rather than hidden.

### 8.4 The kept flags

`make bench-locomo-source` over all 1,986 LoCoMo questions (1,536 answerable), the phase-7
corpus reused for every query-side arm (`--adopt-corpus`: its ingestion key changed only in
the model-use list, inert with the model off; the artifact records the key it held), the
relevance floor on in all arms. Recall / complete-source coverage at depth:

| arm | all @10 | all @50 | all @100 | multi-hop @10 | multi-hop @50 | multi-hop @100 |
|---|---|---|---|---|---|---|
| control | 0.6508 / 0.5938 | 0.8110 / 0.7480 | 0.8704 / 0.8112 | 0.3997 / 0.1844 | 0.6337 / 0.3723 | 0.7235 / 0.4787 |
| `entity_prefetch` on | 0.6512 / 0.5944 | 0.8110 / 0.7480 | 0.8704 / 0.8112 | 0.3997 / 0.1844 | 0.6337 / 0.3723 | 0.7235 / 0.4787 |
| `memory_entity_search` on | 0.6568 / 0.5990 | 0.8167 / 0.7533 | 0.8742 / 0.8164 | 0.4249 / 0.2128 | 0.6588 / 0.3972 | 0.7437 / 0.5071 |

(`benchmark/results/overhaul/locomo_source_*.summary.json`; the 18 MB per-question records
are not committed.)

- **`entity_prefetch`: removed.** Indistinguishable from the control at every depth now that
  its two halves meet (phase 9 found it a no-op before the fix). It cost one more prefetch
  arm and an indexed payload field on every memory and chunk; both are gone.
- **`memory_entity_search`: on.** +2.5 / +2.8 points of complete multi-hop coverage at @50 /
  @100, +0.6 / +0.6 / +0.4 recall at @10 / @50 / @100, no depth worse - consistent with the
  earlier paired validation on conversations 3-9 (docs/ENTITY-TOPIC-RETRIEVAL-2026-09-26.md).
  Its cost, paired per question against the arm run just before it: the 250 questions with
  an English multi-hop cue pay +122 ms at the median (p95 484 → 699 ms on this box), the
  all-question p95 moves 503 → 547 ms. That is a judgment against the 300 ms target, taken
  because it is the largest multi-hop gain measured in this repository and its work is
  bounded by its own timeout; `BENCH_MEMORY_ENTITY_SEARCH=off` is the control arm that can
  take it back.
- **`query_decomposition`: removed.** Five multi-hop questions through the local gateway
  (gemini-3.8-flash): 6.4, 8.3, 3.9, 12.2 (a failed call) and 3.2 s; one of the five came
  back decomposed. A model call on the read path cannot fit a 300 ms budget at any quality
  (`benchmark/results/overhaul/query_decomposition_latency.json`).
- **`consolidation`: removed.** See 8.6.

### 8.5 The learning constants, reviewed

| constant | value | evidence | decision |
|---|---|---|---|
| `SUMMARY_EVERY` | 20 | 290 windows of 20 LoCoMo turns: p50 637, p95 848, max 986 tokens against the 2,000-token conversation budget, whose window is also 20 messages | kept: summary + window cover the thread with no gap and the window budget never binds |
| `PINNED_SHARE` | 0.5 | at their maxima the pinned sections are ~3.8k tokens (profile 3 × 4,000 chars, summary 2,000 chars, 3 procedures, hints) - inside half the default 8,000 budget | kept; at small budgets the priority order decides |
| `PROCEDURE_MIN_SUPPORT` / `_MIN_SUCCESS_RATE` | 2 / 0.6 | no labelled multi-run traffic exists yet | kept, unmeasured |
| prefetch (`PREFETCH_MIN_PULLS`, `_MIN_RATE`, `_MAX`, settle) | 3, 0.5, 5, 10 min | no agent-tool pull traffic exists yet | kept, unmeasured |
| `APPROVAL_MIN_SUPPORT`, approve/reject rates | 5, 0.95 / 0.5 | no approval traffic | kept, unmeasured |
| standing factor | ±15% | no feedback traffic; bounded so it only reorders near-ties | kept, unmeasured |

The unmeasured ones need traffic the benchmarks do not generate (labelled runs, pulls,
verdicts); they are named here so the first deployment that has it knows what to measure.

### 8.6 Consolidation (on-landing beliefs and entity summaries)

The flag changes the write path, so it cannot share the phase-7 corpus: LoCoMo conversations
0 and 1 (304 questions, 231 answerable) were ingested twice from one snapshot of this tree,
each arm into its own database and its own tenants (`BENCH_CORPUS_TENANT_PREFIX`).

| arm | memories (conv 0 / 1) | all @10 | all @50 | all @100 | multi-hop @50 | multi-hop @100 | tokens p50 |
|---|---|---|---|---|---|---|---|
| off | 591 / 489 | 0.6696 / 0.6147 | 0.8279 / 0.7749 | 0.8882 / 0.8442 | 0.6376 / 0.3953 | 0.7519 / 0.5581 | 3,239 |
| on | 711 / 632 | 0.6696 / 0.6147 | 0.8279 / 0.7749 | 0.8889 / 0.8442 | 0.6376 / 0.3953 | 0.7558 / 0.5581 | 3,726 |

Identical at every depth up to 50; one multi-hop question gained a source at @100. It mints
263 beliefs and entity summaries over the two conversations, adds 15% to the packed tokens
and +86 ms to the median read, paired per question (the derived memories are validated
against their sources on every read). **Removed**, with `LandingReflection`, the belief and
entity-summary services, the pipeline's per-subject serialization it needed and the
`belief_reach` diagnostic; the background ReflectionService (model insights with citations)
is unaffected.

**A retraction on the way.** The first control arm for this table showed consolidation
+3.6 points at @50. Its conversation-1 tenant held 898 memories with 488 distinct contents:
a first attempt killed mid-ingest was still writing into the same database, so the control
retrieved against duplicates. The clean rerun (`p7_locomo_ctl3`) is the row above; the
contaminated one was discarded.
