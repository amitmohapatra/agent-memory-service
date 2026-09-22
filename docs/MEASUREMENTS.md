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
