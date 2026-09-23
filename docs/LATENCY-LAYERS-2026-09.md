# Where the latency and the throughput actually go

Measured, not argued. Every number below came from `benchmark/results/locomo_judged_v6.json`
(304 questions, per-stage timings) or from `--measure-onnx`, run inside the runtime image on
one pinned CPU. Where a number is a projection it says so.

The two targets are p99 < 300 ms and 20 rps on the laptop.

## The measured budget

| stage | p50 | p90 | p99 | max |
|---|---|---|---|---|
| encode | 98.1 | 174.6 | **409.5** | 481.0 |
| search (Qdrant) | 51.5 | 89.2 | 204.2 | 2320.8 |
| scope (authz) | 8.6 | 14.5 | 53.1 | 80.1 |
| verify (NLI) | 3.3 | 5.0 | 11.2 | 22.0 |
| graph | 0.0 | 0.7 | 1.5 | 19.6 |
| expansion / window / visibility | ~0 | ~0 | ≤1.0 | 12.5 |
| **total** | **172.1** | **302.7** | **668.1** | 2522.4 |

`encode` is 61% of the p99 budget. Nothing else is close, and the four smallest stages
together are under 2 ms at p50. There is no distributed-systems problem here: there is one
matrix multiply that is too slow.

## Layer 1 - the encoder. This is the whole finding.

One `embed_query` per retrieval (`modules/retrieval/engine.py:485`), so 98 ms p50 is a single
short query through `granite-embedding-small-english-r2` on two torch threads. That is not a
defect, it is arithmetic: ~28.3M non-embedding parameters x ~12 tokens x 2 FLOPs is ~0.7
GFLOP, which is tens of milliseconds of fp32 CPU.

The three runtimes, one pinned CPU, one thread each. **MEASUREMENTS section 7 established
this first**, on the same day and with the same conclusion; this is an independent
reproduction on a differently-loaded box, which is the only thing a second run of a
contention-sensitive measurement is good for:

| runtime | p50 | p90 | p99 | speedup | cosine vs torch |
|---|---|---|---|---|---|
| torch fp32 (today) | 111.4 | 154.7 | 195.2 | 1.00x | - |
| **onnx fp32** | **39.8** | **61.1** | **111.6** | **2.80x** | **1.00000** |
| onnx int8 | 50.5 | 80.9 | 112.4 | 2.21x | 0.99237 |

Two things fall out of this table.

**ONNX fp32 is free.** The vectors are bit-identical to torch - cosine 1.00000 at both mean
and minimum, across 40 real LoCoMo questions, matching section 7's 1.000000 over its own
fifty fixed texts - for 2.8x the speed. The reindex the runtime
flip forces (the fingerprint is in the collection name, by design) cannot change a single
retrieval result, because it cannot change a single vector. This is the rare optimisation
with no accuracy risk to trade off.

**int8 is worse on both axes, and that settles the freeze.** It is *slower* than fp32 ONNX
here, not faster, and it is the only variant that moves the vectors. The cause is the
hardware: this CPU has no AVX-512 VNNI, so the int8 GEMM is emulated and the dequantise
overhead is not repaid. `docs/FREEZE-multilingual.md` already chose fp32 as the shipping
default on an argument, and section 7 measured 0.966638 worst-case cosine for int8 over its
fifty texts against my 0.98842 over forty queries - the same verdict from two different
corpora. Revisit only on a VM whose CPU reports VNNI.

The measuring tool is `python -m memory_service.tools.download_models --measure-onnx DIR`.
It now rotates the runners one query at a time instead of timing each to completion, because
timing them in sequence compares three different machines on a shared box - the discarded
1.7x-slow reading in section 7 is exactly that failure.

## Layer 2 - concurrency. Why 20 rps is impossible today.

`adapters/models/_runner.py` admits one caller into a model at a time, deliberately: a model
already fans its GEMMs over every core, so letting twelve event-loop threads in at once
spends the box in the scheduler. The docstring is honest that the number which would settle
it had never been taken.

Here is that number. A single gate with service time S saturates at 1/S requests per second:

- torch at 111 ms  ->  ~9 encodes/second ceiling. **20 rps is arithmetically unreachable**,
  no matter what is done to Qdrant, Postgres, the API or the event loop.
- onnx fp32 at 39.8 ms (one thread; less at two)  ->  ~25/second, so 20 rps sits at ~80% gate
  utilisation. Reachable, though with queueing in the tail.

This is the mechanism behind the 0.66 rps the first HTTP load test produced. The encoder
flip is a throughput fix before it is a latency fix.

If 20 rps needs more headroom than one gate gives, the shape of the change is a small **pool**
of k runners with `intra_op_num_threads = cores/k`, which multiplies throughput by k while
preserving the invariant the class exists for (total model threads never exceeds cores). That
is a measurement to take after the flip, not before.

## Layer 3 - network. Already optimal; no action.

- Qdrant is addressed over **gRPC** (`prefer_grpc=True`), not HTTP+JSON, so vectors go over
  the wire as protobuf.
- Responses are **gzipped** above 1024 bytes at the innermost middleware position.
- Both middlewares are plain ASGI callables rather than `BaseHTTPMiddleware` subclasses,
  chosen explicitly to avoid the two anyio streams and body round trip per middleware per
  request that the base class costs at 20 rps.
- Payloads are projected (`PAYLOAD_FIELDS`), so Qdrant returns the fields used and no more.

There is nothing worth changing at this layer. Recording it so it is not re-audited.

## Layer 4 - search. One cold start, not a systemic cost.

p50 51.5, p90 89.2, p99 204.2. The 2320 ms maximum is a single request: the first query of
conversation 0, against a collection that had just been written. Exactly one of 304 queries
exceeds 300 ms in this stage. The fix is a warm-up query at startup, not an index change -
and `docs/STORAGE-AUDIT-2026-09.md` already established HNSW is inactive at this corpus size
(`index.type: plain`), so there is no graph to warm, only page cache.

## Layer 5 - authz. Fixed this session; confirmed by measurement.

`scope` p50 fell from 36.0 ms (v5) to 8.6 ms (v6) after the MEMBERSHIP revision fix. The
scope cache is now keyed on grants alone, so content writes no longer discard it. Nothing
further outstanding.

## Layer 6 - Postgres. Not a bottleneck at this scale.

Largest table is 1.3 MB / 574 rows. Four indexes show zero scans; all are <= 56 kB and serve
background paths the benchmark does not exercise (`ix_memories_unindexed`,
`ix_observations_pending`) or the graph write path. Nothing to add, nothing worth dropping.
Re-examine only at a corpus two orders of magnitude larger.

## What this projects to

Replacing torch with ONNX fp32 per record, holding every other stage fixed:

| | p50 | p90 | p95 | p99 |
|---|---|---|---|---|
| measured v6 (torch) | 172.1 | 302.7 | 440.5 | 668.1 |
| projected (onnx fp32) | **107.7** | **185.0** | **267.9** | **381.2** |

Queries over 300 ms: 10.2% -> 3.9%.

p99 lands at 381 ms, not under 300. The honest reading of the remaining 81 ms is that it is
mostly *contention*, not service cost: the benchmark runs on a 4-CPU laptop that is also
hosting Postgres, Qdrant and the gateway, and the records still above 300 ms in the
projection have an encode p50 of 286 ms - four to seven times the model's own cost, which a
fixed-size matrix multiply cannot produce on its own.

## The caveat that governs all of this

**The LoCoMo harness is strictly sequential.** There is no `gather`, no `TaskGroup`, no
semaphore in `benchmark/locomo.py`: questions run one at a time. So this benchmark has never
measured p99 under concurrent load, and the gate in layer 2 is never contended while it runs.
It measures single-query latency on a noisy shared box, which is why its tail is wide.

Throughput and concurrent p99 are the load test's job, and the load test must be re-run after
the encoder flip - its 0.66 rps was taken against the torch encoder, whose 9/second ceiling
it was never going to beat.

## Order of work

The flip needs a reindex, and a reindex rewrites the collections the 1,986-question run is
reading. So:

1. let the full run finish (it is the accuracy baseline, on today's encoder);
2. flip `DenseModel.runtime` to `onnx`, `graph_file` to `onnx/model.onnx`;
3. `make reindex` - vectors are identical, so results must not move; that is the regression test;
4. re-run LoCoMo for the latency column, and the HTTP load test for rps;
5. only then consider a runner pool, and only if 20 rps is still short.

## Layer 7 - code shape. One real duplication; the hot path is clean.

Checked rather than assumed, because "no duplicate code" and "right data structure" are easy
to assert and easy to get wrong in both directions.

**Complexity on the query path is already right.** `rrf_fuse` is three dicts and one sort -
O(N log N), which is the floor for a ranked fusion. The membership tests inside loops across
`modules/` resolve to dicts or sets in every case that runs per query (`existing_ids` is a
set, `by_id` is a dict); the one genuine O(n*m) substring scan, `graph/service.py:199`, is
gated behind `assist.wants("entity_resolution")` and bounded by `LLM_MAX_QUERY_NAMES`. The
measured graph stage is 0.0 ms at p50 and 1.5 ms at p99, which agrees. Nothing to change.

**A 7-line-window duplicate scan over 168 files** returns 94 cross-file repeats, and almost
all of them are adapters restating a port's signature - `blob/{filesystem,gcs,memory}.py`
against `ports/blob.py`, the two graph stores against `ports/intelligence.py`. That is the
hexagonal shape working, not duplication, and collapsing it would be the bug.

One is real. The same **fourteen** execution-context fields are copied verbatim onto an
`Observation` at three sites:

| site | fields copied |
|---|---|
| `modules/conversation/service.py` | 14/14 |
| `modules/ingestion/service.py` | 14/14 |
| `modules/memory/service.py` | 14/14 |

(`conversation/service.py` does it twice; the second is an `AgentRun` with a different
subset, so it is a near-miss rather than the same block.)

The cost is not the fifty-odd lines. It is that adding a field to `MemoryExecutionContext`
and to the observation row now requires finding three call sites, and missing one drops
provenance silently on exactly one ingest path - with no test that would fail, because each
site is individually correct. The fix is for the domain object to own the stamping, e.g.
`Observation.for_context(ctx, kind=..., content=..., ...)` reading the provenance from one
place, which is also the Single Responsibility reading: three services should not each know
the shape of an observation's provenance.

**Deferred on purpose.** These three modules are the ingest path the 1,986-question run is
executing right now. A running interpreter holds its imported modules, so editing them on
disk cannot affect it - but a module it has not yet imported would be read in its new form
mid-run, and a three-hour accuracy baseline is not worth that. Applied after the run, with
the existing ingest tests as the check.
