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

## Layer 2c - the CPU budget is sized for a box this is not running on

`deploy/Dockerfile` pins `WEB_CONCURRENCY=3` with `OMP_NUM_THREADS=2` and
`MKL_NUM_THREADS=2`, and says why: three API workers with two math threads each is 6 of 8
vCPU, leaving two for the event loops. That is correct arithmetic for the target VM.

It is not the box any measurement has been taken on. Docker here has **4 CPUs**, so the same
image asks for 6 math threads on 4 cores - 150% subscribed before counting three event
loops, and before Postgres, Qdrant and the gateway, which on this laptop share those same
four cores.

The subtlety worth writing down: **`SerialRunner` bounds one caller per *process*, not per
box.** With three workers there are three independent gates, so three encodes can be inside
three copies of the model at once, each with two threads. The invariant the class enforces -
"two encodes are never inside the model at once" - is true per worker and false for the
machine. On 8 vCPU that is the intended design. On 4 it is the thrash the class exists to
prevent, arrived at from the other direction.

This is very likely a real part of why the load run collapsed to 0.66 rps with CPU pinned at
382% of 400%, independently of the encoder's own ceiling: a saturated box with more runnable
threads than cores spends its time in the scheduler.

**Not changed here, deliberately.** The fix is a coupled choice - workers x threads has to
fit the cores actually present - and it spans the image's `ENV`, `_workers_default`, and
`DenseModel.threads`, which is a frozen constant precisely so deployments cannot drift. Two
of the three are read before Python starts, so it wants a small entrypoint that derives both
from `nproc` and still obeys an explicit `WEB_CONCURRENCY`. That is a deployment change whose
benefit is a throughput number, and no throughput number can be taken until the encoder flip
lands - so it would be built blind and unverified. It is the next thing to decide after the
flip, and on a 4-core box the candidate is 2 workers x 2 threads, or 4 x 1.

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

## Layer 4b - the cold start, now paid at startup

Nothing warmed the models. The first encode through an ONNX session or a torch module
allocates the runtime's arenas and picks its kernels, so it costs several times what the
ones after it cost - the encoder's slowest single encode is 481 ms against a 98 ms median -
and a load generator ramps the instant the port opens. Without a warm-up that difference
lands inside the p99 the load test exists to measure and is read as service latency.

`api/app.py` now warms both encoders in the lifespan, before the port accepts anything, and
logs-and-continues if a model will not load: readiness is what reports a model that cannot
serve, and refusing to boot would replace a degraded service with no service at all.

Qdrant's cold page cache - the single 2320 ms outlier in layer 4 - is *not* warmed, because
a warm-up query needs a tenant and a scope that do not exist at startup. Left alone
deliberately rather than overlooked.

## Layer 4c - the query embedding is not cached (open, unquantified)

`Indexer` keeps an embedding cache keyed by content hash, but `engine.py:485` calls
`embedding.embed_query` directly and bypasses it, so an identical query re-encodes every
time. Whether that is worth fixing depends entirely on the repeat rate of real traffic,
which nothing here has measured: LoCoMo asks each question once, so the benchmark would show
no benefit, and a cache would make an unsalted load test report a speed the service does not
have. Recorded as an option with its precondition, not proposed.

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

## The depth caveat, which cuts both ways

Every latency figure above was taken at `BENCH_DEPTH=judged`, which is **twice the shipping
depth**:

| knob | shipped (`constants`) | benchmark (`benchmark/env.py`) |
|---|---|---|
| `final_k` | 50 | 100 |
| `prefetch_k` / `fused_k` | 100 | 200 |
| `memories_max` | 50 | 100 |
| `token_budget` | 8000 | 12000 |

A deployment on the shipped defaults therefore fetches half the candidates, fuses half as
many, and renders half the memories. Some of the measured p99 is depth a deployment would
not buy.

The trap is to report that as free speed. **The accuracy numbers come from the same runs, at
the same doubled depth** - the depth is doubled precisely because it scores better. So the
honest statement is that latency and accuracy are two readings of one dial, and a target on
one is meaningless without naming the depth it was measured at.

Which means the gate has to be: pick the depth that is going to ship, then measure *both*
p99 and LoCoMo at that depth. Quoting a p99 from shipped depth beside an accuracy from
judged depth would be the most flattering pair of numbers available and would describe no
system that exists. Nothing here has yet measured accuracy at shipped depth; until it has,
the 300 ms target should be read against the judged-depth figures, which are the pessimistic
ones.

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

0. **Fix the bootstrap first - the flip breaks `docker compose up` without it.** The hub
   publishes no ONNX graph for `granite-embedding-small-english-r2`, so the graph has to be
   exported from the checkpoint. But in `tools/download_models.py` the `--dir` branch that
   compose runs and the `--export-onnx` branch are mutually exclusive: the bootstrap
   downloads the weights and never exports the graph. Flip `runtime` today and a fresh
   deployment raises `DependencyUnavailable: the ONNX encoder needs .../onnx/model.onnx`.
   The bootstrap must export the dense model's graph when the frozen runtime is `onnx`.
   This is the difference between "one package started by plain docker compose" and not.
1. let the full run finish (it is the accuracy baseline, on today's encoder);
2. flip `DenseModel.runtime` to `onnx` - **one line**, because `DEFAULT_GRAPH_FILE` is
   already `onnx/model.onnx`, so no `graph_file` is needed and the collection fingerprint
   becomes `onnx-granite-embedding-small-english-r2-model-d384`;
3. update `tests/unit/test_settings.py`, which pins the frozen dense model's shape - it is
   the only test coupled to the default (the fingerprint tests all build their own specs);
4. `make reindex` - vectors are identical, so results must not move; that is the regression test;
5. re-run LoCoMo for the latency column, and the HTTP load test for rps;
6. only then consider a runner pool, and only if 20 rps is still short.

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

## Layer 2b - what concurrency actually did, measured

Layer 2 argued from arithmetic that a one-permit gate at 111 ms caps the service near 9
encodes a second. `benchmark/results/load_4core_10rps_cold.json` is that argument happening.
Ten users, 10 rps offered, 4 cores, 180 seconds:

| endpoint | p50 under load | the same work, sequential (v6) |
|---|---|---|
| `POST /v1/context` | **17,000 ms** | 172 ms |
| `POST /v1/recall` | 14,000 ms | - |
| `POST /v1/context` (verified) | **42,000 ms**, max 96,410 | - |
| `POST /v1/messages` (ingest) | 1,700 ms | - |
| `POST /v1/files` (docling) | 3,500 ms | - |

Achieved 0.66 rps against 10 offered, 12% failures, CPU 382% of 400%.

Three things in that table are worth more than the headline.

**Retrieval is the thing that collapses, not ingest.** The intuition that writes are expensive
and reads are cheap is exactly backwards under concurrency: ingest is the *fastest* endpoint
here at 1.7 s, while the read path is 10x slower. Ingest does its expensive work once per
message; the read path queues behind a gate that every read must pass.

**A hundredfold degradation is queueing, not cost.** 172 ms to 17,000 ms is not the service
doing more work per request - it is each request waiting for the ones in front. Thirteen of
the fourteen task weights need an encode, so 10 rps offered demands ~9.3 encodes a second
against a ~9-10/s ceiling. That is utilisation ~0.95, and every queue goes to infinity as
utilisation approaches one. The measurement and the arithmetic agree.

**`core_seconds_per_request` is 5.88.** Taken at face value, 20 rps would need ~118 cores.
That number is real but it is a *saturated* number: it is CPU-seconds divided by the few
requests that escaped, so it prices the queue, not the work. It should not be quoted as the
cost of a request, and the capacity question has to be re-asked below saturation.

### Correction: the gate is not what blocks 20 rps

Layer 2 said a one-permit gate at 111 ms caps the service near 9 encodes a second and that
20 rps is therefore arithmetically unreachable. That is the ceiling of **one process**, and
the deployment runs **three** worker processes, each with its own `SerialRunner`. Corrected,
with two threads costing roughly 1.7x rather than 2x:

| encoder | 1-thread ms | core-ms of work | core-s/s at 20 rps | share of 4 cores | gate ceiling, 3 workers |
|---|---|---|---|---|---|
| torch fp32 | 111.4 | 111 | 2.07 | **52%** | 46/s |
| onnx fp32 | 39.8 | 40 | 0.74 | **18%** | 128/s |

Twenty rps needs 18.6 encodes a second (13 of the 14 task weights reach the encoder). Both
ceilings clear that. **So the gate was never the binding constraint at this scale - CPU is.**

That changes what the encoder flip buys. It is not the difference between impossible and
possible; it is the difference between spending **52%** of a four-core box on encoding and
spending **18%** - on a box that is also running Postgres, Qdrant and the gateway, and that
was measured pinned at 382% of 400%. The flip is still the largest single lever, for a
different reason than the one first written here.

It also redistributes the blame for 0.66 rps. On that run the verified-context task was 7%
of weights, and each such request entered DeBERTa up to `max_claims` (40) times
sequentially - plausibly a larger CPU consumer than the encoder at that offered rate. Today's
cross-claim batching addresses exactly that, and was worth doing on its own evidence rather
than as a consequence of the gate argument.

### What this means for 20 rps

With the ONNX encoder at ~35-40 ms the gate's ceiling moves to ~25-28/s. Twenty rps then
demands ~19 encodes/s against that, utilisation ~0.7, where a queue is finite and small
(order tens of milliseconds rather than tens of seconds). So the encoder flip is not an
optimisation for throughput, it is the precondition for it.

**But verification is a second, independent bottleneck.** `POST /v1/context (verified)` sits
at 42 s p50 and 96 s max. It is the only task that enters the NLI model - DeBERTa-v3-base,
its own `SerialRunner`, a far larger model than the encoder - and at weight 1 of 14 it is 7%
of traffic, which at 20 rps is ~1.4 verifications a second against a gate that can serve
roughly one. Fixing the encoder alone leaves this saturated.

There is one concrete inefficiency inside it, found by reading rather than measuring.
`modules/grounding/cascade.py:270` verifies claims strictly sequentially -
`for claim in claims: await self._verify_claim(claim, evidence)` - and each claim makes its
own `nli.entail(premises, claim)` call. `entail` takes a single hypothesis, so it batches the
five premises *for that claim* and no further. An answer of N claims therefore pays N
acquisitions of the NLI gate and N forward passes of five pairs, where the same work is one
forward pass of 5N pairs: the adapter already pads and batches (`nli.py:_score`), so the
shape it needs is already there, and the padding waste is small because premises for one
answer are of similar length. `_scan_unused` can add a second `entail` per claim on top.

Batching across claims cuts the gate acquisitions to one and lets a single GEMM do the work
of N, which matters more under load than it does sequentially - it is service time, and
service time is what sets the queue. It does not on its own explain 42 s; that is saturation.
It reduces the service time that caused the saturation.

Beyond that the options are a second runner, an ONNX export of the NLI head (the encoder's
2.8x says the runtime is worth something here too), a smaller model, or an explicit statement
that verification is opt-in and separately rate-limited. Nothing here has measured which, and
the cross-claim batching should be done first because it is free of that choice.

`max_claims` defaults to **40**, so this is not a small multiple: one verified request can
make up to forty sequential `entail` calls, each acquiring the gate and tokenising
separately, plus whatever `_scan_unused` adds. Forty small GEMMs where one would do is a
much better explanation of 42 s than anything measured so far.

**The seam, so this can be executed against a measurement rather than rediscovered.**
Everything in `_verify_claim` before `scores = await self.nli.entail(...)` is pure - citation
resolution, `_closest`, the coverage check - and may return an early `ClaimReport` without
touching a model. Everything after it needs only `scores`, `premises`, `cited` and `notes`,
and may await the borderline LLM judge. So the split is:

1. `_prepare(claim, evidence)` -> an early `ClaimReport`, or `(premises, cited, notes)`;
2. one batched call over every surviving claim's pairs;
3. `_decide(...)` per claim, which alone may await the judge.

The blocker is the port, not the cascade: `entail(premises, hypothesis)` is fixed to a single
hypothesis, so batching across claims needs a pair-taking method on `NLIProvider`, which
means changing the contract, the DeBERTa adapter and the `LexicalNLI` stand-in together.

**Not attempted yet, deliberately.** This is hallucination detection, its benefit is
throughput, and throughput cannot be measured until the load test can run - which needs the
encoder flip, which needs the run to finish. Doing it blind would mean changing a correctness
surface with no number to show for it. The safety net is already in place for when it is
done: `tests/eval/test_grounding_gate.py` pins verdicts against
`golden/grounding_claims.json` with a lexical stand-in and runs without weights (11 tests
green as of this writing), so a refactor that moves any verdict fails immediately.

### The two targets are not measured on the same workload

Worth stating plainly, because it is easy to quote them together and be wrong: p99 < 300 ms
is measured on LoCoMo, which is **retrieval only and strictly sequential**. 20 rps is measured
on a load mix that is 36% ingest, 7% file upload through docling, and 7% NLI verification.
They are different systems under different conditions. A capacity statement has to name its
mix, and a latency statement has to name its concurrency, or neither means anything.
