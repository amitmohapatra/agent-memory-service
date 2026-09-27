# Memory research and RAG verification — 2026-09-25

This audit extends [the measured LoCoMo/ONNX work](OPTIMIZATION-2026-09-25.md).
It reviews primary sources and actual production wiring, not feature names. “nmesis” is
interpreted as Microsoft **Mnemis**, consistent with this repository's existing research;
the user has not yet confirmed the spelling. This is a dated review, not a claim to have
exhausted the literature or independently reproduced competitors.

## What the published scores mean

| System / source | LoCoMo | LongMemEval-S | Interpretation |
|---|---:|---:|---|
| Hindsight original benchmark, OSS-120B | 85.67% | 89.0% | LLM-free **recall**, with model-based fact extraction and answering |
| Hindsight original benchmark, Gemini-3 | 89.61% | 91.4% | Different backbone; LoCoMo adversarial questions excluded |
| Hindsight v0.4.19, March 23 report | 92.0% | 94.6% | Single retrieval call followed by LLM answering; not an LLM-free end-to-end system |
| Mnemis, GPT-4.1-mini, System-1 RAG + graph | 89.1% | — | RRF-only ablation; an 8B reranker is **not required** for this score |
| Mnemis, both routes, k=30 | 93.9% | — | Includes LLM hierarchical selection and neural reranking |
| Mem0 managed v3, top-200 | 92.5% (1425/1540) | 94.4% (472/500) | Managed platform; not an OSS reproduction |

Sources: [Hindsight benchmark repository](https://github.com/vectorize-io/hindsight-benchmarks),
[Hindsight's dated v0.4.19 report](https://github.com/vectorize-io/hindsight/blob/main/hindsight-docs/blog/2026-03-23-agent-memory-benchmark.mdx),
[Mnemis paper, Tables 1 and 4](https://arxiv.org/html/2602.15313v1),
[Mem0 benchmark results](https://github.com/mem0ai/memory-benchmarks).
These numbers use different extraction, embedding, reader, judge, context and retrieval
budgets. The newer Hindsight report does not establish that its 94.6% is a LoCoMo result;
it is LongMemEval. Do not attribute an exact reader model to that report without checking
its run configuration.

Mnemis's k=10 setting includes **10 episodes, 20 entities and 20 edges**. Table 4 reports
89.1% both with RRF and with Qwen3-Reranker-8B. The hierarchy uses LLM-based global selection;
Table 3 gives 3637.65 seconds of aggregate selection runtime. Dividing by 1540 is about
2.36 seconds/question, **not a measured p99**. One System-2 ablation discusses only
1387/1540 valid queries: preserve each experiment's denominator. These details correct the
older “89.1 requires an 8B reranker” statement in our gap document.
[Mnemis methodology and ablations](https://arxiv.org/html/2602.15313v1).

Mem0's April update describes ADD-only extraction, retaining agent-generated facts,
entity linking, parallel semantic/keyword/entity retrieval and temporal ranking. Its
managed optimizations are not all available in the OSS SDK; the reported LoCoMo p50 is
0.88 seconds, which does not demonstrate our <300 ms p99 requirement.
[Mem0 official README](https://github.com/mem0ai/mem0).

## Transferable mechanisms, and actual coverage here

Hindsight's current recall combines semantic, lexical, graph and temporal retrieval, then
fusion, reranking and token packing. Search depth is independent of output tokens; the
fixed low/mid/high budgets are 100/300/1000, with a separate reranking cap. Current graph
retrieval uses bounded link expansion, including precomputed semantic links. This is a
useful architectural comparison, not justification for blindly enabling our previously
harmful reranker. [Hindsight recall documentation](https://hindsight.vectorize.io/developer/retrieval).

Hindsight observations consolidate supporting facts in the background, retain evidence
quotes/counts, evolve with new evidence and account for stale consolidation. Their
existence is distinct from labeling a memory `BELIEF` or `ENTITY_SUMMARY`.
[Hindsight consolidation documentation](https://hindsight.vectorize.io/developer/observations).

| Capability | Actual code / status | Evidence or remaining gap |
|---|---|---|
| Semantic + BM25 retrieval and fusion | Built: `retrieval/engine.py`, `adapters/search/qdrant_store.py` | Real SciFact test below. `rrf_k` config applies to client fallback, not native Qdrant fusion. |
| Conversation/source provenance | Built: `memory/pipeline.py`, `rag/indexer.py`, `context/builder.py` | Canonical references now survive search and exact reads; regression tests. Older indexes need reindexing for new payload fields. |
| Graph retrieval | Built for authorized facts and document evidence | `graph/retrieval.py` follows `EvidenceRef.chunk_id`, not memory IDs. Conversational entity-to-memory retrieval is still partial. |
| Date-aware memories and historical queries | Built in domain, memory and graph services | Does not equal a separate time-window candidate arm fused with dense/BM25. That arm is absent. |
| Document hierarchy / companions | Built: ingestion hierarchy, context graph, expansion and verifier | New multi-document, concurrency, filter and budget tests; full SciFact retrieval run. |
| Document/global summaries | Built, indexed as summaries | Section/document summaries do not demonstrate Mnemis's semantic many-to-many hierarchy over conversations. |
| Entity summaries and beliefs | Implementations exist in `memory/derived.py` and `landing.py` | `ObservationPipeline.landing` defaults to `None`; `_wire_memory` does not inject it. Not a shipped automatic capability. |
| Background LLM reflection | Wired via jobs and `ReflectionService` | Returns without work unless the reflection LLM use is enabled. The benchmark's LLM-free ingestion does not exercise it. |
| Whole-session semantic extraction | Partial: observation extraction and optional assists | No demonstrated equivalent of evidence-grounded session/event synthesis. Previous LLM-assist experiments did not establish a gain. |
| Standing, refreshable mental-model queries | Missing as a demonstrated product workflow | A summary memory type alone is insufficient evidence of parity. |
| Tenant/scope isolation, durability and lifecycle | Built, independently testable | PostgreSQL UoW, visibility specification, revisioned caches, archive/jobs; security/failure/contract suites remain required. |
| External RAG evaluation | Built and repaired in this change | Retrieval relevance is measured; real-reader answer faithfulness on PDFs/tables is not established by SciFact. |
| Multilingual and large-scale memory accuracy | Unvalidated | English corpus, local model and existing fixtures cannot establish either. |

Do not create parallel implementations just to match competitor terminology. Reuse the
existing evidence model, repositories, UoW, worker jobs, visibility rules and retrieval
stages. In particular, decide whether to wire and test the dormant landing/admission
features or remove them in a separate ingestion change; turning them on silently would
change admission and supersession across the product without a measured accuracy result.
The older audit already records this problem; it is not claimed fixed here.

## Changes made and their cost

1. **Request isolation:** verification requirements are request-local values. Removed shared
   `_last_seed_*` fields and the unused expansion-stage constructor dependency. A controlled
   interleaving test reproduces the old cross-request contamination; a memory-only request
   also cannot inherit a previous document's requirements.
2. **Companion identity:** evidence groups include target node identity, so “footnote 1” in
   two documents creates two obligations. Parent summaries inherit the matching seed's
   document and citation source, not the first seed's document. Cache format is bumped.
3. **Document selection:** exact hits, extra retrievers and every post-stage are constrained
   to requested documents before the next stage verifies evidence. Empty selection preserves
   existing unrestricted behavior. O(candidates) filtering, no new network call. This is not
   a substitute for tenant/authorization filtering, which remains in place.
4. **Expansion cost:** target and sibling chunks share one bulk repository read (four calls
   instead of five). Source and sibling positions use dictionaries rather than repeated
   scans: O(seeds + edges + fetched chunks), apart from the existing edge sort. A test checks
   both returned companions and the single bulk read.
5. **Graph bounds:** enforce the evidence-chunk budget inside a relation's evidence list.
   Previously one relation could exceed the configured cap before the outer loop checked.
   A set provides membership checks; removed the unused `names_of` argument.
6. **Benchmark integrity:** reuse shared IR metrics, report true recall separately from hit
   rate, retain full queries/IDs, timings, p99, actual model class, configuration and stand-ins.
   An O(documents + chunks) preflight checks exact tenant corpus identity, checksums, READY
   current versions, indexed chunk fingerprints and Qdrant chunk IDs. Two bulk SQL reads
   and an ID-only vector scroll run outside query timing. Subset qrels are explicitly
   intersected with the indexed corpus. Gates reject legacy metric versions, wrong corpus
   size/cutoff, duplicate query IDs, unmapped candidates and a source hash that no longer
   matches the checked-out runtime/benchmark code.

Regression tests for the first four correctness cases and document-filter bypasses failed
before their corresponding fixes. New corpus-audit tests reject missing, duplicate,
unexpected, stale and unindexed data, including equal vector counts with different IDs.

7. **Stable fusion ties:** a fixed-vector diagnostic reproduced five different returned
   orderings in five calls for each of five queries. Scores for shared IDs were identical;
   membership at the store cutoff also varied for two queries. Native fusion results now
   sort by descending score, then record ID before engine dedup/cuts. This adds
   O(k log k) work over the already bounded pool, no wider search or extra RPC, and leaves
   all scores unchanged. It stabilizes the returned pool, not ANN membership or ties cut
   off by the server. Record-ID ordering also assumes the same index, not a fresh reingest.
   Three permutation tests cover the tie behavior, including score precedence and RPC count.
   Cache format is bumped again. Diagnostic: `codex_rag_rank_repeat.json`; its saved
   reproduction script is `codex_rag_rank_repeat_script.txt`.

## Measurement results

The preserved before artifact is `benchmark/results/codex_rag_before.json`; its `recall_at_10` field is
legacy **hit rate**, not true recall. Recompute recall from its saved rankings and qrels for
comparison. That run returned nDCG@10 0.7427 and hit rate@10 0.8867 on all 5183 documents and
300 queries. Its first attempt failed on a Qdrant deadline at the first query; the completed
retry is retained. Do not omit that failure when interpreting availability or cold starts.

| Full SciFact, k=10 | Before | RAG fixes, before tie-break (first run) |
|---|---:|---:|
| Documents / queries | 5183 / 300 | 5183 / 300 |
| nDCG@10 | 0.7427 | 0.7424 |
| True recall@10 | 0.8731 (recomputed) | 0.8697 |
| Hit rate@10 | 0.8867 | 0.8833 |
| Serial p50 / p95 / p99, ms | 151.3 / 413.5 / 740.1 | 138.5 / 256.6 / 362.1 |
| Unmapped candidates | 0 | 0 |

The after audit verified all 6814 current chunks in both PostgreSQL and Qdrant, exact
checksums for 5183 documents, and the ONNX Granite/BM25 fingerprint. No indexing was
repeated and no relevance labels were supplied to retrieval. The previous canonical result is preserved as `codex_rag_historical.json`. The canonical
`external_retrieval.json` is refreshed again after the tie-break change so its source hash
matches the final code.

`codex_rag_comparison.json` recomputes the old true recall and pairs all 300 query records.
84 rankings changed; 10 nDCG values changed (4 improved, 6 worsened), with one hit lost.
The mean nDCG difference is -0.00028; its paired query-bootstrap 95% interval is
[-0.00574, +0.00582] (5000 resamples, fixed seed). This is **not an accuracy improvement**.
Nor does a lower observed latency establish a causal speedup of that size: cold state and
host load were not controlled by an alternating A/B design for this RAG patch. The new
preflight also scrolls vector IDs before querying, unlike the baseline, and can warm store
caches. Treat the displayed before/after latency as observations, not an isolated speedup.
The first after run's p99 is still above 300 ms. An unchanged-code repeat
(`codex_rag_after_repeat.json`) measured nDCG 0.7396, recall 0.8731, hit rate 0.8867 and
p50/p95/p99 145.9/333.2/654.7 ms. That variability prompted the fixed-vector diagnostic
and tie-break fix; neither earlier after-run is a measurement of the final tie-break code.

Final tie-break code, first full run (`codex_rag_stable.json`): **nDCG@10 0.7436,
recall@10 0.8731, hit rate@10 0.8867**, with p50/p95/p99 **133.9/253.3/445.1 ms**.
The canonical external artifact uses this run, not a retrospectively selected best score.
This preserves measured retrieval quality within the observed earlier variation, but does
not establish a statistically reliable improvement or meet the latency target. An independent
final-code repeat (`codex_rag_stable_repeat.json`) returned **identical top-10 document
rankings for all 300 queries**, identical quality metrics, and p50/p95/p99
**127.0/227.8/315.6 ms**. `codex_rag_stability.json` records both runs and matching source
and corpus identities. The lower p99 still misses 300 ms; neither is a production load test.

Per-stage p50/p99 in the first after run: encoder wait 40.7/122.5 ms, search 39.5/154.8,
expansion 27.9/92.8, verification 8.2/42.1, visibility 11.4/54.4. These percentiles are not
additive and encoder work overlaps other stages. They identify storage and scheduling tails
as well as inference cost; they do not justify a blanket claim that every layer is optimized.

SciFact measures retrieval, not answer accuracy, citation correctness, unanswerable
questions, complex tables or PDF extraction fidelity. The hermetic suites cover correctness
with stand-ins; their passing status is not a real-model accuracy score. The real-model
harness uses real PostgreSQL, Qdrant and ONNX embeddings, but in-process authorization/cache
and lexical NLI. Its serial engine timing is not HTTP p99 at 20 requests/second.

Final-run tail examples show why a single “encoder optimization” is insufficient: query 294
spent 364.6 ms in search; query 51 in the repeat spent 213.5 ms in expansion; query 1175
spent 183.4 ms waiting for encoding. Different stages dominate different slow requests.
The next profile needs SQL pool/statement and RPC timings within those stages before deciding
whether batching, deployment topology, queue limits or model work is the limiting factor.

## Next experiments in priority order

1. Establish an identical full-set reader/judge protocol before a competitor claim: the same
   1540 answerable LoCoMo questions, exact prompts/model versions, token budgets, extraction
   settings and judge-failure denominators. Keep strict scoring and adversarial abstention
   separate. Persist contexts/answers, then rejudge them without regenerating answers.
2. Partition existing errors into evidence absent, source detail lost, budget omission and
   reader failure. The saved full DeepSeek result is 72.92%; a two-conversation 81.12% control
   is a different population. Source-turn promotion did not earn default enablement.
3. Add an evidence-grounded session representation at ingestion using existing jobs and
   provenance, then ablate it on conversation-held-out data. Preserve named actors, dates,
   numbers and negation; inspect extraction failures independently of reader failures.
4. Complete graph-to-memory hydration with batched authorized reads and bounded candidates,
   then fuse an entity arm. Test cross-tenant, current/as-of, missing/deleted/superseded source,
   repeated edge and many-evidence cases before benchmarking. No N+1 memory reads.
5. Evaluate temporal candidate windows independently of recency, and bounded fusion/reranking
   variants using both LoCoMo and SciFact. Budget search depth, returned evidence and tokens
   separately. Do not tune solely against the final test set or assume an 8B reranker fits CPU.
6. Measure the actual SLO: separate load generator, fixed hardware, remote stores and real
   authorization/cache, uncached requests at 20 rps for five minutes; count errors, cold
   starts and queue wait. Profile DB pool wait, OpenFGA calls, graph expiry, embedding queue,
   Qdrant RPCs, response bytes and serialization. Current graph hydration still checks each
   distinct document's visibility separately (bounded by the now-enforced chunk cap).
7. Extend external RAG evaluation to multi-document citations, tables/footnotes, stale
   versions, unanswerable claims, and answer-level faithfulness with a fixed real reader.
   Add LongMemEval for updates/multi-session coverage and a larger corpus for scalability.

The evidence supports these experiments; it does not guarantee 85%, 94%, universal feature
parity, zero dead code throughout the repository, or <300 ms production p99.

## Regression validation

The broad test run completed with 1067 passes, 21 skips and one stale test-fixture failure:
`grant_membership(workspaces=...)` used an argument removed from the service API. The fixture
now grants tenant membership with the current signature; the rebuild assertions are unchanged,
and the corrected test passed. Four originally skipped real OpenFGA contracts subsequently
passed with Docker Desktop's internal socket setting.

The final per-case union is **1093 passed, 19 skipped, zero unresolved failures**.
The last final-code evaluation/e2e/security run passed 18 tests and skipped two optional
model/Docling gates, including all six distractor-corpus capability cases among the passes.

The final tie-break change passed a focused 72-test suite covering adapter payload/retry/
ordering behavior, search contracts, architecture boundaries, retrieval stages, request
isolation, document context, and retrieval/graph gates. Ruff check and format, `git diff
--check`, and Pyright passed (0 type errors; 21 warnings for optional model/GCP dependencies
absent on the macOS host). The complete per-test merge and skip reasons are in
`benchmark/results/codex_rag_validation.json`; later successful reruns supersede the original
fixture failure. This is a union of runs, not one uninterrupted all-green invocation.

Model/Docling/GCS/Bifrost and worker-dependent skipped lanes remain explicitly recorded.
Real ONNX retrieval was measured separately in Linux Docker; this does not turn skipped
real PDF parsing or answer-faithfulness checks into passes. Gate-generated artifacts from
this session are preserved in `codex_rag_gate_results.json` rather than overwriting unrelated
historical gate files.

## Reproduce the real retrieval run

From the repository root, with the existing isolated full index and local backing services:

```bash
docker run --rm --name memory-rag-verify \
  -v "$PWD:/app" -v "$PWD/models:/models:ro" \
  --add-host host.docker.internal:host-gateway \
  -e PYTHONPATH=/app/src:/app -e GIT_COMMIT=05a45b7-working-tree \
  -e MEMORY__DATABASE__URL=postgresql+psycopg://memory:memory@host.docker.internal:5432/memory_bench_docs \
  -e MEMORY__SEARCH__QDRANT_URL=http://host.docker.internal:6333 \
  -e BENCH_SEARCH=qdrant -e BENCH_EMBEDDING=frozen -e BENCH_DEPTH=shipped \
  --entrypoint /opt/venv/bin/python memory-service-memory-api \
  -m benchmark.external_retrieval --reuse-index --out rag_verification.json
```

The index audit deliberately rejects a different subset, outdated document contents,
incomplete versions or mismatched model/index IDs. A passing retrieval benchmark is not
an index of real-reader RAG answers. Model/runtime source hashes are embedded in results.
For Docker Desktop's OpenFGA contract tests on this host, use
`TESTCONTAINERS_DOCKER_SOCKET_OVERRIDE=/var/run/docker.sock`; this fixes the helper's socket
mount rather than disabling authorization tests or changing service authorization.
