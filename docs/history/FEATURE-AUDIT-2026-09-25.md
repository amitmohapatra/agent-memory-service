# Feature audit and research follow-up — 2026-09-25

This audit supersedes the pasted feature inventory where they disagree. Historical scores
describe their saved runs, not the current working tree. Runtime changes and experiments
from this follow-up are recorded below; unfinished capabilities are explicitly unfinished.

For the later wider-memory/fusion experiments and their reader results, see
[ACCURACY-EXPERIMENTS-2026-09-26.md](ACCURACY-EXPERIMENTS-2026-09-26.md).

## What the implementation actually does

| Claim | Verified status | Consequence |
|---|---|---|
| Source IDs disappear | Repaired in the preceding work: index, exact retrieval and context preserve `EvidenceRef`; graph memory hydration now shares the exact-read projection. | Source evidence remains distinct from the memory citation. |
| Expansion is discarded after the cut | Repaired previously: primary evidence is capped separately from bounded companions; packing enforces required document evidence groups. | Do not remove companion retrieval to improve a document-only metric. |
| Graph cannot return source memories | Fixed in this follow-up: direct `Relation.memory_id` and memory evidence pointers hydrate through one bounded canonical batch read. | Maximum six new memory candidates; deduplication, visibility, deletion, expiry and current/as-of checks. This is evidence expansion, not a new graph fusion arm. |
| Landing reflection is live | False. The constructor argument remains absent. `landing.py` and `derived.py` exist, but are not production continuous consolidation. | Enabling it needs the lifecycle work below, not just wiring. |
| We have no continuous adaptation | Too broad. Native REINFORCE/MERGE, supersession, forgetting, tool outcome aggregation and procedure mining exist. Optional scheduled LLM reflection exists separately. | These are memory-state changes, not online training of model weights. Neither dormant landing aggregation nor optional reflection establishes Hindsight parity. |
| No global route | A `GLOBAL_SUMMARY` route retrieves document node summaries. | It is not Mnemis global selection over a conversation hierarchy. |
| Atomic-only storage | Facts, preserved verbatim turns and document node summaries exist. | No maintained conversation topic → episode → fact hierarchy. |
| No summary producer | Document ingestion produces node summaries (`sum_...`). | Distinguish representation enums, document summaries and canonical `ENTITY_SUMMARY` memories. |
| Temporal data is unused | Graph traversal has as-of handling; native consolidation has validity/supersession. | Query-time temporal windows and ranking across memory search remain missing. Recency alone would not fix historical questions. |
| `rrf_k=60` controls Qdrant | It controls outer strategy fusion. The 26 September follow-up adds explicit `hybrid_rrf_k` for inner dense/sparse fusion, preserving the legacy default wire request and scores. | Inner constant 60 reduced measured source recall; retain the default and keep historical benchmark configurations distinct. |
| SPLADE++ sparse retrieval | False for this tree: frozen sparse configuration is BM25. | Benchmark provenance must name BM25. |
| ColBERT is configured and unused | False for the frozen model catalogue inspected here. | A downloaded historical model is not a configured runtime capability. |
| `contextual_chunks` has no effect | Its reported query-time ablation was invalid: it belongs to ingestion, not retrieval. | The harness now rejects unknown/ingestion flags. Testing it requires a separate re-indexed corpus. |
| MCP facade | No implemented MCP server/tool facade found; accepting `source="mcp"` on tool records is not one. | Useful integration work, but it does not directly improve retrieval accuracy. |

The supplied database counts and percentages are snapshots, not enduring properties. A hub
can represent a frequent speaker rather than an entity-resolution error. Evaluate alias
precision/recall, typed-relation correctness and source coverage before merging entities.
API route counts and claims that competitors lack entire capabilities are not quality evidence.

A read-only check of the isolated `memory_bench_conv` / `bench_conv` snapshot confirms 472
undeleted memories, 317 entities and 816 relations: 714/816 (87.5%) are `mentions`, with zero
BELIEF and ENTITY_SUMMARY records. The 350 OBSERVATION records include preserved raw turns;
that enum name does not mean Hindsight-style consolidated observations. The degree-337 hub
is `user:jon`, followed by `user:gina` at 317. More concrete extraction noise is the entity
`thanks` with 99 endpoint appearances. Improve and validate entity extraction before blindly
merging speaker hubs or adding semantic edges. Exact predicate counts and scope are saved
in `codex_feature_snapshot.json`; the supplied “5% typed” needs a definition of which generic
predicates count as typed and is not independently asserted here.

## Why landing reflection cannot just be switched on

Code inspection found these concrete blockers in `landing.py`, `derived.py` and mutation paths:

1. `_NARROWNESS` imposes an ordering on PRIVATE/RUN/USER/GROUP/THREAD/TENANT. Those audiences
   overlap; they are not nested. Reconstructing keys from the current writer can broaden
   access or change a RUN audience. Related facts are selected by scope/subject without
   checking an identical audience. Preserve/intersect actual stored keys; never infer
   authority from a visibility enum or the writer's identity.
2. Forgetting a source does not invalidate its derived beliefs/summaries. Source update,
   supersession, expiry, withdrawal and permission changes need dependency invalidation,
   durable re-indexing and cache revision updates. An old summary must not keep deleted facts.
3. Derived lookup scans only 200 records and filters subject in Python. Concurrent writers
   can create duplicate current slots. Use indexed exact slot/audience lookup and serialize
   updates per slot with a database invariant or lock.
4. The eight-fact landing window can replace an entity summary with a partial recent view.
   Make bounded snapshots explicit or incrementally retain valid provenance; never advertise
   such a window as a complete profile. Same text with different sources also needs a
   provenance update, and revision history must be bounded.
5. Arrival order is not event order. Numeric disagreement currently reaches contradiction
   handling before single-value supersession. Test dated corrections, delayed arrivals,
   true contradiction and agent echoes separately. Native consolidation already performs
   some of these operations; a second pass must not double-strengthen an observation.

Required tests before enabling: sibling-run/private/thread isolation; concurrent same-slot
writes; idempotent retries; zero/one/many facts; changed sources with identical summary text;
source correction/deletion/expiry; outbox replay and index/cache convergence; bounded reads
and token/input sizes. This follow-up does **not** claim those lifecycle gaps are repaired.

## Competitor research and implications

[Hindsight's benchmark repository](https://github.com/vectorize-io/hindsight-benchmarks)
reports **89.0% on LongMemEval with OSS-120B**, while that configuration's LoCoMo figure is
85.67%. “LLM-free” describes recall, not extraction and answer generation. Its newer
[v0.4.19 report](https://github.com/vectorize-io/hindsight/blob/main/hindsight-docs/blog/2026-03-23-agent-memory-benchmark.mdx)
reports 92.0% LoCoMo and 94.6% LongMemEval with a single recall plus a reader. These are
different configurations/benchmarks, not evidence that an entirely LLM-free system scores 89%.

[Hindsight observations](https://hindsight.vectorize.io/developer/observations) consolidate
facts asynchronously into evidence-backed beliefs, track source quotes/proof counts,
support scoped consolidation, and expose freshness. Its
[retrieval documentation](https://hindsight.vectorize.io/developer/retrieval) combines
semantic, lexical, graph and temporal retrieval, then fusion and optional reranking. Current
temporal search spreads selection across a requested interval and uses a bounded proximity
signal. The transferable idea is richer precomputed evidence with cheap retrieval. It is
not a reason to put an unbounded reflection agent in our 300 ms query path.

[Mem0's current repository](https://github.com/mem0ai/mem0) describes its managed v3 pipeline,
which must be distinguished from the OSS implementation. Its
[benchmark suite](https://github.com/mem0ai/memory-benchmarks) separates ingestion, search,
answering and judging; reader/judge defaults are not proof of every published run's model.
The [current LoCoMo judge prompt](https://github.com/mem0ai/memory-benchmarks/blob/main/benchmarks/locomo/prompts.py)
accepts partial lists and allows 14-day date and 50% duration tolerances. Keep our strict
metric and adversarial score; an additional published-ruler rescore must be separately
labelled. The asserted +6 points is not newly verified by this follow-up.

[Mnemis](https://arxiv.org/html/2602.15313v1) combines similarity retrieval with global
selection over a hierarchy. Its 90.06% global-route valid-response figure is a particular
ablation, not universal query coverage. The quoted 2.36 seconds is an aggregate-derived
average, not p99. Its System-1 RAG+graph RRF ablation already reaches 89.1 without an 8B
reranker gain. A conversation-wide completeness route is promising but needs a bounded
candidate/token budget, recall tests for lists and multi-hop evidence, and independent timing.

[HyperMem](https://arxiv.org/html/2604.08256v1) builds topics, episodes and facts with
hyperedges. It uses LLMs for streaming episode boundaries, topic aggregation and contextual
fact extraction; the reported 92.73% is an LLM-judged result. It motivates session/episode
evidence instead of isolated assertions, but does not establish that its construction cost
or accuracy transfers to our model, CPU budget or stricter evaluator.

[ColBERT](https://arxiv.org/abs/2004.12832) precomputes document token vectors and performs
query-document token MaxSim at retrieval. Scoring scales with query tokens, document tokens
and vector dimension; it is not literally O(1) per candidate. A trial needs frozen weights,
token-vector storage, memory/CPU measurements and a fair no-reranker control.

## Selective LLM ingestion and multi-hop work

### Historical failure diagnosis, rechecked from saved records

The full `v2_judged_shipped.json` run is historical, not final-source validation.
Among answerable records with a valid judge and an explicit evidence-completeness flag,
994/1102 (90.20%) were answered correctly when `complete_in_candidates` was true,
versus 126/431 (29.23%) when false. Four answerable rows lack that flag and three
multi-hop judge calls failed; neither group enters these conditional rates.
This is an association using token-overlap evidence matching, not an oracle experiment
or a proven accuracy ceiling. It supports prioritizing evidence coverage before blaming
the reader model. Across all 282 multi-hop rows, 96 (34.04%) have complete candidate
evidence and 77 (27.30%) have complete top-30 evidence under that diagnostic.
The full run's 72.92% answer score would need about 186 additional correct answers
out of 1540 answerable questions to reach 85%. New code has not yet undergone a full
reader rerun. The next discriminating reader experiment should compare retrieved context
with gold evidence under the same reader and strict judge, then test extraction,
consolidation and bounded multi-source retrieval on held-out conversations.

The user's latest instruction allows necessary LLM ingestion for complex inputs. Existing
`LLMAssist` already supports ambiguous extraction/worthiness and conflict adjudication with
bounded consultations and native fallback. Preserve that adapter and job path. Do not enable
all LLM uses globally: query expansion, entity resolution and summaries may run on reads.

The next justified extraction experiment is a bounded source window for third-person,
multi-actor or multi-topic narrative and unresolved references. Extract multiple facts with
literal source spans, explicit actors, dates, numbers, negation and causal predicates; never
let the model choose visibility, scope or source IDs. Trigger only on demonstrated native
coverage failures; record the trigger, model, tokens, latency and source hash. Validate false
relations, actor confusion and unsupported facts as well as recall. A failed gateway must
preserve the raw turn and native output. Ingestion cost is measured separately from recall.

Precompute typed links and episode membership from those grounded facts. At retrieval,
follow bounded links to raw evidence (the memory hydration fixed here supplies this path).
Do not infer causality from co-occurrence. Evaluate multi-hop support coverage before the
reader, then fixed-reader answer accuracy and adversarial abstention. Use conversation-held-out
development splits and LongMemEval updates/multi-session tasks before claiming generalization.

## RAG experiment protocol

`python -m benchmark.rag_ablation` reuses and audits the full existing 5183-document,
6814-chunk SciFact index, runs all 300 queries sequentially per arm and writes each result.
Arms: control, soft document cap 1/2, neighbor off, verification off, parent off,
definition off, control repeat. Each artifact records actual settings, source hash, corpus
identity, rankings, per-stage latency and true recall separately from hit rate. Comparisons
use paired query bootstrap intervals with a fixed seed; these are exploratory comparisons
without multiple-testing correction, not held-out model selection.

The O(n) soft cap operates before the primary cut, preserves score order within each tier,
fills spare slots from overflow and exempts exact hits/companions. A query selecting exactly
one document bypasses it. It defaults off pending evidence across document types.

Repeated document chunks are not automatically waste: they can hold complementary evidence.
The full index actually has 6814 chunks for 5183 abstracts, so “one chunk per abstract” is
not literally true for this parser. However SciFact still cannot validate hierarchical
parent/footnote/table/definition completeness. Keep the golden multi-hop corpus and negative
controls alongside external relevance metrics. Fielded BM25 also needs an indexed ablation
and fingerprint change; payload metadata alone does not implement field-aware scoring.

## Completed full-corpus ablations

All eight arms used the same audited index and frozen source hash, real Granite ONNX,
BM25, Qdrant and PostgreSQL. Authorization/cache and NLI use the documented benchmark
stand-ins. Each arm includes its first query; these are serial engine timings, not HTTP/load.

| Arm | nDCG@10 | True recall@10 | p99 ms |
|---|---:|---:|---:|
| Control | 0.7436 | 0.8731 | 450.8 |
| Document cap 1 | 0.7459 | 0.8806 | 699.0 |
| Document cap 2 | 0.7436 | 0.8731 | 403.0 |
| Neighbor off | 0.7436 | 0.8731 | 414.5 |
| Verification off | 0.7436 | 0.8731 | 354.2 |
| Parent off | 0.7436 | 0.8731 | 328.5 |
| Definition off | 0.7436 | 0.8731 | 333.7 |
| Control repeat | 0.7436 | 0.8731 | 395.6 |

`codex_rag_gap_comparison.json` contains the paired comparisons. Control/repeat have zero
changed rankings. Cap 1 changes 210 rankings but improves nDCG on only three queries, with
297 unchanged and none worse; mean delta +0.00234, bootstrap 95% interval [0, +0.00564].
Cap 2 changes 11 rankings without changing quality. Each off-arm has identical top-10
document rankings to control. The earlier small-corpus negative effects did not reproduce.
No flag was disabled on that basis, and diversity remains experimental/off. The larger
candidate count under cap 1 includes companions, which deliberately remain available.

These document experiments preceded a subsequent memory-only context-packing fix. Its
first LoCoMo probe (`codex_graph_memory_probe_before_packing.json`) found hydration happened
but graph memories never reached full contexts. Primary memories now retain their configured
cap while graph companions use the graph stage's separate bounded allowance; the token
budget still bounds the complete bundle. Regression tests cover both a full cap and a tight
token budget, plus an actual SQL/graph/context path with lexical/dense hits deliberately
absent. Final-source document controls and corrected graph measurements are recorded below.

No 85% LoCoMo or 300 ms production p99 claim follows from a feature implementation or a
passing hermetic test suite.

## Graph memory result after packing was corrected

`codex_graph_memory_probe.json` verifies all 369 source-turn bodies for conversation 1 in
the existing isolated index. It runs 105 questions twice, alternating cap-0/cap-6 order
within each question, with ten warmups and bundle caching disabled (420 timed requests).
Graph memory companions reach 36/105 questions per pass; all 72 treatment requests fetch
six companions, and 72 memory lists change. Neither arm expires its graph traversal budget
during timed queries. The paired source-presence metrics remain identical: mean annotated
source recall 0.7992 and complete-source presence 0.7619. These figures include annotated
adversarial questions and are **not answer correctness or abstention scores**.
`codex_graph_memory_analysis.json` also separates the 81 unique answerable questions:
source recall **0.7891**, complete-source presence **0.7407** in both arms; 32 receive graph
companions. Repeated queries are not treated as additional independent quality samples.

Control p50/p95/p99 is 120.7/305.0/542.5 ms; hydration is 125.9/396.9/485.9 ms. This does
not establish a speedup or the 300 ms SLO: tails vary, p95 is worse, and this is one warm
serial corpus. The median increases about 5 ms. Current graph relation ranking does not
recover additional gold sources here, despite the now-functional evidence path. Improving
typed relations, entity resolution and temporal candidate selection remains necessary work
to test; adding more co-occurrence evidence is not demonstrated to improve accuracy.

Reproduce using the existing isolated LoCoMo index with the same Docker mounts/environment
as the latency harness, `MEMORY__DATABASE__URL` ending in `memory_bench_conv`, and:

```bash
python -m benchmark.graph_memory --conversation 1 --repeats 2 --out graph_probe.json
```

No store reset occurs. The old diagnostic script and before-packing artifact are retained
for the failed first hypothesis; the maintained probe is now `benchmark/graph_memory.py`.

## Final-source RAG controls

After the memory packing fix, two further full SciFact controls
(`codex_rag_gap_final_baseline.json` and `codex_rag_gap_final_baseline_repeat.json`) reproduce
nDCG@10 **0.7436**, true recall@10 **0.8731** and hit rate@10 **0.8867**. All 300 rankings
agree. Serial p50/p95/p99 is **140.3/267.8/337.5 ms** and **142.8/253.0/405.6 ms**.
The canonical `external_retrieval.json` is the first final-source control, not the fastest
retrospectively selected run. Source hashes include both maintained benchmark modules and
match the corrected graph probe. No quality gain or achieved production SLO is claimed.

## Final regression validation

The broader final run passed **845 tests**, skipped **2**, and had **0 failures**. It covers
all unit tests, affected PostgreSQL graph/memory/retrieval/context integrations, security,
evaluation gates (including all six 150-distractor capability cases), retrieval end-to-end,
search contracts and OpenAPI contracts. A separate additional golden-definition ablation
passed, giving **846 passed / 2 skipped / 0 failed** across these two final runs.

That golden test isolates definition expansion on the actual ACME cross-page document:
turning it off removes the DEFINED_BY contribution, while the independent verifier fetches
the missing page-1 definition and restores completeness. This is a mechanism/interaction
test with stand-in embeddings, not a new real-reader accuracy benchmark. It explains why
a flat end-to-end ablation score alone cannot establish that a stage is disconnected.

The two skipped lanes are real DeBERTa grounding and real Docling PDF evaluation; the host
lacks their optional dependencies. Linux ONNX/SciFact runs do not substitute for those tests.
Ruff lint/format passed (384 Python files at the full check), Pyright reported 0 errors and
21 existing optional-dependency warnings, and `git diff --check` passed. The graph functional
integration case uses a 2-second traversal deadline to avoid testing host speed; dedicated
budget tests and all real-model probes retain the shipped 150 ms deadline.

Results: `codex_feature_validation.json` records every final test case and skip reason;
`codex_feature_gate_results.json` preserves regenerated gate reports. Historical gate files
were restored; canonical external retrieval intentionally uses the final-source control.
The four temporary benchmark containers were removed. No code was committed or deployed.
