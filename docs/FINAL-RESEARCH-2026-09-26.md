# Final accuracy investigation — 26 September 2026

This is a primary-source review and a test plan, not a claim to have reproduced
competitors or achieved their scores. Results and deployment status belong in the
companion handoff. The user authorized up to $10 of newly funded OpenRouter use;
target new spend is at most $8, retaining $2 margin. Credentials stay in Bifrost.

## What the competitor numbers actually measure

- [Hindsight's benchmark repository](https://github.com/vectorize-io/hindsight-benchmarks)
  reports 89.0% on **LongMemEval-S using OSS-120B**, not a model-free LoCoMo system.
  Its corresponding LoCoMo table and backbone must be compared separately.
- [Hindsight's March 23 report](https://hindsight.vectorize.io/blog/2026/03/23/agent-memory-benchmark)
  describes a single retrieval call followed by model answering. Its 92.0 LoCoMo and
  94.6 LongMemEval results are different benchmarks. “No query-time LLM retrieval”
  does not mean no extraction model, embedding model, reranker, reader or judge.
- [Mem0's current evaluation repository](https://github.com/mem0ai/memory-benchmarks)
  reports LoCoMo 1425/1540 at top-200 and 1414/1540 at top-50. The 446 adversarial
  questions are outside those denominators. This is answer correctness, not source recall.
- [Mem0's April algorithm report, updated September 22](https://mem0.ai/blog/mem0-the-token-efficient-memory-algorithm)
  reports 92.5 LoCoMo and 94.4 LongMemEval with roughly 7,000 retrieved tokens.
  Managed results include proprietary optimizations. The public method uses one-pass
  additive extraction, preserves agent facts and historical state, and combines
  semantic, normalized keyword and entity signals. This does not establish 94%
  LoCoMo without LLMs or sub-300-ms p99.

## Methods, implications and decisions

| Method / primary source | Relevant mechanism | Implication for this service |
|---|---|---|
| [Hindsight observations](https://hindsight.vectorize.io/developer/observations) | Background consolidation of evidence-linked observations; scope determines which facts may combine | Existing reflection must be reachable by its sources' audience and must reconnect recent updates with older facts. Test visibility and invalidation, not just successful writes. |
| [Hindsight September 23 consolidation notes](https://hindsight.vectorize.io/blog/2026/09/23/bring-facts-not-beliefs) | Keep raw facts; reconcile observations; count distinct supporting sources; preserve differences in numbers, entities, negation and conditions | Text equality is insufficient to merge memory occurrences. Preserve source/speaker/time provenance during retrieval. Keep source deletion and revision checks. |
| [Mem0 temporal reasoning](https://mem0.ai/blog/introducing-temporal-reasoning-in-mem0) | Separate temporal enrichment on writes; deterministic query intent; additive temporal ranking without hard pre-filtering | Distinguish event time from ingestion time. Do not indiscriminately boost recent facts for historical questions. Unknown or approximate dates must survive. Temporal scoring needs a separate ablation before promotion. |
| [LongMemEval](https://arxiv.org/abs/2410.10813) | Session decomposition, fact-augmented retrieval keys, time-aware query expansion; tests extraction, multi-session reasoning, temporal reasoning, updates and abstention | An isolated source sentence may omit an antecedent. Evaluate representation granularity and contextual keys, retaining original text and provenance. LoCoMo alone cannot validate continuous learning. |
| [HippoRAG 2](https://arxiv.org/abs/2502.14802) | Graph-based associative retrieval and non-parametric continual memory | A mentions-only graph is not equivalent. Relation quality and passage links matter; PPR over low-quality links can amplify noise. Benchmark graph contribution separately from embeddings. |
| [Mnemis](https://arxiv.org/html/2602.15313v1) | Local retrieval plus a global route over a hierarchical memory graph | Broad enumeration needs coverage across episodes, not only nearest facts. Hierarchical selection uses models and adds work; its aggregate runtime is not our p99. Bounded query decomposition remains experimental until measured. |
| [HyperMem](https://arxiv.org/abs/2604.08256) | Higher-order associations and hierarchical memory | Topic/episode/fact representations are useful hypotheses. Merely adding a SUMMARY enum does not implement hierarchical retrieval. Require a producer, lifecycle, source expansion and ablation. |
| [HiMem](https://arxiv.org/abs/2601.06377) | Episode and stable-note memory with dynamic updates | Separate transient episodes from durable knowledge. Old facts must remain recoverable where historical questions require them, while current-state answers distinguish supersession. |

## Concrete changes under evaluation

1. **Provenance-aware retrieval deduplication.** Equal words from different speakers,
   dates or source turns remain distinct. Collapse repeated representations only within
   an attribution group. Preserve document deduplication. Normalize once, hash aliases,
   and restrict containment comparisons to groups: O(total text + sum(group size²)),
   O(candidate count + normalized text) space, within the bounded retrieval pool.
2. **Continuous reflection history.** Discover updated as well as newly created current
   facts. For each active scope, reserve bounded capacity for older facts with the same
   subject, owner, scope and audience. At most four indexed subject lookups; model calls
   stay outside transactions and outside `/context`. A partial index supports recent
   update discovery. Durable per-source revision receipts include successful empty
   responses, survive restarts, and leave changed revisions pending. Original verbatim
   turns are eligible history without adding them to native landing aggregation.
   Each worker invocation scans at most 1,000 returned sources and makes at most 25
   bounded consultations. This is memory updating, not model-weight training.
3. **Reflection audience correctness.** Persist the intersection of cited source audiences,
   as the derived-memory layer already does. Do not force every shared insight private.
   Commit-time source revisions and audience validation remain mandatory. No private or
   run-only source grants a new reader access. Existing stored private reflections are
   not automatically republished by changing this code.
4. **Bounded reflection prompts.** Construct prompt lengths linearly, retain complete
   source text and allow a larger output request. The transport's configured maximum
   still caps output; record truncation rather than treating HTTP 200 as valid JSON.
5. **Affordable evaluation.** Reuse exact paid extraction responses. Keep reader model,
   prompt and ruler fixed within each comparison. Parallel readers reuse the existing
   resumable runner, with bounded workers, request pacing and a conservative cost check
   before sending requests. Never score provider failure as incorrect.

## Measurement contract

- Complete the frozen 37-question comparison first: 11 multi-hop and 26 temporal
  questions on all 369 turns of conversation 1. This is development evidence only.
- Preserve original full-corpus artifacts and their hashes. Copy the old native database
  before applying newer schema migrations; original indexes remain read-only.
- Evaluate all 1,986 LoCoMo questions for retrieval; use all 1,540 answerable questions
  for the answerable denominator and report 446 adversarial questions separately.
  Four answerable questions lack source annotations, so retrieval coverage has n=1,536.
- Compare the archived pre-change deduplication function with the candidate at equal
  retrieval depth and token budget, alternating arm order. No gold enters retrieval.
- Split development conversations 0–2 from validation conversations 3–9, while admitting
  that historical baseline scores for the whole dataset were already inspected.
- Keep strict scoring primary. Any competitor-style ruler is a separately labeled
  sensitivity analysis on identical predictions; never replace failures by easier gold.
- Record CPU builder latency separately from HTTP/load p99 and answer-generation latency.
  Concurrent ingestion/testing makes timings exploratory; a quiet rerun is required.
- Run cross-thread, source-update/delete, private/run isolation and document-RAG regression
  tests. Synthetic tests establish contracts, not an external benchmark score.

## Limits that must not be hidden

The old 24-hour discovery cutoff is removed. Revision receipts prevent unchanged work
from being repeatedly charged after restarts; a successful empty response also advances
progress. Discovery uses an ordered anti-join against receipts: the returned batch is
bounded, but PostgreSQL may examine more historical rows. It is not an O(batch-size)
durable work queue or a fair scheduler for arbitrarily many busy tenants. Concurrent
worker invocations can still consult the model twice before either writes receipts;
admission remains serialized and current derived slots remain unique. The existing
periodic schedule is every six hours, not immediate per-message learning.
Topic-aware exhaustive consolidation, canonical entity
resolution, contextual multi-turn extraction, calibrated event-time reranking, a learned
late-interaction reranker and a global memory hierarchy are separate capabilities; they
must not be claimed from these changes. Each has write/storage/read costs and requires
its own corpus-level and lifecycle measurements. No finite review covers all research.

## Findings from completing the frozen complex comparison

The raw existing judge returned native 34/37 and assisted 32/37 with identical GPT-4o
readers. Those are one-conversation development results. Reviewing all 74 predictions
found incomplete lists/qualifiers that the nominally strict judge accepted. A separate,
conservative complete-key-content audit gives 31/37 and 30/37; it is manual sensitivity
analysis by the coding assistant, not a new independent benchmark. Original predictions,
judgments, gold and the raw score remain unchanged. Neither result establishes overall
90% accuracy or a reliable assisted-ingestion gain.

The apparent verbatim count drop (350 to 241) is a category-count artifact: both arms
preserve exactly the same 355 full original turns. The same 14 greeting/question turns
are filtered in both. Audit text and evidence identity before diagnosing source loss.

The full retrieval dataset contains duplicate question text across answerable/adversarial
categories. Evaluation/resume keys must use conversation plus original question ordinal.
The revised harness preserves all 1,986 rows and checks metadata before reusing a control.
