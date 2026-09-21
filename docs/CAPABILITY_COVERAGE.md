# Capability coverage: what is actually covered, verified in code

Written because a cut list argued from flag names is worthless. The question is never "is
`raptor` on?" — it is "does anything answer a corpus-level question?". Each row below was
checked against the source, and the verdict says which mechanism provides it.

> **Status, 2026-09-20.** All seven flags named as redundant below have since been *removed*
> — not merely left off. This document is the argument for that, and
> `tests/eval/test_capability_coverage.py` is the executable form of it: one test per row,
> each demonstrating the capability surviving without the flag. The fixture also asserts that
> none of the seven names has reappeared in `RetrievalSettings`, so if one comes back the
> proof fails rather than quietly going stale.

## 1. Retrieval capabilities

| Capability | Covered by | Verified | Redundant flag |
|---|---|---|---|
| Exact match on identifiers, error codes, SKUs | `exact` + BM25 sparse with server-side IDF | on by default | — |
| Semantic / paraphrase match | dense (Granite 384-d) | on | — |
| Rank fusion across retrievers | native Qdrant RRF (`FusionQuery`), `rrf_k=60` | on | — |
| Precision at the top | hybrid fusion (dense + sparse, RRF) | on | **`colbert`** — late interaction approximates a cross-encoder, and the cross-encoder itself measured *worse* than no reranking at all (p = 0.012, MEASUREMENTS.md §3e), so `rerank` is now off by default too |
| Chunk understood in document context | contextual header (title, section path, page, entities) prepended before indexing | on | **`late_chunking`** — see §2, plus it is incompatible with a served model tier |
| Referent resolution at answer time | `PARENT`, `PREVIOUS`, `NEXT` expansion | on | — |
| Term definitions | `DEFINED_BY` expansion | on | — |
| Qualifying footnotes, cross-references | `FOOTNOTE`, `CROSS_REFERENCE` always in the expansion kinds | on | — |
| Multi-hop reasoning (A→B→C) | graph retrieval, `hops=2` for multi-hop query types | on | **`graph_ppr`** — a ranking refinement over a traversal that already happens |
| Corpus/document-level questions ("what does this say overall") | `build_summaries()` — one summary per section, subsection and document, indexed as `kind="summary"` so `GLOBAL_SUMMARY` questions find them | on | **`raptor`**, **`graphrag_global`** — hierarchical summarisation already exists and is indexed |
| Long-document navigation | document hierarchy + `section_path` on every chunk | on | **`pageindex`** |
| Vocabulary expansion (synonym not present in text) | dense embedding, partially | on | **`minicoil`** — a lighter sparse variant; `splade` is the better single experiment |
| Evidence supports the claim | NLI cascade, `evidence_verification` | on | — |
| Refuse when evidence is insufficient | `abstain_when_insufficient`, `escalation_max_rounds=2` | on | — |

**Conclusion.** Seven of the eight off-by-default flags name a capability that another
mechanism already provides, and that mechanism is on. `splade` is the exception: learned
sparse expansion is not covered by anything on, and it is the one worth keeping as the single
experiment slot — especially since a measurement in this repository showed recall staying
high with a *random* embedding, i.e. the lexical path is carrying the result.

## 2. The one gap this review found

`late_chunking` is redundant **at answer time** — `PARENT`/`PREVIOUS`/`NEXT` expansion already
delivers surrounding text to the model, which is what resolves "the city". It is *not* fully
redundant at **retrieval time**: a chunk whose referent was named in an earlier, different
node will not match a query using the proper noun.

Coverage of that case today:

| Where the entity appears | Covered |
|---|---|
| Document title | yes — contextual header |
| Section path | yes — contextual header |
| Same node as the chunk | yes — `node.entities` |
| An earlier, different node | **no** |

`entities = node.entities or extract_entities(node.text)` is node-local, and ancestor entities
do not propagate down the tree.

**But `late_chunking` cannot fix it here.** It requires token-level embeddings over a whole
document; a served embed endpoint returns one pooled vector per input, so it is mutually
exclusive with the separate model tier (wiring now refuses the combination at startup).

**The fix that fits the architecture: propagate ancestor entities into the contextual header.**
Deterministic, no model call, works with a remote embedder, and it also helps BM25 because
the proper noun is literally present in the indexed text — late chunking only ever helped the
dense side. Given that the lexical path is doing the work here, that is the more valuable
half.

## 3. Memory types — measured, not assumed

Of 25 types, **11 are produced by extraction**: `SEMANTIC`, `PREFERENCE`, `USER`, `TOOL`,
`OBSERVATION`, `AGENT`, `PROCEDURAL`, `TASK`, `EPISODIC`, `BELIEF`, `ENTITY_SUMMARY`.

**14 are never produced anywhere** and are reachable only through an explicit caller hint:
`WORKING`, `CONVERSATION`, `SHARED`, `WORK`, `SKILL`, `DECISION`, `FAILURE`, `OUTCOME`,
`ARTIFACT`, `KNOWLEDGE_RAG`, `SUMMARY`, `DERIVED`, `POLICY`, `CUSTOM`.

They are not dead code — each has an admission weight and a lifetime default, so a caller who
sets one gets correct behaviour. They are an *import surface*, which is a legitimate thing for
an enterprise product to have. **Keep them; document them as such.**

This review corrected an error introduced earlier in the API documentation, which told callers
that `SUMMARY`, `DERIVED` and `KNOWLEDGE_RAG` were "written by the pipeline". They are produced
at zero sites. Telling a caller not to set a value that nothing else sets makes it
unreachable. A test now asserts the description matches what the pipeline produces.

## 4. What this changes about the earlier cut list

The earlier document argued for cuts from surface area. This one argues from coverage, which
is the defensible basis, and it reaches a *narrower* conclusion:

- **Cut the seven flags** — each names a capability demonstrably provided by something on.
- **Keep `splade`** — the only genuinely uncovered retrieval capability.
- **Keep all 25 memory types** — reversing the earlier recommendation. They are an import
  surface with correct admission behaviour, not unused enum values.
- **Alternative providers** (`mem0`, `cognee`, `langmem`, `graphiti`, `docling_graph`) are still
  unverified either way. They are not capabilities, they are substitutes for `native`; nothing
  in this review establishes whether they beat it. **No decision until measured.**

## 5. Still blocking

Everything above establishes *redundancy*, not *quality*. It shows that removing a flag does
not remove a capability. It cannot show that the covering mechanism is as good — only a
baseline on a corpus nobody here wrote can do that, and it still does not exist.

Cut the seven flags on the coverage argument. Do not tune `candidate_k`, choose between
`native` and its alternatives, or claim an accuracy number until that baseline lands.
