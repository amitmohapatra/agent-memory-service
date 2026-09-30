# Where the service consults a model

Every path is complete without a model: each use below is optional work layered on a
deterministic path, and any model failure (no key, gateway error, rate limit, invalid output)
falls back to that path. A use runs only when all of these hold (`LLMAssist.wants`):

1. the deployment allows it: `MEMORY__MODELS__LLM__USES` (default: every use) and
   `MEMORY__MODELS__LLM__ENABLED` is not `false`;
2. the tenant policy of the identity that owns the work allows it (`PUT /v1/model-key/policy`,
   resolved agent → workspace → tenant, the same order as keys);
3. something can pay: a registered key at one of those levels, or the operator key;
4. on a read, the read may consult the model: the request's `use_llm`, or when the request
   does not say, the policy's `read_assist`.

The tier is the model the call goes to: `fast` (`MEMORY__MODELS__LLM__FAST_MODEL`) for the
uses in `MEMORY__MODELS__LLM__FAST_USES` (default `contextual_extraction`, `query_expansion`,
`chunk_context`), `strong` (`MEMORY__MODELS__LLM__MODEL`) for the rest. Every system prompt
ends with one rule (`modules/llm/assist.py:SOURCE_LANGUAGE_RULE`): text is returned in the
language of its source, never translated; only schema labels (query types, snake_case
predicates, field names) are fixed English.

`/v1/context` and `/v1/recall` make no model call unless the read is assisted (rule 4), and
then only `query_expansion` - the `/v1/context` latency target (p95 < 300 ms) is measured
with the model off (`docs/MEASUREMENTS.md`, section 8).

## Ingestion (background jobs; never on the request that wrote the data)

| use | when it runs | tier | what it produces | without it |
|---|---|---|---|---|
| `contextual_extraction` | a user message the rules cannot fully read. **English**: two or more sentences no rule parsed → the model selects source spans (narrative units, `modules/memory/narrative.py`), never new wording. **Any other language** (`Observation.lang`, `domain/language.py`): every message with a sentence that is not a question or an acknowledgement → typed facts in the message's language, each citing its sentences (`modules/memory/source_facts.py`); a slot (`lives_in`, `works_at`, `name`, ...) only when its value is copied from the cited text, so a German message can supersede an English fact | fast | OBSERVATION spans (English); PREFERENCE / USER / SEMANTIC / EPISODIC / TASK facts (other languages) | English rules only; the verbatim turn is kept in every language, so the text stays retrievable through the multilingual dense space |
| `relation_extraction` | graph enrichment of a memory. English: ≥ 2 entities found by the rules and only `mentions` edges between them → typed relations among those entities. Other languages: the model names both ends and the relation (`graph/native.py:open_relations`), and both names must occur verbatim in the text. Documents: the top co-occurring entity pairs (English), and up to `LLM_MAX_OPEN_CHUNKS_PER_DOCUMENT` = 6 chunks not in English | strong | typed edges (`extraction: llm`, confidence ≤ 0.8) | `mentions` / `co_occurs_with` / structural edges |
| `chunk_context` | document ingestion: parts of a split node, tables, and chunks not in English, at most 48 per document | fast | a 1-2 sentence situating context indexed with the chunk (`text` never changes) | the deterministic header (title, section path, salient entities) |
| `conflict_adjudication` | a new fact in the grey band of lexical similarity to an existing one (same numbers, same negation) | strong | supersede / keep-both decision | the deterministic lexical/dense thresholds |
| `summaries` | document node summaries (bounded number per document), thread summaries (`summary.refresh`, every `SUMMARY_EVERY` messages), the `user` profile block, graph entity summaries (≤ 4 model calls per enrichment job) | strong | abstractive text | the extractive / template text, stored the same way |
| `reflection` | periodic job, per principal with a key, over recent memories | strong | cited insights (≥ 2 sources) | none |
| `memory_connections` | periodic job over recent memory pairs | strong | typed edges between memories (supersedes / contradicts / relates) | none |
| `procedure_abstraction` | the tool-learning job, for a procedure that clears support and success-rate gates | strong | title and strategy text distilled from successes and failures | the miner's own rendering |

## Reads (only when the read is assisted)

| use | when it runs | tier | fallback |
|---|---|---|---|
| `query_expansion` | `/v1/context` or `/v1/recall` whose question no rule classified - which includes every question not in English, since the router's cue patterns are English and never route another language (`modules/retrieval/router.py`) | fast | the unexpanded hybrid search (every dense space, BM25, the graph by entity name) |
| `entity_resolution` | `/v1/graph/query` names that match no entity lexically | strong | lexical match only; the retrieval-time graph stage never uses it (it runs under the graph budget) |
| `grounding_judge` | `/v1/verify` and `/v1/context` with `answer`: claims the NLI cascade could not decide | strong | the claim stays undecided |

Removed in 0.3.0: `query_decomposition`. A model call on the read path took 3.2-12.2 s
(median 6.4 s) through the local gateway against a 300 ms budget, and one of five multi-hop
questions came back decomposed (`docs/MEASUREMENTS.md`, section 8).
