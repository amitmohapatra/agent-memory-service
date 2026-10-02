# ADR 0027: A conversation turn restated at ingest, searched with the turn

Date: 2026-10-02. Status: accepted (opt-in per tenant policy).

## Context

After ADR 0026 the memories are ranked by a general fusion that nothing was fitted to.
Through the service LoCoMo reads recall@10 0.800, with multi-hop (0.571) and open-domain
(0.511) questions furthest behind: their evidence is turns written for the person they
answer ("Yeah, we went last weekend, the kids loved it"), which share no words with the
question asked months later. Every non-LLM lever measured on both LoCoMo and LongMemEval
(session keys, diversity, entity hops, query expansion, a window key, small cross-encoders,
doc2query) either did not transfer or did not help.

The published evidence on what an LLM at ingest should write is specific
(LongMemEval, arXiv 2410.10813, Table 10): appending the model's facts to the turn's own key raised
round-level recall@10 from 0.692 to 0.784, while indexing the same facts as separate keys,
merged by rank, lowered it to 0.568. The LoCoMo paper finds the same for "observations".

## Decision

1. **A new model use, `memory_restatement`** (`modules/memory/restatement.py`). For a
   conversation message, the model is shown the turn, the turn before it, the speakers and
   the date it was said, and returns a standalone restatement (names for pronouns, absolute
   dates for relative ones, who/what/where/with whom/when/why) and up to three facts.
2. **Appended to the turn's own index key** (`memory_index_text`), never replacing it: the
   memory's content, payload text and rendering stay the turn verbatim; the restatement is
   stored as `system_metadata["restatement"]` and only searched. No new vector, no new
   collection, no new fusion weight.
3. **Untrusted output is checked**: a line is kept only when every number and every
   capitalised name in it occurs in what the model was shown (computed dates excepted), and
   lengths are bounded. A weak model can add less; it cannot add an invented person, place or
   quantity.
4. **Through the gateway only.** It is an ordinary `LLMAssist` use: it runs when the bound
   agent's or tenant's virtual key (or the operator's) can pay and the tenant's policy allows
   it, on the model the policy names for it (a fast use by default) - OpenAI, Gemini or a
   local model behind Bifrost alike. It is **opt-in**: the default policy (no policy row)
   leaves it out, so registering a key does not start a model call per message on its own;
   a tenant names it in its policy (`PUT /v1/model-key/policy`, `uses`). Without a model nothing
   changes.

## Evidence

Offline, LoCoMo conversation 1 (150 answerable questions), turns restated by a 2B CPU model
(IBM Granite 3.3 2B, one line per turn) and fused with the shipped ranking unchanged:

| key | recall@10 | multi-hop | single-hop |
|---|---|---|---|
| no restatement (ADR 0026) | 0.839 | 0.557 | 0.914 |
| restatement as a separate key | 0.844 | 0.612 | 0.914 |
| turn + restatement as an extra key | 0.851 | 0.612 | 0.914 |
| **turn + restatement as the turn's own key** | **0.858** | **0.617** | **0.943** |

One conversation is a noisy estimate (about three points); the full-corpus run and the
through-the-service run with a gateway model are what this ADR is re-measured by. A 2B
model is the floor: it resolves few dates and writes no facts; a hosted model through the
gateway writes both.

## Consequences

- One model call per conversation message at ingest (a few hundred tokens in, under a
  hundred out); none at query time.
- The index text of a restated turn changes; turns indexed before the model was configured
  keep their key until they are re-indexed.
- `benchmark.native_source_retrieval --ingest-uses memory_restatement` (with `BENCH_LLM=on`
  and the gateway) measures it through the service; the use is part of the corpus key.
