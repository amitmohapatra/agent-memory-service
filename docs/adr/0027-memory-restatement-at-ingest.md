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
   dates for relative ones, who/what/where/with whom/when/why), up to three facts, and up
   to four relations (person, snake_case predicate, object): one call for all three.
2. **Appended to the turn's own index key** (`memory_index_text`), never replacing it: the
   memory's content, payload text and rendering stay the turn verbatim; the restatement is
   stored as `system_metadata["restatement"]` and only searched. No new vector, no new
   collection, no new fusion weight.
3. **Untrusted output is checked**: a line is kept only when every number and every
   capitalised name in it occurs in what the model was shown, and lengths are bounded (a
   computed date - "2023-05-07", "May 7, 2023", "the 7th of May" - is
   the one exception). A relation is kept only when every word of its person and its object
   (pronoun-like function words and computed dates aside) occurs in what the model was
   shown, and its predicate is short snake_case. A weak model can add less; it cannot add an
   invented person, place or quantity.
4. **Through the gateway only.** It is an ordinary `LLMAssist` use: it runs when the bound
   agent's or tenant's virtual key (or the operator's) can pay and the tenant's policy allows
   it, on the model the policy names for it (a fast use by default) - OpenAI, Gemini or a
   local model behind Bifrost alike. It is **opt-in**: the default policy (no policy row)
   leaves it out, so registering a key does not start a model call per message on its own;
   a tenant names it in its policy (`PUT /v1/model-key/policy`, `uses`). Without a model nothing
   changes.
5. **The relations link the turn in the knowledge graph.** They are stored as
   `system_metadata["restatement_relations"]`; graph enrichment (`modules/graph/native.py`)
   writes each as a model-extracted relation (confidence at most 0.6, `extraction: llm`)
   bound to the turn, and links the turn to each object it names, as the open extractor does
   for non-English text. The English rules found typed facts only in sentences shaped like
   "I work at X"; a casual turn ("we took the kids to a pottery class") now reaches the graph
   too, with no second model call. The speaker the model names is the speaker's own node.
6. **Turns stored before the model could restate them** are restated by a tool, exactly as
   ingest would (bound to each turn's owner, so their key pays and their policy decides),
   then re-indexed and re-linked: `python -m memory_service.tools.restate --tenant acme
   [--limit N] [--force]`. The turn's content is unchanged, so nothing derived from it is
   invalidated.
7. **Which model, per use.** The tenant's policy names the gateway model for each use
   (`PUT /v1/model-key/policy`, `models: {"memory_restatement": "openai/gpt-4.1-mini"}`);
   a use it does not name calls the service's fast model (`LLMTuning.fast_model`).

8. **The model is told who and when.** The prompt names the turn's speaker and addressee
   (who "I" and "you" are) and hands over the dates the service already resolved in the
   turn (`modules/memory/temporal.py`, ADR 0024 decision 7), so the model copies a date
   rather than computing one. A line that only echoes the turn (90% of its words from it)
   and a relation whose person or object is a placeholder ("N/A", "none", "unknown") are
   dropped.

## Evidence

Offline, LoCoMo conversation 1 (150 answerable questions), turns restated by a 2B CPU model
(IBM Granite 3.3 2B, one line per turn) and fused with the shipped ranking unchanged:

| key | recall@10 | multi-hop | single-hop |
|---|---|---|---|
| no restatement (ADR 0026) | 0.839 | 0.557 | 0.914 |
| restatement as a separate key | 0.844 | 0.612 | 0.914 |
| turn + restatement as an extra key | 0.851 | 0.612 | 0.914 |
| **turn + restatement as the turn's own key** | **0.858** | **0.617** | **0.943** |

Through the service (`native_source_retrieval --ingest-uses memory_restatement`), the same
conversation restated by the same 2B model behind a gateway stand-in, with the facts and
relations of the final prompt, read 0.838 -> 0.842 (multi-hop 0.557 -> 0.586, open-domain
0.545 -> 0.591, temporal 1.000 -> 0.973, one question), and 0.900 -> 0.897 at 20: within
noise. The 2B model leaves pronouns in and writes facts without names; two of 419 calls
returned invalid JSON and kept the turn as it was.

One conversation is a noisy estimate (about three points); the full-corpus run and the
through-the-service run with a gateway model are what this ADR is re-measured by. A 2B
model is the floor: it resolves few dates and writes no facts; a hosted model through the
gateway writes both.

**Answers, not just recall (2B model, judged).** Recall says whether the evidence reached
the bundle; it cannot say whether the restatement helps a model answer. LoCoMo conversation
1, a seeded per-category sample of 39 questions (`benchmark.locomo --sample 40 --judge`), the
shipped bundle, the same 2B model answering and grading behind the gateway stand-in, with
and without `memory_restatement` at ingest (the prompt of decision 8):

| ingest | answerable, strict ruler | answerable, lenient ruler | all evidence in top 10 |
|---|---|---|---|
| no model | 17 / 30 | 27 / 30 | 24 / 30 |
| 2B restatement | 13 / 30 | 21 / 30 | 24 / 30 |

Paired on the same questions, the restated corpus lost 5 answers and won 1 under the strict
ruler (exact test p = 0.22), lost 6 and won none under the lenient one (p = 0.03). The
evidence reached the bundle as often; 35 of 39 answers differed in wording, and the losses
are the small answerer misreading a bundle that holds the answer (a pet and a slipper given
to the wrong person). The 2B grader is consistent with itself (one verdict of 39 changed on a
re-grade) but strict about form: it rejected "2023-07-02" for "2 July 2023" and "twice" for
"2", which is why the lenient ruler reads ten points higher on both arms. So with a 2B model
the restatement does not pay; it stays opt-in, and the run that decides it is the same
judged pair with a hosted model through the gateway.

**Tried and removed: the graph as a ranked list.** Fusing the traversal's memories into
the ranking by reciprocal rank (instead of appending them after it) read 0.807 / 0.807 /
0.802 recall@10 on LoCoMo at weights 0 / 0.25 / 0.5 without a model, and 0.842 / 0.842 /
0.828 / 0.779 at 0 / 0.25 / 0.5 / 1 on conversation 1 with the 2B model's relations. It was
neutral at best, so it was removed rather than shipped off.

## Consequences

- One model call per conversation message at ingest (a few hundred tokens in, about a
  hundred and fifty out with the relations); none at query time.
- The index text of a restated turn changes; turns indexed before the model was configured
  keep their key until they are re-indexed.
- `benchmark.native_source_retrieval --ingest-uses memory_restatement` (with `BENCH_LLM=on`
  and the gateway) measures it through the service; the use is part of the corpus key.
