# ADR 0011: Context preservation — expansion, hierarchical summaries, evidence verification

**Status:** accepted · **Date:** 2026-09-15

## Decision
- **Expansion follows the Document Context Graph, not similarity.** For the best-ranked
  chunks the `ExpansionStage` pulls in what a reader would need next to them: the
  definition of a term the chunk uses (DEFINED_BY), the footnote it refers to (FOOTNOTE), the
  section it points at (CROSS_REFERENCE), the parent section's summary (PARENT) and the
  previous/next chunk (neighbours). Expansions never leave the source document (so they
  inherit its visibility), are marked with `expanded_from` / `expansion_edge`, and are
  bounded by `retrieval.expansion_budget_items`. Exact-id lookups are never expanded.
- **Hierarchical summaries are extractive and deterministic.** At index time every
  document/section/subsection node gets a summary built from the chunks beneath it (lead +
  keyword-density sentence scoring, tables skipped, bounded length). Summaries are stored on
  the node, indexed as `kind="summary"` records (`sum_<node_id>`) in the knowledge
  collection, answer GLOBAL_SUMMARY questions, arrive as PARENT expansions, and land in the
  bundle's `summaries` bucket — they never replace chunks as evidence.
- **Conversation windows carry a rolling summary.** Turns that fall out of the token window
  are digested (role + first sentence, bounded) into `conversation.summary` so long threads
  keep their arc without spending the budget.
- **Evidence-group verification is derived from structure.** The `VerificationStage`
  computes the *required companions* of the top chunks from the same edges — definition,
  footnote, referenced section (satisfied by any descendant of the section) — verifies
  they are present, and escalates (fetches them directly, up to `escalation_max_rounds`)
  when they are not. The report (`COMPLETE | INCOMPLETE | INSUFFICIENT`, required /
  satisfied / missing groups, escalations, notes) is returned by `/v1/recall` and
  `/v1/context` and rendered into the bundle.
- **Honest abstention.** No evidence, or no evidence sharing a content term with the
  question, yields `INSUFFICIENT`; the bundle renders `## Evidence status`, and the SDK's
  `context(..., require_evidence=True)` raises `InsufficientEvidence` instead of returning
  something an agent might answer from anyway.
- **The report describes the bundle, not the retrieval.** `ContextBuilder` re-checks the
  required groups against the items that actually fit the token budget; a dropped
  companion downgrades the report to INCOMPLETE with a note. Companions, expansions, facts
  and summaries are not counted against the caller's `limit` — that cap applies to ranked
  evidence only, so completing the evidence never silently evicts it.
- **Near-duplicate collapse.** Chunks with identical `text_hash` (the same document
  uploaded twice) collapse onto one candidate before reranking, with the twins recorded in
  the payload; this removes the crowding effect seen in the M6 benchmark.

## Evidence
- Critical gate (`benchmark/results/retrieval_gate.json`): Recall@20 = 1.00,
  Evidence-Group Recall = 1.00, and the new `critical_evidence_complete_rate` = 1.00 — every
  critical question ends with a COMPLETE report; for the EBITDA question the required groups
  are the Adjusted EBITDA definition (p1), footnote 3 (p20) and Section 8 (p14), all satisfied.
- `tests/integration/test_context_preservation.py`: expansion + verification complete the
  cross-page chain; escalation with `limit=1` and no graph/expansion help still fetches the
  Section-8 chunk; unrelated questions abstain; a 260-token budget yields INCOMPLETE with a
  budget note; GLOBAL_SUMMARY returns the document summary; long threads get a rolling summary.
- 234 tests pass; contract test regenerated.

## Consequences
- Required groups exist only where the Document Context Graph found structure; prose without
  definitions/footnotes/references verifies trivially (COMPLETE with no groups), which is
  honest but weak — the golden evidence groups in `tests/eval` remain the quality gate.
- Extractive summaries are faithful but not fluent; `llm.uses` includes summarisation for
  deployments that enable an LLM.
