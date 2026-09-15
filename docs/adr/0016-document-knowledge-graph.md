# ADR 0016: The document knowledge graph is extracted by a deterministic business grammar

**Status:** accepted · **Date:** 2026-09-15

## Context
The first document graph (ADR 0010) was an entity index: capitalised phrases typed THING,
`mentioned_in` / `co_occurs_with` / `discusses` relations. It routed retrieval to evidence
but could not answer "what was revenue in FY26", confused `ACME` with `ACME Corporation`,
`ARR` with `Recurring Revenue`, and produced noise entities ("Operating", "EUR m",
"Annualised"). An enterprise knowledge graph needs typed entities with resolved aliases,
factual relations with attributes and provenance, and a way to keep counterfactuals apart
from facts — without an LLM, because LLMs are off by default (ADR 0002).

## Decision
- **Two-pass extraction** (`modules/graph/document_facts.py`): a document-level lexicon
  (defined terms, a financial metric lexicon, numeric-table row labels, organisations by
  suffix, people with roles, a location gazetteer, programmes/events from headings,
  sections) with aliases (`ARR`, `ACME`, `total revenue` → `Revenue`, "the acquisition of
  Initech"), then longest-first case-insensitive linking per chunk so every surface form
  resolves to one entity with page evidence and mention counts.
- **A grammar of business statements** produces typed facts with attributes:
  `has_value` (period, currency, amount, change, previous value, direction, estimate),
  table cells as `has_value` per row × period with the table as provenance,
  `segment_of`, `driven_by`, `excludes` (from definitions and "excluded from" sentences),
  `provides` / `serves` / `operates_in` / `headquartered_in` / `employs` / `founded_in`,
  `approved_by`, `acquired` / `involves` / `closed_on` / `consideration`,
  `reduced` / `consolidated`, `refers_to`, `adopted_in`, `has_role` / `works_at`.
  "Would have been" statements become `would_have_value` with `hypothetical: true` so a
  counterfactual never masquerades as the actual figure; values attached to an excluded
  item ("excludes a EUR 7 million settlement") are attributed to the item, not the metric.
- **Precision over recall.** Every rule is anchored on an explicit cue, every fact keeps its
  sentence and chunk, single capitalised words that appear in lower case elsewhere are not
  names, and lexicon entries that are never linked are dropped. The structural layer stays
  (one `mentioned_in` per entity with pages, `discusses` per section, bounded
  `co_occurs_with` at low confidence) so the graph remains a router to evidence.
- **Resolution is alias-aware** in both stores (`aliases ?| [...]` in PostgreSQL), so
  `/v1/graph/query` and the retrieval graph stage resolve abbreviations and short forms.
- **Retrieval uses the facts.** The graph stage ranks facts by predicate cues in the
  question ("exclude", "drove", "pay"), de-duplicates identical triples across document
  copies, and the router sends relation-shaped questions ("what items does X exclude",
  "how much did X pay for Y") to the graph.
- **A KG gate.** `tests/eval/golden/kg_facts.json` lists the entities, aliases, facts,
  forbidden facts and noise names for the fixtures; `kg_gate.json` must show fact recall
  1.00, false facts 0, noise 0 and every golden question resolving to its fact. It is part
  of the release gate.

## Consequences
- English business/financial documents are covered well; other genres need lexicon and
  rule additions (the golden file is where a new document type is onboarded).
- LLM-backed enrichment providers (Graphiti, Docling Graph) remain pluggable and can add
  open-vocabulary relations on top; the native layer is the floor, not the ceiling.
- Graph API and SDK expose `aliases` on entities and `attributes` on facts.
