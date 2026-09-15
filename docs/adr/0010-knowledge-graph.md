# ADR 0010: Knowledge graph — PostgreSQL-native, evidence-carrying, bounded traversal

**Status:** accepted · **Date:** 2026-09-15

## Decision
- **Graph in PostgreSQL, next to the canonical rows.** `graph_entities` (unique on tenant +
  scope + canonical name) and `graph_relations` (temporal: `valid_from`, `valid_to`,
  `status`, `superseded_by`) with JSONB audience keys and GIN indexes. The graph is derived
  state — rebuildable from memories and chunks — but one backup restores everything and no
  extra database is required. Neo4j/Graphiti is an optional adapter, not a dependency.
- **Every relation carries evidence.** Facts point at the chunk/node/page or message they
  came from; the retrieval stage uses the graph as a *router to evidence* and pulls the
  source chunks into the bundle. The graph never substitutes for the text it summarises.
- **Native enrichment is deterministic.** Memories contribute typed facts from their
  consolidated triple (`user:u1 —works_at→ acme corp`) plus `mentions`; documents contribute
  `defined_in` (definitions), `mentioned_in` (entity → document, per chunk/page),
  bounded `co_occurs_with` (≤ 6 entities per chunk) and `discusses` from section titles
  ("Restructuring Programme" → "Annualised"). Ids are deterministic hashes so re-enrichment
  upserts; `delete_for_document` + re-run gives an exact rebuild.
- **Temporal facts follow memories.** When a memory is superseded, expired or forgotten, its
  relations are closed (`SUPERSEDED`, `valid_to`) by the same index job. Traversal shows
  CURRENT facts by default and `as_of=<instant>` shows what was valid then.
- **Isolation is store-side and bounded.** Traversal is hop-by-hop with one indexed query per
  hop, filtered by `tenant_id` and `visibility_keys ?| audience`, capped by `max_visited`;
  entities reached through a visible edge are re-checked against the audience, so a visible
  hub cannot tunnel into invisible leaves. The `global` audience key is tenant-scoped
  (`global:<tenant>`) — found while writing the gate: a bare `global` key would have made
  the tenant column the only barrier.
- **Retrieval integration.** `GraphStage` runs as a post-stage for entity/relation, multi-hop,
  temporal and decision queries: entities are resolved from the question (extracted names +
  lowercased n-grams), 1 hop (2 for multi-hop, `max_visited=80`), typed facts first, then
  the facts' evidence chunks are inserted after the ranked evidence as `GRAPH_EVIDENCE`
  expansions. Evidence is capped at the caller's `limit` after post-stages; facts ride along
  into the bundle's `graph_facts` bucket with `relation_id:` citations. `/v1/recall` returns
  facts only when `kinds` includes `fact`.
- **API/SDK.** `POST /v1/graph/query` (free text or explicit entities, `hops`, `as_of`,
  `max_visited`) and `ctx.graph.query(...)`.

## Evidence
- `tests/security/test_graph_isolation.py`: 96 reader configurations × 144 objects across two
  tenants, 2-hop traversal from a tenant-wide hub — returned == oracle-allowed, 0 leaks.
- `tests/integration/test_graph.py`: fixture graph (definitions/mentions/co-occurrence with
  pages 1/11/14/20), idempotent rebuild, multi-hop stage reaches pages 14 and 20 for the
  EBITDA question, thread/workspace/tenant visibility, memory facts with supersession and
  `as_of`, forget retires facts.
- Retrieval gates unchanged with the stage on (Recall@20 = 1.00, EGR = 1.00); benchmark with
  25 salted copies: recall p95 ≈ 84 ms (from 36 ms without the graph stage; 209 ms before the
  GIN indexes and the `max_visited=80` retrieval cap), still inside the 300 ms budget.

## Consequences
- Native entity extraction is capitalisation/definition based; lowercase concepts are only
  reachable through section titles or memory triples. The LLM `relation_extraction` use and
  the Graphiti / Docling Graph adapters (both `requires_llm`) are the escalation path.
- Graph fan-out grows with near-duplicate corpora; per-document dedup (M10) will also shrink
  the graph.
