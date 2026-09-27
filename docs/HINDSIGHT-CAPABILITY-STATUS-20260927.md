# Hindsight integration: current capability status

This describes `codex/agent-capabilities-20260927`, including the integration stack; it is not the deployed service.
Implementation is not benchmark validation, and a native counterpart is not API parity.
The older `CAPABILITY_COVERAGE.md` documents a September 20 design decision; its model,
fusion and measured-default statements are not a current deployment manifest.

| Capability | Current implementation | Validation and remaining boundary |
|---|---|---|
| Retain/extract | Native source retention, optional assisted extraction; Hindsight 0.10.1 SDK extraction preview for enabled contextual extraction | Mock HTTP SDK contract plus isolated PostgreSQL ingestion/deletion tests. No paid extraction quality measurement. |
| Consolidate observations | Native beliefs/entity summaries and background reflection; source dates preserved | Source revisions, audience intersection, recursive invalidation, replay and chronology tested. Native consolidation is not promoted by default. |
| Recall | Dense/BM25 fusion, exact lookup, bounded graph traversal, document expansion, memory source lineage | Full LoCoMo source-recall arms and document encoder screening are separate measurements. This does not establish answer accuracy. |
| Temporal recall | Validity, observation time, supersession and graph temporal queries | General query-time date ranking is not equivalent to Hindsight's temporal retrieval strategy. |
| Reflect | Configured query expansion and grounding assistance, independently permitted on each read | A multi-step Hindsight-style reflect agent is not implemented. `use_llm=false` denies read model calls. |
| Mental models | Persistent standing questions with native excerpts or explicitly assisted synthesis; scheduled refresh and model-free reads | Scope, revisions, expiry, definition races and citation IDs tested. Incremental per-source delta synthesis is not implemented. |
| Knowledge pages | Maintained titled briefs reuse the standing-question engine and native evidence | Not full wiki parity: folders, history, collaborative editing and filesystem projection are missing. |
| Multilingual | Shared Unicode lexical/sentence primitives, source retention; CPU embedding challengers evaluated | Twelve-language paragraph retrieval measured. Multilingual NLI, extraction and end-to-end answers are not certified. |
| Semantic caching | Revision/scope/budget/model/policy-bound context caching with cosine and ordered lexical guards | Conservative equivalent-query reuse; not arbitrary paraphrase or generated-answer caching. Forgetting and ACL invalidation tested. |
| Documents/storage | PostgreSQL source of truth, search projections, hierarchy, blob/archive lifecycle, outbox reconciliation | Reparse removes stale chunks and summaries. Generated search summaries remain separate from extractive SQL evidence. |
| Memory management | List/get/forget, reinforcement, supersession, evidence and lifecycle | Native API/SDK; no remote Hindsight memory bank replication. |
| Banks/configuration | Native tenant/principal/run/thread/group visibility | Hindsight bank templates/configuration API is not exposed. Tags are not a replacement for native authorization. |
| Operations | Durable job queue, job status, retry/reconciliation | Hindsight operation API compatibility is not provided. |
| Webhooks/templates | No matching implementation | Outbound webhook subscriptions and bank-template catalog are missing. |
| Agent memory | Run-tree visibility, tool outcomes/procedures, cross-thread source-backed learning | Existing isolation and lifecycle gates cover these. A scored multi-run task/cost evaluation remains outstanding. |
| Agent VKs | Encrypted owner-bound registration, rotation/revocation, retry-time validation and background-job identity | Mocked wire/SQL/API tests pass. Native Bifrost requests exclude MCP. All agents use the native extraction engine because the pinned Hindsight SDK cannot carry their model credential. |
| CPU performance | Bounded serial model executor, cached embeddings, batch yielding; neural reranking stays off | Real component timings are recorded separately from context-builder and HTTP timings. No production p99 claim. |

## Integration boundary

The SDK is a client for the Hindsight server, not an in-process extraction library.
Current reuse is stateless extraction preview: source storage, authorization, evidence,
indexing and deletion remain native. It does not mirror user memories into a remote bank.
This avoids creating a second authorization/deletion authority just to reuse extraction.
All agent identities use the native Bifrost path because the pinned extraction request
cannot carry a model credential. Keys are bound per request, never by mutating global defaults.

Hindsight mental models move synthesis into background work and serve a stored result;
knowledge pages use the same underlying model with wiki-oriented defaults. They are
valuable agent capabilities. The new native brief engine provides a bounded standing-answer
lifecycle; existing entity summaries alone did not implement it. [Mental models](https://hindsight.vectorize.io/developer/mental-models),
[knowledge pages](https://hindsight.vectorize.io/developer/knowledge-pages).

The current Hindsight retain documentation explicitly includes LLM extraction. A recall
benchmark with no generation call during retrieval therefore must not be described as a
fully LLM-free ingest-and-answer result. [Retain](https://hindsight.vectorize.io/developer/retain),
[recall](https://hindsight.vectorize.io/developer/retrieval).

## Promotion rules

- Keep each corpus-changing arm in its own database; preserve model/data/source hashes.
- Keep answer accuracy, direct source recall, complete-source recall, and paragraph
  retrieval separate. A fragment carrying a gold source ID may still omit the answer.
- Require fresh-context reader predictions to claim answer gains after retrieval changes.
- Keep model-call-free tests and mocked gateway tests separate from live LLM evaluation.
- Reindex for Unicode/model fingerprints and to remove legacy summaries without provenance.
- Do not declare whole-product feature parity or 90% accuracy from these component gates.

See `HINDSIGHT-INTEGRATION-HANDOFF.md` for commands, isolation details and current results.

See `AGENT-CAPABILITIES-HANDOFF-20260927.md` for the later credential/brief stack and its isolated schema.
