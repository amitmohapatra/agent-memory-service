# Multilingual multi-team memory platform — plan of record, 28 September 2026

Status: decisions and execution order. Supersedes `MEMORY-PLATFORM-IMPLEMENTATION-PLAN-2026-09-27.md`
(Hindsight-SDK-centric) wherever the two disagree; keeps its authorization and deletion contracts.
Every number below names its artifact. Nothing here is a new accuracy claim.

## 1. Where we start (verified 2026-09-28)

| Fact | Evidence |
|---|---|
| `main` = `efaf008`; codex checkpoint = `6b99db6`; merge-base `66b8fa2` (fix C). The checkpoint merges onto `main` with one comment-line conflict in `.env.example`; `src/` merges clean. | `git merge-tree` 3-way |
| Full LoCoMo: **72.92 %** strict / **81.17 %** on Mem0's ruler; `multi_hop complete@50` **0.333**; p99 **300.8 ms**. Loss buckets: 226 fragmentation / 109 reader / 81 absent. | `docs/HANDOFF.md` |
| Multilingual paragraph retrieval R@10, 12 languages: English Granite **65.96 %** → Bekko a8m **98.83 %**. English SciFact: Granite 0.7409/0.8912, Bekko alone 0.7256/0.8626, **Granite + Bekko + BM25 RRF 0.7557/0.8926** (offline list fusion; search p99 55 → 98 ms). | `CPU-MULTILINGUAL-DECISION-20260928.md` |
| Multi-hop source recall@50: Granite 58.04 % → E5 61.92 %; complete@50 32.27 → 35.82. Bekko LoCoMo arm never ran. | `LOCOMO-OFFLINE-COMPARISON-20260927.md` |
| The runtime dual-encoder ensemble is **not implemented**; default encoder and NLI are still English. | `PR-CHECKPOINT-20260928.md` |
| Auth is `trusted_dev` static keys or external JWT. There is **no tenant registry, no key issuance, no super-admin, no workspace/team management API, no revocation** (grants are never deleted), no retention job, no read audit, no per-tenant quota. | `modules/auth/authentication.py`, `modules/authz/service.py` (grant_* only), `PRODUCT_DECISIONS.md §4` |
| OpenFGA already models `tenant`, `group`, `workspace` (admin/member/viewer). `WORKSPACE`/`GROUP` audiences were withdrawn only because nothing issued membership grants. | `deploy/openfga/model.fga`, ADR 0005 |
| HITL exists only as `ObservationKind.FEEDBACK` folded into memories. No separate feedback store, no correction API. | `modules/memory/native.py:464` |
| The codex checkpoint adds per-agent encrypted Bifrost keys, LLM `auto` mode with gateway model discovery (Chinese families excluded), standing briefs (mental models / knowledge pages), semantic context cache, Unicode BM25, ONNX NLI, Tesseract OCR, migrations 0009–0013. 1,561 offline tests pass; the worker-kill requeue test is unresolved. | `AGENT-CAPABILITIES-HANDOFF-20260927.md` |
| **A working LLM path exists**: Bifrost on `:8091` (providers anthropic/deepseek/gemini/openrouter, `enforce_auth_on_inference=false`). Anthropic: no credit. Gemini 2.5-flash: retired. **`gemini/gemini-3.8-flash` answered a one-token probe.** `.env` still names `openrouter/openai/gpt-4o`. | probes this session |
| The harness (`agent-harness`) uses one process-wide `MemoryClient(url, api_key)` and calls `context, recall, observe, remember, verify, memories, get_memory, forget, files.add, graph.query, tools.record, runs.outcome, chat.*`. Framework adapters (LangGraph, deep agents, agents SDK) live there, not here (ADR 0020). | `agent-harness/src/universal_agent_harness/memory/runtime.py` |

## 2. What is being built

1. A **fully multilingual** stack — dense, sparse, chunking, grounding, OCR, temporal parsing — with **no Chinese-origin model or derivative** anywhere (runtime, catalogue, benchmark challengers, gateway discovery), enforced by a test.
2. **Accuracy up, latency down**: multi-hop is the defect; the fast path stays LLM-free and under p99 300 ms in-process.
3. A **multi-team platform**: super-admin onboards tenants; tenants issue keys; teams (workspaces) hold separate or shared data; boundaries are tested from the agent's side through the SDK only.
4. **Team-owned model credentials**: a team registers its Bifrost virtual key once; our prompts, their bill; ingestion may use it, reads stay model-free unless asked.
5. **HITL** stored separately and projected into memory state; **continuous learning** with or without it.
6. Parity with Hindsight/Mem0 on the surfaces that matter: mental models/pages (briefs), fact edit/history, webhooks, pagination and filters, deep mode.
7. SOLID, no duplicate paths, no dead code; the architecture and complexity ratchets are not raised.

## 3. Decisions

### D1 — Base
Integration branch `platform/multilingual-2026-09` = `main` + `codex/multilingual-agent-memory-checkpoint-20260928`.
Corpus-changing ingestion stays off by default (`consolidation_enabled=False`; contextual extraction only with a
registered credential). Measurement arms stay isolated per `CODEX-STACK-HANDOFF-2026-09-27.md`.

### D2 — Model stack (origin column is the compliance record)

| Role | Model | Origin / licence | Decision |
|---|---|---|---|
| Dense, English | `ibm-granite/granite-embedding-small-english-r2`, 384-d | IBM, Apache-2.0 | keep, as named vector `dense_en` |
| Dense, multilingual | `hotchpotch/bekko-embedding-v1-a8m`, 384-d, ONNX (`ModernBertModel`, 4 layers, 256k vocab) | Japanese maintainer; ModernBERT (Answer.AI/LightOn) + mmBERT (JHU) lineage; MIT | **add**, as named vector `dense_ml` |
| Fusion | Qdrant multi-prefetch `dense_en` + `dense_ml` + `bm25` → RRF; `dense_en` prefetch only for Latin-script queries | — | **implement** the measured 0.7557/0.8926 ensemble at runtime; both encoders run concurrently |
| Sparse | Unicode BM25 `v2-unicode` | none | keep |
| Reranker | none | — | keep off (measured −5.2 nDCG, 21× latency) |
| Grounding NLI | `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`, FP32 ONNX | Microsoft base; MIT | **freeze**; drop the English DeBERTa (same 36/40 golden score; one model, one path). Off the `/context` path. |
| OCR | Docling + Tesseract, all language packs | IBM / Google-HP; MIT / Apache | keep; the 12-language screen must pass in the container |
| Temporal | `dateparser` (BSD-3, 200 locales) with `RELATIVE_BASE=occurred_at` | — | **add** |
| Script tag | deterministic Unicode-script detection on every record, indexed payload `script` | none | **add** (drives prefetch pruning and per-script gates) |
| Generative | any gateway model the team's key exposes, minus excluded families | — | `_EXCLUDED` gains `moonshot|yi-|baichuan|internlm|hunyuan|doubao|ernie|gte|m3e|bce|bge`; Makefile default becomes `gemini/gemini-3.8-flash` |
| Purge | `bge-*`, `bge-reranker-v2-m3`, `gte-multilingual-base`, `qwen3-embedding-0.6b` | BAAI / Alibaba | **remove** from `benchmark/challengers.txt`, README, and `models/`; add `tests/unit/test_model_provenance.py` |

Multilingual **fact extraction** is LLM-assisted with the team's key using the source-span selector
(`modules/memory/narrative.py`: the model picks sentence ranges; no generated text is stored). The English
rule extractor remains the model-free floor. Verbatim retention already covers every script.

### D3 — Tenancy, keys, teams (the platform layer)
- Auth mode `api_key` (DB-backed) beside `trusted_dev`/`jwt`. Key format `mk_<prefix>.<secret>`; only a SHA-256 is
  stored; row = `key_id, tenant_id, role ∈ {admin, service, agent}, workspace_id?, expires_at, revoked_at, last_used_at`.
  Tenant is **derived from the key**; a `X-Memory-Tenant` header must match or be absent. This closes the
  "one key reaches every tenant by changing a header" hole for shared deployments.
- One bootstrap secret, `MEMORY__AUTHENTICATION__BOOTSTRAP_ADMIN_KEY`, is the platform super-admin. That is the whole
  configuration a new deployment needs beyond store URLs.
- Admin API: `POST/GET /v1/admin/tenants`, `POST /v1/admin/tenants/{id}/keys` (returned once), `DELETE /v1/admin/keys/{id}`.
  Tenant-admin API: `POST /v1/workspaces`, `POST/DELETE /v1/workspaces/{id}/members` (users, agents, groups),
  `POST /v1/keys` (scoped to the caller's tenant). Every write goes through the outbox; revocation bumps the
  tenant revision so scope caches die immediately.
- Visibility gains **`WORKSPACE`** (team-shared data; key `workspace:<tenant>/<ws>`); membership grants are issued by the
  API above, so the audience is usable — the reason it was withdrawn no longer holds. Revocation deletes tuples.
- Data lifecycle: per-tenant `retention_days` by record kind with a verifiable purge job; an append-only read audit
  (`tenant, principal, scope fingerprint, query hash, record ids`) written off the hot path; per-tenant token-bucket
  quota in Dragonfly returning 429 + `Retry-After`.

### D4 — Team model credentials
Generalise the checkpoint's per-agent credential store to `(tenant, scope_kind ∈ {tenant, workspace, agent}, scope_id)`
with precedence agent > workspace > tenant; no operator fallback in `api_key` mode. Same AES-256-GCM cipher, same
rotation/revocation semantics, same `auto` discovery. A team registers one key at workspace level and every agent
inherits it. SDK: `ctx.model_key.set/status/revoke` at any level the caller is authorised for.

### D5 — HITL and continuous learning
`POST /v1/feedback` → `human_feedback` table: `target_kind ∈ {memory, answer, brief, procedure}, target_id, verdict ∈
{confirm, reject, correct}, correction, reviewer, evidence`. Stored separately, never merged into content, listable and
auditable. An outbox projector applies it through the existing machinery: `correct` → new revision superseding the
old with `EvidenceRef(source_type="feedback")`; `reject` → invalidation edge; `confirm` → reinforcement and a
confidence floor. Learning without HITL is what already runs: observation extraction, supersession, tool-procedure
mining, brief refresh, bounded reflection. SDK: `ctx.feedback.confirm/reject/correct/list`.

### D6 — Accuracy programme (multi-hop), in this order, each gated
1. Instrument: per-item `[record_id, kind, retrievers, fused_rank]` and gold-turn ranks per arm (zero runtime cost).
2. Weighted RRF (`Rrf(weights=…)`, server ≥ 1.17) fitted offline from those dumps and spent on **halving depth**.
   Gate: `evidence_all_recall ≥ 0.99` at the lower depth; p99 down.
3. Entity → memory routing as an RRF list (needs `entities` in the payload). Gate: strict multi-hop.
4. Dated per-(subject, predicate) aggregates from `BeliefService`, rendered as one line. Gate: strict multi-hop ≥ +3,
   `false_merge == 0.0`. Targets the 226-question fragmentation bucket.
5. Rank-anchored rendering block, weekday dates, resolved relative dates. Reader-side, zero query cost.
6. Reader protocol for the 143 abstain-with-evidence questions — needs the Gemini path (see §6).
7. Whole-session extraction with turn citations (extends `narrative.py`), async, gated on +3 strict and unchanged p50.
8. Paraphrase dedup against top-20 vector neighbours — precondition for 7.
Rule kept: no delta is accepted unless the metric is shown sensitive to the exact change; fixed ladder 10/20/30/50;
host load recorded; Mem0's ruler reported beside the strict one.

### D7 — Parity without a second engine
Hindsight stays an optional extra (`memory-service[hindsight]`), not a core dependency: its SDK cannot carry a team key
and its server is not deployed here. Parity is native: briefs (mental models/pages), feedback (fact edit/history),
`POST /v1/webhooks` (outbox → signed per-tenant HTTP delivery), cursor pagination and filters on `/v1/memories`,
and an opt-in deep mode (`use_llm=true` on `/context` with the team key) later.

### D8 — Latency
Fast path p99 < 300 ms in-process on 8 vCPU; both encoders concurrent; semantic context cache; weighted RRF at
reduced depth; the serial model executor yields between indexing batches. Load gate: 20 RPS with ingestion running.

### D9 — Quality gates
`tests/unit/test_architecture.py` and `test_complexity_budget.py` are not raised. Dead paths go: English DeBERTa
adapter, Chinese-origin challengers, `hindsight-client` as core, the measured-harmful `ambiguous_*` LLM uses,
and any experimental flag not promoted by its arm. A provider value without a contract test does not ship.

## 4. Milestones and gates

| M | Deliverable | Gate |
|---|---|---|
| **M0** | Integration branch; `uv sync`; suite green on isolated DBs; worker-kill test rerun sequentially; model-provenance purge + test; `.env`/Makefile point at `gemini/gemini-3.8-flash` | offline suite green; provenance test green |
| **M1** | Tenancy: `api_key` mode, bootstrap admin, tenants/keys/workspaces/members, `WORKSPACE` audience, revocation, retention, read audit, quotas; SDK; `tests/agent/` boundary suite via SDK only | zero leaks in the oracle + matrix suites incl. workspace and revocation; read-after-revoke and read-after-delete denied |
| **M2** | Multilingual runtime: named vectors + concurrent encoders + script-aware prefetch; mDeBERTa freeze; script tag; `dateparser`; 12-language SDK suite; reindex tool | container gates: SciFact ≥ 0.7557/0.8926; XQuAD mean R@10 ≥ 0.98; LoCoMo source arms (Granite, ensemble); p99 < 300 ms |
| **M3** | Team credentials (3 levels); LLM-assisted multilingual extraction; HITL feedback + projector; webhooks; pagination/filters | credential precedence and revocation tests; feedback projection visible on next `/context`; webhook delivery signed and retried |
| **M4** | Accuracy steps 1–6; judged arms on Gemini 3.8-flash; the published table (strict, Mem0 ruler, category 5 separate, judge named) | each step's own gate; budget approved before any 2,000-call arm |
| **M5** | Deep mode; workspace templates; harness longitudinal eval (10/50/100 runs); 20 RPS load gate | separate deadline and cost report for deep mode |

## 5. Measurement protocol (unchanged)
Per-arm database, Qdrant namespace and cache; corpus, model and dataset hashes in every artifact; host load recorded;
failed calls are unmeasured, never wrong; no golden set is edited to pass; no production-readiness claim while any
gate fails.

## 6. Decisions the owner must make
1. **Budget for judged runs.** Gemini 3.8-flash works. A judged LoCoMo arm is ~2,000 paced calls; a 20-question smoke
   will report the cost per arm before any full run.
2. **Base branch.** D1 merges the checkpoint, including migrations 0009–0013. Ingestion that changes the corpus stays off
   by default; say so if you want the platform work on `main` without it.
3. **Vocabulary.** "Team" is `workspace` in the API and OpenFGA model; it is not renamed.
