# Memory Service

Enterprise Multi-Agent Memory Service: durable, scope-aware, context-preserving memory for
chat applications, plain Python agents, multi-agent systems and RAG — without coupling the
core to any agent framework.

```python
from universal_memory import MemoryClient

memory = MemoryClient("http://memory-service:8080", api_key="dev-key")

ctx = memory.bind(
    tenant_id=request.tenant_id,
    workspace_id=request.workspace_id,
    user_id=request.user_id,
    group_ids=request.group_ids,
    thread_id=request.thread_id,
    session_id=request.session_id,
    turn_id=request.turn_id,
)

await ctx.chat.user(request.message, attachments=request.files)
bundle = await ctx.context(request.message)
answer = await my_agent.run(bundle.rendered)
await ctx.chat.assistant(answer)
```

That is the whole developer surface for the 90% path. Qdrant, BM25, embeddings, rerankers,
RRF, GCS compaction, graph enrichment, dedup, memory types, TTLs and task queues are
configuration, not API.

## What it guarantees (and what it does not)

| Guarantee | Definition | Enforced by |
|---|---|---|
| Acknowledged data loss = 0 | A `2xx/202` is returned only after the source record **and** its processing job are committed to PostgreSQL. | Unit of Work + transactional job enqueue; failure-injection tests |
| Unauthorized retrieval = 0 | No cross-tenant, cross-user or private-agent data reaches any model. | OpenFGA decisions turned into store-side filters **before** search; security test suite |
| Critical Recall@K = 1.00, Evidence-Group Recall = 1.00 | Every business-critical golden case retrieves all required evidence groups. | Retrieval eval gate (`make eval`), release gate |
| Honest evidence | If evidence is insufficient after escalation, the response says `INSUFFICIENT_EVIDENCE`. | Evidence verifier + abstention |

It does **not** claim universal 100% semantic retrieval accuracy for arbitrary questions.
All performance numbers are benchmark targets recorded with provenance, not promises.

## Stack

PostgreSQL (canonical state) · Qdrant (BM25 + dense + RRF) · Dragonfly (cache/working memory) ·
OpenFGA (authorization) · Procrastinate (PostgreSQL task queue) · GCS (raw archive) ·
Docling (parsing) · Granite Embedding R2 + MiniLM reranker on CPU · OpenTelemetry +
OpenLineage · DeepEval. LLMs are **off by default**.

## Quick start

```bash
git clone <repo> memory-service && cd memory-service
uv venv .venv --python 3.12 && source .venv/bin/activate
uv sync --dev                # core + dev tooling
uv sync --all-extras --dev   # + docling, local models, providers, eval (large)

cp .env.example .env
docker compose up -d         # postgres, qdrant, dragonfly, openfga, api, worker
make migrate
uv run memory-api            # http://localhost:8080/docs
```

Standard `venv` fallback:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[all]" && pip install -e sdk/python -e integrations/langgraph
```

Local embedding/reranker models are read from `models/` (see `.env.example`). Download
`ibm-granite/granite-embedding-small-english-r2` and `cross-encoder/ms-marco-MiniLM-L6-v2`
into that directory, or point `MEMORY__MODELS__*__MODEL_PATH` elsewhere.

## Developer commands

```
make setup | dev-up | dev-down | migrate
make lint | format | typecheck
make unit | integration | contract-test | e2e | security-test | performance-test
make eval | bench-retrieval | bench-memory | bench-embedding | bench-reranker | bench-storage | load-test
make validate            # full release smoke gate
```

`make validate` runs every suite and then `memory_service.tools.release_gate`, which fails the
build if any hard gate lacks evidence or is violated.

## API

Swagger UI at `/docs`, ReDoc at `/redoc`, schema at `/openapi.json`, health at
`/health/live` and `/health/ready`, metrics at `/metrics`, build info at `/version`.
Public routes live under `/v1` (see `docs/openapi.json`).

## Repository layout

See [ARCHITECTURE.md](ARCHITECTURE.md). Decisions are recorded in `docs/adr/`.

## Status

Milestones M0–M13 are implemented sequentially with hard quality gates; see
`docs/MILESTONES.md` for what is complete and what each gate measured.
