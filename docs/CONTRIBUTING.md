# Contributing

## Workflow per milestone

A. Inspect the repository and existing implementation.
B. Read the specification section for the milestone.
C. Validate third-party assumptions against official docs (record versions).
D. Write an ADR in `docs/adr/` when changing an architectural default.
E. List files/components to add or change.
F. Implement the smallest complete vertical slice.
G–K. Add unit, integration, contract, E2E and security/failure tests.
L–M. Run everything; fix every failure. Never weaken a gate to make a test pass.
N–O. Run the milestone benchmark/eval; save outputs under `benchmark/results/` with provenance.
P. Update OpenAPI (`make openapi`) and docs.
Q. `make lint typecheck`.
R. Do not proceed if a release gate fails.

## Conventions

- Python 3.12+, `uv`, `src` layout, Ruff (format + lint), Pyright (standard).
- Typed Pydantic models at every public/application boundary; `dict[str, Any]` only for
  custom metadata.
- Provider SDKs only inside `src/memory_service/adapters/`. LLM provider SDKs (`openai`,
  `anthropic`, `litellm`, …) are banned everywhere: the only generative path is the Bifrost
  gateway adapter (`adapters/models/llm.py`), enforced by Ruff `banned-api` and
  `tests/unit/test_architecture.py`.
- Tests are marked: `unit`, `integration`, `contract`, `e2e`, `security`, `performance`,
  `failure`, `eval`; plus `docker` (needs a Docker daemon with registry access), `models`
  (needs local model files / Docling) and `bifrost` (needs a running gateway). CI runs
  everything that is not `docker`/`models`/`bifrost`; the release gates run the rest.
- Test settings are hermetic by default (hash embedding, lexical reranker, in-process
  Qdrant/cache/authorization, builtin parser). `MEMORY_TEST_PROVIDERS=env` makes every
  fixture take the *models / search / cache / authorization / documents / retrieval*
  sections from the environment instead, so the same suites run against real weights and
  servers; the fixtures then reset the Qdrant collections and the cache between tests.
- Commit messages: `M<n>: <what>`; no tool or AI attribution trailers.

## Real-component validation

Everything that needs model weights or Docling runs in the `memory-validate` compose
service (a Linux image with every extra; the repository is bind-mounted at `/app`, weights
at `/models`):

```bash
docker compose --profile validation up -d memory-validate
docker compose exec memory-validate uv sync --frozen --all-extras --dev   # workspace members
docker compose exec -e MEMORY_TEST_PROVIDERS=env memory-validate make validate
```

Secrets never enter the repository: the Bifrost virtual key lives in the git-ignored
`secrets.env` (or `MEMORY__MODELS__LLM__API_KEY`); provider keys live only in Bifrost.

## Pre-commit

```bash
uv run pre-commit install
```
