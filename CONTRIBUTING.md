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
- Provider SDKs only inside `src/memory_service/adapters/`.
- Tests are marked: `unit`, `integration`, `contract`, `e2e`, `security`, `performance`,
  `failure`, `eval`; plus `docker` (needs a Docker daemon with registry access) and `models`
  (needs local model files). CI runs everything that is not `docker`/`models`; nightly runs
  the rest.
- Commit messages: `M<n>: <what>`.

## Pre-commit

```bash
uv run pre-commit install
```
