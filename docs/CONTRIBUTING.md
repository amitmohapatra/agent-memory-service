# Contributing

## Workflow per change

The milestone plan (M0 to M13) is finished; a change now follows this loop.

1. **Read first.** The code you are changing, its tests, the ADR that decided it
   ([index](adr/README.md)) and the page that documents it. Check a third-party assumption
   against the library's own documentation, and record the version.
2. **Decide in an ADR** when the change alters an architectural default, the API surface or a
   guarantee. Amend an older ADR's header rather than rewriting its text.
3. **Make the smallest complete vertical slice**: the code, its unit test, and the
   integration, contract, end-to-end and security or failure tests the change touches.
4. **Measure before claiming.** A retrieval or latency change runs its benchmark and saves the
   output under `benchmark/results/` with provenance; a dated write-up goes to
   `benchmark/reports/`, never into the user docs.
5. **Update what describes it, in the same change**: `make openapi` for the HTTP contract, the
   page in `docs/` that explains it, the [configuration reference](configuration.md) for a
   setting, an example in `examples/` when the change is something a reader would copy, and
   the [CHANGELOG](../CHANGELOG.md).
6. **Run the gates**: `make lint typecheck`, the suites (`make unit integration e2e`, or the
   CI list in [chapter 12](guide/12-testing-and-gates.md#what-ci-runs)), `make examples` and
   `make docs-check`. Fix every failure; never weaken a gate to make a test pass, and do not
   merge when a release gate fails.

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
- Test settings are hermetic by default (hash embedding, in-process
  Qdrant/cache/authorization, builtin parser). `MEMORY_TEST_PROVIDERS=env` makes every
  fixture take the *models / search / cache / authorization / documents / retrieval*
  sections from the environment instead, so the same suites run against real weights and
  servers; the fixtures then reset the Qdrant collections and the cache between tests.
- Commit messages: a first line that says what changed, then why in the body.
- **Migrations are online** ([deploy/database.md](deploy/database.md#online-migrations),
  ADR 0031). `migrations/env.py` sets `lock_timeout` (5 s) and `statement_timeout` (15 min)
  for every migration, so a revision that cannot get its lock fails instead of queueing the
  live service behind it. An index on an existing table is `CREATE INDEX CONCURRENTLY IF
  NOT EXISTS` inside `op.get_context().autocommit_block()` (migration 0024 is the pattern).
  Schema changes go expand-then-contract across releases. Test `upgrade head`,
  `downgrade -1`, `upgrade head` on a scratch database.

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
`secrets.env` (or the `BIFROST_VIRTUAL_KEY` environment variable); provider keys live only in Bifrost.

## Pre-commit

```bash
uv run pre-commit install
```
