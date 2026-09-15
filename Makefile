.DEFAULT_GOAL := help
SHELL := /bin/bash
UV ?= uv
PY ?= $(UV) run
COMPOSE ?= docker compose

.PHONY: help setup dev-up dev-down migrate lint format typecheck unit integration contract-test e2e security-test \
        performance-test failure-test eval bench-retrieval bench-advanced bench-memory bench-embedding bench-reranker bench-storage \
        load-test gates gates-network validate openapi reindex examples clean

help: ## Show targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

setup: ## Create .venv with uv and install all extras + dev tools
	$(UV) venv .venv --python 3.12
	$(UV) sync --all-extras --dev
	$(PY) pre-commit install || true

setup-min: ## Minimal install (core + dev, no models/docling/cognee)
	$(UV) venv .venv --python 3.12
	$(UV) sync --dev

dev-up: ## Start dev stack (postgres, qdrant, dragonfly, openfga, api, worker)
	$(COMPOSE) up -d --build

dev-down: ## Stop dev stack
	$(COMPOSE) down -v

migrate: ## Apply database migrations
	$(PY) alembic upgrade head
	$(PY) python -m memory_service.adapters.tasks.procrastinate_schema || true

lint: ## Ruff lint
	$(PY) ruff check .

format: ## Ruff format + import sort
	$(PY) ruff format .
	$(PY) ruff check --fix .

typecheck: ## Pyright
	$(PY) pyright

unit: ## Unit tests (no external services)
	$(PY) pytest tests/unit sdk/python/tests integrations/langgraph/tests -m "not docker and not models" -q

integration: ## Integration tests (needs Postgres/Redis; uses embedded fallbacks where possible)
	$(PY) pytest tests/integration -m "not docker and not models" -q

contract-test: ## OpenAPI + provider contract tests
	$(PY) pytest tests/contract -q

e2e: ## End-to-end API flows
	$(PY) pytest tests/e2e -m "not docker and not models" -q

security-test: ## Isolation / authorization gates (release blocking)
	$(PY) pytest tests/security -q

performance-test: ## Latency budgets (writes benchmark/results/performance.json)
	$(PY) python -m benchmark.performance

eval: ## Deterministic retrieval/memory evaluation gates
	$(PY) pytest tests/eval -m "not models" -q

bench-retrieval: ## Retrieval benchmark
	$(PY) python -m benchmark.retrieval

bench-advanced: ## Advanced retrieval strategies vs baseline (adoption verdicts)
	$(PY) python -m benchmark.advanced

bench-memory: ## Memory intelligence benchmark
	$(PY) python -m benchmark.memory

bench-embedding: ## Embedding runtime benchmark
	$(PY) python -m benchmark.embedding

bench-reranker: ## Reranker benchmark
	$(PY) python -m benchmark.reranker

bench-storage: ## Archive segment / storage benchmark
	$(PY) python -m benchmark.storage

BASE_URL ?= http://localhost:8080
API_KEY ?= dev-key

load-test: ## Locust load test (headless) -> benchmark/results/load_test.json (BASE_URL, API_KEY)
	$(PY) python -m benchmark.load.run --base-url $(BASE_URL) --api-key $(API_KEY) -u 20 -r 5 -t 60s

gates-network: ## Network-hop gates against a deployed api + real workers (BASE_URL, API_KEY): durability_network, performance_network, load_test
	$(PY) python -m benchmark.deployed --base-url $(BASE_URL) --api-key $(API_KEY)
	$(PY) python -m benchmark.load.run --base-url $(BASE_URL) --api-key $(API_KEY) -u 20 -r 5 -t 60s

openapi: ## Export OpenAPI schema
	$(PY) python -m memory_service.tools.export_openapi docs/openapi.json

failure-test: ## Failure-injection scenarios (worker kill, cache flush, blob outage, index rebuild, authz outage)
	$(PY) pytest tests/failure -m failure -q

gates: ## Produce every release-gate artifact under benchmark/results/
	$(PY) pytest tests sdk/python/tests integrations/langgraph/tests -m "not docker and not models" -q -p benchmark.pytest_results
	$(PY) python -m benchmark.security
	$(PY) python -m benchmark.failure_injection
	$(PY) python -m benchmark.durability
	$(PY) python -m benchmark.performance
	$(PY) python -m benchmark.retrieval
	$(PY) python -m benchmark.memory

examples: ## Run the SDK tour and the LangGraph crew against a running server (./examples/run_server.sh)
	$(PY) python examples/sdk_tour.py
	$(PY) python examples/langgraph_crew/app.py

reindex: ## Rebuild the search index from PostgreSQL (add --drop for a full rebuild)
	$(PY) python -m memory_service.tools.reindex

validate: ## Full release gate: lint, types, every suite, gate artifacts, then the gate evaluator
	$(MAKE) lint
	$(MAKE) typecheck
	$(MAKE) gates
	$(PY) python -m memory_service.tools.release_gate

clean:
	rm -rf .venv .pytest_cache .ruff_cache .mypy_cache dist build .blob
