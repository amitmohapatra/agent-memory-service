.DEFAULT_GOAL := help
SHELL := /bin/bash
UV ?= uv
PY ?= $(UV) run
COMPOSE ?= docker compose

.PHONY: help setup dev-up dev-down migrate lint format typecheck unit integration contract-test e2e security-test \
        performance-test failure-test eval bench-retrieval bench-memory bench-embedding bench-reranker bench-storage \
        load-test validate openapi clean

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

performance-test: ## Latency budgets
	$(PY) pytest tests/performance -q

failure-test: ## Failure injection
	$(PY) pytest tests/failure -q 2>/dev/null || $(PY) pytest tests/integration -m failure -q

eval: ## Deterministic retrieval/memory evaluation gates
	$(PY) pytest tests/evals -m "not models" -q

bench-retrieval: ## Retrieval benchmark
	$(PY) python -m benchmark.retrieval

bench-memory: ## Memory intelligence benchmark
	$(PY) python -m benchmark.memory

bench-embedding: ## Embedding runtime benchmark
	$(PY) python -m benchmark.embedding

bench-reranker: ## Reranker benchmark
	$(PY) python -m benchmark.reranker

bench-storage: ## Archive segment / storage benchmark
	$(PY) python -m benchmark.storage

load-test: ## Locust load test (headless)
	$(PY) locust -f benchmark/load/locustfile.py --headless -u 20 -r 5 -t 60s --host http://localhost:8080

openapi: ## Export OpenAPI schema
	$(PY) python -m memory_service.tools.export_openapi docs/openapi.json

validate: ## Full release smoke gate
	$(MAKE) lint
	$(MAKE) typecheck
	$(MAKE) unit
	$(MAKE) contract-test
	$(MAKE) integration
	$(MAKE) security-test
	$(MAKE) e2e
	$(MAKE) eval
	$(PY) python -m memory_service.tools.release_gate

clean:
	rm -rf .venv .pytest_cache .ruff_cache .mypy_cache dist build .blob
