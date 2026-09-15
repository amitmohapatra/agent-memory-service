#!/usr/bin/env bash
# Start a single-process Memory Service for the examples (API + inline jobs).
#
# Needs PostgreSQL (migrated) and a Redis/Dragonfly. Without a Qdrant server the search
# index runs in-process (qdrant-client local mode), which is why jobs run *inline* here:
# a separate worker process would index into its own local Qdrant. With the compose stack
# (`make dev-up`) use MEMORY__TASKS__PROVIDER=procrastinate and `uv run memory-worker`.
#
#   ./examples/run_server.sh            # http://localhost:8080/docs, API key "dev-key"
#   MEMORY__SERVICE__PORT=9000 ./examples/run_server.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export MEMORY__SERVICE__ENVIRONMENT="${MEMORY__SERVICE__ENVIRONMENT:-dev}"
export MEMORY__SERVICE__LOG_JSON="${MEMORY__SERVICE__LOG_JSON:-false}"
export MEMORY__SERVICE__LOG_LEVEL="${MEMORY__SERVICE__LOG_LEVEL:-WARNING}"
export MEMORY__AUTHENTICATION__MODE="${MEMORY__AUTHENTICATION__MODE:-trusted_dev}"
export MEMORY__AUTHENTICATION__TRUSTED_DEV_API_KEYS="${MEMORY__AUTHENTICATION__TRUSTED_DEV_API_KEYS:-[\"dev-key\"]}"
export MEMORY__AUTHORIZATION__PROVIDER="${MEMORY__AUTHORIZATION__PROVIDER:-memory}"
export MEMORY__DATABASE__URL="${MEMORY__DATABASE__URL:-postgresql+psycopg://memory:memory@localhost:5432/memory}"
export MEMORY__CACHE__PROVIDER="${MEMORY__CACHE__PROVIDER:-redis}"
export MEMORY__CACHE__URL="${MEMORY__CACHE__URL:-redis://localhost:6379/0}"
export MEMORY__SEARCH__PROVIDER="${MEMORY__SEARCH__PROVIDER:-memory}"
export MEMORY__BLOB__PROVIDER="${MEMORY__BLOB__PROVIDER:-filesystem}"
export MEMORY__BLOB__FILESYSTEM_ROOT="${MEMORY__BLOB__FILESYSTEM_ROOT:-./.blob}"
export MEMORY__TASKS__PROVIDER="${MEMORY__TASKS__PROVIDER:-inline}"
export MEMORY__MODELS__EMBEDDING__PROVIDER="${MEMORY__MODELS__EMBEDDING__PROVIDER:-hash}"
export MEMORY__MODELS__EMBEDDING__DIMENSION="${MEMORY__MODELS__EMBEDDING__DIMENSION:-64}"
export MEMORY__MODELS__RERANKER__PROVIDER="${MEMORY__MODELS__RERANKER__PROVIDER:-lexical}"
export MEMORY__MODELS__LLM__ENABLED="${MEMORY__MODELS__LLM__ENABLED:-false}"
export MEMORY__DOCUMENTS__PARSER="${MEMORY__DOCUMENTS__PARSER:-builtin}"
export MEMORY__OBSERVABILITY__OTEL_ENABLED="${MEMORY__OBSERVABILITY__OTEL_ENABLED:-false}"
uv run alembic upgrade head >/dev/null
exec uv run memory-api
