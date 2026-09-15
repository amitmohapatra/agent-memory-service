# syntax=docker/dockerfile:1.7
#
# Targets:
#   runtime     (default) production image: core + gcp + models + docling extras, no dev tools
#   validation  runtime + dev tooling + every extra; the repository is bind-mounted at /app so
#               the release gates run against the working tree with real weights and services
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv
RUN apt-get update && apt-get install -y --no-install-recommends libpq5 curl && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
# workspace member metadata (sources are copied later); keeps the dependency layer cacheable
COPY sdk/python/pyproject.toml sdk/python/README.md sdk/python/
COPY integrations/langgraph/pyproject.toml integrations/langgraph/README.md integrations/langgraph/
COPY integrations/crewai/pyproject.toml integrations/crewai/README.md integrations/crewai/
COPY integrations/mcp/pyproject.toml integrations/mcp/README.md integrations/mcp/

FROM base AS runtime-deps
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-workspace --extra gcp --extra models --extra docling

FROM runtime-deps AS runtime
COPY src ./src
COPY sdk ./sdk
COPY integrations ./integrations
COPY migrations ./migrations
COPY alembic.ini ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra gcp --extra models --extra docling
ENV PATH="/opt/venv/bin:$PATH"
EXPOSE 8080
USER nobody
CMD ["memory-api"]

FROM base AS validation
RUN apt-get update && apt-get install -y --no-install-recommends make git procps && rm -rf /var/lib/apt/lists/*
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --all-extras --dev --no-install-workspace
ENV PATH="/opt/venv/bin:$PATH" HF_HOME=/models/hf-cache MEMORY_MODELS_DIR=/models
# the repository is mounted at /app at run time; workspace members are installed then
CMD ["sleep", "infinity"]
