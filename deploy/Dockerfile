# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
RUN apt-get update && apt-get install -y --no-install-recommends libpq5 curl && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY sdk/python/pyproject.toml sdk/python/README.md sdk/python/
COPY integrations/langgraph/pyproject.toml integrations/langgraph/README.md integrations/langgraph/
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-workspace --extra gcp

COPY src ./src
COPY sdk ./sdk
COPY integrations ./integrations
COPY migrations ./migrations
COPY alembic.ini ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --extra gcp

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8080
USER nobody
CMD ["memory-api"]
