# syntax=docker/dockerfile:1.7
#
# Targets:
#   runtime     (default) production image: core + the extras named by the EXTRAS build arg
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

# Which optional extras go into the image. The default is what a deployment needs to run
# with real embedding/rerank/NLI models. `docling` (rich document parsing) is opt-in: it
# brings opencv, rapidocr, pandas and litellm — roughly 300 MB — and the service falls back
# to its builtin parser without it.
#
#   docker compose build --build-arg EXTRAS="gcp models docling"
ARG EXTRAS="gcp models"

FROM base AS runtime-deps
ARG EXTRAS
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-workspace $(for e in $EXTRAS; do echo --extra $e; done)

FROM runtime-deps AS runtime
ARG EXTRAS
COPY src ./src
COPY sdk ./sdk
COPY integrations ./integrations
COPY migrations ./migrations
COPY alembic.ini ./
# The OpenFGA authorization model. A deployment writes it to OpenFGA the first time it finds
# no model there, so leaving it out of the image works right up until the first fresh
# install — which then fails on a missing file, reported as "OpenFGA unavailable".
COPY deploy/openfga ./deploy/openfga
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev $(for e in $EXTRAS; do echo --extra $e; done)
# COPY preserves the builder's permissions: a source tree checked out with a restrictive
# umask (0700 directories) produces an image whose own package is unreadable to the
# unprivileged runtime user, and the container exits with ModuleNotFoundError. The image
# must not depend on how the build host happened to be set up. Only the copied sources need
# this — the virtualenv is created inside the image with normal permissions.
RUN chmod -R a+rX /app/src /app/sdk /app/integrations /app/migrations /app/alembic.ini /app/deploy
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
