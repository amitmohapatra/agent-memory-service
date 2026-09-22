"""OpenAPI customization: authentication, standard headers, error schema, tags, versioning."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from memory_service.api.errors import ErrorEnvelope
from memory_service.api.middleware import (
    HEADER_CORRELATION_ID,
    HEADER_IDEMPOTENCY_KEY,
    HEADER_REQUEST_ID,
    HEADER_TRACE_ID,
)

DESCRIPTION = """
Enterprise Multi-Agent Memory Service.

**Durable**: a `2xx/202` is returned only after the source record and its processing job
have committed to PostgreSQL. **Scoped**: every retrieval is authorized and filtered by
tenant / workspace / user / group / thread / agent *before* any model sees data.
**Honest**: when evidence is insufficient the service says so (`INSUFFICIENT_EVIDENCE`)
rather than pretending retrieval succeeded.

### Authentication
The calling *service* authenticates with `trusted_dev` (API key, dev only) or `jwt`
(JWKS). The end-user identity and scope travel in trusted context headers
(`X-Memory-Tenant`, `X-Memory-Workspace`, `X-Memory-User`, `X-Memory-Groups`) which are
only honored from an authenticated caller. Fine-grained authorization is evaluated by
OpenFGA on every request.

### Standard headers
| Header | Direction | Purpose |
|---|---|---|
| `Idempotency-Key` | request | Makes persistent writes safe to retry (24h window). |
| `X-Request-ID` | both | Per-request id; generated when absent. |
| `X-Correlation-ID` | both | Groups related requests (e.g. one user turn). |
| `X-Trace-ID` | both | OpenTelemetry trace correlation. |

### Errors
All errors use one envelope (`ErrorEnvelope`). `retryable=true` means the same request
may succeed later.

### Versioning
Public routes live under `/v1`. Breaking changes ship under a new prefix; `/v1` remains
available for at least one release after deprecation.
"""

TAGS: list[dict[str, Any]] = [
    {"name": "operations", "description": "Health, readiness, metrics and version."},
    {"name": "threads", "description": "Conversation threads, sessions and turns."},
    {
        "name": "messages",
        "description": "Visible and internal messages with durable acknowledgement.",
    },
    {
        "name": "observations",
        "description": "'This happened.' The service decides what to remember.",
    },
    {"name": "files", "description": "File ingestion into RAG memory with structural context."},
    {"name": "retrieval", "description": "Scope-filtered recall and ContextBundle assembly."},
    {"name": "memories", "description": "Canonical memory access and deletion."},
    {"name": "jobs", "description": "Background job status."},
    {"name": "admin", "description": "Administrative and benchmark routes (separate auth)."},
]


def _header_param(name: str, description: str, required: bool = False) -> dict[str, Any]:
    return {
        "name": name,
        "in": "header",
        "required": required,
        "description": description,
        "schema": {"type": "string"},
    }


def custom_openapi(app: FastAPI, *, version: str) -> dict[str, Any]:
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title="Memory Service API",
        version=version,
        description=DESCRIPTION,
        routes=app.routes,
        tags=TAGS,
        servers=[{"url": "/", "description": "current host"}],
    )
    components = schema.setdefault("components", {})
    components.setdefault("schemas", {})["ErrorEnvelope"] = ErrorEnvelope.model_json_schema(
        ref_template="#/components/schemas/{model}"
    )
    components["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "trusted_dev mode only",
        },
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "jwt mode",
        },
    }
    schema["security"] = [{"ApiKeyAuth": []}, {"BearerAuth": []}]

    common = [
        _header_param(HEADER_REQUEST_ID, "Client request id (generated when absent)."),
        _header_param(HEADER_CORRELATION_ID, "Correlation id shared by related requests."),
        _header_param(HEADER_TRACE_ID, "Trace id for OpenTelemetry correlation."),
    ]
    idem = _header_param(
        HEADER_IDEMPOTENCY_KEY, "Idempotency key for safe retries of persistent writes."
    )
    for path, methods in schema.get("paths", {}).items():
        for method, op in methods.items():
            if method not in ("get", "post", "put", "patch", "delete"):
                continue
            if path.startswith(("/health", "/metrics", "/version")):
                continue
            params = op.setdefault("parameters", [])
            existing = {p.get("name") for p in params}
            for p in common:
                if p["name"] not in existing:
                    params.append(dict(p))
            if method in ("post", "put", "patch", "delete") and idem["name"] not in existing:
                params.append(dict(idem))
    app.openapi_schema = schema
    return schema
