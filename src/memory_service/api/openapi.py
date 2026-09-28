"""OpenAPI customization: authentication, standard headers, problem schema, tags, ids."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.routing import APIRoute

from memory_service.api.errors import Problem, error_responses
from memory_service.api.headers import (
    CORRELATION_ID_HEADER,
    DEPRECATION_HEADER,
    IDEMPOTENCY_KEY_HEADER,
    IDEMPOTENT_REPLAYED_HEADER,
    LINK_HEADER,
    RATE_LIMIT_LIMIT_HEADER,
    RATE_LIMIT_REMAINING_HEADER,
    REQUEST_ID_HEADER,
    RETRY_AFTER_HEADER,
    TRACE_ID_HEADER,
    deprecation_headers_for,
)
from memory_service.config.constants import (
    ALIASES_REMOVED_IN,
    DEPRECATED_HEADER_ALIASES,
    DEPRECATED_RESPONSE_HEADER_ALIASES,
    HEADERS,
)
from memory_service.observability.tracing import TRACEPARENT_HEADER

TITLE = "trellis-memory API"

DESCRIPTION = f"""
trellis-memory: durable, scope-aware, context-preserving memory for chat, agents and RAG.

**Durable**: a `2xx/202` is returned only after the source record and its processing job
have committed to PostgreSQL. **Scoped**: every retrieval is authorized and filtered by
tenant / workspace / user / group / thread / agent *before* any model sees data.
**Honest**: when evidence is insufficient the service says so (`INSUFFICIENT_EVIDENCE`)
rather than pretending retrieval succeeded.

### Authentication
The calling *service* authenticates with `api_key` (keys the service issues:
`mk_<key_id>.<secret>` in `X-API-Key` or as a Bearer token, each naming its tenant and
optionally a workspace; one bootstrap secret onboards tenants), `jwt` (JWKS) or
`trusted_dev` (static key, laptops only). The end-user identity and scope travel in
trusted context headers (`X-Trellis-Tenant`, `X-Trellis-Workspace`, `X-Trellis-User`) which
are only honored from an authenticated caller; in `api_key` mode the tenant comes from the
key and a header may agree with it, never contradict it. Fine-grained authorization is
evaluated by OpenFGA on every request. The pre-0.2 spellings `X-Memory-Tenant`,
`X-Memory-Workspace` and `X-Memory-User` are still read; they are removed in {ALIASES_REMOVED_IN}.

### Standard headers
| Header | Direction | Purpose |
|---|---|---|
| `Idempotency-Key` | request | Makes persistent writes safe to retry (24h window). |
| `traceparent` | both | W3C Trace Context: continued when sent; the response names the trace. |
| `X-Request-ID` | both | Per-request id; generated when absent or not an id. |
| `X-Correlation-ID` | both | Groups related requests; echoed when it is an id, else replaced. |
| `X-Trace-ID` | response | The 32-hex trace id the request ran under (as in `traceparent`). |
| `X-Trellis-LLM-Tokens` | response | LLM tokens the request spent, when it spent any. |

An id starts with a letter or digit, continues with letters, digits and `._:-`, and is at
most 200 characters. A request carrying a scope header under both spellings with different
values, or any scope or credential header more than once with different values, is refused
(422) before the credential is read: a gateway that stamps one spelling must strip both.

### Errors
Every error is an RFC 9457 problem (`application/problem+json`, schema `Problem`):
`type` is `urn:trellis:problem:<code in kebab case>`, `code` the stable category,
`retryable=true` means the same request may succeed later, and `trace_id` / `request_id`
are what to quote.

### Versioning
Public routes live under `/v1`. 0.2.0 (ADR 0022) changed `/v1` in place once, because its
only consumers were the owner's own repositories: problem details replaced the error
envelope, operation ids became `<tag>.<function>`, `X-Trace-ID` became a response header.
From 0.2.0 on, breaking changes ship under a new prefix and `/v1` remains available for at
least one release after deprecation. Routes marked deprecated are aliases kept for one
release and removed in {ALIASES_REMOVED_IN}.
"""

TAGS: list[dict[str, Any]] = [
    {"name": "operations", "description": "Health, readiness, metrics and version."},
    {"name": "threads", "description": "Conversation threads, sessions and turns."},
    {
        "name": "messages",
        "description": "Visible and internal messages with durable acknowledgement.",
    },
    {
        "name": "memory",
        "description": "Observations ('this happened'; the service decides what to remember) "
        "and canonical memory access and deletion.",
    },
    {
        "name": "documents",
        "description": "Document ingestion into RAG memory with structural context, and "
        "document status.",
    },
    {
        "name": "files",
        "description": (
            f"Deprecated alias of `documents` (ADR 0022); removed in {ALIASES_REMOVED_IN}."
        ),
    },
    {
        "name": "retrieval",
        "description": "Scope-filtered recall, ContextBundle assembly, and "
        "grounding verification of an answer against evidence.",
    },
    {"name": "graph", "description": "Entity and relation queries over the knowledge graph."},
    {"name": "briefs", "description": "Standing questions and pages kept current by the service."},
    {
        "name": "tools",
        "description": "Tool memory: declared tools, invocation records, the output cache, "
        "chains and run outcomes.",
    },
    {
        "name": "agents",
        "description": "An agent's own model credential (a Bifrost virtual key): status, "
        "set and revoke.",
    },
    {"name": "jobs", "description": "Background job status."},
    {
        "name": "admin",
        "description": "Platform operator: onboarding tenants with the bootstrap key.",
    },
    {
        "name": "tenancy",
        "description": "Tenant administration: keys, workspaces (teams), groups, read audit.",
    },
]


def operation_id(route: APIRoute) -> str:
    """``<tag>.<function>``: stable across path edits and readable in generated clients.

    Every public route carries a tag (``tests/contract/test_openapi_conventions.py`` holds
    them to it); a route added without one, as tests do, keeps its function name.
    """
    return f"{route.tags[0]}.{route.name}" if route.tags else route.name


def _header_param(
    name: str, description: str, *, required: bool = False, deprecated: bool = False
) -> dict[str, Any]:
    param: dict[str, Any] = {
        "name": name,
        "in": "header",
        "required": required,
        "description": description,
        "schema": {"type": "string"},
    }
    if deprecated:
        param["deprecated"] = True
    return param


#: Response headers every public operation carries (the correlation middleware sets them).
RESPONSE_HEADERS: dict[str, str] = {
    REQUEST_ID_HEADER: "The request id: the caller's when it was a valid id, else a generated one.",
    TRACE_ID_HEADER: "The 32-hex trace id the request ran under.",
    TRACEPARENT_HEADER: "W3C Trace Context naming the trace the request ran under.",
    CORRELATION_ID_HEADER: "The correlation id: the body scope's when it named one (an "
    "invalid one is refused), else the header's when it is an id, else a generated one.",
    HEADERS.llm_tokens: "LLM tokens the request spent; absent when it spent none.",
    RATE_LIMIT_LIMIT_HEADER: "The tenant's request budget per minute; present when the rate "
    "limiter counted the request.",
    RATE_LIMIT_REMAINING_HEADER: "Requests left in the current window; present when the rate "
    "limiter counted the request.",
}
#: On the responses of persistent writes: the replay of an Idempotency-Key record.
WRITE_RESPONSE_HEADERS: dict[str, str] = {
    IDEMPOTENT_REPLAYED_HEADER: "true when the response is the stored result of an earlier "
    "request with the same Idempotency-Key."
}
DEPRECATED_ROUTE_HEADERS: dict[str, str] = {
    DEPRECATION_HEADER: "RFC 9745 structured-field date (@<unix seconds>) of this alias "
    f"route's deprecation; the route is removed in {ALIASES_REMOVED_IN}.",
    LINK_HEADER: 'RFC 8288 link with rel="successor-version" (RFC 5829) naming the canonical '
    "route.",
}
#: Every public operation can answer these before the route runs.
EDGE_STATUSES = (413, 429)


def _header_components() -> dict[str, Any]:
    described = {**RESPONSE_HEADERS, **WRITE_RESPONSE_HEADERS, **DEPRECATED_ROUTE_HEADERS}
    described[RETRY_AFTER_HEADER] = "Seconds until the rate-limit window resets (on 429)."
    components: dict[str, Any] = {
        name: {"description": text, "schema": {"type": "string"}}
        for name, text in described.items()
    }
    for name, example in deprecation_headers_for("POST", "/v1/files").items():
        components[name]["example"] = example  # what the middleware sends for that alias
    for name, alias in DEPRECATED_RESPONSE_HEADER_ALIASES.items():
        components[alias] = {
            "description": f"Deprecated spelling of {name}; sent until {ALIASES_REMOVED_IN}.",
            "schema": {"type": "string"},
            "deprecated": True,
        }
    return components


def _response_header_refs(names: list[str]) -> dict[str, Any]:
    return {name: {"$ref": f"#/components/headers/{name}"} for name in names}


def _document_responses(path: str, method: str, op: dict[str, Any]) -> None:
    """Attach the edge statuses, the response headers and a per-operation problem instance."""
    responses = op.setdefault("responses", {})
    for status in EDGE_STATUSES:
        responses.setdefault(str(status), error_responses(status)[status])
    headers = list(RESPONSE_HEADERS) + list(DEPRECATED_RESPONSE_HEADER_ALIASES.values())
    if method != "get":
        headers += list(WRITE_RESPONSE_HEADERS)
    if op.get("deprecated"):
        headers += list(DEPRECATED_ROUTE_HEADERS)
    for status, response in responses.items():
        response.setdefault("headers", {}).update(_response_header_refs(headers))
        if status == "429":
            response["headers"].update(_response_header_refs([RETRY_AFTER_HEADER]))
        for media in response.get("content", {}).values():
            example = media.get("example")
            if isinstance(example, dict) and "instance" in example:
                example["instance"] = path


def _rename_body_schemas(schema: dict[str, Any]) -> None:
    """FastAPI names a form body model ``Body_<operationId>``; with a dot in the operation id
    pydantic's ref shortener keeps only the part after the dot, so the component would be
    named after the function. Give it the full, underscored name and repoint the refs."""
    schemas = schema.get("components", {}).get("schemas", {})
    renames = {
        key: definition["title"].replace(".", "_")
        for key, definition in schemas.items()
        if str(definition.get("title", "")).startswith("Body_") and key != definition["title"]
    }
    for old, new in renames.items():
        schemas[new] = schemas.pop(old)

    def repoint(node: Any) -> None:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.rsplit("/", 1)[-1] in renames:
                node["$ref"] = "#/components/schemas/" + renames[ref.rsplit("/", 1)[-1]]
            for value in node.values():
                repoint(value)
        elif isinstance(node, list):
            for value in node:
                repoint(value)

    repoint(schema)


def _scope_header_params() -> list[dict[str, Any]]:
    described = {
        HEADERS.tenant: "The tenant acted for. In api_key mode the key names it and this must "
        "agree with the key.",
        HEADERS.workspace: "The workspace (team) acted in; membership is checked.",
        HEADERS.user: "The end user acted for.",
    }
    params = [_header_param(name, text) for name, text in described.items()]
    params.extend(
        _header_param(
            alias,
            f"Deprecated spelling of {name}; read until {ALIASES_REMOVED_IN}. A request "
            "carrying both spellings, or either more than once, with different values is "
            "refused.",
            deprecated=True,
        )
        for name, alias in DEPRECATED_HEADER_ALIASES.items()
    )
    return params


def custom_openapi(app: FastAPI, *, version: str) -> dict[str, Any]:
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=TITLE,
        version=version,
        description=DESCRIPTION,
        routes=app.routes,
        tags=TAGS,
        servers=[{"url": "/", "description": "current host"}],
    )
    components = schema.setdefault("components", {})
    schemas = components.setdefault("schemas", {})
    # No route names Problem as a response model (that would document it under
    # application/json too), so its schema and the enums it references are added here.
    problem = Problem.model_json_schema(ref_template="#/components/schemas/{model}")
    for name, definition in problem.pop("$defs", {}).items():
        schemas.setdefault(name, definition)
    schemas[Problem.__name__] = problem
    components["headers"] = _header_components()
    components["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "api_key mode: a key the service issued (mk_...); trusted_dev "
            "mode: a configured development key",
        },
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "jwt mode: a JWKS-verified token; api_key mode: an issued key "
            "(mk_...) may be sent as a Bearer token too",
        },
    }
    schema["security"] = [{"ApiKeyAuth": []}, {"BearerAuth": []}]

    common = [
        _header_param(
            TRACEPARENT_HEADER, "W3C Trace Context to continue (00-<trace>-<span>-<flags>)."
        ),
        _header_param(
            REQUEST_ID_HEADER,
            "Client request id; generated when absent or not an id (a letter or digit, then "
            "letters, digits and ._:-, at most 200 characters).",
        ),
        _header_param(
            CORRELATION_ID_HEADER,
            "Correlation id shared by related requests; echoed when it is an id (a letter or "
            "digit, then letters, digits and ._:-, at most 200 characters), else replaced.",
        ),
        *_scope_header_params(),
    ]
    idem = _header_param(
        IDEMPOTENCY_KEY_HEADER, "Idempotency key for safe retries of persistent writes."
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
            _document_responses(path, method, op)
    _rename_body_schemas(schema)
    app.openapi_schema = schema
    return schema
