"""OpenAPI customization: authentication, standard headers, problem schema, tags, ids."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.routing import APIRoute

from memory_service.api.errors import Problem, error_responses
from memory_service.api.headers import (
    CORRELATION_ID_HEADER,
    IDEMPOTENCY_KEY_HEADER,
    IDEMPOTENT_REPLAYED_HEADER,
    LINK_HEADER,
    LOCATION_HEADER,
    RATE_LIMIT_LIMIT_HEADER,
    RATE_LIMIT_REMAINING_HEADER,
    REQUEST_ID_HEADER,
    RETRY_AFTER_HEADER,
    TRACE_ID_HEADER,
)
from memory_service.config.constants import HEADERS
from memory_service.observability.tracing import TRACEPARENT_HEADER

TITLE = "trellis-memory API"

DESCRIPTION = """
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
evaluated by OpenFGA on every request.

### Standard headers
| Header | Direction | Purpose |
|---|---|---|
| `Idempotency-Key` | request | Makes every write safe to retry (24h window). |
| `Location` | response | 201: the created resource; 202 that queued a job: `/v1/jobs/{id}`. |
| `traceparent` | both | W3C Trace Context: continued when sent; the response names the trace. |
| `X-Request-ID` | both | Per-request id; generated when absent or not an id. |
| `X-Correlation-ID` | both | Groups related requests; echoed when it is an id, else replaced. |
| `X-Trace-ID` | response | The 32-hex trace id the request ran under (as in `traceparent`). |
| `X-Trellis-LLM-Tokens` | response | LLM tokens the request spent, when it spent any. |

The read-only POSTs (`/v1/recall`, `/v1/context`, `/v1/verify`, `/v1/tools/hints`) take no
`Idempotency-Key`: repeating them changes nothing. An id starts with a letter or digit,
continues with letters, digits and `._:-`, and is at most 200 characters. A request carrying
any scope or credential header more than once with different values is refused (422) before
the credential is read.

### Errors
Every error is an RFC 9457 problem (`application/problem+json`, schema `Problem`):
`type` is `urn:trellis:problem:<code in kebab case>`, `code` the stable category,
`retryable=true` means the same request may succeed later, and `trace_id` / `request_id`
are what to quote. A retryable 429, 503 or 504 carries `Retry-After` (seconds): 503 is a
dependency that is down (PostgreSQL unreachable or its pool exhausted, the search or
authorization server away), 504 a database statement stopped at its time budget.

### Versioning
Public routes live under `/v1`. 0.2.0 (ADR 0022) changed `/v1` in place once, because its
only consumers were the owner's own repositories: problem details replaced the error
envelope, operation ids became `<tag>.<function>`, `X-Trace-ID` became a response header.
0.3.0 removed the one-release aliases 0.2.0 kept (`POST /v1/files`, the `X-Memory-*` header
spellings): each operation has exactly one route and each header one spelling.
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
        "name": "feedback",
        "description": "Human, judge and interrupt judgements on runs (and the answers they gave), "
        "memories, tool calls and procedures, projected into what they judge (memory "
        "standing, run outcomes, tool statistics and approval patterns, procedures).",
    },
    {
        "name": "retrieval",
        "description": "Scope-filtered recall, ContextBundle assembly, and "
        "grounding verification of an answer against evidence.",
    },
    {"name": "graph", "description": "Entity and relation queries over the knowledge graph."},
    {
        "name": "profile",
        "description": "Pinned profile blocks of the user, the agent and the workspace, part of "
        "every pushed context.",
    },
    {
        "name": "tools",
        "description": "Tool memory: the catalog, call records, run outcomes, tool hints "
        "(candidates, learned plan, next step, prefilled arguments) and approval suggestions.",
    },
    {
        "name": "agent_tools",
        "description": "The memory tools an agent calls itself (search, remember, update, "
        "forget, history, profile edit, procedures, tools, outcome); every call is a pull the "
        "pushed context learns from.",
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


def _header_param(name: str, description: str, *, required: bool = False) -> dict[str, Any]:
    return {
        "name": name,
        "in": "header",
        "required": required,
        "description": description,
        "schema": {"type": "string"},
    }


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
#: On the responses of cursor-paged list routes: the next page, when there is one.
PAGED_RESPONSE_HEADERS: dict[str, str] = {
    LINK_HEADER: 'RFC 8288 link: rel="next" names the next page of a list (present exactly '
    "when one exists, ADR 0023)."
}
#: On a 201 (the created resource) and on the 202s that queue a job (its status).
LOCATION_RESPONSE_HEADERS: dict[str, str] = {
    LOCATION_HEADER: "Where the result lives: the created resource on a 201 (e.g. "
    "/v1/memories/{memory_id}), the status of the first queued job on a 202 "
    "(/v1/jobs/{job_id}). An idempotent replay carries the same Location."
}
#: The 202s whose body names a job, and so a Location (``POST /v1/tools/invocations`` records
#: the call in the request and queues none).
JOB_ACCEPTS = frozenset({"messages.create_messages", "documents.upload_document"})
#: POSTs that only read - a query too large or too structured for a URL - so neither take an
#: ``Idempotency-Key`` nor answer ``Idempotent-Replayed``. ``/v1/verify`` records the judge's
#: verdict, under an id derived from the run, the bundle and the answer: verifying the same
#: answer twice is one verdict without a key.
UNKEYED_POSTS = frozenset(
    {"retrieval.recall", "retrieval.context", "retrieval.verify", "tools.tool_hints"}
)
#: Every public operation can answer these before the route runs.
EDGE_STATUSES = (413, 429)
#: The retryable statuses whose responses say when to retry (``Retry-After``).
RETRY_AFTER_STATUSES = frozenset({"429", "503", "504"})


def _header_components() -> dict[str, Any]:
    described = {
        **RESPONSE_HEADERS,
        **WRITE_RESPONSE_HEADERS,
        **PAGED_RESPONSE_HEADERS,
        **LOCATION_RESPONSE_HEADERS,
    }
    described[RETRY_AFTER_HEADER] = (
        "Seconds to wait before retrying: until the rate-limit window resets (429), or until "
        "an unavailable or timed-out dependency is worth asking again (503, 504)."
    )
    components: dict[str, Any] = {
        name: {"description": text, "schema": {"type": "string"}}
        for name, text in described.items()
    }
    return components


def _response_header_refs(names: list[str]) -> dict[str, Any]:
    return {name: {"$ref": f"#/components/headers/{name}"} for name in names}


def _document_responses(path: str, method: str, op: dict[str, Any]) -> None:
    """Attach the edge statuses, the response headers and a per-operation problem instance."""
    responses = op.setdefault("responses", {})
    for status in EDGE_STATUSES:
        responses.setdefault(str(status), error_responses(status)[status])
    headers = list(RESPONSE_HEADERS)
    if takes_idempotency_key(method, op):
        headers += list(WRITE_RESPONSE_HEADERS)
    if any(p.get("name") == "cursor" for p in op.get("parameters", [])):
        headers += list(PAGED_RESPONSE_HEADERS)
    for status, response in responses.items():
        response.setdefault("headers", {}).update(_response_header_refs(headers))
        if status in RETRY_AFTER_STATUSES:
            response["headers"].update(_response_header_refs([RETRY_AFTER_HEADER]))
        if status == "201" or (status == "202" and op.get("operationId") in JOB_ACCEPTS):
            response["headers"].update(_response_header_refs(list(LOCATION_RESPONSE_HEADERS)))
        for media in response.get("content", {}).values():
            example = media.get("example")
            if isinstance(example, dict) and "instance" in example:
                example["instance"] = path


def takes_idempotency_key(method: str, op: dict[str, Any]) -> bool:
    """Whether the operation is a write that honours ``Idempotency-Key`` (every write goes
    through ``api/idempotent.run_idempotent``), as opposed to a GET or a read-only POST."""
    return method in ("post", "put", "patch", "delete") and op.get("operationId") not in (
        UNKEYED_POSTS
    )


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
    return [_header_param(name, text) for name, text in described.items()]


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
        IDEMPOTENCY_KEY_HEADER,
        "Makes a retry of this write safe for 24 hours: the same key with the same body gets "
        "the first response (status, body, Location) with Idempotent-Replayed: true, never a "
        "second effect; the same key with a different body is a 409. Any string of at most "
        "255 characters, unique per logical write (a UUID).",
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
            if takes_idempotency_key(method, op) and idem["name"] not in existing:
                params.append(dict(idem))
            _document_responses(path, method, op)
    _rename_body_schemas(schema)
    app.openapi_schema = schema
    return schema
