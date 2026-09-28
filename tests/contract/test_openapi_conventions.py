"""API conventions every trellis package codes against (ADR 0022): stable operation ids,
the trellis headers with their deprecated spellings, problem details, deprecated aliases."""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.contract

OPERATION_ID = re.compile(r"^[a-z_]+\.[a-z_]+$")
METHODS = ("get", "post", "put", "patch", "delete")
SCOPE_HEADERS = ("X-Trellis-Tenant", "X-Trellis-Workspace", "X-Trellis-User")
DEPRECATED_HEADERS = ("X-Memory-Tenant", "X-Memory-Workspace", "X-Memory-User")
ALIASES = {
    ("post", "/v1/files"): "/v1/documents",
    ("post", "/v1/tools/record"): "/v1/tools/invocations",
}


def _operations(schema: dict) -> list[tuple[str, str, dict]]:
    return [
        (method, path, op)
        for path, methods in schema["paths"].items()
        for method, op in methods.items()
        if method in METHODS
    ]


def test_operation_ids_are_tag_dot_function_and_unique(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    seen: dict[str, str] = {}
    for method, path, op in _operations(schema):
        operation_id = op["operationId"]
        assert OPERATION_ID.match(operation_id), f"{method.upper()} {path}: {operation_id}"
        assert operation_id.split(".")[0] == op["tags"][0], f"{method.upper()} {path}"
        assert operation_id not in seen, f"{operation_id} used by {seen[operation_id]} and {path}"
        seen[operation_id] = path


def test_every_public_operation_documents_the_scope_and_trace_headers(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    for method, path, op in _operations(schema):
        if not path.startswith("/v1/"):
            continue
        params = {p["name"]: p for p in op.get("parameters", [])}
        for name in (*SCOPE_HEADERS, "traceparent"):
            assert name in params, f"{method.upper()} {path}: {name} undocumented"
            assert not params[name].get("deprecated"), f"{method.upper()} {path}: {name}"
        for name in DEPRECATED_HEADERS:
            assert params[name].get("deprecated") is True, f"{method.upper()} {path}: {name}"


def test_errors_are_problem_details(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    problem = schema["components"]["schemas"]["Problem"]
    assert set(problem["required"]) >= {"type", "title", "status", "detail", "instance", "code"}
    assert "ErrorEnvelope" not in schema["components"]["schemas"]
    for method, path, op in _operations(schema):
        if not path.startswith("/v1/"):
            continue  # /health/ready answers 503 with its own body on purpose
        for status, response in op.get("responses", {}).items():
            if not status.startswith(("4", "5")):
                continue
            content = response.get("content", {})
            assert set(content) == {"application/problem+json"}, f"{method.upper()} {path} {status}"
            assert content["application/problem+json"]["schema"] == {
                "$ref": "#/components/schemas/Problem"
            }


def test_the_renamed_nouns_keep_a_deprecated_alias(client: TestClient) -> None:
    from memory_service.api.headers import DEPRECATED_ROUTES

    schema = client.get("/openapi.json").json()
    for (method, path), canonical in ALIASES.items():
        assert schema["paths"][path][method].get("deprecated") is True, path
        assert not schema["paths"][canonical][method].get("deprecated"), canonical
    deprecated = {path for method, path, op in _operations(schema) if op.get("deprecated")}
    assert deprecated == {path for _, path in ALIASES}
    # the middleware marks responses by path: the table and the document must agree
    assert dict(DEPRECATED_ROUTES) == {
        (method.upper(), path): canonical for (method, path), canonical in ALIASES.items()
    }


def test_x_trace_id_is_documented_as_a_response_header_only(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    assert "X-Trace-ID" in schema["components"]["headers"]
    for method, path, op in _operations(schema):
        names = {p["name"] for p in op.get("parameters", [])}
        assert "X-Trace-ID" not in names, f"{method.upper()} {path}"


RESPONSE_HEADERS = ("X-Request-ID", "X-Trace-ID", "traceparent", "X-Correlation-ID")


def test_every_public_operation_documents_edge_statuses_and_response_headers(
    client: TestClient,
) -> None:
    schema = client.get("/openapi.json").json()
    components = schema["components"]["headers"]
    for method, path, op in _operations(schema):
        if not path.startswith("/v1/"):
            continue
        responses = op["responses"]
        assert {"413", "429"} <= set(responses), f"{method.upper()} {path}"
        for status, response in responses.items():
            headers = response.get("headers", {})
            assert set(RESPONSE_HEADERS) <= set(headers), f"{method.upper()} {path} {status}"
            for name, ref in headers.items():
                assert ref == {"$ref": f"#/components/headers/{name}"} and name in components
            if status == "429":
                assert "Retry-After" in headers
            if method != "get":
                assert "Idempotent-Replayed" in headers, f"{method.upper()} {path} {status}"
            if op.get("deprecated"):
                assert {"Deprecation", "Link"} <= set(headers), f"{method.upper()} {path}"
            problem = response.get("content", {}).get("application/problem+json", {})
            example = problem.get("example")
            if example is not None:
                assert example["instance"] == path
    assert components["X-Memory-LLM-Tokens"]["deprecated"] is True


def test_form_body_schemas_keep_their_full_names(client: TestClient) -> None:
    """A dotted operation id must not leak a bare function name into the schema names."""
    schemas = client.get("/openapi.json").json()["components"]["schemas"]
    assert not [name for name in schemas if "." in name]
    for name, definition in schemas.items():
        title = str(definition.get("title", ""))
        if title.startswith("Body_"):
            assert name == title.replace(".", "_"), name
