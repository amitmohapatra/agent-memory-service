"""API conventions every trellis package codes against (ADR 0022): stable operation ids,
one spelling per trellis header, problem details, one route per operation."""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from memory_service.api.openapi import UNKEYED_POSTS

pytestmark = pytest.mark.contract

OPERATION_ID = re.compile(r"^[a-z_]+\.[a-z_]+$")
METHODS = ("get", "post", "put", "patch", "delete")
SCOPE_HEADERS = ("X-Trellis-Tenant", "X-Trellis-Workspace", "X-Trellis-User")
REMOVED_HEADERS = ("X-Memory-Tenant", "X-Memory-Workspace", "X-Memory-User")


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
        for name in REMOVED_HEADERS:
            assert name not in params, f"{method.upper()} {path}: {name}"


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


#: The one deprecated alias (ADR 0030): the review queue moved to GET /v1/feedback?review=pending.
DEPRECATED_ALIASES = {"feedback.pending_feedback"}


def test_the_only_deprecated_operation_is_the_review_queue_alias(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    assert "/v1/files" not in schema["paths"]
    deprecated = {op["operationId"] for _, _, op in _operations(schema) if op.get("deprecated")}
    assert deprecated == DEPRECATED_ALIASES
    assert "X-Memory-LLM-Tokens" not in schema["components"]["headers"]


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
            if method != "get" and op["operationId"] not in UNKEYED_POSTS:
                assert "Idempotent-Replayed" in headers, f"{method.upper()} {path} {status}"
            if status == "201":
                assert "Location" in headers, f"{method.upper()} {path} {status}"
            problem = response.get("content", {}).get("application/problem+json", {})
            example = problem.get("example")
            if example is not None:
                assert example["instance"] == path
    assert "X-Memory-LLM-Tokens" not in components


def test_form_body_schemas_keep_their_full_names(client: TestClient) -> None:
    """A dotted operation id must not leak a bare function name into the schema names."""
    schemas = client.get("/openapi.json").json()["components"]["schemas"]
    assert not [name for name in schemas if "." in name]
    for name, definition in schemas.items():
        title = str(definition.get("title", ""))
        if title.startswith("Body_"):
            assert name == title.replace(".", "_"), name


def test_the_operational_routes_need_no_credential(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    for method, path, op in _operations(schema):
        if path.startswith(("/health", "/metrics", "/version")):
            assert op["security"] == [], f"{method.upper()} {path}"
        else:
            assert "security" not in op, f"{method.upper()} {path}: the document's default"
    assert schema["info"]["license"]["identifier"] == "Apache-2.0"
    assert schema["info"]["contact"]["url"].startswith("https://")


def test_no_read_documents_a_conflict_it_cannot_raise(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    reads = [f"GET {path}" for method, path, op in _operations(schema) if method == "get"]
    conflicted = [
        f"GET {path}"
        for method, path, op in _operations(schema)
        if method == "get" and "409" in op["responses"]
    ]
    assert reads and not conflicted


def test_every_request_instant_is_read_as_utc(client: TestClient) -> None:
    """A naive request instant is UTC, and the document says so wherever one is taken."""
    schema = client.get("/openapi.json").json()
    schemas = schema["components"]["schemas"]
    for name, definition in schemas.items():
        if not name.endswith(("Request", "In", "Body_documents_upload_document")):
            continue
        for field, prop in (definition.get("properties") or {}).items():
            formats = {prop.get("format")} | {o.get("format") for o in prop.get("anyOf", [])}
            if "date-time" in formats:
                assert "UTC" in prop.get("description", ""), f"{name}.{field}"
    for method, path, op in _operations(schema):
        for param in op.get("parameters", []):
            param_schema = param.get("schema", {})
            formats = {param_schema.get("format")} | {
                o.get("format") for o in param_schema.get("anyOf", [])
            }
            if "date-time" in formats:
                assert "UTC" in param.get("description", ""), f"{method} {path} {param['name']}"
