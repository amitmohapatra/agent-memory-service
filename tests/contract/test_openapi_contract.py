"""OpenAPI contract: schema is valid, every public route documents examples and errors."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from openapi_spec_validator import validate

pytestmark = pytest.mark.contract

PUBLIC_PREFIX = "/v1/"


def _schema(client: TestClient) -> dict:
    return client.get("/openapi.json").json()


def test_schema_is_valid_openapi(client: TestClient) -> None:
    validate(_schema(client))


def test_every_public_operation_has_error_responses_and_standard_headers(
    client: TestClient,
) -> None:
    schema = _schema(client)
    problems: list[str] = []
    for path, methods in schema["paths"].items():
        if not path.startswith(PUBLIC_PREFIX):
            continue
        for method, op in methods.items():
            if method not in ("get", "post", "put", "patch", "delete"):
                continue
            responses = op.get("responses", {})
            for status in ("401", "403", "422"):
                if status not in responses:
                    problems.append(f"{method.upper()} {path}: missing {status} response")
            params = {p["name"] for p in op.get("parameters", [])}
            for header in ("X-Request-ID", "X-Trace-ID", "X-Correlation-ID"):
                if header not in params:
                    problems.append(f"{method.upper()} {path}: missing {header} header")
            if method in ("post", "put", "patch", "delete") and "Idempotency-Key" not in params:
                problems.append(f"{method.upper()} {path}: missing Idempotency-Key header")
            body = op.get("requestBody")
            if body:
                content = body.get("content", {})
                for media, spec in content.items():
                    ref = spec.get("schema", {}).get("$ref")
                    has_example = "example" in spec or "examples" in spec
                    if ref:
                        name = ref.rsplit("/", 1)[-1]
                        comp = schema["components"]["schemas"].get(name, {})
                        has_example = (
                            has_example
                            or "example" in comp
                            or "examples" in comp
                            or _all_fields_have_examples(comp, schema)
                        )
                    if not has_example:
                        problems.append(
                            f"{method.upper()} {path}: request body ({media}) has no example"
                        )
    assert not problems, "\n".join(problems)


def _all_fields_have_examples(component: dict, schema: dict) -> bool:
    props = component.get("properties") or {}
    if not props:
        return False
    for name, prop in props.items():
        if (
            name in component.get("required", [])
            and "examples" not in prop
            and "example" not in prop
            and "default" not in prop
            and "$ref" not in prop
            and "allOf" not in prop
        ):
            return False
    return True


def test_committed_schema_matches_generated(client: TestClient) -> None:
    committed = Path(__file__).resolve().parents[2] / "docs" / "openapi.json"
    if not committed.is_file():
        pytest.skip("docs/openapi.json not committed yet")
    assert json.loads(committed.read_text()) == json.loads(
        json.dumps(_schema(client), sort_keys=True)
    )
