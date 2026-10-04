"""Every field, parameter and operation of the API says what it is (ADR 0030).

A property with no description is a question every client author asks of the source code:
the document had 389 of them, and 199 undescribed parameters, most of them the same scope
query parameters repeated on twenty routes. This test keeps the count at zero, so a field
added without saying what it is, what it is for, its unit or format and its allowed values
fails the build rather than the next reader.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.contract

METHODS = ("get", "post", "put", "patch", "delete")


def _properties(owner: str, schema: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
    for name, prop in (schema.get("properties") or {}).items():
        yield f"{owner}.{name}", prop
        if isinstance(prop, dict) and "properties" in prop:
            yield from _properties(f"{owner}.{name}", prop)


def _undescribed(spec: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for name, schema in spec["components"]["schemas"].items():
        missing += [
            where for where, prop in _properties(name, schema) if not prop.get("description")
        ]
    for path, item in spec["paths"].items():
        for method, op in item.items():
            if method not in METHODS:
                continue
            label = f"{method.upper()} {path}"
            if not (op.get("summary") or op.get("description")):
                missing.append(f"{label}: the operation")
            missing += [
                f"{label}: parameter {param['name']}"
                for param in op.get("parameters", [])
                if not param.get("description")
            ]
            for media, body in (op.get("requestBody") or {}).get("content", {}).items():
                missing += [
                    f"{label}: {where}"
                    for where, prop in _properties(media, body.get("schema") or {})
                    if not prop.get("description")
                ]
    return missing


def test_every_property_parameter_and_operation_is_described(client: TestClient) -> None:
    missing = _undescribed(client.get("/openapi.json").json())
    assert not missing, f"{len(missing)} undescribed:\n  " + "\n  ".join(missing)


def test_the_check_sees_a_missing_description() -> None:
    spec = {
        "components": {"schemas": {"A": {"properties": {"x": {"type": "string"}}}}},
        "paths": {
            "/v1/a": {"get": {"summary": "s", "parameters": [{"name": "limit", "in": "query"}]}}
        },
    }
    assert _undescribed(spec) == ["A.x", "GET /v1/a: parameter limit"]
