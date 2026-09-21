"""OpenAPI contract: schema is valid, every public route documents examples and errors."""

from __future__ import annotations

import json
import re
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


def test_every_enum_field_tells_the_caller_which_value_to_use(client: TestClient) -> None:
    """A 25-value enum with no description is a question, not an API.

    ``memory_type`` offers 25 values, ``visibility`` 10, and most of them are written by the
    pipeline rather than chosen by a caller. Without a description the only way to pick one
    is to read our source, and a wrong choice is silent: a hint overrides the classifier, so a
    mislabelled memory simply stops being retrievable by the queries that should find it.
    """
    spec = _schema(client)
    schemas = spec.get("components", {}).get("schemas", {})
    undocumented: list[str] = []
    for name, schema in schemas.items():
        for field, prop in (schema.get("properties") or {}).items():
            values = prop.get("enum")
            if not values:
                for key in ("allOf", "anyOf", "oneOf"):
                    for ref in prop.get(key, []):
                        target = ref.get("$ref", "").rsplit("/", 1)[-1]
                        if target and schemas.get(target, {}).get("enum"):
                            values = schemas[target]["enum"]
            if values and not prop.get("description"):
                undocumented.append(f"{name}.{field} ({len(values)} values)")
    assert not undocumented, (
        "enum fields with no guidance on which value to use:\n  " + "\n  ".join(undocumented)
    )


def test_the_docs_do_not_describe_endpoints_that_no_longer_exist(client: TestClient) -> None:
    """Documentation that outlives its endpoint is worse than none.

    ``docs/TOOL_MEMORY.md`` carried a status note saying four endpoints were removed and then
    documented three of them in full eighty lines further down, request bodies and all. A
    reader reaching §30.2 first has no way to know.

    An endpoint path may still be *mentioned* — saying why something was removed is useful —
    but not as a live instruction with a request body.
    """
    spec = _schema(client)

    # Path parameters are named freely in prose ("{id}" for the spec's "{run_id}"), so both
    # sides are normalised to "{}" — the check is about the endpoint existing, not its spelling.
    def normalise(path: str) -> str:
        return re.sub(r"\{[^}]*\}", "{}", path.rstrip("/"))

    live = {normalise(path) for path in spec.get("paths", {})}
    docs = Path(__file__).resolve().parents[2] / "docs"
    instruction = re.compile(r"`(?:POST|GET|PUT|PATCH|DELETE)\s+(/v1/[A-Za-z0-9/_{}-]+)")
    # A doc may legitimately describe an endpoint that does not exist — explaining why one
    # was removed, or specifying one not built yet — as long as it says so. The marker can be
    # on the line itself or on the section heading it sits under, so a design section is
    # marked once rather than line by line.
    marker = re.compile(r"(?i)\bremoved\b|not implemented|\bplanned\b|\bfuture\b|\bproposed\b")
    offenders: list[str] = []
    for page in sorted(docs.rglob("*.md")):
        section_exempt = False
        for number, line in enumerate(page.read_text().splitlines(), start=1):
            if line.startswith("#"):
                section_exempt = bool(marker.search(line))
            if marker.search(line) or section_exempt:
                continue
            for match in instruction.finditer(line):
                path = normalise(match.group(1))
                if path not in live:
                    offenders.append(f"{page.relative_to(docs)}:{number}: {match.group(1)}")
    assert not offenders, (
        "docs give instructions for endpoints the API does not serve:\n  " + "\n  ".join(offenders)
    )
