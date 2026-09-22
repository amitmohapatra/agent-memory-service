"""The public API's closed sets are enums and its free-form fields are bounded.

Every case here fails validation before any store is touched, so a wrong value is a 422
with the allowed list in it — not an empty result (an unknown recall kind used to be dropped
silently) and not a 500 (``Visibility("NOPE")`` used to raise ValueError out of the handler).
"""

from __future__ import annotations

import json
from typing import Any, get_args

import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient
from pydantic import BaseModel, TypeAdapter, ValidationError

from memory_service.api.app import create_app
from memory_service.api.validation import (
    METADATA_MAX_BYTES,
    METADATA_MAX_KEYS,
    TOOL_JSON_MAX_BYTES,
    CustomMetadata,
    container_depth,
)
from memory_service.domain import enums
from memory_service.domain.grounding import ClaimVerdict, GroundingMethod
from memory_service.domain.tools import CacheScope, SideEffects, ToolSource, ToolStatus
from memory_service.modules.grounding import cascade
from universal_memory import models as sdk

HEADERS = {"X-API-Key": "test-key", "X-Memory-Tenant": "acme", "X-Memory-User": "u1"}
SCOPE = {"thread_id": "thr_1"}


def _errors(response: Any) -> list[dict[str, Any]]:
    body = response.json()
    assert response.status_code == 422, response.text
    assert body["error"]["code"] == "VALIDATION"
    return body["error"]["details"]["errors"]


def _locs(response: Any) -> set[str]:
    return {".".join(e["loc"]) for e in _errors(response)}


# ------------------------------------------------------------------ request enums


def test_unknown_recall_kind_is_rejected_not_dropped(client: TestClient) -> None:
    r = client.post(
        "/v1/recall", headers=HEADERS, json={"scope": SCOPE, "query": "q", "kinds": ["fact"]}
    )
    assert "body.kinds.0" in _locs(r)
    assert "'chunk', 'memory' or 'summary'" in r.text


def test_recall_kinds_must_name_at_least_one_kind(client: TestClient) -> None:
    r = client.post("/v1/recall", headers=HEADERS, json={"scope": SCOPE, "query": "q", "kinds": []})
    assert "body.kinds" in _locs(r)


def test_tool_record_rejects_unknown_visibility_status_and_source(client: TestClient) -> None:
    base = {"scope": SCOPE, "tool": "t", "args": {}}
    r = client.post("/v1/tools/record", headers=HEADERS, json={**base, "visibility": "NOPE"})
    assert _locs(r) == {"body.visibility"}
    r = client.post("/v1/tools/record", headers=HEADERS, json={**base, "status": "meh"})
    assert _locs(r) == {"body.status"}
    r = client.post(
        "/v1/tools/plan",
        headers=HEADERS,
        json={"scope": SCOPE, "task": "t", "available_tools": [{"name": "x", "source": "zzz"}]},
    )
    assert _locs(r) == {"body.available_tools.0.source"}


def test_tool_record_sub_calls_are_typed_and_bounded(client: TestClient) -> None:
    base = {"scope": SCOPE, "tool": "t", "args": {}}
    r = client.post(
        "/v1/tools/record", headers=HEADERS, json={**base, "sub_calls": [{"ordinal": -1}]}
    )
    assert _locs(r) == {"body.sub_calls.0.ordinal", "body.sub_calls.0.tool"}
    r = client.post(
        "/v1/tools/record",
        headers=HEADERS,
        json={**base, "sub_calls": [{"ordinal": i, "tool": "x"} for i in range(65)]},
    )
    assert _locs(r) == {"body.sub_calls"}


def test_memory_type_filter_is_an_enum(client: TestClient) -> None:
    r = client.get("/v1/memories", headers=HEADERS, params={"memory_type": ["SEMANTIC", "nope"]})
    assert _locs(r) == {"query.memory_type.1"}


def test_verify_item_kind_is_closed(client: TestClient) -> None:
    r = client.post(
        "/v1/verify",
        headers=HEADERS,
        json={"scope": SCOPE, "answer": "a", "items": [{"item_id": "i", "text": "t", "kind": "x"}]},
    )
    assert _locs(r) == {"body.items.0.kind"}


def test_file_visibility_form_field_is_an_enum(client: TestClient) -> None:
    r = client.post(
        "/v1/files",
        headers=HEADERS,
        files={"file": ("a.txt", b"hello", "text/plain")},
        data={"scope": json.dumps(SCOPE), "visibility": "NOPE"},
    )
    assert _locs(r) == {"body.visibility"}


def test_pydantic_validation_error_raised_inside_a_handler_is_a_422(settings) -> None:
    """A model validated by hand inside a handler is still the caller's input."""

    class Inner(BaseModel):
        n: int

    app = create_app(settings)
    router = APIRouter()

    @router.post("/coerce")
    async def coerce(body: dict[str, Any]) -> dict[str, Any]:
        return Inner.model_validate(body).model_dump()

    app.include_router(router)
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.post("/coerce", json={"n": "not a number"})
        assert r.status_code == 422, r.text
        errors = _errors(r)
        assert errors[0]["loc"] == ["n"] and errors[0]["type"] == "int_parsing"
        # the input value is not echoed back: the envelope never contains source text
        assert "not a number" not in r.text


# ------------------------------------------------------------------ bounds


@pytest.mark.parametrize(
    ("path", "body", "loc"),
    [
        ("/v1/context", {"query": "q", "token_budget": 16_001}, "body.token_budget"),
        ("/v1/context", {"query": "q", "answer": "x" * 8_001}, "body.answer"),
        ("/v1/context", {"query": "q", "document_ids": ["d"] * 101}, "body.document_ids"),
        ("/v1/recall", {"query": "q", "document_ids": ["d"] * 101}, "body.document_ids"),
        ("/v1/graph/query", {"query": "q", "max_visited": 501}, "body.max_visited"),
        ("/v1/verify", {"answer": "x" * 8_001, "query": "q"}, "body.answer"),
        (
            "/v1/verify",
            {"answer": "a", "items": [{"item_id": "i", "text": "t"}] * 51},
            "body.items",
        ),
        (
            "/v1/verify",
            {"answer": "a", "items": [{"item_id": "i", "text": "x" * 4_001}]},
            "body.items.0.text",
        ),
        (
            "/v1/verify",
            {
                "answer": "a",
                "items": [{"item_id": "i", "text": "t"}],
                "unused": [{"item_id": "u", "text": "t"}] * 21,
            },
            "body.unused",
        ),
        (
            "/v1/observations",
            {"content": "c", "hints": {"custom_type": "x" * 65}},
            "body.hints.custom_type",
        ),
        ("/v1/observations", {"content": "c", "source_system": "x" * 101}, "body.source_system"),
        (
            "/v1/messages",
            {"role": "USER", "content": "c", "source_system": "x" * 101},
            "body.source_system",
        ),
    ],
)
def test_request_knobs_are_clamped(
    client: TestClient, path: str, body: dict[str, Any], loc: str
) -> None:
    r = client.post(path, headers=HEADERS, json={"scope": SCOPE, **body})
    assert loc in _locs(r), r.text


def test_tool_payloads_are_bounded_by_serialised_size(client: TestClient) -> None:
    big = {"k": "x" * TOOL_JSON_MAX_BYTES}
    base = {"scope": SCOPE, "tool": "t"}
    r = client.post("/v1/tools/record", headers=HEADERS, json={**base, "args": big})
    assert _locs(r) == {"body.args"}
    r = client.post(
        "/v1/tools/record", headers=HEADERS, json={**base, "output": "x" * (256 * 1024 + 1)}
    )
    assert _locs(r) == {"body.output"}
    r = client.post(
        "/v1/tools/plan",
        headers=HEADERS,
        json={"scope": SCOPE, "task": "t", "available_tools": [{"name": "x", "schema": big}]},
    )
    assert _locs(r) == {"body.available_tools.0.schema"}


def test_custom_metadata_is_bounded_everywhere_it_appears(client: TestClient) -> None:
    too_many = {f"k{i}": i for i in range(METADATA_MAX_KEYS + 1)}
    too_deep = {"a": {"b": {"c": 1}}}
    too_big = {"blob": "x" * METADATA_MAX_BYTES}
    for path, body in (
        ("/v1/threads", {}),
        ("/v1/messages", {"role": "USER", "content": "c"}),
        ("/v1/observations", {"content": "c"}),
    ):
        for bad in (too_many, too_deep, too_big):
            r = client.post(
                path, headers=HEADERS, json={"scope": SCOPE, **body, "custom_metadata": bad}
            )
            assert "body.custom_metadata" in _locs(r), (path, r.text)
    # the scope's own metadata, on any route that carries a scope
    r = client.post(
        "/v1/recall",
        headers=HEADERS,
        json={"scope": {**SCOPE, "custom_metadata": too_deep}, "query": "q"},
    )
    assert "body.scope.custom_metadata" in _locs(r)
    # the multipart form: parsed by hand, bounded by the same rule
    r = client.post(
        "/v1/files",
        headers=HEADERS,
        files={"file": ("a.txt", b"hello", "text/plain")},
        data={"scope": json.dumps(SCOPE), "custom_metadata": json.dumps(too_deep)},
    )
    assert r.status_code == 422 and "custom_metadata" in r.text
    # a flat object of objects is the allowed shape
    ok = {"source": "upload", "tags": ["a", "b"], "meta": {"k": 1}}
    assert TypeAdapter(CustomMetadata).validate_python(ok) == ok


def test_container_depth_counts_nested_containers() -> None:
    assert container_depth({}) == 1
    assert container_depth({"a": 1}) == 1
    assert container_depth({"a": [1, 2]}) == 2
    assert container_depth({"a": {"b": {"c": 1}}}) == 3
    with pytest.raises(ValidationError):
        TypeAdapter(CustomMetadata).validate_python({"a": [[1]]} | {"b": {"c": {"d": 1}}})


# ------------------------------------------------------------------ the SDK speaks the same vocabulary


def _values(literal: Any) -> set[str]:
    return set(get_args(literal))


@pytest.mark.parametrize(
    ("sdk_literal", "service"),
    [
        (sdk.MemoryType, enums.MemoryType),
        (sdk.Lifetime, enums.Lifetime),
        (sdk.Visibility, enums.Visibility),
        (sdk.ScopeLevel, enums.ScopeLevel),
        (sdk.TemporalStatus, enums.TemporalStatus),
        (sdk.Representation, enums.Representation),
        (sdk.QueryType, enums.QueryType),
        (sdk.EvidenceStatus, enums.EvidenceStatus),
        (sdk.MessageRole, enums.MessageRole),
        (sdk.MessageKind, enums.MessageKind),
        (sdk.ObservationKind, enums.ObservationKind),
        (sdk.JobStatus, enums.JobStatus),
        (sdk.DocumentStatus, enums.DocumentStatus),
        (sdk.ArchiveStatus, enums.ArchiveStatus),
    ],
)
def test_sdk_literals_match_the_service_enums(sdk_literal: Any, service: Any) -> None:
    assert _values(sdk_literal) == {m.value for m in service}


@pytest.mark.parametrize(
    ("sdk_literal", "service"),
    [
        (sdk.EvidenceKind, cascade.EvidenceKind),
        (sdk.ClaimVerdictValue, ClaimVerdict),
        (sdk.GroundingMethod, GroundingMethod),
        (sdk.ToolSource, ToolSource),
        (sdk.ToolStatus, ToolStatus),
        (sdk.SideEffects, SideEffects),
        (sdk.CacheScope, CacheScope),
    ],
)
def test_sdk_literals_match_the_service_literals(sdk_literal: Any, service: Any) -> None:
    assert _values(sdk_literal) == _values(service)


def test_recall_kinds_are_the_three_record_kinds_the_engine_serves() -> None:
    from memory_service.api.routers.v1.retrieval import RecallKind

    assert _values(RecallKind) == _values(sdk.RecallKind) == {"chunk", "memory", "summary"}


def test_verify_accepts_exactly_the_kinds_it_ranks() -> None:
    """/v1/verify's ``kind`` is the key set of the citation-preference table, so every kind a
    bundle emits (a packed item's representation, an unused item's record kind) round-trips."""
    assert _values(cascade.EvidenceKind) == set(cascade._SOURCE_RANK)
    packed = {"CHUNK", "TABLE", "PARAGRAPH", "SECTION", "SUBSECTION", "CODE_BLOCK"}
    packed |= {"SUMMARY", "ENTITY", "RELATION", "MEMORY"}
    unused = {"chunk", "memory", "summary", "fact"}
    assert packed | unused <= _values(cascade.EvidenceKind)
