"""The public API's closed sets are enums and its free-form fields are bounded.

Every case here fails validation before any store is touched, so a wrong value is a 422
with the allowed list in it — not an empty result (an unknown recall kind used to be dropped
silently) and not a 500 (``Visibility("NOPE")`` used to raise ValueError out of the handler).
"""

from __future__ import annotations

import json
import re
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
from memory_service.domain.tools import SideEffects, ToolStatus
from memory_service.modules.grounding import cascade
from trellis.memory import models as sdk

HEADERS = {"X-API-Key": "test-key", "X-Trellis-Tenant": "acme", "X-Trellis-User": "u1"}
SCOPE = {"thread_id": "thr_1"}


def _errors(response: Any) -> list[dict[str, Any]]:
    body = response.json()
    assert response.status_code == 422, response.text
    assert body["code"] == "VALIDATION"
    return body["details"]["errors"]


def _locs(response: Any) -> set[str]:
    return {".".join(e["loc"]) for e in _errors(response)}


# ------------------------------------------------------------------ request enums


def test_unknown_recall_kind_is_rejected_not_dropped(client: TestClient) -> None:
    r = client.post(
        "/v1/recall", headers=HEADERS, json={"scope": SCOPE, "query": "q", "kinds": ["fact"]}
    )
    assert "body.kinds.0" in _locs(r)
    assert "'memory', 'chunk', 'summary', 'episode' or 'message'" in r.text


def test_recall_kinds_must_name_at_least_one_kind(client: TestClient) -> None:
    r = client.post("/v1/recall", headers=HEADERS, json={"scope": SCOPE, "query": "q", "kinds": []})
    assert "body.kinds" in _locs(r)


def test_tool_record_rejects_unknown_visibility_and_status(client: TestClient) -> None:
    base = {"scope": SCOPE, "tool": "t", "args": {}}
    r = client.post("/v1/tools/invocations", headers=HEADERS, json={**base, "visibility": "NOPE"})
    assert _locs(r) == {"body.visibility"}
    r = client.post("/v1/tools/invocations", headers=HEADERS, json={**base, "status": "meh"})
    assert _locs(r) == {"body.status"}


def test_a_catalog_entry_s_side_effects_are_the_closed_set(client: TestClient) -> None:
    entry = {"name": "erp-create_po", "side_effects": "unknown"}
    r = client.put("/v1/tools/catalog", headers=HEADERS, json={"tools": [entry]})
    assert _locs(r) == {"body.tools.0.side_effects"}
    r = client.put("/v1/tools/catalog", headers=HEADERS, json={"tools": []})
    assert _locs(r) == {"body.tools"}
    r = client.post("/v1/tools/hints", headers=HEADERS, json={"task": "t", "k": 0})
    assert _locs(r) == {"body.k"}


def test_tool_record_sub_calls_are_typed_and_bounded(client: TestClient) -> None:
    base = {"scope": SCOPE, "tool": "t", "args": {}}
    r = client.post(
        "/v1/tools/invocations", headers=HEADERS, json={**base, "sub_calls": [{"ordinal": -1}]}
    )
    assert _locs(r) == {"body.sub_calls.0.ordinal", "body.sub_calls.0.tool"}
    r = client.post(
        "/v1/tools/invocations",
        headers=HEADERS,
        json={**base, "sub_calls": [{"ordinal": i, "tool": "x"} for i in range(65)]},
    )
    assert _locs(r) == {"body.sub_calls"}


def test_memory_type_filter_is_an_enum(client: TestClient) -> None:
    r = client.get("/v1/memories", headers=HEADERS, params={"memory_type": ["SEMANTIC", "nope"]})
    assert _locs(r) == {"query.memory_type.1"}


def test_verify_needs_the_bundle_it_checks_against(client: TestClient) -> None:
    r = client.post("/v1/verify", headers=HEADERS, json={"scope": SCOPE, "answer": "a"})
    assert _locs(r) == {"body.bundle_id"}
    r = client.post(
        "/v1/verify",
        headers=HEADERS,
        json={"scope": SCOPE, "answer": "a", "bundle_id": "b", "items": []},
    )
    assert "body.items" in _locs(r), "the one evidence source is the bundle"


def test_context_format_is_closed(client: TestClient) -> None:
    r = client.post(
        "/v1/context", headers=HEADERS, json={"scope": SCOPE, "query": "q", "format": "xml"}
    )
    assert _locs(r) == {"body.format"}
    r = client.post(
        "/v1/context", headers=HEADERS, json={"scope": SCOPE, "query": "q", "use_llm": True}
    )
    assert "body.use_llm" in _locs(r), "the model policy decides; a request does not"


def test_file_visibility_form_field_is_an_enum(client: TestClient) -> None:
    r = client.post(
        "/v1/documents",
        headers=HEADERS,
        files={"file": ("a.txt", b"hello", "text/plain")},
        data={"scope": json.dumps(SCOPE), "visibility": "NOPE"},
    )
    assert _locs(r) == {"body.visibility"}


def test_pydantic_validation_error_raised_inside_a_handler_is_a_500(settings, overrides) -> None:
    """Input is validated at the edge; a ValidationError past it is a bug, not a 422.

    Every request field is typed on the route signature, and the two by-hand coercions that
    remain (the /v1/documents form, the execution context) catch their own ValidationError and
    raise ValidationFailed. So a pydantic failure that escapes a handler means the service
    built a bad model from its own data, and it must surface as an INTERNAL 500 carrying the
    request id, not be dressed up as the caller's mistake.
    """

    class Inner(BaseModel):
        n: int

    app = create_app(settings, overrides=overrides)
    router = APIRouter()

    @router.post("/coerce")
    async def coerce(body: dict[str, Any]) -> dict[str, Any]:
        return Inner.model_validate(body).model_dump()

    app.include_router(router)
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.post("/coerce", json={"n": "not a number"}, headers={"X-Request-ID": "req_test"})
        assert r.status_code == 500, r.text
        error = r.json()
        assert error["code"] == "INTERNAL" and error["retryable"] is False
        # the ids ride in the problem (the 500 path bypasses the header middleware)
        assert error["request_id"] == "req_test"
        assert re.fullmatch(r"[0-9a-f]{32}", error["trace_id"])
        # the input value is not echoed back: the problem never contains source text
        assert "not a number" not in r.text


# ------------------------------------------------------------------ bounds


@pytest.mark.parametrize(
    ("path", "body", "loc"),
    [
        ("/v1/context", {"query": "q", "token_budget": 16_001}, "body.token_budget"),
        ("/v1/context", {"query": "q", "document_ids": ["d"] * 101}, "body.document_ids"),
        ("/v1/recall", {"query": "q", "document_ids": ["d"] * 101}, "body.document_ids"),
        ("/v1/recall", {"query": "q", "limit": 101}, "body.limit"),
        ("/v1/verify", {"answer": "x" * 8_001, "bundle_id": "b"}, "body.answer"),
        ("/v1/verify", {"answer": "a", "bundle_id": "b" * 65}, "body.bundle_id"),
        (
            "/v1/messages",
            {"messages": [{"role": "USER", "content": "c", "source_system": "x" * 101}]},
            "body.messages.0.source_system",
        ),
        (
            "/v1/messages",
            {"messages": [{"role": "USER", "content": "c"}] * 101},
            "body.messages",
        ),
        ("/v1/messages", {"messages": []}, "body.messages"),
        (
            "/v1/agent-tools/tool_search",
            {"args": {}, "toolbox": ["t"] * 501},
            "body.toolbox",
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
    r = client.post("/v1/tools/invocations", headers=HEADERS, json={**base, "args": big})
    assert _locs(r) == {"body.args"}
    r = client.post(
        "/v1/tools/invocations", headers=HEADERS, json={**base, "output": "x" * (256 * 1024 + 1)}
    )
    assert _locs(r) == {"body.output"}
    r = client.put(
        "/v1/tools/catalog", headers=HEADERS, json={"tools": [{"name": "x", "input_schema": big}]}
    )
    assert _locs(r) == {"body.tools.0.input_schema"}


def test_custom_metadata_is_bounded_everywhere_it_appears(client: TestClient) -> None:
    too_many = {f"k{i}": i for i in range(METADATA_MAX_KEYS + 1)}
    too_deep = {"a": {"b": {"c": 1}}}
    too_big = {"blob": "x" * METADATA_MAX_BYTES}
    for bad in (too_many, too_deep, too_big):
        r = client.patch(
            "/v1/threads/thr_1", headers=HEADERS, json={"scope": SCOPE, "custom_metadata": bad}
        )
        assert "body.custom_metadata" in _locs(r), r.text
        message = {"role": "USER", "content": "c", "custom_metadata": bad}
        r = client.post(
            "/v1/messages", headers=HEADERS, json={"scope": SCOPE, "messages": [message]}
        )
        assert "body.messages.0.custom_metadata" in _locs(r), r.text
    # the scope's own metadata, on any route that carries a scope
    r = client.post(
        "/v1/recall",
        headers=HEADERS,
        json={"scope": {**SCOPE, "custom_metadata": too_deep}, "query": "q"},
    )
    assert "body.scope.custom_metadata" in _locs(r)
    # the multipart form: parsed by hand, bounded by the same rule
    r = client.post(
        "/v1/documents",
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
        (sdk.EvidenceStatus, enums.EvidenceStatus),
        (sdk.MessageRole, enums.MessageRole),
        (sdk.MessageKind, enums.MessageKind),
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
        (sdk.ClaimVerdictValue, ClaimVerdict),
        (sdk.GroundingMethod, GroundingMethod),
        (sdk.ToolStatus, ToolStatus),
        (sdk.SideEffects, SideEffects),
    ],
)
def test_sdk_literals_match_the_service_literals(sdk_literal: Any, service: Any) -> None:
    assert _values(sdk_literal) == _values(service)


def test_search_kinds_are_the_record_kinds_and_the_history() -> None:
    from memory_service.modules.retrieval.search import SearchKind

    assert (
        _values(SearchKind)
        == _values(sdk.SearchKind)
        == {"memory", "chunk", "summary", "episode", "message"}
    )


def test_grounding_ranks_every_kind_a_bundle_carries() -> None:
    """Grounding ranks evidence by kind through the citation-preference table, so every kind a
    bundle records (a packed item's representation, an unused item's record kind) is in it."""
    assert _values(cascade.EvidenceKind) == set(cascade._SOURCE_RANK)
    packed = {"CHUNK", "TABLE", "PARAGRAPH", "SECTION", "SUBSECTION", "CODE_BLOCK"}
    packed |= {"SUMMARY", "ENTITY", "RELATION", "MEMORY"}
    unused = {"chunk", "memory", "summary", "fact"}
    assert packed | unused <= _values(cascade.EvidenceKind)
