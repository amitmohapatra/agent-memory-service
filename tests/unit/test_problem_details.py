"""RFC 9457 problem details: one shape for every error (ADR 0022)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from memory_service.api.errors import (
    PROBLEM_MEDIA_TYPE,
    PROBLEM_TITLES,
    build_problem,
    error_responses,
    problem_type,
)
from memory_service.domain.enums import ErrorCode

MEMBERS = {
    "type",
    "title",
    "status",
    "detail",
    "instance",
    "code",
    "retryable",
    "trace_id",
    "request_id",
    "details",
}


def test_every_error_code_has_a_title_and_a_urn() -> None:
    assert set(PROBLEM_TITLES) == set(ErrorCode)
    assert problem_type(ErrorCode.SCOPE_DENIED) == "urn:trellis:problem:scope-denied"
    assert problem_type(ErrorCode.DEPENDENCY_UNAVAILABLE) == (
        "urn:trellis:problem:dependency-unavailable"
    )
    assert len({problem_type(code) for code in ErrorCode}) == len(ErrorCode)


def test_build_problem_fills_the_standard_members_from_the_code() -> None:
    problem = build_problem(
        code=ErrorCode.NOT_FOUND,
        message="Thread not found",
        status=404,
        retryable=False,
        instance="/v1/threads/thr_1",
        trace_id="a" * 32,
        request_id="req_1",
    )
    assert problem.model_dump() == {
        "type": "urn:trellis:problem:not-found",
        "title": "Not found",
        "status": 404,
        "detail": "Thread not found",
        "instance": "/v1/threads/thr_1",
        "code": ErrorCode.NOT_FOUND,
        "retryable": False,
        "trace_id": "a" * 32,
        "request_id": "req_1",
        "details": {},
    }


def test_documented_error_responses_are_problems() -> None:
    responses = error_responses(401, 418, 503)
    for status, entry in responses.items():
        content = entry["content"]
        assert set(content) == {PROBLEM_MEDIA_TYPE}
        example = content[PROBLEM_MEDIA_TYPE]["example"]
        assert set(example) == MEMBERS and example["status"] == status
    # an undocumented status falls back to the dependency example, still under its own status
    assert (
        responses[418]["content"][PROBLEM_MEDIA_TYPE]["example"]["code"] == "DEPENDENCY_UNAVAILABLE"
    )


def test_a_handled_error_is_a_problem_on_the_wire(client: TestClient) -> None:
    r = client.get("/does-not-exist")
    assert r.status_code == 404 and r.headers["content-type"] == PROBLEM_MEDIA_TYPE
    body = r.json()
    assert set(body) == MEMBERS
    assert body["type"] == "urn:trellis:problem:not-found" and body["title"] == "Not found"
    assert body["status"] == 404 and body["instance"] == "/does-not-exist"
    assert body["trace_id"] == r.headers["X-Trace-ID"]
    assert body["request_id"] == r.headers["X-Request-ID"]


def test_a_validation_problem_keeps_the_field_errors_in_details(client: TestClient) -> None:
    r = client.post(
        "/v1/recall",
        headers={"X-API-Key": "test-key", "X-Trellis-Tenant": "acme", "X-Trellis-User": "u1"},
        json={"scope": {}, "query": 42},
    )
    assert r.status_code == 422 and r.headers["content-type"] == PROBLEM_MEDIA_TYPE
    body = r.json()
    assert body["code"] == "VALIDATION" and body["type"] == "urn:trellis:problem:validation"
    assert body["details"]["errors"][0]["loc"][:2] == ["body", "query"]
