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


# -- dependencies: what a database failure tells the client ----------------------------


def _failing_app(error: Exception) -> TestClient:
    """A bare app with the service's handlers and one route that raises ``error``."""
    from fastapi import FastAPI

    from memory_service.api.errors import install_error_handlers

    app = FastAPI()
    install_error_handlers(app)

    @app.get("/boom")
    async def boom() -> None:
        raise error

    return TestClient(app, raise_server_exceptions=False)


def _driver_errors() -> dict[str, tuple[Exception, int, str]]:
    import psycopg.errors
    from sqlalchemy.exc import DBAPIError, InterfaceError, OperationalError
    from sqlalchemy.exc import TimeoutError as PoolTimeout

    secret = Exception("password=hunter2 host=db.internal")
    cancelled = psycopg.errors.QueryCanceled("canceling statement due to statement timeout")
    return {
        "operational": (OperationalError("SELECT 1", {}, secret), 503, "DEPENDENCY_UNAVAILABLE"),
        "interface": (InterfaceError("SELECT 1", {}, secret), 503, "DEPENDENCY_UNAVAILABLE"),
        "invalidated": (
            DBAPIError("SELECT 1", {}, secret, connection_invalidated=True),
            503,
            "DEPENDENCY_UNAVAILABLE",
        ),
        "pool timeout": (PoolTimeout("QueuePool limit reached"), 503, "DEPENDENCY_UNAVAILABLE"),
        "statement timeout": (OperationalError("SELECT 1", {}, cancelled), 504, "TIMEOUT"),
    }


def test_a_database_outage_is_a_retryable_problem_with_retry_after() -> None:
    for name, (error, status, code) in _driver_errors().items():
        r = _failing_app(error).get("/boom")
        assert r.status_code == status, name
        body = r.json()
        assert body["code"] == code and body["retryable"] is True, name
        assert r.headers["Retry-After"] == "5", name
        assert "hunter2" not in r.text and "SELECT" not in r.text, name


def test_any_other_driver_error_stays_an_internal_error() -> None:
    from sqlalchemy.exc import IntegrityError

    r = _failing_app(IntegrityError("INSERT", {}, Exception("duplicate key"))).get("/boom")
    assert r.status_code == 500 and r.json()["code"] == "INTERNAL"
    assert "Retry-After" not in r.headers and "duplicate" not in r.text


def test_a_failure_that_names_its_wait_is_told_to_the_client() -> None:
    from memory_service.domain.errors import DependencyUnavailable

    error = DependencyUnavailable("llm circuit open", details={"retry_after_seconds": 2.2})
    r = _failing_app(error).get("/boom")
    assert r.status_code == 503 and r.headers["Retry-After"] == "3"


def test_a_405_keeps_the_allow_header(client: TestClient) -> None:
    r = client.put("/health/live")
    assert r.status_code == 405 and r.headers["content-type"] == PROBLEM_MEDIA_TYPE
    assert r.headers["Allow"] == "GET"


def test_every_documented_code_is_one_the_service_can_produce() -> None:
    """A code in the enum is a promise a client codes against; none is advertised unused."""
    from memory_service.domain import errors

    produced = {
        cls.code
        for cls in vars(errors).values()
        if isinstance(cls, type) and issubclass(cls, errors.MemoryServiceError)
    }
    # the middleware writes RATE_LIMIT (429) itself; nothing else is produced outside a class
    assert set(ErrorCode) == produced | {ErrorCode.RATE_LIMIT}


def test_a_naive_request_instant_is_utc() -> None:
    from datetime import UTC, datetime

    from pydantic import TypeAdapter

    from memory_service.domain.instants import UtcDateTime

    adapter = TypeAdapter(UtcDateTime)
    assert adapter.validate_python("2026-09-14T10:00:00") == datetime(2026, 9, 14, 10, tzinfo=UTC)
    kept = adapter.validate_python("2026-09-14T12:00:00+02:00")
    assert kept.utcoffset() is not None and kept.hour == 12
