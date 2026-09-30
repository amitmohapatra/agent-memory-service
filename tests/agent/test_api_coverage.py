"""The gate: every operation the service publishes is driven by an agent test, and every
operation refuses an unauthenticated caller with a problem document.

The first test is the one that fails when a route is added with no test. It reads the claims
collected from ``@pytest.mark.covers`` marks (``conftest.pytest_collection_modifyitems``), so it
does not care in which order the suite runs, or whether the rest of it passed.

The second is the error path for all 68 ``/v1`` operations in one pass: an agent whose key is
wrong must be refused by every one of them, in RFC 9457 shape. It is one test rather than 68
parametrised ones because each case would otherwise pay for its own service and table
truncation; the assertion reports every operation that misbehaved, not just the first.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.agent import coverage
from tests.agent.conftest import sdk
from trellis.memory import MemoryError

pytestmark = pytest.mark.e2e

HERE = Path(__file__).resolve().parent
#: The four operations no agent authenticates for: they are the operator's, and the liveness
#: and metrics routes are what a scrape and a load balancer call before any key exists.
PUBLIC = {"operations.live", "operations.ready", "operations.metrics", "operations.version"}
PROBLEM_MEMBERS = {"type", "title", "status", "detail", "instance", "code"}
#: Multipart routes: an empty JSON body would be refused for the wrong reason.
MULTIPART = {"documents.upload_document"}


def test_every_operation_is_driven_by_an_agent_test(request: pytest.FixtureRequest) -> None:
    """Fails when an operation exists that no test in tests/agent claims a happy path for."""
    on_disk = {path.name for path in HERE.glob("test_*.py")}
    collected = {item.path.name for item in request.session.items if item.path.is_relative_to(HERE)}
    if on_disk - collected:
        pytest.skip(
            "partial collection ("
            + ", ".join(sorted(on_disk - collected))
            + " not collected): the coverage gate needs the whole tests/agent suite"
        )

    operations = coverage.operations()
    unclaimed = sorted(op for op in operations if op not in coverage.CLAIMED)
    assert not unclaimed, (
        f"{len(unclaimed)} of {len(operations)} operations have no agent test: "
        + ", ".join(f"{op} ({operations[op]})" for op in unclaimed)
        + ". Add a test in tests/agent that drives it through the SDK and marks it with "
        "@pytest.mark.covers(...)."
    )


def test_every_operation_that_needs_a_key_documents_its_error_path() -> None:
    """Every ``/v1`` operation is claimed by a ``covers_error`` test somewhere in the suite."""
    operations = coverage.operations()
    missing = sorted(
        op for op in operations if op not in PUBLIC and op not in coverage.CLAIMED_ERRORS
    )
    assert not missing, f"no error-path test claims: {missing}"


@pytest.mark.covers_error(*sorted(op for op in coverage.operations() if op not in PUBLIC))
async def test_a_wrong_key_is_refused_by_every_operation_with_a_problem(app, running) -> None:
    client = sdk(app, "mk_this_key_was_never_issued")
    operations = coverage.operations()
    wrong: list[str] = []
    for operation in sorted(op for op in operations if op not in PUBLIC):
        method, template = operations[operation].split(" ", 1)
        path = coverage.concrete_path(template)
        kwargs: dict[str, object] = {}
        if operation in MULTIPART:
            kwargs["files"] = {"file": ("a.txt", b"unreachable", "text/plain")}
        elif method in ("POST", "PUT", "PATCH"):
            kwargs["json"] = {}
        try:
            await client.transport.request(method, path, **kwargs)  # type: ignore[arg-type]
        except MemoryError as refused:
            problem = coverage.last_problem()
            if refused.status != 401:
                wrong.append(f"{operation}: status {refused.status}, expected 401")
            elif not problem.keys() >= PROBLEM_MEMBERS:
                wrong.append(f"{operation}: problem is missing {PROBLEM_MEMBERS - problem.keys()}")
            elif not str(problem.get("content_type", "")).startswith("application/problem+json"):
                wrong.append(f"{operation}: content type {problem.get('content_type')!r}")
            elif problem.get("status") != 401 or not problem.get("code"):
                wrong.append(f"{operation}: problem body says {problem.get('status')!r}")
        else:
            wrong.append(f"{operation}: answered an unissued key")
    assert not wrong, "\n".join(wrong)
