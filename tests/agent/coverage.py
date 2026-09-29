"""What the agent suite actually reached, and the gate that says it reached all of it.

Coverage here is *recorded*, never declared: :func:`tests.agent.conftest.sdk` hands its client
a transport that resolves every request against the service's own routes and records the
operation id it hit. A test therefore earns coverage only by driving ``trellis.memory`` - a
raw ``TestClient`` call records nothing - which is what makes "through the SDK only" a property
of the suite rather than a habit.

Two gates sit on top of that record:

* ``@pytest.mark.covers("tag.operation", ...)`` claims a *happy path*: the operation must have
  answered that test with a status under 400. ``@pytest.mark.covers_error(...)`` claims the
  error path: a status of 400 or more. Both are verified after the test, so a claim cannot be
  a lie (``conftest._verify_coverage_claims``).
* ``tests/agent/test_api_coverage.py`` asserts at collection time that every operation in
  ``docs/openapi.json`` is claimed by some test. That is the assertion that fails when a route
  is added with no test, and it does not depend on test order.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx

#: The committed schema is the list of operations; a contract test already pins it against the
#: generated one (tests/contract/test_openapi_contract.py), so this file needs no running app.
OPENAPI = Path(__file__).resolve().parents[2] / "docs" / "openapi.json"

_METHODS = ("get", "post", "put", "patch", "delete")


@lru_cache(maxsize=1)
def operations() -> dict[str, str]:
    """``operation id -> "METHOD /path"`` for every operation in the committed schema."""
    schema = json.loads(OPENAPI.read_text())
    out: dict[str, str] = {}
    for path, item in schema["paths"].items():
        for method, op in item.items():
            if method in _METHODS:
                out[op["operationId"]] = f"{method.upper()} {path}"
    return out


@lru_cache(maxsize=1)
def deprecated() -> frozenset[str]:
    """The operation ids the schema marks deprecated (the one-release aliases of ADR 0022)."""
    schema = json.loads(OPENAPI.read_text())
    return frozenset(
        op["operationId"]
        for item in schema["paths"].values()
        for method, op in item.items()
        if method in _METHODS and op.get("deprecated")
    )


def concrete_path(template: str, **values: str) -> str:
    """``template`` with every ``{placeholder}`` filled, defaulting to a value that exists
    nowhere: the unauthenticated sweep must be refused before anything is looked up."""
    out = template
    while "{" in out:
        head, _, rest = out.partition("{")
        name, _, tail = rest.partition("}")
        out = head + values.get(name, "nonexistent") + tail
    return out


# --------------------------------------------------------------------------- the record

#: operation id -> the statuses it answered, for the whole session.
_SESSION: dict[str, set[int]] = {}
#: the same, for the test that is running now.
_CURRENT: dict[str, set[int]] = {}
#: The last problem body an SDK request received, so a test can assert the RFC 9457 envelope
#: the service actually sent rather than the fields the SDK keeps.
_LAST_PROBLEM: dict[str, Any] = {}
#: operation id -> the response headers of its most recent call in this test. The SDK hands a
#: caller the decoded body, so this is where a test asserting a *header* contract (the alias
#: routes' Deprecation/Link, the echoed request id) reads what the service sent.
_LAST_HEADERS: dict[str, dict[str, str]] = {}

#: operation id -> node ids that claim it. Filled at collection time by
#: ``conftest.pytest_collection_modifyitems``, which is why the gate is order-independent.
CLAIMED: dict[str, list[str]] = {}
#: The same for error-path claims.
CLAIMED_ERRORS: dict[str, list[str]] = {}


def begin_test() -> None:
    _CURRENT.clear()
    _LAST_PROBLEM.clear()
    _LAST_HEADERS.clear()


def record(operation: str, status: int) -> None:
    _SESSION.setdefault(operation, set()).add(status)
    _CURRENT.setdefault(operation, set()).add(status)


def reached(*, errors: bool) -> set[str]:
    """Operations the running test reached with an error status (``errors``) or without it."""
    return {
        op for op, seen in _CURRENT.items() if any((status >= 400) is errors for status in seen)
    }


def session_statuses() -> dict[str, set[int]]:
    return {op: set(seen) for op, seen in _SESSION.items()}


def last_problem() -> dict[str, Any]:
    """The problem body of the most recent error response an SDK request received."""
    return dict(_LAST_PROBLEM)


def headers_of(operation: str) -> dict[str, str]:
    """The response headers of ``operation``'s most recent call in the running test."""
    return dict(_LAST_HEADERS.get(operation, {}))


@lru_cache(maxsize=1)
def _matchers() -> tuple[tuple[str, re.Pattern[str], str], ...]:
    """``(METHOD, compiled template, operation id)``, literal templates first.

    Matching against the committed schema rather than the app's route table is deliberate:
    ``include_router`` nests its routes behind a private ``_IncludedRouter`` in FastAPI 0.141,
    so a walk of ``app.routes`` depends on framework internals, while the schema is the surface
    the service publishes and the thing this suite is measuring coverage of. The deprecated
    aliases are separate paths in it, so they resolve to themselves.
    """
    out = []
    for operation, spec in operations().items():
        method, template = spec.split(" ", 1)
        literals = re.split(r"\{[^}]+\}", template)
        pattern = re.compile("^" + "[^/]+".join(re.escape(part) for part in literals) + "$")
        out.append((len(literals) - 1, method, pattern, operation))
    return tuple((m, p, o) for _, m, p, o in sorted(out, key=lambda row: row[0]))


def resolve(method: str, path: str) -> str | None:
    """The operation id ``method path`` names in the published schema, or None.

    A template with no placeholder wins over one with placeholders, so a literal route is never
    credited to a path-parameter route that also matches it.
    """
    for route_method, pattern, operation in _matchers():
        if route_method == method.upper() and pattern.match(path):
            return operation
    return None


class RecordingTransport(httpx.AsyncBaseTransport):
    """``inner`` with a note kept of which operation each request reached.

    Wrapping the transport rather than the app is deliberate: only traffic a ``MemoryClient``
    produced passes through here, so coverage cannot be earned by reaching around the SDK.
    """

    def __init__(self, inner: httpx.AsyncBaseTransport) -> None:
        self._inner = inner

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        response = await self._inner.handle_async_request(request)
        operation = resolve(request.method, request.url.path)
        if operation is not None:
            record(operation, response.status_code)
            _LAST_HEADERS[operation] = {
                name.lower(): value for name, value in response.headers.items()
            }
        if response.status_code >= 400:
            # Small bodies, and the only reason to read one here: a test that asserts the
            # problem envelope should assert what the service sent, not what the SDK kept.
            body = await response.aread()
            _LAST_PROBLEM.clear()
            _LAST_PROBLEM.update(_problem(body, response.headers.get("content-type", "")))
        return response

    async def aclose(self) -> None:
        await self._inner.aclose()


def _problem(body: bytes, content_type: str) -> dict[str, Any]:
    try:
        parsed = json.loads(body)
    except ValueError:
        return {"content_type": content_type}
    if not isinstance(parsed, dict):
        return {"content_type": content_type}
    return {**parsed, "content_type": content_type}
