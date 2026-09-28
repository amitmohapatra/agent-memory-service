"""``api/headers.py``: the wire header names, the deprecated scope-header spellings and the
alias-route deprecation headers (ADR 0022)."""

from __future__ import annotations

import pytest
from packaging.version import Version
from starlette.datastructures import Headers

from memory_service.__about__ import __version__
from memory_service.api.headers import (
    DEPRECATED_ROUTES,
    deprecation_headers_for,
    refuse_ambiguous_headers,
    require_one_spelling,
    route_path,
    scope_header,
    scope_values,
)
from memory_service.config.constants import ALIASES_REMOVED_IN, DEPRECATED_HEADER_ALIASES, HEADERS
from memory_service.domain.errors import ValidationFailed


def test_two_spellings_that_disagree_are_refused_and_two_that_agree_are_one() -> None:
    """A gateway that still stamps the old spelling strips only the old spelling; a client
    behind it must not pick its own tenant or user by adding the new one."""
    conflict = Headers({"X-Trellis-Tenant": "acme", "X-Memory-Tenant": "globex"})
    assert scope_header(conflict, HEADERS.tenant) == "acme"  # the lenient reader, for counting
    with pytest.raises(ValidationFailed, match="X-Trellis-Tenant was sent more than once"):
        require_one_spelling(conflict, HEADERS.tenant)
    with pytest.raises(ValidationFailed):
        refuse_ambiguous_headers(conflict)
    agree = Headers({"X-Trellis-User": "u1", "x-memory-user": "u1"})
    assert require_one_spelling(agree, HEADERS.user) == "u1"
    assert require_one_spelling(Headers({"X-Memory-Workspace": "fin"}), HEADERS.workspace) == "fin"
    refuse_ambiguous_headers(agree)


def test_a_duplicated_header_of_one_spelling_is_refused_too() -> None:
    """A proxy that appends rather than replaces leaves the client's line in front."""
    twice = Headers(raw=[(b"x-trellis-user", b"attacker"), (b"x-trellis-user", b"stamped")])
    assert scope_values(twice, HEADERS.user) == ["attacker", "stamped"]
    with pytest.raises(ValidationFailed):
        require_one_spelling(twice, HEADERS.user)
    same = Headers(raw=[(b"x-trellis-user", b"u1"), (b"x-memory-user", b"u1")])
    assert require_one_spelling(same, HEADERS.user) == "u1"


def test_an_empty_header_is_no_header() -> None:
    blank = Headers({"X-Trellis-Tenant": "  ", "X-Memory-Tenant": "acme"})
    assert scope_values(blank, HEADERS.tenant) == ["acme"]
    assert require_one_spelling(blank, HEADERS.tenant) == "acme"
    assert require_one_spelling(Headers({"X-Trellis-Tenant": ""}), HEADERS.tenant) is None


def test_two_different_credentials_are_refused_before_anyone_picks_one() -> None:
    """The limiter keys on the first credential header, the authenticator would verify the
    last: a request that sends two is refused before either reads it."""
    two_keys = Headers(raw=[(b"x-api-key", b"mk_a.1"), (b"x-api-key", b"mk_b.2")])
    with pytest.raises(ValidationFailed, match="X-API-Key was sent more than once"):
        refuse_ambiguous_headers(two_keys)
    repeated = Headers(raw=[(b"authorization", b"Bearer t"), (b"authorization", b"Bearer t")])
    refuse_ambiguous_headers(repeated)


def test_a_header_without_an_alias_is_read_plainly() -> None:
    assert scope_header(Headers({"X-API-Key": "k"}), HEADERS.api_key) == "k"
    assert scope_header(Headers({}), HEADERS.api_key) is None
    assert require_one_spelling(Headers({}), HEADERS.user) is None


def test_every_scope_header_has_exactly_its_pre_trellis_alias() -> None:
    assert DEPRECATED_HEADER_ALIASES == {
        "X-Trellis-Tenant": "X-Memory-Tenant",
        "X-Trellis-Workspace": "X-Memory-Workspace",
        "X-Trellis-User": "X-Memory-User",
    }


def test_only_alias_routes_are_marked_deprecated_by_method_and_route_path() -> None:
    assert deprecation_headers_for("POST", "/v1/documents") == {}
    assert deprecation_headers_for("GET", "/v1/files") == {}  # a 405, not the alias
    marked = deprecation_headers_for("post", "/v1/files")
    assert marked["Deprecation"].startswith("@") and marked["Deprecation"][1:].isdigit()
    assert marked["Link"] == '</v1/documents>; rel="successor-version"'
    # behind a mount the router sees the path without the root, and the successor is under it
    scope = {"path": "/memory/v1/files", "root_path": "/memory", "method": "POST"}
    assert route_path(scope) == "/v1/files"
    mounted = deprecation_headers_for("POST", route_path(scope), "/memory")
    assert mounted["Link"] == '</memory/v1/documents>; rel="successor-version"'
    assert route_path({"path": "/v1/files", "root_path": ""}) == "/v1/files"
    assert set(DEPRECATED_ROUTES) == {("POST", "/v1/files"), ("POST", "/v1/tools/record")}


def test_the_deprecation_window_is_still_open() -> None:
    """When the version reaches the removal release, delete the aliases with it: the header
    spellings, the response header alias, ScopeBody.trace_id, the alias routes."""
    assert Version(__version__) < Version(ALIASES_REMOVED_IN), (
        f"{__version__} has reached {ALIASES_REMOVED_IN}: remove the ADR 0022 aliases"
    )
