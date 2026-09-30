"""``api/headers.py``: the wire header names and the one-value rule on scope and credential
headers. Each header has one spelling: the pre-0.2 ``X-Memory-*`` spellings were removed in
0.3.0 and are not read at all."""

from __future__ import annotations

import pytest
from starlette.datastructures import Headers

from memory_service.api.headers import (
    refuse_ambiguous_headers,
    require_one_value,
    scope_header,
    scope_values,
)
from memory_service.config.constants import HEADERS
from memory_service.domain.errors import ValidationFailed


def test_the_removed_spellings_are_not_read() -> None:
    old = Headers({"X-Memory-Tenant": "globex", "X-Memory-User": "u9"})
    assert scope_values(old, HEADERS.tenant) == []
    assert require_one_value(old, HEADERS.user) is None
    mixed = Headers({"X-Trellis-Tenant": "acme", "X-Memory-Tenant": "globex"})
    assert require_one_value(mixed, HEADERS.tenant) == "acme"
    refuse_ambiguous_headers(mixed)


def test_a_duplicated_header_is_refused() -> None:
    """A proxy that appends rather than replaces leaves the client's line in front."""
    twice = Headers(raw=[(b"x-trellis-user", b"attacker"), (b"x-trellis-user", b"stamped")])
    assert scope_values(twice, HEADERS.user) == ["attacker", "stamped"]
    assert scope_header(twice, HEADERS.user) == "attacker"  # the lenient reader, for counting
    with pytest.raises(ValidationFailed, match="X-Trellis-User was sent more than once"):
        require_one_value(twice, HEADERS.user)
    with pytest.raises(ValidationFailed):
        refuse_ambiguous_headers(twice)
    same = Headers(raw=[(b"x-trellis-user", b"u1"), (b"x-trellis-user", b"u1")])
    assert require_one_value(same, HEADERS.user) == "u1"


def test_an_empty_header_is_no_header() -> None:
    blank = Headers(raw=[(b"x-trellis-tenant", b"  "), (b"x-trellis-tenant", b"acme")])
    assert scope_values(blank, HEADERS.tenant) == ["acme"]
    assert require_one_value(blank, HEADERS.tenant) == "acme"
    assert require_one_value(Headers({"X-Trellis-Tenant": ""}), HEADERS.tenant) is None


def test_two_different_credentials_are_refused_before_anyone_picks_one() -> None:
    """The limiter keys on the first credential header, the authenticator would verify the
    last: a request that sends two is refused before either reads it."""
    two_keys = Headers(raw=[(b"x-api-key", b"mk_a.1"), (b"x-api-key", b"mk_b.2")])
    with pytest.raises(ValidationFailed, match="X-API-Key was sent more than once"):
        refuse_ambiguous_headers(two_keys)
    repeated = Headers(raw=[(b"authorization", b"Bearer t"), (b"authorization", b"Bearer t")])
    refuse_ambiguous_headers(repeated)


def test_a_header_is_read_plainly() -> None:
    assert scope_header(Headers({"X-API-Key": "k"}), HEADERS.api_key) == "k"
    assert scope_header(Headers({}), HEADERS.api_key) is None
    assert require_one_value(Headers({}), HEADERS.user) is None
