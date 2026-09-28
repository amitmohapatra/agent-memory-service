"""Cursor pagination (ADR 0023): one convention for every list route.

A page is asked for with ``cursor`` (opaque, copied from the previous page) and ``limit``.
The response carries ``Link: <url>; rel="next"`` (RFC 8288) exactly when a next page exists,
and envelope bodies also carry ``next_cursor`` so a client that only sees the body can page.

The cursor is base64url JSON of the keyset the repository ordered by. It is checked on the
way in, so a malformed or foreign cursor is a 422 and never a database error, and it is not
signed: it only names a position, and every query is scoped by the caller's tenant anyway.
"""

from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import Annotated, Any, Final

from fastapi import Query, Request, Response

from memory_service.domain.errors import ValidationFailed

LINK_HEADER: Final = "Link"
NEXT_REL: Final = "next"
CURSOR_MAX_CHARS: Final = 1024
_SCALARS: Final = (str, int, float)

CursorQuery = Annotated[
    str | None,
    Query(
        max_length=CURSOR_MAX_CHARS,
        description="Opaque position of the next page, copied from the previous response "
        '(its `next_cursor` or `Link: <...>; rel="next"`). Omit for the first page.',
    ),
]


def encode_cursor(position: Mapping[str, Any]) -> str:
    raw = json.dumps(dict(position), sort_keys=True, separators=(",", ":"), default=str)
    return base64.urlsafe_b64encode(raw.encode()).decode().rstrip("=")


def decode_cursor(
    cursor: str | None, *, fields: Mapping[str, type] | Sequence[str]
) -> dict[str, Any] | None:
    """The position a cursor names, with exactly ``fields`` as members converted to their
    declared types (``str``, ``int`` or timezone-aware ``datetime``), else 422. A sequence of
    names declares string members."""
    if cursor is None or cursor == "":
        return None
    spec: Mapping[str, type] = fields if isinstance(fields, Mapping) else dict.fromkeys(fields, str)
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        position = json.loads(base64.urlsafe_b64decode(padded.encode()).decode())
    except (binascii.Error, UnicodeDecodeError, ValueError) as exc:
        raise ValidationFailed("invalid cursor", details={"cursor": cursor[:64]}) from exc
    if not isinstance(position, dict) or set(position) != set(spec):
        raise ValidationFailed("invalid cursor", details={"cursor": cursor[:64]})
    try:
        return {name: _member(position[name], kind) for name, kind in spec.items()}
    except (TypeError, ValueError) as exc:
        raise ValidationFailed("invalid cursor", details={"cursor": cursor[:64]}) from exc


def _member(value: Any, kind: type) -> Any:
    if isinstance(value, bool) or not isinstance(value, _SCALARS):
        raise TypeError("a cursor member is a scalar")
    if kind is datetime:
        instant = datetime.fromisoformat(str(value))
        if instant.tzinfo is None:
            raise ValueError("a cursor instant is timezone-aware")
        return instant
    if kind is int:
        if not isinstance(value, int) or value < 0:
            raise TypeError("a cursor sequence is a non-negative integer")
        return value
    if not isinstance(value, str):
        raise TypeError("a cursor identifier is a string")
    return value


def page[T](
    rows: Sequence[T], *, limit: int, position: Callable[[T], Mapping[str, Any]]
) -> tuple[list[T], str | None]:
    """Split ``limit + 1`` fetched rows into the page and the cursor of the row after it.

    Repositories are asked for one row more than the page: its presence is the proof that a
    next page exists, and the cursor names the position *after the last row returned*. With
    a keyset of immutable columns a row written between two requests is neither skipped nor
    duplicated; a keyset on a bare instant (the read audit) can skip rows sharing it."""
    items = list(rows[:limit])
    if len(rows) <= limit or not items:
        return items, None
    return items, encode_cursor(position(items[-1]))


def next_link(request: Request, cursor: str) -> str:
    url = request.url.include_query_params(cursor=cursor)
    return f'<{url}>; rel="{NEXT_REL}"'


def link_next(request: Request, response: Response, cursor: str | None) -> None:
    """Set the ``Link`` header when there is a next page; leave it absent otherwise."""
    if cursor is not None:
        response.headers[LINK_HEADER] = next_link(request, cursor)
