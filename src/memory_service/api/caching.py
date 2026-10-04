"""Conditional GETs: an ``ETag`` on a read whose answer a client polls, and ``304 Not
Modified`` when the client already holds it (RFC 9110 §8.8.3, §13.1.2).

The tag is a digest of the response bytes, so it changes exactly when the answer does and
needs no counter kept in step with every write that could change it. It is weak (``W/``):
the compression middleware may re-encode the bytes, and a weak tag still validates the
representation a client cached. Only GETs use it; a 304 carries the tag and the
``Cache-Control`` again and no body.
"""

from __future__ import annotations

import hashlib
from typing import Final

from fastapi import Request, Response
from pydantic import BaseModel

from memory_service.api.headers import CACHE_CONTROL_HEADER, ETAG_HEADER, IF_NONE_MATCH_HEADER

JSON_MEDIA_TYPE: Final = "application/json"
NOT_MODIFIED: Final = 304


def etag_of(body: bytes) -> str:
    """The weak tag of ``body``: 128 bits of BLAKE2b, quoted as RFC 9110 asks."""
    return f'W/"{hashlib.blake2b(body, digest_size=16).hexdigest()}"'


def matches(if_none_match: str | None, etag: str) -> bool:
    """Whether ``If-None-Match`` names ``etag`` (weak comparison) or is ``*``."""
    if not if_none_match:
        return False
    wanted = etag.removeprefix("W/")
    for candidate in if_none_match.split(","):
        tag = candidate.strip()
        if tag == "*" or tag.removeprefix("W/") == wanted:
            return True
    return False


def conditional(
    request: Request,
    body: bytes,
    *,
    etag: str | None = None,
    cache_control: str,
    headers: dict[str, str] | None = None,
) -> Response:
    """``body`` as JSON with its ``ETag``, or a bodiless 304 when the client holds it."""
    tag = etag or etag_of(body)
    out = {**(headers or {}), ETAG_HEADER: tag, CACHE_CONTROL_HEADER: cache_control}
    if matches(request.headers.get(IF_NONE_MATCH_HEADER), tag):
        return Response(status_code=NOT_MODIFIED, headers=out)
    return Response(content=body, media_type=JSON_MEDIA_TYPE, headers=out)


def conditional_model(
    request: Request,
    model: BaseModel,
    *,
    cache_control: str,
    headers: dict[str, str] | None = None,
) -> Response:
    """:func:`conditional` for a response model, serialised the way FastAPI would."""
    return conditional(
        request, model.model_dump_json().encode(), cache_control=cache_control, headers=headers
    )
