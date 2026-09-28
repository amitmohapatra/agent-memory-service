"""One cursor convention (ADR 0023): opaque, validated, and announced in a Link header."""

from __future__ import annotations

import pytest
from fastapi import FastAPI, Request, Response
from fastapi.testclient import TestClient

from memory_service.api.pagination import (
    LINK_HEADER,
    decode_cursor,
    encode_cursor,
    link_next,
    next_link,
    page,
)
from memory_service.domain.errors import ValidationFailed
from trellis.memory.transport import next_cursor


def test_a_cursor_round_trips_and_only_its_own_fields_are_accepted() -> None:
    cursor = encode_cursor({"updated_at": "2026-09-28T07:00:00+00:00", "memory_id": "mem_1"})
    assert "=" not in cursor and "+" not in cursor and "/" not in cursor
    assert decode_cursor(cursor, fields=("updated_at", "memory_id")) == {
        "updated_at": "2026-09-28T07:00:00+00:00",
        "memory_id": "mem_1",
    }
    assert decode_cursor(None, fields=("x",)) is None
    assert decode_cursor("", fields=("x",)) is None
    with pytest.raises(ValidationFailed):  # a cursor from another route
        decode_cursor(cursor, fields=("tenant_id",))


@pytest.mark.parametrize(
    "bad",
    [
        "not base64 at all!",
        encode_cursor({"a": [1, 2]}),  # a list is not a scalar position
        encode_cursor({"a": True}),  # neither is a boolean
        "W10",  # "[]": not an object
        "eyJhIjoxfQ.extra",
    ],
)
def test_a_malformed_cursor_is_a_validation_error_not_a_database_error(bad: str) -> None:
    with pytest.raises(ValidationFailed) as exc:
        decode_cursor(bad, fields=("a",))
    assert exc.value.http_status == 422


def test_members_are_converted_to_their_declared_types_or_refused() -> None:
    from datetime import UTC, datetime

    spec = {"created_at": datetime, "sequence": int, "id": str}
    good = encode_cursor({"created_at": "2026-09-28T07:00:00+00:00", "sequence": 3, "id": "k"})
    assert decode_cursor(good, fields=spec) == {
        "created_at": datetime(2026, 9, 28, 7, tzinfo=UTC),
        "sequence": 3,
        "id": "k",
    }
    for bad in (
        {"created_at": "garbage", "sequence": 3, "id": "k"},
        {"created_at": "2026-09-28T07:00:00", "sequence": 3, "id": "k"},  # naive instant
        {"created_at": "2026-09-28T07:00:00+00:00", "sequence": -1, "id": "k"},
        {"created_at": "2026-09-28T07:00:00+00:00", "sequence": "3", "id": "k"},
        {"created_at": "2026-09-28T07:00:00+00:00", "sequence": 3, "id": 7},
    ):
        with pytest.raises(ValidationFailed):
            decode_cursor(encode_cursor(bad), fields=spec)


def test_page_keeps_limit_rows_and_names_the_position_after_the_last_one() -> None:
    rows = [{"id": i} for i in range(4)]
    items, cursor = page(rows, limit=3, position=lambda r: {"id": r["id"]})
    assert [r["id"] for r in items] == [0, 1, 2]
    assert decode_cursor(cursor, fields={"id": int}) == {"id": 2}
    items, cursor = page(rows[:3], limit=3, position=lambda r: {"id": r["id"]})
    assert len(items) == 3 and cursor is None  # exactly a page: no proof of more
    assert page([], limit=3, position=lambda r: {"id": 0}) == ([], None)


def test_the_link_header_repeats_the_query_with_the_cursor_replaced() -> None:
    app = FastAPI()

    @app.get("/v1/things")
    def things(request: Request, response: Response, cursor: str | None = None, limit: int = 2):
        link_next(request, response, None if cursor == "last" else "last")
        return {"cursor": cursor}

    with TestClient(app) as client:
        first = client.get("/v1/things?limit=2&kind=a")
        link = first.headers[LINK_HEADER]
        assert link.startswith("<http://testserver/v1/things?limit=2&kind=a&cursor=last>")
        assert link.endswith('; rel="next"')
        assert next_cursor(link) == "last"
        last = client.get("/v1/things?limit=2&kind=a&cursor=last")
        assert LINK_HEADER not in last.headers
        assert next_cursor(last.headers.get(LINK_HEADER)) is None


def test_the_sdk_reads_the_next_cursor_out_of_any_link_header_shape() -> None:
    assert next_cursor('<https://x/v1/keys?cursor=abc&limit=2>; rel="next"') == "abc"
    assert next_cursor("<https://x/v1/keys?cursor=abc>; rel=next") == "abc"
    assert next_cursor('<https://x/docs>; rel="help", <https://x/v1/k?cursor=z>; rel="next"') == "z"
    assert next_cursor('<https://x/v1/keys?cursor=abc>; rel="prev"') is None
    assert next_cursor(None) is None


def test_next_link_keeps_the_mount_prefix_a_proxy_put_in_the_path() -> None:
    """Behind ``--root-path /memory`` the ASGI path carries the prefix (ASGI 2.x), and the
    link must point back through the proxy, not at the bare route."""
    app = FastAPI(root_path="/memory")

    @app.get("/v1/things")
    def things(request: Request):
        return {"link": next_link(request, "c")}

    with TestClient(app) as client:
        link = client.get("/memory/v1/things").json()["link"]
        assert link.startswith("<http://testserver/memory/v1/things?cursor=c>")
