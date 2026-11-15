"""``OpenFGAAuthorizationProvider.read``: every tuple of one object, over OpenFGA's Read pages
(what a deleted thread's revocation deletes), and an error that says what failed."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from memory_service.adapters.authz.openfga_provider import (
    READ_PAGE_SIZE,
    OpenFGAAuthorizationProvider,
)
from memory_service.domain.errors import DependencyUnavailable
from memory_service.ports.authorization import RelationTuple

pytestmark = pytest.mark.unit

THREAD = "thread:acme/thr1"


def _tuple(user: str, relation: str) -> Any:
    return SimpleNamespace(key=SimpleNamespace(user=user, relation=relation, object=THREAD))


class _Client:
    """OpenFGA's Read, two pages long; or failing as a gateway that refuses it does."""

    def __init__(self, *, fails: bool = False) -> None:
        self.fails = fails
        self.asked: list[tuple[str | None, dict[str, Any]]] = []
        self.pages = {
            "": ([_tuple("tenant:acme", "tenant"), _tuple("user:u1", "owner")], "next"),
            "next": ([_tuple("agent:u1/bot", "participant")], ""),
        }

    async def read(self, body: Any, options: dict[str, Any]) -> Any:
        self.asked.append((body.object, dict(options)))
        if self.fails:
            raise ValueError("store not found")
        tuples, token = self.pages[str(options.get("continuation_token", ""))]
        return SimpleNamespace(tuples=tuples, continuation_token=token)


def _provider(client: _Client) -> OpenFGAAuthorizationProvider:
    """The instance without its constructor: this exercises the read, not the connection."""
    provider = OpenFGAAuthorizationProvider.__new__(OpenFGAAuthorizationProvider)

    async def connected() -> _Client:
        return client

    provider._get_client = connected  # type: ignore[method-assign]
    return provider


async def test_every_page_of_an_objects_tuples_is_read() -> None:
    client = _Client()
    assert await _provider(client).read(THREAD) == [
        RelationTuple(user="tenant:acme", relation="tenant", object=THREAD),
        RelationTuple(user="user:u1", relation="owner", object=THREAD),
        RelationTuple(user="agent:u1/bot", relation="participant", object=THREAD),
    ]
    assert client.asked == [
        (THREAD, {"page_size": READ_PAGE_SIZE}),
        (THREAD, {"page_size": READ_PAGE_SIZE, "continuation_token": "next"}),
    ]


async def test_a_read_openfga_refuses_says_what_failed() -> None:
    with pytest.raises(DependencyUnavailable, match="OpenFGA read failed") as failed:
        await _provider(_Client(fails=True)).read(THREAD)
    assert failed.value.details == {"object": THREAD}
