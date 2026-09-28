"""A primary-key collision at insert time is the same 409 a sequential duplicate gets."""

from __future__ import annotations

import pytest
from sqlalchemy.exc import IntegrityError

from memory_service.adapters.db.tenancy_repository import _insert
from memory_service.domain.errors import Conflict

pytestmark = pytest.mark.unit


class _Session:
    def __init__(self, fail: bool) -> None:
        self.fail, self.added = fail, []

    def add(self, row: object) -> None:
        self.added.append(row)

    async def flush(self) -> None:
        if self.fail:
            raise IntegrityError("INSERT", {}, Exception("duplicate key value"))


async def test_a_primary_key_collision_is_a_conflict_not_a_500() -> None:
    with pytest.raises(Conflict, match="tenant acme already exists"):
        await _insert(_Session(fail=True), object(), "tenant acme")  # type: ignore[arg-type]


async def test_a_clean_insert_adds_and_flushes() -> None:
    session = _Session(fail=False)
    row = object()
    await _insert(session, row, "tenant acme")  # type: ignore[arg-type]
    assert session.added == [row]
