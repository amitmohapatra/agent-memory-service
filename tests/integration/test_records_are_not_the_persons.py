"""An event, an import or a file is a record the request carried, not the person's own words:
what is read from it is kept as what happened, never as the person's rule, preference or
profile line - however it is worded, with a model or without."""

from __future__ import annotations

import pytest
from benchmark.common import submit_observation
from sqlalchemy import select

from memory_service.adapters.db.orm import MemoryRow
from memory_service.domain.enums import ObservationKind
from tests.integration.test_statements_follow_the_user import _chat

pytestmark = pytest.mark.integration

PERSONS = {"rule", "instruction", "preference", "attribute"}


@pytest.mark.parametrize("kind", [ObservationKind.EVENT, ObservationKind.IMPORT])
async def test_a_record_never_becomes_the_persons_rule_or_preference(
    container, uow_factory, kind
) -> None:
    from memory_service.modules.jobs.registry import register_handlers  # noqa: PLC0415

    register_handlers(container)
    ctx = _chat()
    for text in (
        "Never call the customer after 6pm.",
        "I prefer email over phone.",
        "My name is Dana and I work at Globex.",
    ):
        async with uow_factory() as uow:
            await submit_observation(uow, ctx, content=text, kind=kind)
            await uow.commit()
        for _ in range(3):
            await container.tasks.drain()
    async with container.database.session_factory() as s:
        rows = (await s.execute(select(MemoryRow.system_metadata, MemoryRow.memory_type))).all()
    assert rows, "what happened is kept"
    made = {(m or {}).get("category") for m, _ in rows}
    assert not made & PERSONS, made
    assert not {t for _, t in rows} & {"USER", "PREFERENCE"}
    profile = await container.services["profile"].refresh_user("acme", "u1")
    assert profile is None or not profile.text.strip()
    found = await container.services["context_builder"].build(_chat(), "my name and email?")
    assert not [b for b in found.profile if b.text.strip()], "no line of the person's own"
