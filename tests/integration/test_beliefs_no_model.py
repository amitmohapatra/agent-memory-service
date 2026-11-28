"""Without the model, beliefs change nothing: the rendered context is the memories as said,
byte for byte - and what Caroline said of herself is something she said, never the asker's
own profile line (it was, before a profile was read from the asker's own audiences only).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from benchmark.common import submit_observation

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import Visibility
from memory_service.domain.observation import ProcessingHints
from memory_service.modules.jobs.registry import register_handlers

pytestmark = pytest.mark.integration

ASKER = MemoryExecutionContext(tenant_id="acme", user_id="asker", workspace_id="ws")
SAID = [
    (
        "caroline",
        "I work as a counselor at a youth center.",
        datetime(2023, 5, 8, 14, 0, tzinfo=UTC),
    ),
    (
        "melanie",
        "Caroline told me she loves painting sunsets.",
        datetime(2023, 6, 9, 9, 0, tzinfo=UTC),
    ),
    ("melanie", "I took my kids camping last weekend.", datetime(2023, 6, 9, 9, 5, tzinfo=UTC)),
]

#: ``ContextBuilder.build(ASKER, QUESTION).render()``
QUESTION = "What does Caroline do for work, and what does she love?"
EXPECTED = (
    "## How to read memory\n"
    "- [m1] a memory: something said or learned: cite it by that id, which the memory tools "
    "take.\n"
    "- A memory line: the day it was said (year-month-day, weekday), who said it (their id; "
    "assistant for an agent's reply), what was said.\n"
    "\n"
    "## Memories\n"
    "- [m1] 2023-05-08 Mon caroline: I work as a counselor at a youth center.\n"
    "- [m2] 2023-06-09 Fri melanie: Caroline told me she loves painting sunsets."
)


async def _ingest(container, uow_factory) -> None:
    register_handlers(container)
    for speaker, text, when in SAID:
        async with uow_factory() as uow:
            await submit_observation(
                uow,
                ASKER.model_copy(update={"user_id": speaker}),
                content=text,
                hints=ProcessingHints(visibility=Visibility.TENANT),
                occurred_at=when,
            )
            await uow.commit()
    for _ in range(3):
        await container.tasks.drain()


async def test_the_context_without_a_model_is_todays_byte_for_byte(container, uow_factory) -> None:
    await _ingest(container, uow_factory)
    bundle = await container.services["context_builder"].build(ASKER, QUESTION)
    assert bundle.render() == EXPECTED
