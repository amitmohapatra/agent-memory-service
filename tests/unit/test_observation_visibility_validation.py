"""A visibility the context cannot express must be refused at submission.

Audience keys are built from the context's anchors, so AGENT_GROUP without an agent group
(or WORKSPACE without a workspace, and so on) cannot be expressed. Accepting such an
observation and failing later in ``memory.process_observation`` loses the write silently:
the caller has a 202 and no memory ever appears.
"""

from __future__ import annotations

import pytest

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import ObservationKind, Visibility
from memory_service.domain.errors import ValidationFailed
from memory_service.domain.observation import ProcessingHints
from memory_service.modules.memory.service import MemoryService


def context(**fields) -> MemoryExecutionContext:
    return MemoryExecutionContext(
        tenant_id="acme", request_id="req-1", correlation_id="corr-1", trace_id="t" * 32, **fields
    )


class _Repo:
    def __init__(self) -> None:
        self.added = []

    async def add(self, observation) -> None:
        self.added.append(observation)


class _Uow:
    def __init__(self) -> None:
        self.observations = _Repo()

    async def enqueue(self, spec) -> int:
        return 1


@pytest.mark.parametrize(
    ("visibility", "missing"),
    [
        (Visibility.AGENT_GROUP, "agent_group_id"),
        (Visibility.WORKSPACE, "workspace_id"),
        (Visibility.USER, "user_id"),
        (Visibility.THREAD, "thread_id"),
        (Visibility.WORK, "work_id"),
    ],
)
async def test_unsatisfiable_visibility_is_rejected(visibility, missing):
    service = MemoryService(authz=None)  # authorization is not reached
    with pytest.raises(ValidationFailed) as exc:
        await service.submit_observation(
            _Uow(),
            context(user_id=None),
            kind=ObservationKind.EVENT,
            content="something happened",
            hints=ProcessingHints(visibility=visibility),
        )
    assert missing in str(exc.value)


async def test_a_satisfiable_visibility_is_accepted():
    uow = _Uow()
    service = MemoryService(authz=None)
    ack = await service.submit_observation(
        uow,
        context(user_id="u1"),
        kind=ObservationKind.EVENT,
        content="something happened",
        hints=ProcessingHints(visibility=Visibility.USER),
    )
    assert ack.observation_id
    assert uow.observations.added


async def test_no_visibility_hint_is_left_to_the_pipeline():
    uow = _Uow()
    ack = await MemoryService(authz=None).submit_observation(
        uow, context(), kind=ObservationKind.EVENT, content="something happened"
    )
    assert ack.observation_id
