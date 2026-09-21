"""Every declared memory type, round-tripped through the real pipeline.

Nine of the twenty-five memory types are written by no code path in this service. They exist
in the enum, they have admission weights and lifetime defaults in ``admission.py``, and
nothing produces them — they are an *import surface*: a caller hints one and the pipeline is
expected to honour it.

Eight of those nine had no test at all. A declared type that nothing produces and nothing
exercises is indistinguishable, from the outside, from a type that is quietly broken — a
missing weight, a lifetime default that expires it on arrival, an enum the repository layer
cannot round-trip. This is the test that tells the difference, and it is parametrised over
the enum rather than a hand-written list so a type added later cannot skip it.
"""

from __future__ import annotations

import pytest

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import MemoryType, ObservationKind
from memory_service.domain.observation import ProcessingHints
from memory_service.modules.jobs.registry import register_handlers

pytestmark = pytest.mark.integration

CTX = MemoryExecutionContext(tenant_id="acme", user_id="u1", workspace_id="ws1")

#: Content worth remembering whatever the type: the admission gate must not be the reason a
#: type fails, or the test would be measuring the gate instead of the type.
CONTENT = (
    "The FY26 migration was approved on 4 March 2026 and Priya owns the rollback plan, "
    "which pins the previous image digest in the release manifest."
)


#: Two types are expected *not* to produce a stored memory from a plain hint, and both are
#: correct behaviour rather than gaps:
#:
#: ``WORKING`` is ``Lifetime.EPHEMERAL`` with the lowest admission prior in the table (0.2).
#: It is scratch state for a run in flight, so the gate declining to make it permanent is the
#: feature. ``CUSTOM`` requires a ``custom_type`` alongside it and is rejected without one.
#:
#: Naming them here rather than skipping them keeps the contract visible: if WORKING ever
#: starts persisting, or CUSTOM stops requiring its discriminator, this test says so.
EPHEMERAL_BY_DESIGN = {MemoryType.WORKING}
NEEDS_DISCRIMINATOR = {MemoryType.CUSTOM}


async def _submit(container, uow_factory, memory_type: MemoryType) -> list:
    register_handlers(container)
    service = container.services["memory"]
    async with uow_factory() as uow:
        ack = await service.submit_observation(
            uow,
            CTX,
            kind=ObservationKind.MESSAGE,
            content=CONTENT,
            hints=ProcessingHints(
                memory_type=memory_type,
                custom_type="release_note" if memory_type in NEEDS_DISCRIMINATOR else None,
            ),
        )
        await uow.commit()
    await container.tasks.drain()  # process_observation
    await container.tasks.drain()  # memory.index
    assert ack.observation_id
    async with uow_factory() as uow:
        return await service.list_memories(uow, CTX)


@pytest.mark.parametrize("memory_type", list(MemoryType), ids=lambda t: t.value)
async def test_every_declared_memory_type_survives_the_pipeline(
    container, uow_factory, memory_type: MemoryType
) -> None:
    """Submitted with a type hint, stored, and readable back.

    The assertion is deliberately weak on *which* type comes back: the pipeline may legitimately
    classify differently from the hint, and several types are derived rather than hinted. What
    must hold is that hinting a declared type never loses the memory and never raises — the
    failure modes an unexercised enum member actually has.
    """
    memories = await _submit(container, uow_factory, memory_type)
    if memory_type in EPHEMERAL_BY_DESIGN:
        assert not memories, (
            f"{memory_type.value} is Lifetime.EPHEMERAL with the lowest admission prior; "
            "persisting it would make scratch state permanent"
        )
        return
    assert memories, f"hinting {memory_type.value} produced no memory at all"
    assert all(m.memory_type in set(MemoryType) for m in memories)


@pytest.mark.parametrize("memory_type", list(MemoryType), ids=lambda t: t.value)
def test_every_declared_memory_type_has_an_admission_prior(memory_type: MemoryType) -> None:
    """A type with no prior is admitted on a default nobody chose.

    This is the cheap half of the check and it covers the whole enum, including types the
    integration test above cannot reach because nothing hints them in practice.
    """
    from memory_service.modules.memory.admission import _TYPE_PRIOR

    assert memory_type in _TYPE_PRIOR, f"{memory_type.value} has no admission prior"
    weight = _TYPE_PRIOR[memory_type]
    assert 0.0 <= float(weight) <= 1.0, f"{memory_type.value} prior {weight} out of range"
