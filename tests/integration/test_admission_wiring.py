"""The admission gate on the write path, behind its tenant switch (ADR 0032).

The gate scores each extracted candidate; the switch is ``Tenant.admission_gate``, off by
default, because the retrieval gates were measured with every candidate kept. Extraction is
stood in for here so each test controls exactly which candidates reach the gate.
"""

from __future__ import annotations

import secrets
from datetime import UTC, datetime
from typing import Any

import pytest
from benchmark.common import submit_observation

from memory_service.config.constants import MEMORY_INTELLIGENCE
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import DedupDecision, Lifetime, MemoryType
from memory_service.domain.evidence import EvidenceRef
from memory_service.domain.tenancy import Tenant
from memory_service.modules.memory.admission import AdmissionGate
from memory_service.modules.memory.pipeline import ObservationPipeline
from memory_service.ports.intelligence import ConsolidationOutcome, MemoryCandidate

pytestmark = pytest.mark.integration

WORTH_KEEPING = MemoryCandidate(
    content="The Berlin office moved to Hamburg.",
    memory_type=MemoryType.USER,
    lifetime=Lifetime.LONG_TERM,
    subject="user:u1",
    confidence=0.9,
    importance=0.8,
)
CHATTER = MemoryCandidate(
    content="brb, on my way",
    memory_type=MemoryType.CONVERSATION,
    lifetime=Lifetime.SHORT_TERM,
    confidence=0.2,
    importance=0.1,
)


class FixedExtraction:
    """Extracts the given candidates from any observation and creates each one."""

    def __init__(self, *candidates: MemoryCandidate) -> None:
        self.candidates = candidates

    async def extract(self, observation: Any, ctx: Any) -> list[MemoryCandidate]:
        seen = EvidenceRef(
            source_type="message",
            source_id=observation.observation_id,
            observed_at=observation.occurred_at,
        )
        return [c.model_copy(update={"evidence": [seen]}) for c in self.candidates]

    async def classify(self, candidate: MemoryCandidate, ctx: Any) -> MemoryCandidate:
        return candidate

    async def consolidate(
        self, candidate: MemoryCandidate, existing: Any, ctx: Any
    ) -> ConsolidationOutcome:
        return ConsolidationOutcome(decision=DedupDecision.CREATE, candidate=candidate)


async def _run(uow_factory, *, admission_gate: bool | None) -> tuple[list, list]:
    """Process one observation for a fresh tenant; ``None`` means the tenant has no row."""
    tenant_id = f"gate-{secrets.token_hex(4)}"
    ctx = MemoryExecutionContext(tenant_id=tenant_id, user_id="u1")
    async with uow_factory() as uow:
        if admission_gate is not None:
            await uow.tenants.add(
                Tenant(tenant_id=tenant_id, name="Gate", admission_gate=admission_gate)
            )
        observed = await submit_observation(uow, ctx, content="brb. Berlin moved to Hamburg.")
        await uow.commit()
    pipeline = ObservationPipeline(
        uow_factory,
        FixedExtraction(WORTH_KEEPING, CHATTER),  # type: ignore[arg-type]
        settings=MEMORY_INTELLIGENCE,
        gate=AdmissionGate(MEMORY_INTELLIGENCE),
    )
    outcomes = await pipeline.run(
        {"tenant_id": tenant_id, "observation_id": observed.observation_id}
    )
    async with uow_factory() as uow:
        stored = await uow.memories.list_recent(
            since=datetime(2000, 1, 1, tzinfo=UTC), tenant_id=tenant_id
        )
    return outcomes, stored


async def test_a_tenant_with_the_gate_on_stores_only_what_it_admits(uow_factory) -> None:
    outcomes, stored = await _run(uow_factory, admission_gate=True)
    assert [m.content for m in stored] == [WORTH_KEEPING.content]
    admission = stored[0].system_metadata["admission"]
    assert admission["verdict"] == "ADMIT" and admission["score"] >= 0.4
    assert {"worthiness", "novelty", "confidence", "expected_utility"} <= admission.keys()
    rejected = next(o for o in outcomes if o.candidate.content == CHATTER.content)
    assert rejected.decision is DedupDecision.IGNORE
    assert rejected.reason.startswith("reject: ")


@pytest.mark.parametrize("admission_gate", [False, None], ids=["switched-off", "no-tenant-row"])
async def test_without_the_switch_every_candidate_is_stored(
    uow_factory, admission_gate: bool | None
) -> None:
    """The default, and the development tenant's (it has no row): unchanged behaviour."""
    outcomes, stored = await _run(uow_factory, admission_gate=admission_gate)
    assert {m.content for m in stored} == {WORTH_KEEPING.content, CHATTER.content}
    assert all("admission" not in m.system_metadata for m in stored)
    assert all(o.decision is DedupDecision.CREATE for o in outcomes)
