"""Graph pointers resolve canonical memory evidence without inheriting relation access."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import Lifetime, MemoryType, ScopeLevel, TemporalStatus, Visibility
from memory_service.domain.evidence import EvidenceRef
from memory_service.domain.memory import CanonicalMemory, Scope, TemporalState
from memory_service.modules.authz.visibility import VisibilitySpecification
from memory_service.modules.context.builder import candidate_to_item
from memory_service.modules.graph.retrieval import GraphStage
from memory_service.modules.memory.native import normalized_hash

pytestmark = pytest.mark.integration
CTX = MemoryExecutionContext(tenant_id="acme", user_id="u1")
VIS = VisibilitySpecification(tenant_id="acme", keys=frozenset({"tenant:acme"}))
NOW = datetime.now(UTC)


def memory(identifier, **changes):
    content = f"Evidence for {identifier}"
    return CanonicalMemory(
        memory_id=identifier,
        tenant_id="acme",
        scope=Scope(tenant_id="acme", level=ScopeLevel.TENANT),
        visibility=Visibility.TENANT,
        owner_principal="user:u1",
        lifetime=Lifetime.LONG_TERM,
        memory_type=MemoryType.SEMANTIC,
        content=content,
        normalized_hash=normalized_hash(content),
        temporal=TemporalState(observed_at=NOW, valid_from=NOW - timedelta(days=2)),
        evidence=[EvidenceRef(source_type="message", source_id="msg_original", observed_at=NOW)],
    ).model_copy(update=changes)


async def test_batch_hydration_checks_canonical_access_lifecycle_and_lineage(uow_factory):
    rows = [
        memory("mem_allowed"),
        memory("mem_private", visibility=Visibility.PRIVATE),
        memory("mem_deleted", deleted_at=NOW),
        memory(
            "mem_expired", system_metadata={"expires_at": (NOW - timedelta(days=1)).isoformat()}
        ),
        memory(
            "mem_old",
            temporal=TemporalState(
                observed_at=NOW - timedelta(days=2),
                valid_from=NOW - timedelta(days=2),
                valid_to=NOW - timedelta(days=1),
                status=TemporalStatus.SUPERSEDED,
            ),
        ),
        memory(
            "mem_future",
            temporal=TemporalState(
                observed_at=NOW,
                valid_from=NOW + timedelta(days=1),
            ),
        ),
    ]
    async with uow_factory() as uow:
        for row in rows:
            keys = (
                ["principal:acme/user:u2"]
                if row.visibility is Visibility.PRIVATE
                else ["tenant:acme"]
            )
            await uow.memories.add(row, visibility_keys=keys)
        await uow.memories.forget("acme", "mem_deleted")
        await uow.commit()
    stage = GraphStage(AsyncMock(), uow_factory)
    ids = [row.memory_id for row in rows] + ["mem_missing"]
    found = await stage._expand_memories(CTX, ids, VIS, as_of=None)
    assert [c.record_id for c in found] == ["mem_allowed"]
    assert found[0].expansion_edge == "GRAPH_EVIDENCE"
    assert candidate_to_item(found[0]).evidence == rows[0].evidence
    historical = await stage._expand_memories(CTX, ids, VIS, as_of=NOW - timedelta(hours=36))
    assert [c.record_id for c in historical] == ["mem_allowed", "mem_old"]
    # Even a corrupted cross-tenant pointer cannot resolve through the canonical query.
    foreign = CTX.model_copy(update={"tenant_id": "other"})
    assert await stage._expand_memories(foreign, ids, VIS, as_of=None) == []
