"""Cache invalidation follows readers, not only the storage anchor."""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from memory_service.domain.enums import Lifetime, MemoryType, Visibility
from memory_service.domain.evidence import EvidenceRef
from memory_service.domain.revisions import RevisionKind
from memory_service.modules.memory.pipeline import build_memory
from memory_service.modules.memory.revisions import bump_memory_revisions, memory_revision_keys
from memory_service.ports.intelligence import MemoryCandidate
from tests.unit.test_memory_native import CTX

pytestmark = pytest.mark.unit


def memory(keys):
    candidate = MemoryCandidate(
        content="A shared memory belongs to this audience.",
        memory_type=MemoryType.USER,
        lifetime=Lifetime.LONG_TERM,
        visibility=Visibility.USER,
        evidence=[
            EvidenceRef(source_type="message", source_id="message", observed_at=datetime.now(UTC))
        ],
    )
    result = build_memory(candidate, CTX, now=datetime.now(UTC))
    result.system_metadata["visibility_keys"] = keys
    return result


@pytest.mark.parametrize(
    "audience", ["tenant:acme", "agroup:acme/g", "run:acme/r", "runup:acme/p", "thread:acme/thr"]
)
def test_shared_audience_invalidates_other_principals_even_with_user_anchor(audience):
    keys = memory_revision_keys(memory([audience]))
    assert (RevisionKind.USER, "u1") in keys
    assert (RevisionKind.TENANT, "") in keys


@pytest.mark.parametrize("kind", [RevisionKind.USER])
def test_explicit_audience_adds_its_own_revision(kind):
    keys = memory_revision_keys(memory([f"{kind}:acme/reader"]))
    assert (kind, "reader") in keys
    assert (RevisionKind.TENANT, "") not in keys
    assert (kind, "reader") not in memory_revision_keys(memory([f"{kind}:other/reader"]))


async def test_batch_bumps_each_affected_revision_once():
    uow = SimpleNamespace(revisions=SimpleNamespace(bump=AsyncMock()))
    await bump_memory_revisions(uow, [memory(["tenant:acme"])] * 100)
    assert uow.revisions.bump.await_count == 2


@pytest.mark.parametrize(
    "principal,kind,identifier",
    [
        ("user:reader", RevisionKind.USER, "reader"),
        ("agent:reader/research", RevisionKind.AGENT, "research"),
        ("agent:research", RevisionKind.AGENT, "research"),
        ("service:anonymous", RevisionKind.TENANT, ""),
    ],
)
def test_principal_audience_is_independent_of_storage_anchor(principal, kind, identifier):
    assert (kind, identifier) in memory_revision_keys(memory([f"principal:acme/{principal}"]))


def test_a_documents_revisions_are_its_audiences_and_its_thread() -> None:
    from memory_service.domain.revisions import document_revision_keys

    assert document_revision_keys("acme", None, ["user:acme/u1", "principal:acme/user:u1"]) == {
        (RevisionKind.USER, "u1")
    }
    assert document_revision_keys("acme", "thr_1", ["workspace:acme/ws1"]) == {
        (RevisionKind.TENANT, ""),
        (RevisionKind.THREAD, "thr_1"),
    }
    # another tenant's key names no reader here; a document with no audience left is TENANT
    assert document_revision_keys("acme", None, ["user:globex/u1"]) == {(RevisionKind.TENANT, "")}
    assert document_revision_keys("acme", None, []) == {(RevisionKind.TENANT, "")}
