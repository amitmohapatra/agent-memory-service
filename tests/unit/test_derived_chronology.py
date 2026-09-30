"""Historical evidence remains historical when a summary is built years later."""

from datetime import UTC, datetime

import pytest

from memory_service.domain.enums import MemoryType
from memory_service.modules.context.builder import candidate_to_item
from memory_service.modules.memory.derived import _derived
from memory_service.modules.retrieval.engine import memory_candidate
from tests.unit.test_bundle_rendering import _bundle
from tests.unit.test_llm_memory import CTX, _sources

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "kind", [MemoryType.BELIEF, MemoryType.ENTITY_SUMMARY, MemoryType.PREFERENCE]
)
async def test_source_clock_survives_synthesis_and_context_projection(kind):
    sources = await _sources("I prefer concise answers.", "I prefer bullet points.")
    old, recent, now = (
        datetime(2023, 5, 8, tzinfo=UTC),
        datetime(2023, 6, 9, tzinfo=UTC),
        datetime(2026, 9, 27, tzinfo=UTC),
    )
    for memory, stamp in zip(sources, (old, recent), strict=True):
        memory.temporal = memory.temporal.model_copy(update={"observed_at": stamp})
    derived, keys = _derived(
        CTX,
        memory_type=kind,
        scope=sources[0].scope,
        sources=sources,
        content="The user prefers concise bullet points.",
        subject="user:u1",
        predicate="summary",
        confidence=0.6,
        importance=0.5,
        now=now,
        category="summary",
        extra={},
    )
    assert derived.created_at == now and derived.updated_at == now
    assert derived.temporal.observed_at == recent
    assert derived.temporal.valid_from is None and derived.temporal.valid_to is None
    assert derived.system_metadata["source_observed_from"] == old.isoformat()
    assert derived.system_metadata["source_observed_to"] == recent.isoformat()
    assert {e.observed_at for e in derived.evidence} == {old, recent}
    assert keys == sorted(sources[0].system_metadata["visibility_keys"])
    item = candidate_to_item(memory_candidate(derived, retriever="exact", score=1.0))
    rendered = _bundle([item]).render()
    assert "sources through 2023-06-09 Fri" in rendered
    assert "2026-09-27" not in rendered
