"""What a memory is indexed as: the fact with its date and subject, not the bare content.

Built through the real pipeline (extract -> classify -> build_memory) rather than a hand-made
object, so the text reflects what ingestion actually produces.
"""

from __future__ import annotations

from datetime import UTC, datetime

from memory_service.config.constants import MemoryIntelligenceSettings
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import ObservationKind
from memory_service.domain.ids import content_hash
from memory_service.domain.observation import Observation
from memory_service.modules.memory.native import NativeMemoryIntelligence
from memory_service.modules.memory.pipeline import build_memory
from memory_service.modules.rag.indexer import memory_index_text

NOW = datetime(2023, 5, 25, 10, 0, tzinfo=UTC)


async def _memories(
    text: str, ctx: MemoryExecutionContext, kind: ObservationKind = ObservationKind.MESSAGE
):
    native = NativeMemoryIntelligence(MemoryIntelligenceSettings())
    obs = Observation(
        tenant_id=ctx.tenant_id,
        kind=kind,
        content=text,
        content_hash=content_hash(text),
        user_id=ctx.user_id,
        thread_id=ctx.thread_id,
        workspace_id=ctx.workspace_id,
        principal_id=ctx.principal_id,
        message_id="msg_1",
    )
    cands = [await native.classify(c, ctx) for c in await native.extract(obs, ctx)]
    return [build_memory(c, ctx, now=NOW) for c in cands]


async def test_index_text_carries_date_subject_type_and_content() -> None:
    ctx = MemoryExecutionContext(tenant_id="acme", user_id="caroline", workspace_id="ws1")
    fact = (await _memories("I work at ACME Corp.", ctx))[0]
    text = memory_index_text(fact)
    day = fact.temporal.observed_at.date().isoformat()
    assert text.startswith(f"[{day}] caroline: "), text
    assert fact.memory_type.value.lower() + ":" in text
    assert fact.content in text


async def test_opaque_subjects_are_not_repeated_into_the_text() -> None:
    """A thread id says nothing a query could match; only a named subject helps."""
    ctx = MemoryExecutionContext(
        tenant_id="acme", user_id="u1", workspace_id="ws1", thread_id="thr_1"
    )
    decision = (await _memories("Go with Qdrant for retrieval.", ctx, ObservationKind.DECISION))[0]
    assert decision.subject == "thread:thr_1"
    text = memory_index_text(decision)
    assert text.startswith(f"[{decision.temporal.observed_at.date().isoformat()}] "), text
    assert "thr_1" not in text, text


async def test_a_verbatim_turn_is_indexed_with_the_turn_it_answers() -> None:
    """A reply rarely restates its question, and a bare question is never a memory of its
    own, so the question's words reach the index only beside the reply."""
    ctx = MemoryExecutionContext(tenant_id="acme", user_id="caroline", workspace_id="ws1")
    native = NativeMemoryIntelligence(MemoryIntelligenceSettings())
    reply = "Yes, last weekend with my kids, we loved it."
    obs = Observation(
        tenant_id="acme",
        kind=ObservationKind.MESSAGE,
        content=reply,
        content_hash=content_hash(reply),
        user_id="caroline",
        workspace_id="ws1",
        principal_id=ctx.principal_id,
        message_id="msg_2",
    )
    prior = {"source_id": "msg_1", "speaker": "melanie", "text": "Did you go camping?"}
    memories = [
        build_memory(c.model_copy(update={"preceding_turn": prior}), ctx, now=NOW)
        for c in [await native.classify(c, ctx) for c in await native.extract(obs, ctx)]
    ]
    verbatim = [m for m in memories if m.system_metadata.get("category") == "verbatim_turn"]
    assert verbatim, [m.system_metadata.get("category") for m in memories]
    text = memory_index_text(verbatim[0])
    assert text.startswith("melanie: Did you go camping?\n["), text
    assert text.endswith(reply), text
    # the content - what is rendered and returned - is the reply alone
    assert verbatim[0].content == reply
    # extracted facts stay exact
    for m in memories:
        if m is not verbatim[0]:
            assert "camping" not in memory_index_text(m), memory_index_text(m)
