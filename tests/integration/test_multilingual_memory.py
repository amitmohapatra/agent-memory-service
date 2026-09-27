"""No-LLM ingestion, sparse recall and deletion across scripts, with real storage.

These are functional contracts using hash dense vectors, not model-quality scores.
"""

# ruff: noqa: RUF001 — literal multilingual fixtures intentionally use non-Latin letters.

import pytest

from memory_service.domain.enums import Visibility
from memory_service.domain.observation import ProcessingHints
from tests.integration.test_memory import U1, U2, _memories, _observe

pytestmark = pytest.mark.integration

STATEMENTS = [
    "My office is in Berlin.",
    "Mein Büro befindet sich in Berlin.",
    "Mi oficina está en Madrid.",
    "Biroul meu este în București.",
    "Το γραφείο μου βρίσκεται στην Αθήνα.",
    "Мой офис находится в Москве.",
    "मेरा कार्यालय दिल्ली में है।",
    "يقع مكتبي في القاهرة.",
    "我的办公室在北京。",
    "สำนักงานของฉันอยู่ในกรุงเทพ",
    "Văn phòng của tôi ở Hà Nội.",
    "Benim ofisim İstanbul şehrinde bulunuyor.",
]


@pytest.mark.parametrize("statement", STATEMENTS)
async def test_original_language_survives_ingestion_recall_isolation_and_forgetting(
    container, uow_factory, statement
):
    assert not container.llm.enabled
    await _observe(
        container,
        uow_factory,
        U1,
        statement,
        hints=ProcessingHints(visibility=Visibility.USER),
    )
    memories = await _memories(uow_factory, U1, container)
    original = next(m for m in memories if m.content == statement)
    engine = container.services["retrieval"]
    found = await engine.retrieve(U1, statement, kinds=("memory",))
    assert original.memory_id in {c.record_id for c in found.candidates}
    assert not (await engine.retrieve(U2, statement, kinds=("memory",))).candidates
    async with uow_factory() as uow:
        await container.services["memory"].forget(uow, U1, original.memory_id)
        await uow.commit()
    await container.tasks.drain()
    found = await engine.retrieve(U1, statement, kinds=("memory",))
    assert original.memory_id not in {c.record_id for c in found.candidates}


async def test_similar_transcripts_keep_both_sources_and_exact_repeats_reinforce(
    container, uow_factory
):
    first = "Notre équipe accepte les commandes internationales pour ce produit en Europe."
    corrected = "Notre équipe refuse les commandes internationales pour ce produit en Europe."
    for text in (first, corrected, corrected):
        await _observe(container, uow_factory, U1, text)
    memories = await _memories(uow_factory, U1, container)
    originals = {
        m.content: m for m in memories if m.system_metadata.get("category") == "verbatim_turn"
    }
    assert set(originals) == {first, corrected}
    assert originals[corrected].reinforcement_count > originals[first].reinforcement_count
