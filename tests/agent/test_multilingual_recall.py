"""Twelve languages, through the SDK alone: what a team writes in its own script, it reads
back in its own script, nobody else reads it, and forgetting is forgetting.

    a statement observed in each of the twelve languages is the top recall for its own
        question, and the context bundle carries it
    another tenant's key reads none of them
    a forgotten memory is gone on the next recall

Hermetic under CI (the hash encoder and Unicode BM25 carry the lexical match); with
``MEMORY_TEST_PROVIDERS=env`` in the runtime image the same suite runs on the frozen
encoders, and the recall then rests on the multilingual dense space as well.
"""

# ruff: noqa: RUF001 - literal multilingual fixtures intentionally use non-Latin letters.

from __future__ import annotations

import pytest

from tests.agent.conftest import BOOTSTRAP, sdk

pytestmark = pytest.mark.e2e

#: (language, the statement a team member observes, the question its colleague asks)
LANGUAGES = [
    ("en", "Our office is in Berlin, next to the main station.", "Where is our office?"),
    (
        "de",
        "Unser Büro befindet sich in Berlin, neben dem Hauptbahnhof.",
        "Wo befindet sich unser Büro?",
    ),
    (
        "es",
        "Nuestra oficina está en Madrid, junto a la estación de Atocha.",
        "¿Dónde está nuestra oficina?",
    ),
    ("ro", "Biroul nostru este în București, lângă Gara de Nord.", "Unde este biroul nostru?"),
    ("tr", "Ofisimiz İstanbul'da, Haydarpaşa garının yanında.", "Ofisimiz nerede?"),
    (
        "vi",
        "Văn phòng của chúng tôi ở Hà Nội, cạnh ga trung tâm.",
        "Văn phòng của chúng tôi ở đâu?",
    ),
    (
        "el",
        "Το γραφείο μας βρίσκεται στην Αθήνα, δίπλα στον σταθμό Λαρίσης.",
        "Πού βρίσκεται το γραφείο μας;",
    ),
    ("ru", "Наш офис находится в Москве, рядом с Казанским вокзалом.", "Где находится наш офис?"),
    ("hi", "हमारा कार्यालय दिल्ली में मुख्य स्टेशन के पास है।", "हमारा कार्यालय कहाँ है?"),
    ("ar", "يقع مكتبنا في القاهرة بجوار محطة القطار الرئيسية.", "أين يقع مكتبنا؟"),
    ("zh", "我们的办公室在北京，靠近北京站。", "我们的办公室在哪里？"),
    ("th", "สำนักงานของเราอยู่ในกรุงเทพ ใกล้สถานีรถไฟหัวลำโพง", "สำนักงานของเราอยู่ที่ไหน"),
]


def _texts(items) -> list[str]:
    return [getattr(item, "text", "") or "" for item in items]


@pytest.mark.parametrize(
    ("language", "statement", "question"), LANGUAGES, ids=[lang for lang, _, _ in LANGUAGES]
)
async def test_a_team_reads_back_its_own_language(
    app, running, language, statement, question
) -> None:
    platform = sdk(app, BOOTSTRAP)
    tenant = await platform.admin.create_tenant(f"Team {language}", tenant_id=f"team-{language}")
    admin = sdk(app, tenant.admin_key.token)
    service = await admin.tenant.keys.issue("service", f"harness-{language}")
    harness = sdk(app, service.token)
    writer = harness.bind(user_id="writer")
    ack = await writer.observe(statement, kind="MESSAGE")
    assert ack.observation_id

    # the same-language question finds the statement first, and the bundle carries it
    recalled = await writer.recall(question, kinds=["memory"])
    assert recalled and statement in _texts(recalled)[:1], _texts(recalled)[:3]
    bundle = await writer.context(question)
    assert statement in _texts(bundle.memories)

    # another tenant, another key: none of it
    other = await platform.admin.create_tenant(f"Other {language}", tenant_id=f"other-{language}")
    stranger = sdk(app, other.admin_key.token).bind(user_id="writer")
    assert statement not in _texts(await stranger.recall(question, kinds=["memory"]))

    # forgetting is forgetting
    memory = next(item for item in recalled if getattr(item, "text", "") == statement)
    await writer.forget(memory.item_id)
    assert statement not in _texts(await writer.recall(question, kinds=["memory"]))
