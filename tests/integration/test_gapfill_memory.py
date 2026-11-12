"""The gap-fill call through durable ingestion: stored, scoped, recalled, replay-safe and
forgettable; and nothing at all without a model."""

import pytest

from memory_service.domain.enums import Visibility
from memory_service.domain.observation import ProcessingHints
from memory_service.modules.memory.gapfill import GapFill
from tests.integration.test_memory import U1, U2, _memories, _observe
from tests.support_llm import mocked_gateway

pytestmark = pytest.mark.integration

FIRST = "I finally adopted a dog last month."
SECOND = "Her name is Biscuit, a beagle mix."
FACT = "u1 adopted a beagle mix named Biscuit."
OPS = {
    "ops": [
        {
            "op": "add",
            "fact": FACT,
            "quote": "Her name is Biscuit, a beagle mix.",
            "type": "fact",
            "subject": "",
            "date": "",
            "confidence": 0.9,
            "target": 0,
        }
    ]
}
USER = ProcessingHints(visibility=Visibility.USER)


def _shape(memories):
    return sorted(
        (m.content, m.memory_type.value, m.system_metadata.get("category"), m.visibility.value)
        for m in memories
    )


async def test_a_gap_fill_fact_is_stored_recalled_replay_safe_and_forgettable(
    container, uow_factory
):
    pipeline = container.services["observation_pipeline"]
    with mocked_gateway([{"ops": []}, OPS]) as gateway:
        pipeline.gapfill = GapFill(gateway.assist(uses=["contextual_extraction"]))
        await _observe(container, uow_factory, U1, FIRST, hints=USER)
        ack = await _observe(container, uow_factory, U1, SECOND, hints=USER)
        sent = gateway.prompts()[-1]["messages"][1]["content"]
    assert FIRST in sent  # the turn before it is read with it
    memories = await _memories(uow_factory, U1, container)
    fact = next(m for m in memories if m.content == FACT)
    assert fact.system_metadata.get("provider") == "llm"
    turn = next(m for m in memories if m.content == SECOND)
    assert fact.visibility is turn.visibility and fact.evidence == turn.evidence
    assert await pipeline.run({"tenant_id": "acme", "observation_id": ack.observation_id}) == []
    assert len(await _memories(uow_factory, U1, container)) == len(memories)
    engine = container.services["retrieval"]
    found = await engine.retrieve(U1, "What breed is my dog Biscuit?", kinds=("memory",))
    assert any(
        fact.memory_id in (c.record_id, *c.payload.get("duplicates", ())) for c in found.candidates
    )
    assert (await engine.retrieve(U2, f"show {fact.memory_id}")).candidates == []
    async with uow_factory() as uow:
        await container.services["memory"].forget(uow, U1, fact.memory_id)
        await uow.commit()
    await container.tasks.drain()
    assert fact.memory_id not in {m.memory_id for m in await _memories(uow_factory, U1, container)}


async def test_without_a_model_the_memories_are_the_rules_alone(container, uow_factory):
    """The no-model path is the rules' path: wired or not, the gap-fill adds nothing and
    asks nothing."""
    pipeline = container.services["observation_pipeline"]
    assert pipeline.gapfill is not None
    for text in (FIRST, SECOND):
        await _observe(container, uow_factory, U1, text, hints=USER)
    wired = _shape(await _memories(uow_factory, U1, container))
    pipeline.gapfill = None
    for text in (FIRST, SECOND):
        await _observe(container, uow_factory, U2, text, hints=USER)
    unwired = _shape(await _memories(uow_factory, U2, container))
    assert wired == unwired
    assert all(category != "source_fact" for _, _, category, _ in wired)


async def test_a_rule_reading_the_model_read_again_is_held_once(container, uow_factory):
    """vdate-4 of the hard eval: the rules read "The audit is due the Friday before
    Christmas." and the model the same words with the day resolved: one memory, the
    model's."""
    said = "The audit is due the Friday before Christmas."
    resolved = "The audit is due on 2026-12-18."
    ops = {
        "ops": [
            {
                "op": "add",
                "fact": resolved,
                "quote": said,
                "type": "plan",
                "subject": "audit",
                "date": "2026-12-18",
                "confidence": 0.9,
                "target": 0,
            }
        ]
    }
    pipeline = container.services["observation_pipeline"]
    with mocked_gateway([ops]) as gateway:
        pipeline.gapfill = GapFill(gateway.assist(uses=["contextual_extraction"]))
        await _observe(container, uow_factory, U1, said, hints=USER)
    readings = [
        m.content
        for m in await _memories(uow_factory, U1, container)
        if m.system_metadata.get("category") != "verbatim_turn" and "audit" in m.content
    ]
    assert readings == [resolved]


class _Says:
    """The NLI head as the frozen one reads spread-2: the one fact holding both sentences
    entails the reply kept with its question (0.97); either half alone does not."""

    representative = True

    def __init__(self, whole: str) -> None:  # "" when no fact holds both
        self.whole = whole

    async def entail_groups(self, groups):
        from memory_service.ports.models import NLIScore

        return [
            [
                NLIScore(entailment=e, neutral=1 - e, contradiction=0.0)
                for e in (0.97 if p == self.whole and "(in reply to:" in h else 0.01 for p in ps)
            ]
            for ps, h in groups
        ]


@pytest.mark.parametrize(
    ("facts", "referred"),
    [
        # the model wrote one fact with both sentences and "it" resolved: that one memory
        (("Ada's second warehouse, in Tacoma, handles all the West Coast returns.",), False),
        # the model split the reply: the turn kept with its question is the one holding both
        (("Ada's second warehouse is in Tacoma.", "The warehouse handles returns."), True),
    ],
    ids=["whole", "split"],
)
async def test_a_reply_the_model_read_whole_is_not_kept_again_with_its_question(
    container, uow_factory, facts, referred
):
    """spread-2 of the hard eval: "In Tacoma. It handles all the West Coast returns." after
    "Where is it?" was kept with the question beside it, and in the runs where the model read
    both sentences as one fact with "it" resolved, the same fact was held twice. Skipping it
    whenever the model read the reply cost recall in every run where the model split it (five
    of five), so it is skipped only when one fact alone says it; the turn with the question
    stays searchable by the turn's own second key (``memory_context_text``)."""
    from datetime import UTC, datetime, timedelta

    from tests.integration.test_referred_turn import _contents, _turn

    reply = "In Tacoma. It handles all the West Coast returns."
    ops = {
        "ops": [
            {
                "op": "add",
                "fact": fact,
                "quote": reply,
                "type": "fact",
                "subject": "",
                "date": "",
                "confidence": 0.9,
                "target": 0,
            }
            for fact in facts
        ]
    }
    at = datetime(2026, 10, 30, 9, tzinfo=UTC)
    pipeline = container.services["observation_pipeline"]
    with mocked_gateway([{"ops": []}, ops]) as gateway:
        pipeline.gapfill = GapFill(
            gateway.assist(uses=["contextual_extraction"]), nli=_Says("" if referred else facts[0])
        )
        await _turn(container, uow_factory, "We opened a second warehouse.", at=at, role="user")
        await _turn(
            container, uow_factory, "Where is it?", at=at + timedelta(minutes=1), role="assistant"
        )
        await _turn(container, uow_factory, reply, at=at + timedelta(minutes=2), role="user")
    kept = await _contents(container, uow_factory)
    assert any("(in reply to: Where is it?)" in m for m in kept) is referred, kept
    assert all(f in kept for f in facts), kept


QUESTION = "How do I keep my living room dust-free, especially with a cat that sheds a lot?"
CAT = "u1 has a cat that sheds a lot."


def _add(fact: str, quote: str) -> dict:
    return {
        "ops": [
            {
                "op": "add",
                "fact": fact,
                "quote": quote,
                "type": "fact",
                "subject": "",
                "date": "",
                "confidence": 0.6,
                "target": 0,
            }
        ]
    }


@pytest.mark.parametrize(
    ("said", "fact", "quote"),
    [
        (QUESTION, CAT, "a cat that sheds a lot"),
        ("I'm vegan.", "u1 is vegan.", "I'm vegan."),
    ],
)
async def test_the_model_reads_a_message_whatever_its_shape(
    container, uow_factory, said: str, fact: str, quote: str
):
    """A question, or a statement of a few words, is a message a person wrote like any other:
    the rules keep nothing of the question (no verbatim turn either) and little of the short
    one, and the model reads both and decides what they hold. What it returns is stored
    against the message, as the user's."""
    pipeline = container.services["observation_pipeline"]
    with mocked_gateway([_add(fact, quote)]) as gateway:
        pipeline.gapfill = GapFill(gateway.assist(uses=["contextual_extraction"]))
        ack = await _observe(container, uow_factory, U1, said, hints=USER)
        assert gateway.route.call_count == 1
        assert said in gateway.prompts()[-1]["messages"][1]["content"]
    read = next(m for m in await _memories(uow_factory, U1, container) if m.content == fact)
    assert read.system_metadata.get("provider") == "llm"
    assert read.visibility is Visibility.USER and read.owner_principal == "user:u1"
    assert [e.source_id for e in read.evidence] == [ack.observation_id]


async def test_without_a_model_a_question_forms_nothing_and_asks_nothing(container, uow_factory):
    """The container's own gap-fill, with no model configured: wired, and it asks nothing."""
    assert container.services["observation_pipeline"].gapfill is not None
    with mocked_gateway([_add(CAT, "a cat that sheds a lot")]) as gateway:
        await _observe(container, uow_factory, U1, QUESTION, hints=USER)
        assert gateway.route.call_count == 0
    assert await _memories(uow_factory, U1, container) == []
