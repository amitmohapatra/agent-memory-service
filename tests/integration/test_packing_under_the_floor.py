"""What is packed from under the encoder's relevance floor, through the real write and read
paths.

A question about several conversations ("How many bikes do I own?") is close, in the encoder's
space, to some of the turns that answer it and far from others ("The saddle that came with it
was too narrow ..."): those ranked well - the same session, the same words - and were then
dropped by the floor, with most of the token budget unused. The floor is there for the
question the store cannot answer, which must come back nearly empty. So the few best-ranked
candidates under the floor are packed after everything above it, and only when the question's
best match is close (``related_evidence_relevance``): a question nothing answers still gets
nothing from under the floor.

The hermetic suite's encoder is a stand-in with no calibrated cosine and no floor; these tests
give it the shipped multilingual encoder's (0.2), so the packer decides as it does in a
deployment.
"""

from __future__ import annotations

import pytest

from memory_service.adapters.models.embeddings import HashEmbedding
from memory_service.domain.enums import MessageRole
from tests.integration.test_statements_follow_the_user import _chat, _say

pytestmark = pytest.mark.integration

FLOOR = 0.2
CLOSE = 0.4

#: five conversations, two things said in each
SAID = (
    (
        "I bought a Trek road bike for my commute to the office in March.",
        "The bike shop on Mill Lane fitted mudguards and a rack for free.",
    ),
    (
        "The saddle that came with it was too narrow, so they swapped it for a wider one.",
        "My sister Hanna is visiting from Oslo next month.",
    ),
    (
        "Our kitchen renovation starts after the summer holidays.",
        "The dentist moved my appointment to Thursday morning.",
    ),
    (
        "I finished the online course on watercolour painting.",
        "My neighbour borrowed the ladder again last weekend.",
    ),
    (
        "The library extended my loan of the gardening books.",
        "We booked a cabin by the lake for the autumn break.",
    ),
)


@pytest.fixture
async def said(container, uow_factory, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(HashEmbedding, "relevance_floor", FLOOR)
    for conversation in SAID:
        chat = _chat()
        for statement in conversation:
            await _say(container, uow_factory, chat, statement)
            await _say(container, uow_factory, chat, "Noted.", role=MessageRole.ASSISTANT)


async def _packed(container, asked: str):
    bundle = await container.services["context_builder"].build(_chat(), asked)
    return bundle, [memory.relevance for memory in bundle.memories]


@pytest.mark.usefixtures("said")
async def test_the_next_ranked_under_the_floor_are_packed_after_the_rest_when_a_match_is_close(
    container,
) -> None:
    bundle, relevance = await _packed(
        container, "Which road bike did I buy for my commute to the office?"
    )
    assert max(relevance) >= CLOSE, "the question is about something the user said"
    under = [r for r in relevance if r < FLOOR]
    assert len(under) == 5, "five of the seven ranked candidates under the floor, no more"
    assert all(r >= FLOOR for r in relevance[:-5]), "after everything above the floor"
    assert relevance[-5:] == under
    assert bundle.diagnostics["below_relevance_floor"] == 2, "the two left out are counted"


@pytest.mark.usefixtures("said")
async def test_nothing_under_the_floor_is_packed_when_no_match_is_close(container) -> None:
    bundle, relevance = await _packed(container, "Which bike did I buy?")
    assert relevance, "something clears the floor"
    assert max(relevance) < CLOSE
    assert all(r >= FLOOR for r in relevance)
    assert bundle.diagnostics["below_relevance_floor"] >= 5

    bundle, relevance = await _packed(container, "What should I cook for dinner tonight?")
    assert relevance == [], "a question the store cannot answer comes back empty"
    assert bundle.diagnostics["below_relevance_floor"] >= 5
