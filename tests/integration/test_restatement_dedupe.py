"""A user restating what they said, in other words, keeps one memory of it; two different facts
keep two. Through the real write path (the hard extraction eval's ``dup-1`` and ``dup-2``).

"My favourite cuisine is Thai." is read as a preference, anchored to the user; "Thai food is
my favourite, by the way." as a fact anchored to the conversation, with the same audience (the
user). Compared only within its own anchor, the restatement became a second memory. The
hermetic suite's encoder and NLI head are stand-ins that read no meaning, so the provider gets
the frozen models' scores for these sentences; the thresholds are the real ones.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import pytest

from memory_service.domain.context import MemoryExecutionContext
from memory_service.ports.models import NLIScore
from tests.integration.test_memory import _memories, _observe

pytestmark = pytest.mark.integration

#: (said first, said later, cosine, entailment each way) as the frozen encoder and head read them
DUP_1 = ("My favourite cuisine is Thai.", "Thai food is my favourite, by the way.", 0.85, 0.99)
DUP_2 = (
    "I work at Shopify as a data engineer.",
    "As I said, I'm a data engineer at Shopify.",
    0.87,
    0.96,
)
APART = ("My favourite cuisine is Thai.", "My favourite drink is Thai iced tea.", 0.80, 0.0)


class Frozen:
    """The frozen encoder and NLI head for one pair; every other text is unrelated."""

    representative = True

    def __init__(self, first: str, later: str, cos: float, entail: float) -> None:
        self.vectors = {first: [1.0, 0.0], later: [cos, math.sqrt(1 - cos * cos)]}
        self.pairs = {(first, later), (later, first)}
        self.entail = entail

    def fingerprint(self) -> str:
        return "frozen-scores"

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.vectors.get(t, [0.0, 0.0]) for t in texts]

    async def entail_groups(
        self, groups: Sequence[tuple[Sequence[str], str]]
    ) -> list[list[NLIScore]]:
        out = []
        for premises, hypothesis in groups:
            row = []
            for premise in premises:
                e = self.entail if (premise, hypothesis) in self.pairs else 0.0
                row.append(NLIScore(entailment=e, neutral=1 - e, contradiction=0.0))
            out.append(row)
        return out


async def _readings(container, uow_factory, pair: tuple, user: str) -> list[str]:
    ctx = MemoryExecutionContext(
        tenant_id="acme", user_id=user, workspace_id="ws1", thread_id=f"thr_{user}"
    )
    provider = container.services["memory_provider"]
    frozen = Frozen(*pair)
    provider.embedding, provider.nli = frozen, frozen
    for text in pair[:2]:
        await _observe(container, uow_factory, ctx, text)
    return [
        m.content
        for m in await _memories(uow_factory, ctx, container)
        if m.system_metadata.get("category") != "verbatim_turn"
    ]


@pytest.mark.parametrize("pair", [DUP_1, DUP_2], ids=["dup-1", "dup-2"])
async def test_a_restatement_in_other_words_keeps_one_memory(container, uow_factory, pair) -> None:
    readings = await _readings(container, uow_factory, pair, user=f"u_{len(pair[1])}")
    assert readings == [pair[0]], readings


async def test_two_different_facts_keep_two_memories(container, uow_factory) -> None:
    readings = await _readings(container, uow_factory, APART, user="u_apart")
    assert APART[0] in readings and APART[1] in readings, readings
