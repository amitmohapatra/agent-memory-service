"""Claims enter the NLI model together, not one at a time.

``entail`` takes a single hypothesis, so a cascade with N claims entered the model N times:
N acquisitions of the one-caller gate, N tokenizer calls, N forward passes of a handful of
rows. ``max_claims`` defaults to 40, so that is up to forty where one would do, and the
verified-context endpoint is the slowest thing the service can be asked to do - 42 s p50 and
96 s max under a ten-user load where the same retrieval takes 172 ms sequentially.

The last test is the one that would catch a regression: it counts entries into the model.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from memory_service.adapters.models.nli import LexicalNLI
from memory_service.config.constants import NLISettings
from memory_service.modules.grounding.cascade import Evidence, GroundingCascade
from memory_service.ports.models import NLIScore

pytestmark = pytest.mark.unit


class _CountingNLI(LexicalNLI):
    """The lexical scorer, counting how many times the model is entered."""

    def __init__(self) -> None:
        self.entries = 0
        self.pairs = 0

    async def entail(self, premises: Sequence[str], hypothesis: str) -> list[NLIScore]:
        self.entries += 1
        self.pairs += len(premises)
        return await super().entail(premises, hypothesis)

    async def entail_groups(
        self, groups: Sequence[tuple[Sequence[str], str]]
    ) -> list[list[NLIScore]]:
        self.entries += 1
        self.pairs += sum(len(p) for p, _ in groups)
        return await super().entail_groups(groups)


async def test_groups_score_the_same_as_one_call_each() -> None:
    """Batching is an execution change, not a scoring change."""
    nli = LexicalNLI()
    groups = [
        (["the sky is blue", "grass is green"], "the sky is blue"),
        (["cats purr"], "cats purr"),
    ]
    batched = await nli.entail_groups(groups)
    one_at_a_time = [await nli.entail(premises, hypothesis) for premises, hypothesis in groups]
    assert batched == one_at_a_time


async def test_results_stay_aligned_with_their_group() -> None:
    nli = LexicalNLI()
    groups = [(["a b c"], "a b c"), ([], "nothing"), (["x y", "a b c"], "a b c")]
    scored = await nli.entail_groups(groups)
    assert [len(s) for s in scored] == [1, 0, 2], "a group's scores must not leak into the next"


async def test_no_groups_is_not_an_entry_into_the_model() -> None:
    assert await LexicalNLI().entail_groups([]) == []


async def test_a_cascade_enters_the_model_once_for_many_claims() -> None:
    """The optimisation itself. Three claims, one entry."""
    nli = _CountingNLI()
    cascade = GroundingCascade(nli=nli, assist=None, settings=NLISettings())
    evidence = [
        Evidence(item_id="e1", text="Revenue was EUR 412 million in FY26."),
        Evidence(item_id="e2", text="Adjusted EBITDA rose 8 percent."),
        Evidence(item_id="e3", text="Headcount reached 1200 people."),
    ]
    answer = (
        "Revenue was EUR 412 million in FY26. Adjusted EBITDA rose 8 percent. "
        "Headcount reached 1200 people."
    )
    report = await cascade.verify(answer, evidence)
    assert len(report.claims) == 3, "three assertive sentences should decompose to three claims"
    assert nli.entries == 1, f"entered the model {nli.entries} times for 3 claims; expected 1"
