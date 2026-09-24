"""The best-ranked evidence sits at both ends of the prompt, not only the head.

arXiv 2307.03172 measures retrieval accuracy at 75.8 / 53.8 / 63.2 per cent when the needed
passage is first / in the middle / last. This renderer already put the best-ranked few first
and kept the timeline chronological, but the block was a fixed ten against a bundle of a
hundred - so ninety per cent of the ranked set existed only in the trough, inside ~21,000
characters of date-ordered prose.

Read from a real run, that is exactly what the failures looked like: sixteen of fifty-two
misses declined with the evidence present. Two pairs prove the fact was reachable and the
answerer simply never saw it - "which events has Jon participated in" answered "networking
events, one on 20 June 2023" while "when did Jon visit networking events" answered "I don't
know", same run, same corpus.
"""

from __future__ import annotations

import pytest

from memory_service.domain.context_bundle import (
    MOST_RELEVANT_MAX,
    MOST_RELEVANT_TAIL,
    SHOWN_ABOVE,
    _most_relevant_count,
)

pytestmark = pytest.mark.unit


def test_the_block_is_a_share_of_the_bundle_not_a_fixed_ten() -> None:
    """The defect: at judged depth the block was a tenth of what it was choosing from."""
    assert _most_relevant_count(100) > MOST_RELEVANT_MAX
    assert _most_relevant_count(100) / 100 >= 0.3


def test_a_small_bundle_keeps_the_behaviour_it_had() -> None:
    assert _most_relevant_count(10) == MOST_RELEVANT_MAX
    assert _most_relevant_count(5) == MOST_RELEVANT_MAX


def test_the_share_scales_with_depth() -> None:
    """Halving the depth must not leave the block at the same absolute size."""
    assert _most_relevant_count(50) < _most_relevant_count(100)


def _bundle(n: int):
    from memory_service.domain.context_bundle import (
        ContextBundle,
        ContextItem,
        Representation,
    )

    items = [
        ContextItem(
            item_id=f"m{i}",
            representation=Representation.MEMORY,
            text=f"memory number {i} about the project",
            citation=f"M{i}",
            score=1.0 - i / 1000,
            attributes={"observed_at": f"2026-01-{(i % 28) + 1:02d}T00:00:00Z"},
        )
        for i in range(n)
    ]
    from memory_service.domain.context_bundle import ConversationWindow, EvidenceReport
    from memory_service.domain.enums import EvidenceStatus, QueryType

    return ContextBundle(
        query="q",
        query_type=QueryType.GENERAL_SEMANTIC,
        conversation=ConversationWindow(),
        memories=items,
        evidence=EvidenceReport(status=EvidenceStatus.COMPLETE),
        token_budget=12000,
        token_estimate=0,
    )


def test_the_best_evidence_appears_at_both_ends(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tail repeat is an ablation knob, off by default; this pins what it does when on."""
    import memory_service.domain.context_bundle as cb

    monkeypatch.setattr(cb, "REPEAT_MOST_RELEVANT_AT_END", True)
    rendered = _bundle(100).render()
    assert "## Most relevant" in rendered
    assert "## Most relevant, again" in rendered
    head = rendered.index("## Most relevant")
    timeline = rendered.index("## Memories")
    tail = rendered.index("## Most relevant, again")
    assert head < timeline < tail, "the repeat must come after the timeline, near the question"


def test_the_top_memory_is_in_both_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    import memory_service.domain.context_bundle as cb

    monkeypatch.setattr(cb, "REPEAT_MOST_RELEVANT_AT_END", True)
    rendered = _bundle(100).render()
    head_block = rendered.split("## Memories")[0]
    tail_block = rendered.split("## Most relevant, again")[1]
    assert "memory number 0 " in head_block
    assert "memory number 0 " in tail_block


def test_the_timeline_still_points_rather_than_duplicating() -> None:
    """Promotion is token-neutral: the timeline keeps its shape as pointers."""
    rendered = _bundle(100).render()
    timeline = rendered.split("## Memories")[1]
    assert timeline.count(SHOWN_ABOVE) == _most_relevant_count(100)


def test_the_tail_is_a_reminder_not_a_second_bundle(monkeypatch: pytest.MonkeyPatch) -> None:
    import memory_service.domain.context_bundle as cb

    monkeypatch.setattr(cb, "REPEAT_MOST_RELEVANT_AT_END", True)
    rendered = _bundle(100).render()
    tail_block = rendered.split("## Most relevant, again")[1]
    assert tail_block.count("- [") == MOST_RELEVANT_TAIL
