"""A context says how to read what it shows: what each handle is ([m1] a memory, [s1] a
summary, [d1] a document passage, [h1] an earlier value), what a memory line holds (the day it
was said, who said it, the dates it names worked out, ``a..b`` a range) and what a marker means
- only for what that context carries. The handles and the markers were never explained, so a
model reading "[h1] 2023-05-08 Mon (earlier value, since replaced) caroline: ..." had to guess."""

from __future__ import annotations

from datetime import UTC, datetime

from memory_service.domain.context_bundle import (
    HANDLE_MEANINGS,
    HANDLE_PREFIXES,
    LEGEND_HEADING,
    ContextBundle,
    ContextItem,
    ConversationWindow,
    EvidenceReport,
)
from memory_service.domain.enums import EvidenceStatus, QueryType, Representation
from memory_service.domain.evidence import EvidenceRef, EvidenceSource


def _item(item_id: str, text: str, **attributes: object) -> ContextItem:
    return ContextItem(
        item_id=item_id,
        representation=Representation.MEMORY,
        text=text,
        citation=f"memory_id:{item_id}",
        token_estimate=10,
        evidence=[
            EvidenceRef(
                source_type=EvidenceSource.MESSAGE,
                source_id=f"msg_{item_id}",
                message_id=f"msg_{item_id}",
                observed_at=datetime(2023, 5, 8, tzinfo=UTC),
            )
        ],
        attributes={
            "observed_at": "2023-05-08T10:00:00+00:00",
            "subject": "user:caroline",
            **attributes,
        },
    )


def _bundle(**sections: list[ContextItem]) -> ContextBundle:
    return ContextBundle(
        query="When did Caroline go to the support group?",
        query_type=QueryType.GENERAL_SEMANTIC,
        conversation=ConversationWindow(),
        evidence=EvidenceReport(status=EvidenceStatus.COMPLETE),
        token_budget=1000,
        token_estimate=100,
        **sections,
    )


def _legend(rendered: str) -> str:
    head = f"## {LEGEND_HEADING}\n"
    assert head in rendered, rendered
    return rendered.split(head, 1)[1].split("\n## ", 1)[0]


def test_every_handle_kind_has_a_meaning() -> None:
    assert HANDLE_MEANINGS.keys() == HANDLE_PREFIXES.keys()


def test_a_memory_line_and_its_worked_out_dates_are_explained() -> None:
    said = _item("mem_1", "I went to a support group yesterday.",
                 dated_mentions=[{"text": "yesterday", "date": "2023-05-07"}])  # fmt: skip
    legend = _legend(_bundle(memories=[said]).render())
    assert "[m1]" in legend and HANDLE_MEANINGS["m"] in legend
    assert "the day it was said" in legend and "who said it" in legend
    assert "a..b" in legend
    assert HANDLE_MEANINGS["s"] not in legend and "unverified" not in legend


def test_a_marker_is_explained_only_where_it_is_shown() -> None:
    read = _item("mem_2", "Caroline went to a support group.", category="source_fact",
                 provider="llm")  # fmt: skip
    replaced = _item("mem_3", "Caroline lives in Ohio.", earlier_value=True)
    legend = _legend(_bundle(memories=[read], history=[replaced]).render())
    rendered_line_marks = _bundle(memories=[read], history=[replaced]).render()
    assert ("model-extracted" in legend) == ("model-extracted" in rendered_line_marks)
    assert "[h1]" in legend and HANDLE_MEANINGS["h"] in legend


def test_a_context_with_no_memory_has_no_legend() -> None:
    assert LEGEND_HEADING not in _bundle().render()
