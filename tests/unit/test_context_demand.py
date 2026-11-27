"""``demand.standing``: a bundle with what stands and without what retrieval ranked."""

from __future__ import annotations

from datetime import UTC, datetime

from memory_service.domain.context_bundle import (
    ContextBundle,
    ContextItem,
    ConversationWindow,
    EarlierChatLine,
    EvidenceReport,
    ProfileBlockView,
    UnusedEvidence,
)
from memory_service.domain.enums import EvidenceStatus, MessageRole, QueryType, Representation
from memory_service.modules.context.demand import NOTE, RANKED, SAID, standing
from memory_service.modules.retrieval.engine import UNUSED_MAX


def _item(item_id: str, text: str, **attributes: object) -> ContextItem:
    return ContextItem(
        item_id=item_id,
        representation=Representation.MEMORY,
        text=text,
        citation=f"memory_id:{item_id}",
        token_estimate=10,
        attributes=attributes,
    )


def _bundle(**sections: object) -> ContextBundle:
    return ContextBundle(
        query="which stores are short?",
        query_type=QueryType.GENERAL_SEMANTIC,
        conversation=ConversationWindow(),
        evidence=EvidenceReport(
            status=EvidenceStatus.COMPLETE,
            unused=[UnusedEvidence(item_id="mem_out", kind="memory", text="ranked out")],
        ),
        token_budget=1000,
        token_estimate=400,
        profile=[ProfileBlockView(block="standing_rules", text="- Answer as a table.", version=1)],
        **sections,  # type: ignore[arg-type]
    )


def test_what_is_ranked_is_left_and_what_stands_is_untouched() -> None:
    said = EarlierChatLine(
        message_id="msg_1",
        role=MessageRole.ASSISTANT,
        text="Stores 12 and 40 were short.",
        thread_id="thread_old",
        said_at=datetime(2026, 11, 1, tzinfo=UTC),
    )
    before = _bundle(
        memories=[_item("mem_1", "The tote is short at store 12."), _item("mem_2", "Dentist.")],
        history=[_item("mem_3", "It was short at store 9.")],
        graph_facts=[_item("f_1", "tote short_at store 12")],
        knowledge=[_item("chk_1", "Stock policy.")],
        summaries=[_item("sum_1", "The stock report.")],
        earlier_chats=[said],
    )
    after = standing(before)
    for name in (*RANKED, SAID):
        assert getattr(after, name) == [], name
    assert after.profile == before.profile and after.conversation == before.conversation
    assert after.evidence.status is EvidenceStatus.COMPLETE, (
        "what memory holds, not what is carried"
    )
    assert [u.item_id for u in after.evidence.unused] == ["mem_out", "mem_1", "mem_2"]
    assert after.diagnostics[NOTE] == {
        "carried": False,
        "left": {
            "memories": 2,
            "history": 1,
            "graph_facts": 1,
            "knowledge": 1,
            "summaries": 1,
            "earlier_chats": 1,
        },
    }
    assert 0 < after.token_estimate <= before.token_estimate - 60


def test_nothing_ranked_leaves_the_bundle_as_it_was_but_for_the_note() -> None:
    before = _bundle()
    after = standing(before)
    assert after.diagnostics[NOTE] == {"carried": False, "left": {}}
    assert after.token_estimate == before.token_estimate
    assert after.evidence == before.evidence


def test_the_unused_evidence_stays_bounded_and_takes_no_unverified_reading() -> None:
    many = [_item(f"mem_{n}", f"memory {n}") for n in range(UNUSED_MAX + 3)]
    after = standing(_bundle(memories=many))
    assert len(after.evidence.unused) == UNUSED_MAX
    # a model's unverified restatement of a memory is no evidence to check an answer against
    restated = _item("mem_r", "restated", provider="llm")
    kept = standing(_bundle(memories=[restated, _item("mem_k", "kept")]))
    assert [u.item_id for u in kept.evidence.unused] == ["mem_out", "mem_k"]


def test_the_estimate_never_goes_below_nothing() -> None:
    before = _bundle(memories=[_item(f"mem_{n}", "x") for n in range(60)])
    assert standing(before).token_estimate == 0
