"""The pinned sections of a pushed context: budgeting and rendering."""

from __future__ import annotations

from memory_service.domain.context_bundle import (
    ContextBundle,
    ContextItem,
    ConversationWindow,
    EvidenceReport,
    ProfileBlockView,
    SkillView,
    ThreadSummaryView,
)
from memory_service.domain.enums import EvidenceStatus, QueryType, Representation
from memory_service.domain.tools import MissingArgument, Prefill, ToolCandidate, ToolHints
from memory_service.modules.context.sections import (
    SUMMARY_MIN_TOKENS,
    Pinned,
    ToolsRequest,
    within_budget,
)


def _pinned() -> Pinned:
    return Pinned(
        profile=[ProfileBlockView(block="user", text="name: Ann", version=2)],
        thread_summary=ThreadSummaryView(
            text="Ann ordered paper.", covers_to_sequence=20, version=1
        ),
        procedures=[
            SkillView(
                id="prc_1",
                name="order-paper",
                steps=["erp-get_stock", "erp-create_po"],
                success_rate=0.9,
                support=10,
            )
        ],
        tools=ToolHints(
            candidates=[ToolCandidate(name="erp-create_po", score=1.0, why="step 2")],
            next="erp-create_po",
            prefill={
                "erp-create_po.supplier": Prefill(
                    tool="erp-create_po", value="Acme", source="graph"
                )
            },
            missing=[MissingArgument(tool="erp-create_po", arg="qty", question="How many?")],
        ),
    )


def test_the_pinned_sections_fit_half_the_budget_in_priority_order() -> None:
    kept, used = within_budget(_pinned(), 2000)
    assert kept.profile and kept.thread_summary and kept.procedures and kept.tools
    assert 0 < used <= 1000
    tight, used_tight = within_budget(_pinned(), 30)
    assert tight.profile and not tight.procedures and used_tight <= 15


def test_each_section_costs_what_it_renders_to() -> None:
    """The tools section renders a few candidates, their arguments and what is missing, not
    all twenty: costing the whole hints object dropped it from budgets it fit."""
    from memory_service.domain.context_bundle import tools_section
    from memory_service.modules.ingestion.hierarchy import estimate_tokens

    pinned = _pinned()
    many = pinned.tools.model_copy(  # type: ignore[union-attr]
        update={
            "candidates": [
                ToolCandidate(name=f"tool_{i}", score=0.5, why="matches the task " * 5)
                for i in range(20)
            ]
        }
    )
    cost = estimate_tokens(tools_section(many) or "")
    only_tools = Pinned(tools=many)
    kept, used = within_budget(only_tools, cost * 2)
    assert kept.tools is not None and kept.tools.next == "erp-create_po" and used == cost


def test_a_summary_over_its_share_is_truncated_not_dropped() -> None:
    long = ThreadSummaryView(text="x " * 4000, covers_to_sequence=90, version=3)
    kept, used = within_budget(Pinned(thread_summary=long), 400)
    assert kept.thread_summary is not None
    assert kept.thread_summary.text.startswith("...")
    assert len(kept.thread_summary.text) < len(long.text) and used <= 200
    gone, _ = within_budget(Pinned(thread_summary=long), SUMMARY_MIN_TOKENS)
    assert gone.thread_summary is None, "below the minimum a summary says too little to keep"


def test_a_tools_request_is_part_of_the_cache_identity() -> None:
    assert ToolsRequest().fingerprint() == "tools:8:*"
    assert ToolsRequest(available=["b", "a"], k=3).fingerprint() == "tools:3:a,b"
    assert ToolsRequest(hints=False).fingerprint() == "tools:8:*:nohints"


def _bundle(**extra) -> ContextBundle:
    return ContextBundle(
        query="order paper",
        query_type=QueryType.GENERAL_SEMANTIC,
        conversation=ConversationWindow(thread_id="thr_1", rendered="USER: hi"),
        memories=[
            ContextItem(
                item_id="mem_1",
                representation=Representation.MEMORY,
                text="Acme supplies paper",
                citation="memory_id:mem_1",
            )
        ],
        evidence=EvidenceReport(status=EvidenceStatus.COMPLETE),
        token_budget=2000,
        token_estimate=100,
        **extra,
    )


def test_the_prompt_starts_from_the_pinned_sections() -> None:
    pinned = _pinned()
    rendered = _bundle(
        profile=pinned.profile,
        thread_summary=pinned.thread_summary,
        procedures=pinned.procedures,
        tools=pinned.tools,
    ).render()
    order = ["## Profile", "## Conversation summary", "## Learned skills", "## Tools", "## Recent"]
    positions = [rendered.index(heading) for heading in order]
    assert positions == sorted(positions)
    assert "- order-paper: erp-get_stock -> erp-create_po (worked 90% of 10 runs)" in rendered
    assert (
        "- erp-create_po (confidence 0.00, next step): supplier = 'Acme'; "
        "missing qty: How many?" in rendered
    ), rendered
    assert "step 2" not in rendered, "the why is for the full form, not the prompt"
    assert "[m1] Acme supplies paper" in rendered


def test_a_statement_given_for_a_period_says_its_last_day_and_is_left_out_once_over() -> None:
    """The standing sections read a statement's end from the index (``valid_to``; "no end" is
    indexed as the end of time): its line says the last day it holds, and from the moment
    that day is over it is in no context, before the hourly expiry retires it."""
    from datetime import UTC, datetime, timedelta

    from memory_service.modules.context import standing
    from memory_service.ports.search import PAYLOAD_FIELDS, SearchHit

    now = datetime.now(UTC)

    def hit(valid_to: str | None) -> SearchHit:
        payload = {"text": "Use the outlet channel.", "observed_at": "2026-11-27T09:00:00+00:00"}
        if valid_to is not None:
            payload["valid_to"] = valid_to
        return SearchHit(record_id="mem_1", score=0.5, retriever="dense_en", payload=payload)

    running = hit((now + timedelta(hours=3)).isoformat())
    over = hit((now - timedelta(minutes=1)).isoformat())
    no_end = hit("9999-12-31T00:00:00+00:00")
    assert not standing._ended(running) and standing._ended(over)
    assert not standing._ended(no_end) and not standing._ended(hit(None))
    assert not standing._ended(hit("not a time"))
    # its last day: it lapses when the next one starts
    midnight = hit("2026-11-28T00:00:00+00:00")
    assert standing._dated(midnight, None) == " (said 2026-11-27; until 2026-11-27)"
    assert standing._dated(no_end, None) == " (said 2026-11-27)"
    assert standing.stated("- Use the outlet channel. (said 2026-11-27; until 2026-11-27)") == (
        "Use the outlet channel.",
        "2026-11-27",
    )
    assert "valid_to" in PAYLOAD_FIELDS  # a search returns it, or no section could read it


def test_a_dated_line_says_when_it_was_said_in_the_conversation_the_context_is_for() -> None:
    """``, this conversation`` after the day, for a statement whose message belongs to the
    conversation asked for; a line from another conversation, or a context with none, is as
    it was. The mark is part of the date, not of the statement."""
    from memory_service.modules.context import standing
    from memory_service.ports.search import SearchHit

    def hit(valid_to: str | None = None, **more: object) -> SearchHit:
        payload = {
            "text": "Answer as a table.",
            "observed_at": "2026-11-27T09:00:00+00:00",
            **more,
        }
        if valid_to is not None:
            payload["valid_to"] = valid_to
        return SearchHit(record_id="mem_1", score=0.5, retriever="dense_en", payload=payload)

    here = " (said 2026-11-27, this conversation)"
    assert standing._dated(hit(), None, here=True) == here
    assert standing._dated(hit(), None) == " (said 2026-11-27)"
    until = standing._dated(hit("2026-11-29T00:00:00+00:00"), None, here=True)
    assert until == " (said 2026-11-27, this conversation; until 2026-11-28)"
    # the messages a statement was read from, as indexed; a statement with none has no place
    read = hit(source_refs=[{"source_type": "MESSAGE", "source_id": "o1", "message_id": "msg_1"}])
    assert standing._messages(read) == {"msg_1"}
    assert standing._messages(hit()) == set() == standing._messages(hit(source_refs=["x"]))
    for suffix in (here, until, " (said 2026-11-27)"):
        assert standing.stated(f"- Answer as a table.{suffix}") == (
            "Answer as a table.",
            "2026-11-27",
        )
    block = type(
        "Block", (), {"block": standing.RULES_BLOCK, "text": f"- Answer as a table.{here}"}
    )
    assert standing.listed_rules([block]) == ["Answer as a table."]


async def test_where_a_statement_was_said_is_read_only_when_there_is_something_to_read() -> None:
    """No conversation, or no statement read from a message: no read at all."""
    from memory_service.modules.context import standing
    from memory_service.ports.search import SearchHit

    class Engine:
        def uow_factory(self) -> object:
            raise AssertionError("read without a conversation or a message to look up")

    said = SearchHit(
        record_id="mem_1",
        score=0.5,
        retriever="dense_en",
        payload={"source_refs": [{"message_id": "msg_1"}]},
    )
    unsourced = SearchHit(record_id="mem_2", score=0.5, retriever="dense_en", payload={})
    assert await standing._said_here(Engine(), None, None, [said]) == set()
    assert await standing._said_here(Engine(), None, "thread_a", [unsourced]) == set()
    assert await standing._said_here(Engine(), None, "thread_a", []) == set()
