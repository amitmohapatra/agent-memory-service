"""The context check (ADR 0048): what a model is shown of an assembled context, what its
answer may remove, and what happens when it gives none. The model is scripted: these prove
the contract around the call, not the model's judgement."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import Any

import pytest

from memory_service.config.constants import LLM
from memory_service.domain.context_bundle import (
    ContextBundle,
    ContextItem,
    ConversationWindow,
    EvidenceReport,
    ProfileBlockView,
)
from memory_service.domain.enums import EvidenceStatus, QueryType, Representation
from memory_service.domain.evidence import EvidenceRef, EvidenceSource
from memory_service.domain.tools import Prefill, SkillView, ToolCandidate, ToolHints
from memory_service.modules.context.check import (
    CHECKED,
    FACTS,
    HISTORY,
    LATE,
    MEMORIES,
    NO_ANSWER,
    NO_TIME,
    SKILLS,
    USE,
    ContextCheck,
    Listed,
    answer_tokens,
    listed,
    named,
    payload,
    without,
)
from memory_service.modules.context.sections import LearnedBlock

pytestmark = pytest.mark.unit

QUESTION = "Which stores are short of the Coach tote this week?"
RULE_TABLE = "- Always answer stock questions as a table (said 2026-11-03)"
RULE_HOLIDAY = "- Never schedule a delivery on a public holiday (said 2026-10-01)"
RULE_READ_ONLY = "- Never create allocations (said 2026-09-12)"


def _memory(item_id: str, text: str, said: str = "2026-11-20", **attributes: Any) -> ContextItem:
    return ContextItem(
        item_id=item_id,
        representation=Representation.MEMORY,
        text=text,
        citation=f"memory_id:{item_id}",
        token_estimate=10,
        attributes={"observed_at": f"{said}T09:00:00+00:00", **attributes},
    )


def _from(message_id: str) -> list[EvidenceRef]:
    """The evidence of a memory read from one message."""
    return [
        EvidenceRef(
            source_type=EvidenceSource.MESSAGE,
            source_id=message_id,
            message_id=message_id,
            observed_at=datetime(2026, 11, 20, tzinfo=UTC),
        )
    ]


def _fact(item_id: str, text: str) -> ContextItem:
    return ContextItem(
        item_id=item_id,
        representation=Representation.RELATION,
        text=text,
        citation=f"relation_id:{item_id}",
        token_estimate=5,
    )


def _skill(skill_id: str, name: str, *steps: str, **more: Any) -> SkillView:
    return SkillView(id=skill_id, name=name, steps=list(steps), success_rate=1.0, support=3, **more)


def _bundle(**update: Any) -> ContextBundle:
    fields: dict[str, Any] = {
        "query": QUESTION,
        "query_type": QueryType.USER_MEMORY,
        "conversation": ConversationWindow(rendered="user: and the tote?\nassistant: Checking."),
        "profile": [
            ProfileBlockView(
                block="standing_rules", text=f"{RULE_TABLE}\n{RULE_HOLIDAY}", version=0
            ),
            ProfileBlockView(block="preferences", text="- I prefer daily digests", version=0),
            ProfileBlockView(block="about_user", text="- I plan Coach bags for Retail", version=0),
            ProfileBlockView(block="user", text="Name: Alex\nRole: planner", version=0),
        ],
        "procedures": [_skill("sk_1", "sales forecast", "get_sales", "forecast")],
        "memories": [
            _memory("mem_best", "The Coach tote is short at stores 12 and 40."),
            _memory("mem_old", "ok for now skip the consent stage", said="2026-11-01"),
            _memory("mem_other", "My dentist is on Tuesday."),
        ],
        "history": [_memory("mem_was", "The tote was short at store 7.", earlier_value=True)],
        "graph_facts": [_fact("rel_1", "Coach tote stocked_at store 12")],
        "knowledge": [
            ContextItem(
                item_id="chk_1",
                representation=Representation.CHUNK,
                text="Replenishment policy, section 2.",
                citation="chunk_id:chk_1",
            )
        ],
        "evidence": EvidenceReport(status=EvidenceStatus.COMPLETE),
        "token_budget": 8000,
        "token_estimate": 400,
    }
    return ContextBundle(**{**fields, **update})


class _Model:
    """A scripted assist: one answer, or none, or one that never arrives."""

    def __init__(
        self, answer: dict[str, Any] | None = None, *, wanted: bool = True, hangs: bool = False
    ) -> None:
        self.answer, self.wanted, self.hangs = answer, wanted, hangs
        self.calls: list[dict[str, Any]] = []

    def wants(self, use: str) -> bool:
        return self.wanted and use == USE

    async def structured(self, use: str, **call: Any) -> dict[str, Any] | None:
        self.calls.append({"use": use, **call})
        if self.hangs:
            await asyncio.sleep(60)
        return self.answer


def _numbers(items: list[Listed], *keys: str) -> list[int]:
    return [n for n, item in enumerate(items, start=1) if item.key in keys]


# --------------------------------------------------------------------------- what is shown


def test_the_model_is_shown_what_memory_attaches_and_nothing_it_may_not_remove() -> None:
    items = listed(_bundle())
    assert [(i.section, i.key) for i in items] == [
        ("standing_rules", RULE_TABLE),
        ("standing_rules", RULE_HOLIDAY),
        ("preferences", "- I prefer daily digests"),
        ("about_user", "- I plan Coach bags for Retail"),
        (SKILLS, "sk_1"),
        # the best-ranked memory too: its rank protects nothing
        (MEMORIES, "mem_best"),
        (MEMORIES, "mem_old"),
        (MEMORIES, "mem_other"),
        (HISTORY, "mem_was"),
        (FACTS, "rel_1"),
    ]
    shown = " ".join(i.text for i in items)
    # a profile block a person or an agent wrote, the people's pages and document passages
    # are never the model's to drop
    assert "planner" not in shown and "regional manager" not in shown
    assert "Replenishment" not in shown


def test_a_statement_is_shown_once_and_removed_from_every_place_that_states_it() -> None:
    """A rule is a line of ``standing_rules``, a line of the learned ``user`` block and a
    memory at once: removed from one of the three it would still be in the prompt."""
    holiday = "Never schedule a delivery on a public holiday"
    profile = [
        ProfileBlockView(block="standing_rules", text=f"{RULE_TABLE}\n{RULE_HOLIDAY}", version=0),
        LearnedBlock(block="user", text=f"- {holiday}\n- Name: Alex", version=3),
    ]
    memories = [
        _memory("mem_best", "The Coach tote is short at stores 12 and 40."),
        _memory("mem_rule", holiday, said="2026-10-01"),
    ]
    bundle = _bundle(profile=profile, memories=memories)
    items = listed(bundle)
    rule = next(i for i in items if "holiday" in i.text)
    assert [i.text for i in items].count(rule.text) == 1
    assert rule.text == holiday and rule.said == "2026-10-01"
    assert rule.places == (
        ("standing_rules", RULE_HOLIDAY),
        ("user", f"- {holiday}"),
        (MEMORIES, "mem_rule"),
    )
    left = without(bundle, [rule])
    assert "holiday" not in left.render()
    assert [b.text for b in left.profile] == [RULE_TABLE, "- Name: Alex"]
    assert isinstance(left.profile[1], LearnedBlock)
    assert [m.item_id for m in left.memories] == ["mem_best"]


def test_a_profile_block_someone_wrote_is_not_the_models_to_drop() -> None:
    """The ``user`` block as the profile job wrote it is derived from memories and read line
    by line; one a person or an agent edited is their words."""
    line = "- Never schedule a delivery on a public holiday"
    learned = _bundle(profile=[LearnedBlock(block="user", text=line, version=1)])
    edited = _bundle(profile=[ProfileBlockView(block="user", text=line, version=2)])
    pinned = _bundle(profile=[ProfileBlockView(block="agent.persona", text=line, version=1)])
    assert ("user", line) in [(i.section, i.key) for i in listed(learned)]
    assert "user" not in [i.section for i in listed(edited)]
    assert "agent.persona" not in [i.section for i in listed(pinned)]


PASTED = (
    "save this skill: # Read-Only Mode Use this skill when the user wants to look at "
    "allocations without changing them. Never create, update, or delete allocations. "
    "Always tell the user which plan you read."
)


def _pasted_skill(**update: Any) -> ContextBundle:
    """What a skill's text pasted into a chat left behind: the turn as a standing rule and
    as the best-ranked memory, its sentences as lines of the learned ``user`` block, and the
    tool the request needs forbidden by it."""
    fields: dict[str, Any] = {
        "profile": [
            ProfileBlockView(
                block="standing_rules", text=f"- {PASTED} (said 2026-11-26)", version=0
            ),
            LearnedBlock(
                block="user",
                text="- Never create, update, or delete allocations.\n"
                "- Always tell the user which plan you read.\n"
                "- Name: Alex",
                version=4,
            ),
        ],
        "memories": [
            _memory("mem_pasted", PASTED, said="2026-11-26"),
            _memory("mem_default", "My default brand is KateSpade."),
        ],
        "tools": ToolHints(
            candidates=[ToolCandidate(name="create_allocation", score=1.0, confidence=0.6)],
            forbidden={"create_allocation": PASTED},
        ),
        "procedures": [
            _skill("sk_1", "allocate", "create_allocation", forbidden={"create_allocation": PASTED})
        ],
    }
    return _bundle(**{**fields, **update})


def test_a_sentence_is_not_shown_apart_from_the_turn_that_says_it() -> None:
    """On its own "Never create, update, or delete allocations." reads as the person's rule;
    in the pasted turn it is plainly material. It is asked about once, as the turn."""
    items = listed(_pasted_skill())
    texts = [i.text for i in items]
    assert "Never create, update, or delete allocations." not in texts
    assert "Always tell the user which plan you read." not in texts
    assert "Name: Alex" in texts
    turn = next(i for i in items if i.text == PASTED)
    assert set(turn.places) == {
        ("standing_rules", f"- {PASTED} (said 2026-11-26)"),
        ("user", "- Never create, update, or delete allocations."),
        ("user", "- Always tell the user which plan you read."),
        (MEMORIES, "mem_pasted"),
    }
    # a word or two is in many texts without being read from any of them
    short = _bundle(
        memories=[_memory("m1", "Store 12 is in the north."), _memory("m2", "Store 12")]
    )
    assert [i.key for i in listed(short) if i.section == MEMORIES] == ["m1", "m2"]


def test_nothing_is_kept_for_its_rank_or_its_use_and_a_removed_rule_takes_its_marks() -> None:
    """The ranking put the pasted skill first for the request it blocks, and the tool hints
    forbade the tool the request needs by it: what has to go is exactly what a guard on the
    best match or on an applied rule would keep."""
    bundle = _pasted_skill()
    turn = next(i for i in listed(bundle) if i.text == PASTED)
    left = without(bundle, [turn])
    assert "allocations" not in left.render().replace("create_allocation", "")
    assert [b.text for b in left.profile] == ["- Name: Alex"]
    assert [m.item_id for m in left.memories] == ["mem_default"]
    assert left.tools is not None and left.tools.forbidden == {}
    assert [c.name for c in left.tools.candidates] == ["create_allocation"]
    assert left.procedures[0].forbidden == {}
    # a rule that stays keeps its marks
    kept = without(bundle, [i for i in listed(bundle) if i.text == "Name: Alex"])
    assert kept.tools is not None and kept.tools.forbidden == {"create_allocation": PASTED}
    assert kept.procedures[0].forbidden == {"create_allocation": PASTED}


def test_working_memory_and_what_the_context_would_only_repeat_are_not_shown() -> None:
    window = ConversationWindow(message_ids=["msg_1"], rendered="user: my dentist is on Tuesday")
    said_here = _memory("mem_other", "My dentist is on Tuesday.").model_copy(
        update={"evidence": _from("msg_1")}
    )
    bundle = _bundle(
        conversation=window,
        memories=[
            _memory("wm_0", "scratch note of this run"),
            _memory("mem_best", "The Coach tote is short at stores 12 and 40."),
            said_here,
            _memory("mem_old", "ok for now skip the consent stage"),
        ],
    )
    assert [i.key for i in listed(bundle) if i.section == MEMORIES] == ["mem_best", "mem_old"]


def test_the_call_carries_the_request_the_date_the_conversation_and_each_items_date() -> None:
    bundle = _bundle()
    body = json.loads(payload(QUESTION, bundle, listed(bundle), datetime(2026, 11, 27, tzinfo=UTC)))
    assert body["TODAY"] == "2026-11-27"
    assert body["REQUEST"] == QUESTION
    assert body["CONVERSATION"].endswith("assistant: Checking.")
    old = next(i for i in body["ITEMS"] if "consent" in i["text"])
    assert old == {
        "n": 7,
        "kind": "said in an earlier conversation",
        "text": "ok for now skip the consent stage",
        "said": "2026-11-01",
    }
    assert [i["n"] for i in body["ITEMS"]] == list(range(1, 11))


# --------------------------------------------------------------------------- what an answer does


def test_an_answer_names_only_what_was_shown() -> None:
    items = listed(_bundle())
    assert named({"drop": [2, 2, 99, 0, -1, "3", True]}, items) == [items[1]]
    assert named({"drop": []}, items) == []
    assert named(None, items) is None
    assert named({"drop": "all"}, items) is None
    assert named({"keep": [1]}, items) is None


def test_what_the_model_names_is_removed_and_nothing_else_changes() -> None:
    bundle = _bundle()
    items = listed(bundle)
    dropped = [i for i in items if i.key in (RULE_HOLIDAY, "sk_1", "mem_old", "mem_was", "rel_1")]
    left = without(bundle, dropped)
    assert [b.text for b in left.profile if b.block == "standing_rules"] == [RULE_TABLE]
    assert left.procedures == []
    assert [m.item_id for m in left.memories] == ["mem_best", "mem_other"]
    assert left.history == [] and left.graph_facts == []
    # untouched: the rest of the profile, the people, the passages, the conversation
    assert [b.block for b in left.profile] == [
        "standing_rules",
        "preferences",
        "about_user",
        "user",
    ]
    assert left.knowledge == bundle.knowledge
    assert left.conversation == bundle.conversation
    assert left.memories[0] is bundle.memories[0]
    assert left.token_estimate < bundle.token_estimate


def test_a_section_with_no_line_left_is_gone() -> None:
    bundle = _bundle()
    dropped = [i for i in listed(bundle) if i.section in ("preferences", "about_user")]
    left = without(bundle, dropped)
    assert [b.block for b in left.profile] == ["standing_rules", "user"]
    assert "### preferences" not in left.render()


def test_a_sentence_read_from_a_dropped_turn_goes_with_the_turn() -> None:
    """A sentence memory is not shown beside the turn it was read from
    (``ContextBundle.redundant``); dropping the turn must not bring it back unchecked."""
    source = _from("msg_9")
    turn = _memory("mem_turn", "ok for now skip the consent stage and go on").model_copy(
        update={"evidence": source}
    )
    part = _memory("mem_part", "skip the consent stage").model_copy(update={"evidence": source})
    bundle = _bundle(memories=[_memory("mem_best", "The tote is short at store 12."), turn, part])
    items = listed(bundle)
    assert [i.key for i in items if i.section == MEMORIES] == ["mem_best", "mem_turn"]
    left = without(bundle, [i for i in items if i.key == "mem_turn"])
    assert [m.item_id for m in left.memories] == ["mem_best"]


def test_dropping_the_skill_the_tool_hints_lay_out_takes_its_plan_with_it() -> None:
    plan = _skill("sk_1", "sales forecast", "get_sales", "forecast")
    hints = ToolHints(
        candidates=[
            ToolCandidate(name="get_sales", score=1.0, confidence=0.6),
            ToolCandidate(name="forecast", score=0.9, confidence=0.5),
            ToolCandidate(name="get_stock", score=0.5, confidence=0.4),
        ],
        plan=plan,
        next="get_sales",
        prefill={
            "get_sales.week": Prefill(tool="get_sales", value="W47", source="procedure"),
            "get_stock.item": Prefill(tool="get_stock", value="Coach tote", source="task"),
        },
    )
    bundle = _bundle(procedures=[plan], tools=hints)
    left = without(bundle, [i for i in listed(bundle) if i.section == SKILLS])
    assert left.tools is not None
    assert left.tools.plan is None and left.tools.next is None
    assert list(left.tools.prefill) == ["get_stock.item"]
    assert [c.name for c in left.tools.candidates] == ["get_sales", "forecast", "get_stock"]
    # another skill dropped: the hints are as they were
    other = _skill("sk_2", "returns", "open_return")
    bundle = _bundle(procedures=[plan, other], tools=hints)
    left = without(bundle, [i for i in listed(bundle) if i.key == "sk_2"])
    assert left.tools == hints and left.procedures == [plan]


# --------------------------------------------------------------------------- the call


async def test_one_call_of_the_fast_use_and_its_answer_is_what_is_dropped() -> None:
    bundle = _bundle()
    items = listed(bundle)
    model = _Model({"drop": _numbers(items, RULE_HOLIDAY, "mem_old")})
    dropped, note = await ContextCheck(model).read(QUESTION, bundle)  # type: ignore[arg-type]
    assert [d.key for d in dropped] == [RULE_HOLIDAY, "mem_old"]
    assert len(model.calls) == 1 and model.calls[0]["use"] == "context_check"
    assert note["outcome"] == CHECKED and note["listed"] == len(items)
    assert [d["section"] for d in note["dropped"]] == ["standing_rules", MEMORIES]


async def test_the_answer_has_room_to_name_every_item() -> None:
    """A context that is mostly not about the request needs its whole answer: a list cut
    off by the output limit is no answer, and the context would go out as built."""
    many = [_memory(f"mem_{n}", f"Note number {n} about something else.") for n in range(120)]
    bundle = _bundle(memories=many)
    model = _Model({"drop": []})
    await ContextCheck(model).read(QUESTION, bundle)  # type: ignore[arg-type]
    listed_items = len(listed(bundle))
    assert listed_items > 120
    assert model.calls[0]["max_tokens"] >= 3 * listed_items
    assert answer_tokens(10_000) == LLM.max_tokens


async def test_no_model_allowed_or_nothing_to_ask_about_costs_no_call() -> None:
    off = _Model({"drop": [1]}, wanted=False)
    assert await ContextCheck(off).read(QUESTION, _bundle()) == ([], {})  # type: ignore[arg-type]
    assert off.calls == []
    empty = _bundle(profile=[], procedures=[], memories=[], history=[], graph_facts=[])
    model = _Model({"drop": [1]})
    assert await ContextCheck(model).read(QUESTION, empty) == ([], {})  # type: ignore[arg-type]
    assert model.calls == []


@pytest.mark.parametrize("answer", [None, {"drop": "everything"}, {"keep": [1]}])
async def test_no_answer_leaves_the_context_as_built(answer: dict[str, Any] | None) -> None:
    dropped, note = await ContextCheck(_Model(answer)).read(QUESTION, _bundle())  # type: ignore[arg-type]
    assert dropped == []
    assert note["outcome"] == NO_ANSWER and note["dropped"] == []


async def test_a_late_model_leaves_the_context_as_built() -> None:
    model = _Model({"drop": [1]}, hangs=True)
    check = ContextCheck(model, deadline_seconds=0.25)  # type: ignore[arg-type]
    dropped, note = await check.read(QUESTION, _bundle())
    assert dropped == [] and note["outcome"] == LATE
    assert note["ms"] < 2000


async def test_the_check_never_runs_past_what_the_request_has_left() -> None:
    model = _Model({"drop": [1]}, hangs=True)
    check = ContextCheck(model)  # type: ignore[arg-type]
    dropped, note = await check.read(QUESTION, _bundle(), within=0.3)
    assert dropped == [] and note["outcome"] == LATE and note["ms"] < 2000
    # no time left at all: the model is not asked
    asked = len(model.calls)
    dropped, note = await check.read(QUESTION, _bundle(), within=0.05)
    assert dropped == [] and note["outcome"] == NO_TIME
    assert len(model.calls) == asked


async def test_a_linked_memory_is_kept_or_dropped_with_the_one_it_came_with() -> None:
    """A memory retrieval brought in through another (a second hop, the turn a sentence was
    read from, a linked turn: ``expanded_from``) answers with it: "Charlotte's Web, I loved
    it" means nothing apart from the question it replies to. The model's verdict on either
    is the pair's: one kept keeps both, and only both named go."""
    asked = _memory("mem_asked", "What books do you read with your kids?")
    replied = _memory("mem_replied", "Charlotte's Web, they loved it.").model_copy(
        update={"expanded_from": "mem_asked", "expansion_edge": "SECOND_HOP"}
    )
    bundle = _bundle(memories=[asked, replied, _memory("mem_other", "My dentist is on Tuesday.")])
    items = listed(bundle)
    for named_ in (["mem_replied"], ["mem_asked"]):
        model = _Model({"drop": _numbers(items, *named_, "mem_other")})
        dropped, _ = await ContextCheck(model).read(QUESTION, bundle)  # type: ignore[arg-type]
        assert [d.key for d in dropped] == ["mem_other"], named_
    model = _Model({"drop": _numbers(items, "mem_asked", "mem_replied")})
    dropped, _ = await ContextCheck(model).read(QUESTION, bundle)  # type: ignore[arg-type]
    assert {d.key for d in dropped} == {"mem_asked", "mem_replied"}
