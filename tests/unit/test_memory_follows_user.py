"""What a user says is theirs, kept as they said it.

Two guarantees behind single-fact recall across sessions ("The master lock code for the
hazardous materials cage in Warehouse 3 is 8492" on Monday, "what's the code?" in a new chat
on Tuesday): the statement's default audience is the user, not the conversation it was said
in, and the user's exact words survive whatever a rule makes of them.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from memory_service.config.constants import MemoryIntelligenceSettings
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import (
    DedupDecision,
    Lifetime,
    MemoryType,
    ObservationKind,
    Visibility,
)
from memory_service.domain.ids import content_hash
from memory_service.domain.observation import Observation
from memory_service.modules.memory.native import (
    NativeMemoryIntelligence,
    default_visibility,
    said_by_user,
    verbatim_windows,
)
from memory_service.modules.memory.pipeline import build_memory
from memory_service.modules.retrieval.engine import Candidate, _dedup

pytestmark = pytest.mark.unit

USER = MemoryExecutionContext(tenant_id="acme", user_id="u1", workspace_id="ws1", thread_id="t1")
HARNESS = USER.model_copy(update={"agent_id": "planner", "agent_run_id": "run_1"})
CODE = "The master lock code for the hazardous materials cage in Warehouse 3 is 8492."


def _obs(
    text: str,
    ctx: MemoryExecutionContext = USER,
    *,
    role: str | None = "USER",
    kind: ObservationKind = ObservationKind.MESSAGE,
) -> Observation:
    return Observation(
        tenant_id=ctx.tenant_id,
        kind=kind,
        content=text,
        content_hash=content_hash(text),
        user_id=ctx.user_id,
        workspace_id=ctx.workspace_id,
        thread_id=ctx.thread_id,
        agent_id=ctx.agent_id,
        agent_run_id=ctx.agent_run_id,
        principal_id=ctx.principal_id,
        message_id="msg_1",
        custom_metadata={"role": role} if role else {},
    )


@pytest.fixture
def native() -> NativeMemoryIntelligence:
    return NativeMemoryIntelligence(MemoryIntelligenceSettings())


async def _classified(native, observation: Observation, ctx: MemoryExecutionContext = USER):
    return [await native.classify(c, ctx) for c in await native.extract(observation, ctx)]


# --- whose words ---------------------------------------------------------------------------


def test_a_users_own_words_are_told_apart_from_an_agents() -> None:
    assert said_by_user(_obs(CODE))
    assert said_by_user(_obs(CODE, HARNESS)), "a harness relaying the user's turn"
    assert said_by_user(_obs("Go with Qdrant.", kind=ObservationKind.DECISION, role=None))
    assert not said_by_user(_obs(CODE, role="ASSISTANT")), "the assistant's reply"
    assert not said_by_user(_obs(CODE, HARNESS, role="AGENT"))
    assert not said_by_user(_obs("Draft plan.", HARNESS, kind=ObservationKind.AGENT_RESULT))
    assert not said_by_user(_obs("Deploy finished.", kind=ObservationKind.EVENT)), "a record"
    service = USER.model_copy(update={"user_id": None})
    assert not said_by_user(_obs(CODE, service)), "nobody to follow"


@pytest.mark.parametrize(
    "mtype",
    [MemoryType.SEMANTIC, MemoryType.TASK, MemoryType.EPISODIC, MemoryType.PROCEDURAL],
)
def test_a_users_statement_defaults_to_the_user_in_any_thread(mtype) -> None:
    assert default_visibility(mtype, USER, user_statement=True) is Visibility.USER
    assert default_visibility(mtype, HARNESS, user_statement=True) is Visibility.USER, (
        "relayed by an agent is still the user's, not the agent's private note"
    )
    # not the user's words: the conversation's, as before
    assert default_visibility(mtype, USER) is Visibility.THREAD


def test_an_agents_own_writing_keeps_its_rules() -> None:
    assert default_visibility(MemoryType.TASK, HARNESS) is Visibility.PRIVATE
    assert default_visibility(MemoryType.EPISODIC, HARNESS) is Visibility.PRIVATE
    assert default_visibility(MemoryType.SEMANTIC, HARNESS) is Visibility.THREAD
    assert default_visibility(MemoryType.AGENT, HARNESS, user_statement=True) is Visibility.RUN
    # explicitly shared types keep their audience whoever wrote them
    assert default_visibility(MemoryType.SHARED, USER, user_statement=True) is Visibility.THREAD


async def test_every_memory_of_a_users_message_follows_the_user(native) -> None:
    said = await _classified(native, _obs(CODE))
    assert said and all(c.visibility is Visibility.USER for c in said)
    reply = await _classified(native, _obs("The cage code is 8492.", role="ASSISTANT"))
    assert reply and all(c.visibility is Visibility.THREAD for c in reply), "the assistant's words"


# --- the words as said -----------------------------------------------------------------------


async def test_a_long_named_subject_is_a_fact_and_the_turn_is_kept(native) -> None:
    fact, turn = await _classified(native, _obs(CODE))
    assert fact.category == "fact" and fact.object == "8492"
    assert fact.subject == "master lock code for the hazardous materials cage in warehouse 3"
    assert turn.category == "verbatim_turn" and turn.content == CODE
    assert turn.evidence == fact.evidence, "one source, two readings"


@pytest.mark.parametrize(
    "text",
    [
        "The kids and I went to the beach yesterday and it was great.",
        "The new guy that we hired from Berlin last week is great.",
    ],
)
async def test_a_clause_is_not_read_as_a_long_subject(native, text) -> None:
    cands = await _classified(native, _obs(text))
    assert all(c.category != "fact" for c in cands), [c.subject for c in cands]
    assert cands[-1].category == "verbatim_turn"


async def test_a_short_lived_reading_does_not_take_the_turn_with_it(native) -> None:
    """A rule may read a standing rule as a task: the task lapses in seven days, the user's
    sentence does not."""
    text = "All seasonal holiday merchandise must be routed to Overflow Storage Facility B."
    task, turn = await _classified(native, _obs(text))
    assert task.memory_type is MemoryType.TASK and task.lifetime is Lifetime.SHORT_TERM
    assert turn.category == "verbatim_turn" and turn.lifetime is Lifetime.LONG_TERM
    assert turn.content == text and turn.visibility is Visibility.USER


async def test_a_reading_and_its_turn_never_reinforce_each_other(native) -> None:
    now = datetime.now(UTC)
    first = await _classified(native, _obs(CODE))
    stored = [build_memory(c, USER, now=now) for c in first]
    again = await _classified(native, _obs(CODE))
    outcomes = [await native.consolidate(c, stored, USER) for c in again]
    assert [o.decision for o in outcomes] == [DedupDecision.REINFORCE] * 2
    # each reinforces its own kind: the fact the fact, the turn the turn
    assert [o.target_memory_id for o in outcomes] == [m.memory_id for m in stored]
    # and within one message neither absorbs the other
    fact, turn = first
    assert (await native.consolidate(turn, stored[:1], USER)).decision is DedupDecision.CREATE
    assert (await native.consolidate(fact, stored[1:], USER)).decision is DedupDecision.CREATE


def test_a_long_turn_is_kept_whole_in_sentence_pieces() -> None:
    filler = "Pallet counts for aisle four were checked against the manifest. " * 40
    text = f"{filler}The new dock door code is 7731. {filler}".strip()
    pieces = verbatim_windows(text, 2000)
    assert len(pieces) > 1 and all(len(p) <= 2000 for p in pieces)
    assert all(p in text for p in pieces), "every piece is a slice of what was said"
    assert any("The new dock door code is 7731." in p for p in pieces), "cut at sentence ends"
    assert "".join(text.split()) == "".join("".join(pieces).split()), "nothing dropped"
    assert verbatim_windows("A short turn.", 2000) == ["A short turn."]
    # one unbroken sentence longer than a piece is cut at a space, still losing nothing
    long_one = " ".join(["SKU-12345"] * 300)
    assert "".join("".join(verbatim_windows(long_one, 200)).split()) == "".join(long_one.split())


async def test_a_code_said_after_two_thousand_characters_is_kept(native) -> None:
    filler = "Pallet counts for aisle four were checked against the manifest. " * 40
    text = f"{filler}The new dock door code is 7731."
    turns = [c for c in await _classified(native, _obs(text)) if c.category == "verbatim_turn"]
    assert len(turns) == 2 and "7731" in turns[-1].content


# --- one statement, one slot -----------------------------------------------------------------


def _hit(record_id: str, subject: str, score: float) -> Candidate:
    return Candidate(
        record_id=record_id,
        kind="memory",
        text=CODE,
        score=score,
        retrievers=["fusion"],
        payload={
            "subject": subject,
            "owner_principal": "user:u1",
            "observed_at": "2026-10-06T07:40:00Z",
            "source_refs": [{"source_type": "message", "source_id": "msg_1"}],
        },
    )


def test_a_fact_and_its_turn_take_one_slot() -> None:
    fact = _hit("fact", "master lock code for the hazardous materials cage in warehouse 3", 0.9)
    turn = _hit("turn", "user:u1", 0.7)
    [kept] = _dedup([fact, turn])
    assert kept.record_id == "fact" and kept.payload["duplicates"] == ["turn"]


def test_equal_words_of_different_speakers_still_take_two() -> None:
    assert len(_dedup([_hit("a", "user:u1", 0.9), _hit("b", "user:u2", 0.8)])) == 2
