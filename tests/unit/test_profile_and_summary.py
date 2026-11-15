"""Profile blocks and thread summaries without a database."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.conversation import Message
from memory_service.domain.enums import MemoryType, MessageKind, MessageRole
from memory_service.domain.errors import ValidationFailed
from memory_service.domain.memory import CanonicalMemory
from memory_service.domain.profile import (
    PROFILE_BLOCK_MAX_CHARS,
    ProfileBlock,
    block_scope,
    lines_kept,
    profile_scopes,
)
from memory_service.modules.conversation.summary import (
    LINE_CHARS,
    SUMMARY_MAX_CHARS,
    digest,
    rolled,
)
from memory_service.modules.profile.service import appended, template

AGENT = MemoryExecutionContext(tenant_id="acme", user_id="u1", agent_id="buyer", workspace_id="ops")


def test_a_block_belongs_to_the_caller_s_own_user_agent_or_workspace() -> None:
    assert block_scope("user", AGENT) == "user:u1"
    assert block_scope("agent.persona", AGENT) == "agent:u1/buyer"
    assert block_scope("workspace", AGENT) == "workspace:ops"
    assert set(profile_scopes(MemoryExecutionContext(tenant_id="acme", user_id="u1"))) == {"user"}
    for bad in ("tenant", "user.", "user.Has Space", "../x"):
        with pytest.raises(ValidationFailed):
            block_scope(bad, AGENT)
    with pytest.raises(ValidationFailed, match="needs a agent"):
        block_scope("agent", MemoryExecutionContext(tenant_id="acme", user_id="u1"))


def _memory(content: str) -> CanonicalMemory:
    return CanonicalMemory.model_validate(
        {
            "tenant_id": "acme",
            "content": content,
            "memory_type": MemoryType.PREFERENCE,
            "lifetime": "LONG_TERM",
            "visibility": "USER",
            "scope": {"level": "USER", "tenant_id": "acme", "user_id": "u1"},
            "owner_principal": "user:u1",
            "normalized_hash": content,
            "temporal": {"observed_at": datetime.now(UTC)},
            "evidence": [
                {"source_type": "message", "source_id": "msg_1", "observed_at": datetime.now(UTC)}
            ],
        }
    )


def test_the_learned_block_is_a_line_per_fact_and_an_edit_is_only_appended_to() -> None:
    facts = [_memory("Prefers email"), _memory("Prefers  email"), _memory("Lives in Berlin")]
    assert template(facts) == "- Prefers email\n- Lives in Berlin"
    edited = _edited("name: Ann\n- prefers email")
    held = {f.memory_id for f in facts}
    assert appended(edited, facts, held) == (
        "name: Ann\n- prefers email\n- Lives in Berlin",
        {"- Lives in Berlin": facts[2].memory_id},
    )
    assert appended(_edited("already: Lives in Berlin"), facts[2:], held) == (
        "already: Lives in Berlin",
        {},
    )
    now = _edited(edited.text, at=datetime.now(UTC))
    assert appended(now, facts, held) == (edited.text, {}), "only facts after the edit"
    assert len(template([_memory("x" * 900)] * 1 + [_memory(str(i) * 900) for i in range(9)])) <= (
        PROFILE_BLOCK_MAX_CHARS
    )


def _edited(
    text: str,
    appended_from: dict[str, str] | None = None,
    at: datetime = datetime(2000, 1, 1, tzinfo=UTC),
) -> ProfileBlock:
    return ProfileBlock(
        tenant_id="acme",
        scope_key="user:u1",
        block="user",
        text=text,
        source="edited",
        updated_at=at,
        appended_from=appended_from or {},
    )


def test_a_line_the_job_appended_leaves_with_its_memory_and_the_editor_s_lines_stay() -> None:
    seats, tea = _memory("Prefers window seats"), _memory("Prefers green tea")
    block = _edited(
        "- Prefers window seats\nname: Ann\n- Prefers window seats\n- Prefers green tea",
        {"- Prefers window seats": seats.memory_id, "- Prefers green tea": tea.memory_id},
    )
    # seats is forgotten: its line goes, and only the job's copy of it (the one below); the
    # editor's own line saying the same stays. tea is held, past the newest memories read.
    assert appended(block, [], {tea.memory_id}) == (
        "- Prefers window seats\nname: Ann\n- Prefers green tea",
        {"- Prefers green tea": tea.memory_id},
    )
    # a newer memory saying what a forgotten one said takes its line over
    again = _memory("Prefers  window seats")
    assert appended(
        _edited("name: Ann\n- Prefers window seats", block.appended_from),
        [again],
        {again.memory_id},
    ) == ("name: Ann\n- Prefers window seats", {"- Prefers window seats": again.memory_id})


def test_a_job_s_line_an_editor_changes_or_removes_is_theirs() -> None:
    lines = {"- Prefers tea": "memory_1", "- Lives in Berlin": "memory_2"}
    assert lines_kept("name: Ann\n- Prefers tea, no sugar\n- Lives in Berlin", lines) == {
        "- Lives in Berlin": "memory_2"
    }
    assert lines_kept("", lines) == {}


def _message(sequence: int, content: str, role: MessageRole = MessageRole.USER) -> Message:
    return Message(
        thread_id="thr_1",
        session_id="ses_1",
        turn_id="trn_1",
        tenant_id="acme",
        role=role,
        kind=MessageKind.VISIBLE,
        sequence=sequence,
        content=content,
        content_hash=str(sequence),
        author_principal="user:u1",
    )


def test_the_extractive_summary_rolls_forward_and_drops_the_oldest_lines() -> None:
    first = [_message(1, "Order paper. Quickly."), _message(2, "Done.", MessageRole.ASSISTANT)]
    assert digest(first) == ["user: Order paper. Quickly.", "assistant: Done."]
    summary = rolled("", first)
    later = rolled(summary, [_message(3, "And pens?")])
    assert later.splitlines() == [
        "user: Order paper. Quickly.",
        "assistant: Done.",
        "user: And pens?",
    ]
    long = rolled(later, [_message(i, f"line {i} " + "x" * 150) for i in range(4, 40)])
    assert len(long) <= SUMMARY_MAX_CHARS and "user: Order paper." not in long


def test_a_digest_line_keeps_the_sentences_that_fit_not_only_the_opener() -> None:
    """A correction's opener says nothing on its own ("Correction: the Elm Street depot has
    closed."): the episode a later conversation finds must carry the new value too, or it
    reads as the old one served as current. Whole sentences only, within a line."""
    said = (
        "Correction: the Elm Street depot has closed. Returns now go to the new depot at "
        "45 Harbor Road."
    )
    assert digest([_message(1, said)]) == [f"user: {said}"]
    long = "First sentence. " + "y" * 200 + ". Third."
    assert digest([_message(1, long)]) == ["user: First sentence."]
    assert digest([_message(1, "z" * 300 + ". Short.")]) == ["user: " + "z" * LINE_CHARS]


def _rule(content: str, day: int, category: str = "rule") -> CanonicalMemory:
    m = _memory(content)
    return m.model_copy(
        update={
            "system_metadata": {"category": category},
            "temporal": m.temporal.model_copy(
                update={"observed_at": datetime(2026, 10, day, tzinfo=UTC)}
            ),
        }
    )


@pytest.mark.parametrize(
    ("older", "newer"),
    [
        (
            "Always put the work order number in the email subject, prefixed with WO-E3.",
            "Always put the work order number in the email subject, prefixed with WO-K3.",
        ),
        (
            "ضع دائمًا رقم أمر العمل في عنوان البريد مسبوقًا بـ WO-E3.",  # noqa: RUF001
            "ضع دائمًا رقم أمر العمل في عنوان البريد مسبوقًا بـ WO-K3.",  # noqa: RUF001
        ),
        (
            "Setz die Auftragsnummer immer in den E-Mail-Betreff, mit dem Präfix WO-E3.",
            "Setz die Auftragsnummer immer in den E-Mail-Betreff, mit dem Präfix WO-K3.",
        ),
    ],
)
def test_a_rule_restated_with_new_values_is_listed_once_as_said_last(older, newer) -> None:
    block = template([_rule(newer, 17), _rule(older, 10), _rule("Never book Sundays.", 12)])
    assert block == f"- {newer}\n- Never book Sundays."


def test_facts_that_say_the_same_of_different_things_all_stay() -> None:
    a = _rule("I have a meeting with Ana on Monday.", 10, category="fact")
    b = _rule("I have a meeting with Bob on Tuesday.", 12, category="fact")
    assert template([b, a]).count("\n") == 1
