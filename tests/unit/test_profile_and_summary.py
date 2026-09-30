"""Profile blocks and thread summaries without a database."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.conversation import Message
from memory_service.domain.enums import MemoryType, MessageKind, MessageRole
from memory_service.domain.errors import ValidationFailed
from memory_service.domain.memory import CanonicalMemory
from memory_service.domain.profile import PROFILE_BLOCK_MAX_CHARS, block_scope, profile_scopes
from memory_service.modules.conversation.summary import SUMMARY_MAX_CHARS, digest, rolled
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
    edited = "name: Ann\n- prefers email"
    before = datetime(2000, 1, 1, tzinfo=UTC)
    assert appended(edited, facts, before) == "name: Ann\n- prefers email\n- Lives in Berlin"
    assert appended("already: Lives in Berlin", facts[2:], before) == "already: Lives in Berlin"
    assert appended(edited, facts, datetime.now(UTC)) == edited, "only facts after the edit"
    assert len(template([_memory("x" * 900)] * 1 + [_memory(str(i) * 900) for i in range(9)])) <= (
        PROFILE_BLOCK_MAX_CHARS
    )


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
    assert digest(first) == ["user: Order paper.", "assistant: Done."]
    summary = rolled("", first)
    later = rolled(summary, [_message(3, "And pens?")])
    assert later.splitlines() == ["user: Order paper.", "assistant: Done.", "user: And pens?"]
    long = rolled(later, [_message(i, f"line {i} " + "x" * 150) for i in range(4, 40)])
    assert len(long) <= SUMMARY_MAX_CHARS and "user: Order paper." not in long
