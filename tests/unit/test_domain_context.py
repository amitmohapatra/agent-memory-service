import pytest
from pydantic import ValidationError

from memory_service.domain.context import MemoryExecutionContext


def test_minimal_context_generates_correlation_ids() -> None:
    ctx = MemoryExecutionContext(tenant_id="acme")
    assert ctx.request_id.startswith("req_")
    assert ctx.trace_id
    assert ctx.principal_id == "service:anonymous"


def test_security_fields_cannot_be_overridden_by_metadata() -> None:
    with pytest.raises(ValidationError, match="reserved keys"):
        MemoryExecutionContext(tenant_id="acme", custom_metadata={"tenant_id": "evil"})
    with pytest.raises(ValidationError, match="reserved keys"):
        MemoryExecutionContext(tenant_id="acme", custom_metadata={"user_id": "someone-else"})


def test_context_is_frozen() -> None:
    ctx = MemoryExecutionContext(tenant_id="acme")
    with pytest.raises(ValidationError):
        ctx.tenant_id = "other"  # type: ignore[misc]


def test_lineage_invariants() -> None:
    with pytest.raises(ValidationError, match="session_id requires thread_id"):
        MemoryExecutionContext(tenant_id="t", session_id="ses_1")
    with pytest.raises(ValidationError, match="turn_id requires session_id"):
        MemoryExecutionContext(tenant_id="t", thread_id="thr_1", turn_id="trn_1")
    with pytest.raises(ValidationError, match="agent_run_id requires agent_id"):
        MemoryExecutionContext(tenant_id="t", agent_run_id="run_1")


def test_invalid_ids_rejected() -> None:
    with pytest.raises(ValidationError):
        MemoryExecutionContext(tenant_id="bad tenant id")
    with pytest.raises(ValidationError):
        MemoryExecutionContext(tenant_id="t", group_ids=["ok", "not ok"])


def test_child_agent_inherits_lineage_and_adds_agent_fields() -> None:
    parent = MemoryExecutionContext(
        tenant_id="acme",
        workspace_id="ws",
        user_id="u1",
        thread_id="thr_1",
        session_id="ses_1",
        turn_id="trn_1",
    )
    child = parent.child_agent(agent_id="research", agent_group_id="crew")
    assert child.tenant_id == "acme"
    assert child.user_id == "u1"
    assert child.thread_id == "thr_1" and child.session_id == "ses_1" and child.turn_id == "trn_1"
    assert child.trace_id == parent.trace_id
    assert child.agent_id == "research"
    assert child.agent_run_id and child.agent_run_id.startswith("run_")
    assert child.parent_agent_run_id is None
    assert child.causation_id == parent.request_id
    # Bound to the user it runs for. ``agent_id`` arrives in an unauthenticated request
    # body, so a bare ``agent:research`` let any caller assume this agent by naming it —
    # reproduced live: a different user sending agent_id=worker read another user's PRIVATE
    # memory with HTTP 200.
    assert child.principal_id == "agent:u1/research"
    grandchild = child.child_agent(agent_id="writer")
    assert grandchild.parent_agent_run_id == child.agent_run_id


def test_log_fields_include_tenant_and_trace_only_when_present() -> None:
    ctx = MemoryExecutionContext(tenant_id="t", thread_id="thr_1")
    fields = ctx.log_fields()
    assert fields["tenant_id"] == "t" and fields["thread_id"] == "thr_1"
    assert "session_id" not in fields
