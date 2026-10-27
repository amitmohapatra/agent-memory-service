"""What a user said is theirs to restate or retract: an agent acting for them may not.

Seen live with a small model answering read-only questions: "What is 12 times 7?" made the run
call ``memory_update`` on "I raised PO-77812 for 2,000 corrugated boxes; it's late." and
replace it with "12 times 7 = 84"; "Will payroll be down ...?" replaced "Store 61 stays in
Region South." five times over. The new version kept the user as its owner and ``user:<u>
said`` as its subject and predicate, at a stated confidence, so every later context told the
model the user had said the agent's words. ``POST /v1/memories/{id}/supersede`` and
``DELETE /v1/memories/{id}`` go through the same service, so they are refused the same way.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import MemoryType, TemporalStatus
from memory_service.domain.errors import ScopeDenied
from memory_service.domain.memory import CanonicalMemory
from memory_service.modules.agent_tools.service import AgentTools
from memory_service.modules.memory.pipeline import said_by_a_person
from memory_service.modules.memory.service import MemoryService

pytestmark = pytest.mark.unit

T0 = datetime(2026, 10, 23, 9, 0, tzinfo=UTC)
USER = MemoryExecutionContext(tenant_id="acme", user_id="u1", thread_id="t1")
RUN = USER.model_copy(update={"agent_id": "analyst", "agent_run_id": "run_1"})
ADMIN = MemoryExecutionContext(tenant_id="acme", user_id="root", thread_id="t9")
PO = "I raised PO-77812 for 2,000 corrugated boxes; it's late."


def _memory(content: str, *, owner: str = "user:u1", **fields: Any) -> CanonicalMemory:
    base: dict[str, Any] = {
        "tenant_id": "acme",
        "content": content,
        "memory_type": MemoryType.OBSERVATION,
        "lifetime": "LONG_TERM",
        "visibility": "USER",
        "scope": {"level": "THREAD", "tenant_id": "acme", "thread_id": "t1"},
        "owner_principal": owner,
        "normalized_hash": content,
        "subject": "user:u1",
        "predicate": "said",
        "confidence": 0.99,
        "temporal": {"observed_at": T0},
        "evidence": [{"source_type": "message", "source_id": "msg_1", "observed_at": T0}],
        "system_metadata": {"category": "verbatim_turn"},
        "created_at": T0,
        "updated_at": T0,
    }
    return CanonicalMemory.model_validate({**base, **fields})


class _Rows:
    """The memories table: rows by id, what was added, and what was forgotten."""

    def __init__(self, rows: list[CanonicalMemory]) -> None:
        self.rows = {m.memory_id: m for m in rows}
        self.added: list[CanonicalMemory] = []
        self.forgotten: list[str] = []

    async def get(self, tenant_id: str, memory_id: str) -> CanonicalMemory | None:
        return self.rows.get(memory_id)

    async def visibility_keys(self, tenant_id: str, memory_id: str) -> list[str]:
        return ["user:acme/u1"]

    async def twins(self, *_: Any, **__: Any) -> list[CanonicalMemory]:
        return []

    async def replaced_by_any(self, *_: Any) -> list[CanonicalMemory]:
        return []

    async def lock(self, *_: Any) -> None:
        return None

    async def is_forgotten(self, tenant_id: str, memory_id: str) -> bool:
        return memory_id in self.forgotten

    async def forget(self, tenant_id: str, memory_id: str) -> None:
        self.forgotten.append(memory_id)

    async def add(self, memory: CanonicalMemory, **_: Any) -> None:
        self.added.append(memory)
        self.rows[memory.memory_id] = memory

    async def update(self, memory: CanonicalMemory) -> None:
        self.rows[memory.memory_id] = memory


class _Ignored:
    async def add(self, *_: Any, **__: Any) -> None:
        return None

    async def bump(self, *_: Any, **__: Any) -> None:
        return None

    async def mark_used(self, *_: Any, **__: Any) -> None:
        return None


class _Uow:
    def __init__(self, rows: list[CanonicalMemory]) -> None:
        self.memories, self.pulls, self.revisions = _Rows(rows), _Ignored(), _Ignored()

    async def __aenter__(self) -> _Uow:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def enqueue(self, *_: Any) -> int:
        return 1

    async def commit(self) -> None:
        return None


class _Authz:
    """Everything is visible; ``admins`` are the tenant's admins (an agent counts as the user
    it acts for, as the real check does)."""

    def __init__(self, *admins: str) -> None:
        self.admins = set(admins)

    async def visibility(self, *_: Any, **__: Any) -> Any:
        return SimpleNamespace(allows=lambda *_: True)

    async def is_tenant_admin(self, ctx: MemoryExecutionContext) -> bool:
        return f"user:{ctx.user_id}" in self.admins


class _Handles:
    async def resolve(self, ctx: MemoryExecutionContext, memory_id: str) -> str:
        return memory_id


def _tools(uow: _Uow, authz: _Authz | None = None) -> AgentTools:
    service = MemoryService(authz or _Authz())  # type: ignore[arg-type]
    return AgentTools(lambda: uow, {"memory": service, "bundle_records": _Handles()})


# --- what a person said ---------------------------------------------------------------------


def test_what_a_person_said_is_told_apart_from_an_agents_note() -> None:
    assert said_by_a_person(_memory(PO)), "the user's turn"
    assert said_by_a_person(
        _memory(PO, memory_type=MemoryType.USER, predicate="works_at", subject="user:u1")
    ), "a reading of it, written by the user"
    assert said_by_a_person(_memory(PO, owner="agent:u1/harness")), "a turn a harness relayed"
    note = _memory("Payroll runs on Fridays.", owner=RUN.principal_id, subject=None, predicate=None)
    assert not said_by_a_person(note)
    record = _memory(
        "My timezone is Asia/Kolkata.",
        owner=RUN.principal_id,
        memory_type=MemoryType.USER,
        predicate="timezone",
    )
    assert not said_by_a_person(record), "an event the agent recorded is the agent's"


# --- an agent run's tools -------------------------------------------------------------------


@pytest.mark.parametrize(
    "said",
    [
        _memory(PO),
        _memory("Store 61 stays in Region South.", owner=RUN.principal_id),
        _memory(
            "I work at ACME Corp.",
            memory_type=MemoryType.USER,
            subject="user:u1",
            predicate="works_at",
            object="acme corp",
            system_metadata={"category": "attribute"},
        ),
    ],
    ids=["turn", "relayed-turn", "profile-fact"],
)
@pytest.mark.parametrize("tool", ["memory_update", "memory_forget"])
async def test_an_agent_run_cannot_rewrite_or_forget_what_the_user_said(said, tool) -> None:
    uow = _Uow([said])
    args = {"id": said.memory_id, "content": "12 times 7 = 84"}
    if tool == "memory_forget":
        args.pop("content")
    with pytest.raises(ScopeDenied) as refused:
        await _tools(uow).call(RUN, tool, args)
    # what the model reads back: why, and what to do instead
    assert "user's own statement" in str(refused.value)
    assert "memory_remember" in str(refused.value)
    assert uow.memories.added == [] and uow.memories.forgotten == []
    kept = uow.memories.rows[said.memory_id]
    assert kept.temporal.status is TemporalStatus.CURRENT and kept.content == said.content


async def test_an_agent_acting_for_an_admin_still_cannot_rewrite_the_users_words() -> None:
    uow = _Uow([_memory(PO)])
    [said] = uow.memories.rows.values()
    with pytest.raises(ScopeDenied):
        await _tools(uow, _Authz("user:u1")).call(
            RUN, "memory_update", {"id": said.memory_id, "content": "12 times 7 = 84"}
        )
    assert uow.memories.added == []


async def test_an_agent_still_updates_and_forgets_its_own_memory() -> None:
    mine = _memory(
        "Supplier ETA checks run hourly.",
        owner=RUN.principal_id,
        memory_type=MemoryType.AGENT,
        subject=None,
        predicate=None,
        system_metadata={"category": "stated"},
    )
    uow = _Uow([mine])
    tools = _tools(uow)
    updated = await tools.call(
        RUN, "memory_update", {"id": mine.memory_id, "content": "Supplier ETA checks run daily."}
    )
    assert updated["supersedes"] == mine.memory_id
    [new] = uow.memories.added
    assert new.owner_principal == RUN.principal_id and new.memory_type is MemoryType.AGENT
    assert uow.memories.rows[mine.memory_id].temporal.superseded_by == new.memory_id
    assert await tools.call(RUN, "memory_forget", {"id": new.memory_id}) == {
        "id": new.memory_id,
        "forgotten": True,
    }


async def test_an_agents_new_version_never_reads_as_the_user_speaking() -> None:
    """An agent's own note about the user (an event it recorded, read as the user's profile
    kind) may be updated; the new version is the agent's note, not a statement of the user's."""
    record = _memory(
        "My timezone is Asia/Kolkata.",
        owner=RUN.principal_id,
        memory_type=MemoryType.USER,
        predicate="timezone",
        object="asia/kolkata",
        system_metadata={"category": "attribute"},
    )
    uow = _Uow([record])
    await _tools(uow).call(
        RUN, "memory_update", {"id": record.memory_id, "content": "Timezone is Europe/Berlin."}
    )
    [new] = uow.memories.added
    assert new.owner_principal == RUN.principal_id
    assert new.memory_type is MemoryType.AGENT, "never a kind of the user's profile"
    assert (new.subject, new.predicate) == (None, None), "no person as its subject"
    assert new.evidence[0].agent_id == "analyst" and new.evidence[0].agent_run_id == "run_1"


# --- the user and the admin, unchanged --------------------------------------------------------


async def test_the_user_still_restates_and_forgets_their_own_words() -> None:
    uow = _Uow([_memory(PO)])
    [said] = uow.memories.rows.values()
    service = MemoryService(_Authz())  # type: ignore[arg-type]
    new = await service.supersede(
        uow,  # type: ignore[arg-type]
        USER,
        said.memory_id,
        content="PO-77812 arrived today.",
        reason="updated",
    )
    assert (new.owner_principal, new.subject, new.predicate) == ("user:u1", "user:u1", "said")
    assert new.memory_type is MemoryType.OBSERVATION
    assert await service.forget(uow, USER, new.memory_id)  # type: ignore[arg-type]
    assert uow.memories.forgotten == [new.memory_id]


async def test_a_tenant_admin_still_corrects_a_users_memory() -> None:
    uow = _Uow([_memory(PO)])
    [said] = uow.memories.rows.values()
    service = MemoryService(_Authz("user:root"))  # type: ignore[arg-type]
    new = await service.supersede(
        uow,  # type: ignore[arg-type]
        ADMIN,
        said.memory_id,
        content="I raised PO-77812 for 2,000 corrugated boxes.",
        reason="redacted",
    )
    assert new.owner_principal == "user:u1", "still the user's statement, corrected"
    with pytest.raises(ScopeDenied):  # and nobody else may
        await MemoryService(_Authz()).supersede(  # type: ignore[arg-type]
            uow,  # type: ignore[arg-type]
            ADMIN.model_copy(update={"user_id": "u2"}),
            new.memory_id,
            content="x",
            reason="x",
        )
