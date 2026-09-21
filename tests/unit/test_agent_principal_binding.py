"""An agent principal is bound to the user it runs for.

``agent_id`` arrives in an unauthenticated request body — only tenant_id, user_id and
workspace_id come from trusted headers (api/deps.py:85-104) — but ``principal_id`` makes the
agent *the principal*, and PRIVATE memories are keyed on it. A bare ``agent:{agent_id}``
therefore let any caller assume any agent's identity simply by naming it.

Reproduced against a live service before this was fixed: user ``mallory`` sending
``agent_id=worker`` read a PRIVATE memory owned by ``agent:worker`` and written for a
different user, HTTP 200. Without the agent_id the same request was correctly 403.
"""

from __future__ import annotations

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import Visibility
from memory_service.modules.authz.visibility import visibility_keys


def ctx(user: str | None, agent: str | None = None) -> MemoryExecutionContext:
    return MemoryExecutionContext(tenant_id="acme", user_id=user, agent_id=agent)


def test_two_users_running_the_same_agent_are_two_principals() -> None:
    """The whole bug in one assertion. Both send agent_id="research"; they must not share
    a PRIVATE scope."""
    assert ctx("alice", "research").principal_id != ctx("bob", "research").principal_id


def test_the_principal_names_the_user_it_runs_for() -> None:
    assert ctx("alice", "research").principal_id == "agent:alice/research"


def test_an_agent_with_no_user_keeps_the_bare_form() -> None:
    """An ingestion job or an unattended scheduled run has no user to bind to — and nothing
    for a caller to impersonate their way into, because there is no other user's data under
    that principal."""
    assert ctx(None, "ingest").principal_id == "agent:ingest"


def test_a_user_without_an_agent_is_unchanged() -> None:
    assert ctx("alice").principal_id == "user:alice"
    assert ctx(None).principal_id == "service:anonymous"


def test_a_private_memory_key_cannot_be_reached_by_naming_the_agent() -> None:
    """The key is what authorization actually compares, so the binding has to survive into
    it — asserting only on principal_id would pass even if the key were built some other way.
    """
    alice = ctx("alice", "research")
    mallory = ctx("mallory", "research")
    alice_keys = visibility_keys(
        "acme", Visibility.PRIVATE, owner_principal=alice.principal_id, scope=None
    )
    mallory_keys = visibility_keys(
        "acme", Visibility.PRIVATE, owner_principal=mallory.principal_id, scope=None
    )
    assert set(alice_keys).isdisjoint(mallory_keys)
