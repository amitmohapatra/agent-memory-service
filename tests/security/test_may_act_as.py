"""A key restricted to some principals (``may_act_as``) acts for those and no others.

The list names users (``user:<id>``) and agents (``agent:<id>``). Only the user entries were
ever compared with anything: a key listing one agent acted as any other agent by naming it in
the body, and the agent half of every list was decoration. Each case below is one way a
request names a principal; the last two run against the service to show what a restricted
key reaches, not only what ``build_context`` says.
"""

from __future__ import annotations

from typing import Any

import pytest

from memory_service.api.deps import ScopeBody, build_context
from memory_service.domain.errors import ScopeDenied
from memory_service.modules.auth.authentication import ServicePrincipal
from tests.agent.conftest import BOOTSTRAP, app, running, sdk  # noqa: F401 - fixtures
from trellis.memory import AuthorizationError

pytestmark = pytest.mark.security


class _State:
    def __init__(self, principal: ServicePrincipal) -> None:
        self.service_principal = principal

    def __getattr__(self, name: str) -> str:
        return f"{name}_test"


class _Request:
    def __init__(self, principal: ServicePrincipal, user: str | None) -> None:
        self.headers = {"X-Trellis-User": user} if user else {}
        self.state = _State(principal)


class _Container:
    def __init__(self) -> None:
        from memory_service.config.settings import Settings

        self.settings = Settings()
        self.services: dict[str, Any] = {}


def _key(*may_act_as: str) -> ServicePrincipal:
    return ServicePrincipal(
        service_id="key:k1",
        mode="api_key",
        claims={
            "role": "service",
            "tenant": "acme",
            "workspace": None,
            "key_id": "k1",
            "may_act_as": list(may_act_as),
        },
    )


def _act(principal: ServicePrincipal, *, user: str | None = None, agent: str | None = None):
    return build_context(
        _Request(principal, user),  # type: ignore[arg-type]
        _Container(),  # type: ignore[arg-type]
        ScopeBody(agent_id=agent),
    )


def test_a_listed_user_and_a_listed_agent_are_acted_for() -> None:
    key = _key("user:alice", "agent:reorder")
    assert _act(key, user="alice").principal_id == "user:alice"
    assert _act(key, user="alice", agent="reorder").principal_id == "agent:alice/reorder"
    assert _act(key, agent="reorder").principal_id == "agent:reorder"


def test_an_unlisted_user_is_refused() -> None:
    with pytest.raises(ScopeDenied, match="that user"):
        _act(_key("user:alice", "agent:reorder"), user="bob", agent="reorder")


def test_an_unlisted_agent_is_refused_even_for_a_listed_user() -> None:
    with pytest.raises(ScopeDenied, match="that agent") as exc:
        _act(_key("user:alice", "agent:reorder"), user="alice", agent="pricing")
    assert exc.value.details == {"field": "agent_id"}
    with pytest.raises(ScopeDenied, match="that agent"):
        _act(_key("user:alice"), user="alice", agent="reorder")


def test_an_unlisted_agent_with_no_user_is_refused() -> None:
    """The unattended form (``agent:<id>``, no user) is an identity too: its PRIVATE notes are
    that agent's, and naming it is acting as it."""
    with pytest.raises(ScopeDenied, match="that agent"):
        _act(_key("user:alice"), agent="reorder")


def test_a_restricted_key_naming_nobody_acts_as_itself() -> None:
    assert _act(_key("user:alice")).principal_id == "service:anonymous"
    assert _act(_key()).principal_id == "service:anonymous"  # empty: only the key itself
    with pytest.raises(ScopeDenied):
        _act(_key(), user="alice")
    with pytest.raises(ScopeDenied):
        _act(_key(), agent="reorder")


def test_any_principal_lifts_the_restriction() -> None:
    key = _key("*")
    assert _act(key, user="bob", agent="pricing").principal_id == "agent:bob/pricing"
    assert _act(key, agent="pricing").principal_id == "agent:pricing"


def test_credentials_that_carry_no_list_are_not_restricted() -> None:
    """A development key and an issuer's token name no principals: nothing to check."""
    for mode in ("trusted_dev", "jwt"):
        principal = ServicePrincipal(service_id="svc", mode=mode, claims={})
        ctx = build_context(
            _Request(principal, "bob"),  # type: ignore[arg-type]
            _Container(),  # type: ignore[arg-type]
            ScopeBody(tenant_id="acme", agent_id="pricing"),
        )
        assert ctx.principal_id == "agent:bob/pricing"


# ----------------------------------------------------------------------------- end to end


async def test_a_restricted_key_reaches_only_the_principals_it_lists(app, running) -> None:  # noqa: F811
    platform = sdk(app, BOOTSTRAP)
    acme = await platform.admin.create_tenant("Acme", tenant_id="acme")
    admin = sdk(app, acme.admin_key.token)
    harness = sdk(app, (await admin.tenant.keys.issue("service", "harness")).token)
    bob = harness.bind(user_id="bob")
    await bob.remember("Bob's salary review is on Friday.", visibility="USER")
    await bob.agent("pricing").remember("Pricing note for Bob: margin floor 12%.")

    narrow = await admin.tenant.keys.issue(
        "service", "alice-reorder", may_act_as=["user:alice", "agent:reorder"]
    )
    restricted = sdk(app, narrow.token)
    alice = restricted.bind(user_id="alice")
    await alice.remember("Alice reorders SKU-22 every Monday.", visibility="USER")
    await alice.agent("reorder").search("SKU-22")

    with pytest.raises(AuthorizationError):
        await restricted.bind(user_id="bob").search("salary review")
    with pytest.raises(AuthorizationError):
        await alice.agent("pricing").search("margin floor")
    with pytest.raises(AuthorizationError):
        await restricted.bind(agent_id="pricing").search("margin floor")


async def test_a_restricted_key_naming_nobody_reads_no_user_s_memories(app, running) -> None:  # noqa: F811
    platform = sdk(app, BOOTSTRAP)
    acme = await platform.admin.create_tenant("Acme", tenant_id="acme")
    admin = sdk(app, acme.admin_key.token)
    harness = sdk(app, (await admin.tenant.keys.issue("service", "harness")).token)
    bob = harness.bind(user_id="bob")
    await bob.remember("Bob's salary review is on Friday.", visibility="USER")
    await bob.remember("Bob's private reminder: salary review prep.", visibility="PRIVATE")
    await bob.agent("pricing").remember("Pricing salary review note.", visibility="PRIVATE")
    assert await bob.search("salary review"), "the memories are there to be found"

    narrow = await admin.tenant.keys.issue("service", "alice-only", may_act_as=["user:alice"])
    nobody = sdk(app, narrow.token).bind()
    found = await nobody.search("salary review", limit=50)
    assert found == []
    assert await nobody.advanced.memories.list() == []
