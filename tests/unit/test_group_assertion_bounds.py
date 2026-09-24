"""What a caller may assert about its own group membership, and how much of it.

`X-Memory-Groups` is believed without asking the authorization provider. ADR 0005 decides
that deliberately - the header comes from an authenticated upstream that has already resolved
the caller's memberships - but `ScopeResolver` took a `trust_header_groups` switch that
nothing could reach: not wiring, not constants, not settings. The off position existed only in
the signature and in the ADR's prose, so a deployment whose upstream is not trusted had no way
to say so without editing source.

The list was also uncapped. Each group becomes a `group:{tenant}/{g}` store-side read key, and
the header is split on commas, so one request could assert as many as it liked - the same
unbounded shape the body-supplied group fix closed, left open on the header path because only
the body was examined.
"""

from __future__ import annotations

import pytest

from memory_service.config.constants import AUTHORIZATION
from memory_service.domain.context import MemoryExecutionContext
from memory_service.modules.authz.scope import ScopeResolver

pytestmark = pytest.mark.unit


class _NoMemberships:
    """An authorization provider that would grant nothing if it were asked."""

    async def list_objects(self, *args, **kwargs) -> list[str]:
        return []

    async def batch_check(self, checks) -> list[bool]:
        return [False for _ in checks]


def _ctx(groups: list[str]) -> MemoryExecutionContext:
    return MemoryExecutionContext(tenant_id="acme", user_id="u1", group_ids=groups)


def test_the_switch_is_reachable_from_configuration() -> None:
    """The defect: the parameter existed and nothing could set it."""
    assert hasattr(AUTHORIZATION, "trust_header_groups")
    assert AUTHORIZATION.trust_header_groups is True, "ADR 0005 keeps this on by default"


async def test_trusting_the_header_admits_the_asserted_groups() -> None:
    resolver = ScopeResolver(_NoMemberships(), trust_header_groups=True)  # type: ignore[arg-type]
    scope = await resolver.resolve(_ctx(["legal", "finance"]))
    assert set(scope.group_ids) >= {"legal", "finance"}


async def test_turning_it_off_drops_groups_the_provider_will_not_confirm() -> None:
    """The position that was unreachable. The provider grants nothing, so nothing survives."""
    resolver = ScopeResolver(_NoMemberships(), trust_header_groups=False)  # type: ignore[arg-type]
    scope = await resolver.resolve(_ctx(["legal", "finance"]))
    assert scope.group_ids == [], f"unconfirmed groups survived: {scope.group_ids}"


def test_the_number_of_assertable_groups_is_bounded() -> None:
    assert AUTHORIZATION.max_asserted_groups > 0
