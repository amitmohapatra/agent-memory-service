"""Onboarding grants tenant membership and tenant admin, and nothing wider.

A tenant needs no registration - identifiers are tenant-prefixed everywhere, so the first
write creates it and nothing is required to read your own memories back. What the tool exists
for is the authority that cannot be asserted in a header: tenant admin, which gates forgetting
another principal's memory and widening a tool policy.

``grant_membership`` had no caller anywhere before this - no endpoint, no command - so admin
could never be granted and both branches reading it were unreachable in a running service.
"""

from __future__ import annotations

import asyncio
import runpy
import secrets
import sys
import warnings
from typing import Any

import pytest

from memory_service.__about__ import __version__
from memory_service.application.container import build_container
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.revisions import RevisionKind
from memory_service.modules.authz.service import AuthorizationService
from memory_service.ports.authorization import AccessCheck
from memory_service.tools import onboard
from tests.integration.conftest import integration_overrides, integration_settings

pytestmark = pytest.mark.integration


async def test_admin_is_grantable_at_all(container, uow_factory) -> None:
    authz: AuthorizationService = container.services["authz"]
    root = MemoryExecutionContext(tenant_id="arhaus", user_id="root")
    assert not await authz.is_tenant_admin(root)

    async with uow_factory() as uow:
        await authz.grant_membership("arhaus", "root", admin=True, revisions=uow.revisions)
        await uow.commit()
    assert await authz.is_tenant_admin(root)


async def test_admin_in_one_tenant_is_not_admin_in_another(container, uow_factory) -> None:
    """Membership objects are tenant-prefixed, so the same user id is a different principal."""
    authz: AuthorizationService = container.services["authz"]
    async with uow_factory() as uow:
        await authz.grant_membership("arhaus", "carol", admin=True, revisions=uow.revisions)
        await uow.commit()

    assert await authz.is_tenant_admin(MemoryExecutionContext(tenant_id="arhaus", user_id="carol"))
    assert not await authz.is_tenant_admin(
        MemoryExecutionContext(tenant_id="globex", user_id="carol")
    )


async def test_plain_membership_confers_no_admin(container, uow_factory) -> None:
    authz: AuthorizationService = container.services["authz"]
    async with uow_factory() as uow:
        await authz.grant_membership("arhaus", "dave", revisions=uow.revisions)
        await uow.commit()
    assert not await authz.is_tenant_admin(
        MemoryExecutionContext(tenant_id="arhaus", user_id="dave")
    )


# --------------------------------------------------------------------------- the command


def _command(monkeypatch, make_settings, tmp_path, *argv: str) -> list[Any]:
    """Point ``memory_service.tools.onboard`` at the suite's database and record the
    container it builds (the authorization store is the in-process one, so the grant is read
    back from it)."""
    built: list[Any] = []
    settings = integration_settings(
        make_settings, blob={"provider": "filesystem", "filesystem_root": str(tmp_path)}
    )

    async def build(_settings: Any, version: str) -> Any:
        c = await build_container(settings, version, overrides=integration_overrides(blob=None))
        built.append(c)
        return c

    monkeypatch.setattr(onboard, "build_container", build)
    monkeypatch.setattr(sys, "argv", ["onboard", *argv])
    return built


def _membership_revision(make_settings, tmp_path, tenant_id: str, user_id: str) -> int:
    async def read() -> int:
        settings = integration_settings(
            make_settings, blob={"provider": "filesystem", "filesystem_root": str(tmp_path)}
        )
        c = await build_container(settings, __version__, overrides=integration_overrides())
        try:
            async with c.services["uow_factory"]() as uow:
                found = await uow.revisions.get_many(
                    tenant_id, [(RevisionKind.MEMBERSHIP, user_id)]
                )
        finally:
            await c.close()
        return found.get(f"membership:{user_id}", 0)

    return asyncio.run(read())


def test_the_command_grants_tenant_admin_and_says_so(
    monkeypatch, make_settings, tmp_path, capsys
) -> None:
    tenant = f"onb-{secrets.token_hex(3)}"
    built = _command(
        monkeypatch, make_settings, tmp_path, "--tenant", tenant, "--user", "alice", "--admin"
    )

    assert onboard.main() == 0

    assert capsys.readouterr().out.splitlines()[-1] == f"granted alice in {tenant}: admin=True"
    [c] = built
    authz: AuthorizationService = c.services["authz"]
    alice = MemoryExecutionContext(tenant_id=tenant, user_id="alice")
    assert asyncio.run(authz.is_tenant_admin(alice))
    # the grant bumped alice's membership revision in the same unit of work
    assert _membership_revision(make_settings, tmp_path, tenant, "alice") >= 1


def test_the_command_grants_plain_membership_without_admin(
    monkeypatch, make_settings, tmp_path, capsys
) -> None:
    tenant = f"onb-{secrets.token_hex(3)}"
    built = _command(monkeypatch, make_settings, tmp_path, "--tenant", tenant, "--user", "bob")

    assert onboard.main() == 0

    assert capsys.readouterr().out.splitlines()[-1] == f"granted bob in {tenant}: admin=False"
    authz: AuthorizationService = built[0].services["authz"]
    bob = MemoryExecutionContext(tenant_id=tenant, user_id="bob")
    assert not asyncio.run(authz.is_tenant_admin(bob))
    member = asyncio.run(
        authz.provider.check(
            AccessCheck(user="user:bob", relation="member", object=f"tenant:{tenant}")
        )
    )
    assert member
    assert _membership_revision(make_settings, tmp_path, tenant, "bob") >= 1


def test_the_command_closes_its_container_when_the_grant_fails(
    monkeypatch, make_settings, tmp_path, capsys
) -> None:
    built = _command(monkeypatch, make_settings, tmp_path, "--tenant", "acme", "--user", "carol")
    closed: list[bool] = []
    real_build = onboard.build_container

    async def build_failing(settings: Any, version: str) -> Any:
        c = await real_build(settings, version)
        real_close = c.close

        async def grant_fails(*_: Any, **__: Any) -> None:
            raise RuntimeError("authorization store down")

        async def close() -> None:
            closed.append(True)
            await real_close()

        c.services["authz"].grant_membership = grant_fails
        c.close = close
        return c

    monkeypatch.setattr(onboard, "build_container", build_failing)
    with pytest.raises(RuntimeError, match="authorization store down"):
        onboard.main()
    assert closed == [True] and len(built) == 1
    assert "granted" not in capsys.readouterr().out


@pytest.mark.parametrize("argv", [["--tenant", "acme"], ["--user", "alice"], []])
def test_the_command_requires_a_tenant_and_a_user(monkeypatch, argv: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["onboard", *argv])
    with pytest.raises(SystemExit) as exit_:
        onboard.main()
    assert exit_.value.code == 2


def test_the_module_runs_as_a_command(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["onboard", "--help"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # runpy re-executing an imported module
        with pytest.raises(SystemExit) as exit_:
            runpy.run_module("memory_service.tools.onboard", run_name="__main__")
    assert exit_.value.code == 0
    assert "Grant a user their memberships in a tenant" in capsys.readouterr().out
