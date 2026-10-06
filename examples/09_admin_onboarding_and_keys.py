"""09 · Administration: onboard a tenant, issue and scope keys, create a workspace, revoke.

The platform operator's bootstrap key does one thing: it onboards a tenant and returns the
tenant's first admin key, once. The admin key issues the keys callers use, creates the
tenant's workspaces (teams) and their members, and revokes keys; a revoked key is refused on
its next request. Unset the bootstrap key once onboarding is done.

    uv run python examples/09_admin_onboarding_and_keys.py
"""

from __future__ import annotations

import asyncio

from _support import run_id, service

from trellis.memory import AuthenticationError


async def main() -> None:
    async with service(onboarding=True) as svc:
        tenant_id = f"globex-{run_id()}"
        async with svc.client(svc.bootstrap_key) as platform:
            created = await platform.admin.create_tenant(
                "Globex", tenant_id=tenant_id, retention_days=365
            )
            print("tenant:", created.tenant.tenant_id, "(its first admin key is shown once)")
        assert created.admin_key.token

        async with svc.client(created.admin_key.token) as admin:
            me = await admin.tenant.keys.whoami()
            print("admin key acts for tenant:", me.tenant_id, "role:", me.role)
            assert me.tenant_id == tenant_id

            # a service key that may act only for one user and one agent
            scoped = await admin.tenant.keys.issue(
                "service", "support-bot", may_act_as=["user:u1", "agent:support"]
            )
            team = await admin.tenant.workspaces.create("Finance", workspace_id="finance")
            await admin.tenant.workspaces.set_member(team.workspace_id, "user:u1")
            members = await admin.tenant.workspaces.members(team.workspace_id)
            print("workspace:", team.workspace_id, "members:", [m.principal for m in members])
            keys = await admin.tenant.keys.list()
            print("keys:", sorted((k.name, k.role) for k in keys))

            async with svc.client(scoped.token) as bot:
                ctx = bot.bind(user_id="u1", workspace_id="finance", thread_id=f"thr-{run_id()}")
                await ctx.history.add([("USER", "Close the books for September.")])
                print("the scoped key wrote to the workspace thread")

                await admin.tenant.keys.revoke(scoped.key_id)
                try:
                    await ctx.history()
                    raise AssertionError("a revoked key was accepted")
                except AuthenticationError:
                    print("revoked: refused on its next request (401)")


if __name__ == "__main__":
    asyncio.run(main())
