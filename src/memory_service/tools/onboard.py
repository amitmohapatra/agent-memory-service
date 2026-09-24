"""Grant a user their memberships in a tenant.

    uv run python -m memory_service.tools.onboard --tenant arhaus --user alice \
        [--workspace eng --workspace design] [--group legal] [--admin]

A tenant needs no registration: identifiers are tenant-prefixed everywhere, so the first
write to ``arhaus`` creates it and nothing else is required to read your own memories back.
What DOES need granting is the sharing a tenant is for - WORKSPACE-visible memories resolve
the reader's workspaces from the authorization store, so without a membership tuple a
team-visible memory is readable only by whoever wrote it.

``AuthorizationService.grant_membership`` has always written exactly the right tuples and
has never had a caller: no endpoint, no command, no bootstrap path. This is that caller.
It is an operator tool rather than an API on purpose - it hands out authority, the service
principal that authenticates a request carries no role to check against (it identifies the
calling *service*, and the tenant and user are asserted in headers), and inventing a
super-admin surface is a decision to take deliberately rather than as a side effect.

``--admin`` grants tenant admin, which is read by ``is_tenant_admin``: it lets a principal
forget any memory in the tenant and widen a tool policy. Nothing could grant it before, so
those two branches were unreachable.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from memory_service.__about__ import __version__
from memory_service.application.container import build_container
from memory_service.config.settings import Settings
from memory_service.observability.logging import get_logger

log = get_logger(__name__)


async def _main(args: argparse.Namespace) -> int:
    container = await build_container(Settings(), __version__)
    try:
        authz = container.services["authz"]
        async with container.services["uow_factory"]() as uow:
            # inside the unit of work so the membership revision is bumped with the write:
            # the caller's cached scope is invalidated immediately instead of after the TTL
            await authz.grant_membership(
                args.tenant,
                args.user,
                admin=args.admin,
                revisions=uow.revisions,
            )
            await uow.commit()
    finally:
        await container.close()
    sys.stdout.write(
        f"granted {args.user} in {args.tenant}: admin={args.admin}\n"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Grant a user their memberships in a tenant")
    parser.add_argument("--tenant", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument(
        "--admin",
        action="store_true",
        help="tenant admin: may forget any memory in the tenant and widen tool policy",
    )
    return asyncio.run(_main(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
