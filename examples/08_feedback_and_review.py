"""08 · Feedback and the review queue: a vote waits for the tenant's administrator.

A verdict on a memory or a run is how the platform learns. A vote from a person or an agent
would change what was learned on one word, so it is stored ``pending`` and changes nothing
until the tenant's administrator approves it (ADR 0028). Some verdicts apply as they arrive:
the service's own judge, a run reporting its own status, an owner correcting a memory.

This runs the service the way a deployment does (``api_key`` mode): the platform operator
onboards the tenant, the tenant's admin key issues a service key, and the service key is
what an agent harness holds.

    uv run python examples/08_feedback_and_review.py
"""

from __future__ import annotations

import asyncio

from _support import run_id, service

FACT = "The payments team reviews releases on Thursdays at 15:00."


async def main() -> None:
    async with service(onboarding=True) as svc:
        tenant_id = f"acme-{run_id()}"
        async with svc.client(svc.bootstrap_key) as platform:
            created = await platform.admin.create_tenant("Acme", tenant_id=tenant_id)
        admin_token = created.admin_key.token
        assert admin_token, "the admin key's token is shown once"
        async with svc.client(admin_token) as admin_client:
            issued = await admin_client.tenant.keys.issue("service", "acme-harness")
        assert issued.token

        async with svc.client(issued.token) as harness, svc.client(admin_token) as admin_client:
            user = harness.bind(user_id="u1")
            admin = admin_client.bind(tenant_id=tenant_id)

            await user.remember(FACT, visibility="USER")
            memory = next(m for m in await user.advanced.memories.list() if m.content == FACT)

            vote = await user.feedback("memory", memory.memory_id, "confirm")
            print("user's vote:", vote.review.state if vote.review else None)
            assert vote.review is not None and vote.review.state == "pending"
            unchanged = await user.advanced.memories.get(memory.memory_id)
            assert unchanged.reinforcement_count == memory.reinforcement_count, "nothing moved yet"

            queue = await admin.feedback.pending()
            print("review queue:", [(f.target_kind, f.verdict) for f in queue.items])
            assert [f.feedback_id for f in queue.items] == [vote.feedback_id]

            await admin.feedback.approve(vote.feedback_id, note="matches the calendar")
            # the projection is a later fact (a job; inline here): read the verdict back
            applied = await user.feedback.get(vote.feedback_id)
            assert applied.projection is not None
            print("approved; projection:", applied.projection.action)
            after = await user.advanced.memories.get(memory.memory_id)
            print("reinforcement:", memory.reinforcement_count, "->", after.reinforcement_count)
            assert after.reinforcement_count > memory.reinforcement_count
            assert (await admin.feedback.pending()).items == []

            # a run reporting its own final status is applied as it arrives (source="system")
            run = user.agent("support")
            status = await run.feedback("run", run.scope.agent_run_id, "confirm", source="system")
            print(
                "a run's own status:", "applied" if status.review is None else status.review.state
            )
            assert status.review is None


if __name__ == "__main__":
    asyncio.run(main())
