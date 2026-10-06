"""02 · Conversation history: threads, turns, internal messages, replays, isolation.

The transcript is the service's record of what was said. Ids are yours: a thread is created
the first time it is named. A write is acknowledged only once it is durable, and the same
message sent twice is one message (an idempotent replay). An agent's internal notes are kept
in the transcript but never shown as chat; an ``EVENT`` is something that happened.

    uv run python examples/02_conversation_history.py
"""

from __future__ import annotations

import asyncio

from _support import run_id, service

from trellis.memory import AuthorizationError


async def main() -> None:
    async with service() as svc, svc.client() as memory:
        run = run_id()
        # session and turn are optional: without them a USER message opens the thread's next
        # turn. Naming the turn is what makes a retry of the same message a replay.
        ctx = memory.bind(
            user_id="u1", thread_id=f"thr-{run}", session_id=f"ses-{run}", turn_id=f"trn-{run}-1"
        )

        thread = await ctx.history.update(title="FY26 brief", metadata={"channel": "examples"})
        print("thread:", thread.thread_id, repr(thread.title))

        said = ("USER", "My timezone is Europe/Berlin and I prefer concise answers.")
        [first] = await ctx.history.add([said])
        [again] = await ctx.history.add([said])
        assert again.message_id == first.message_id, "the same message twice is one message"
        print("acknowledged:", first.message_id, "jobs:", first.job_ids)

        [reply] = await ctx.history.add([("ASSISTANT", "Noted: Europe/Berlin, concise answers.")])
        await ctx.agent("planner").history.add(
            [{"role": "AGENT", "kind": "INTERNAL", "content": "Plan: revenue first, then cost."}]
        )
        await ctx.history.add([("EVENT", "The user opened the FY26 report.")])

        visible = await ctx.history()
        everything = await ctx.history(include_internal=True)
        print("visible:", [(m.sequence, m.role) for m in visible])
        print("all:    ", [(m.sequence, m.role, m.kind) for m in everything])
        assert [m.role for m in visible] == ["USER", "ASSISTANT"]
        assert len(everything) == 4

        one = await ctx.history.message(reply.message_id)
        assert one.content.startswith("Noted")

        # another user of the same tenant may not read this thread
        stranger = memory.bind(user_id="mallory", thread_id=ctx.scope.thread_id)
        try:
            await stranger.history()
            raise AssertionError("another user read the thread")
        except AuthorizationError:
            print("another user: refused (403)")

        await ctx.history.delete()
        print("thread deleted (soft delete; the archive is kept per policy)")


if __name__ == "__main__":
    asyncio.run(main())
