"""03 · Memories: what the service learns, what you state, how a fact changes, and recall.

A user message becomes memories in the background: "my timezone is ..." is a fact about the
user. ``remember`` states one directly. When a fact changes, the old one is superseded, not
deleted: it stays readable in a temporal view and is never served as current. ``search``
(``POST /v1/recall``) returns ranked items without assembling a bundle, and ``forget`` takes
one out of everything.

    uv run python examples/03_remember_search_forget.py
"""

from __future__ import annotations

import asyncio

from _support import run_id, service


async def main() -> None:
    async with service() as svc, svc.client() as memory:
        ctx = memory.bind(user_id=f"u-{run_id()}", thread_id=f"thr-{run_id()}")

        # learned from what was said (the job ran inline, so it is there when this returns)
        await ctx.history.add([("USER", "My timezone is Europe/Berlin.")])
        # stated directly, verbatim
        preference = await ctx.remember(
            "Always answer in British English.", memory_type="PREFERENCE"
        )
        task = await ctx.remember("The brief is due on Friday.", memory_type="TASK")

        memories = await ctx.advanced.memories.list()
        for m in memories:
            print(f"{m.memory_type:<11} {m.temporal_status:<10} {m.content}")
        assert any(m.predicate == "timezone" for m in memories)

        # a fact changes: the new version is current, the old one is closed and kept
        moved = await ctx.update(task.memory_id, "The brief is due on Monday.", reason="moved")
        old = await ctx.advanced.memories.get(task.memory_id)
        assert old.temporal_status == "SUPERSEDED" and old.superseded_by == moved.memory_id
        print("superseded:", task.memory_id, "->", moved.memory_id)

        items = await ctx.search("when is the brief due", kinds=["memory"])
        print("recall:", [(i.kind, i.text) for i in items[:3]])
        assert any("Monday" in i.text for i in items)
        assert not any(i.id == task.memory_id for i in items), "a superseded fact is not served"

        await ctx.forget(preference.memory_id)
        left = {m.memory_id for m in await ctx.advanced.memories.list()}
        assert preference.memory_id not in left
        print("forgotten:", preference.memory_id)


if __name__ == "__main__":
    asyncio.run(main())
