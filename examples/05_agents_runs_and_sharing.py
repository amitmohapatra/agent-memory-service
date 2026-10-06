"""05 · Several agents: private run memory, hand-offs down the run tree, explicit sharing.

Each agent run has its own scope. A ``RUN`` note reaches the run that wrote it and the runs
it spawns, and nobody else: not a sibling agent, not the user. Sharing with a crew is
explicit (``AGENT_GROUP``).

    uv run python examples/05_agents_runs_and_sharing.py
"""

from __future__ import annotations

import asyncio

from _support import run_id, service

NOTE = "Source A contradicts source B on FY26 revenue."
SHARED = "FY26 revenue is EUR 412m, confirmed in two sources."


async def main() -> None:
    async with service() as svc, svc.client() as memory:
        user = memory.bind(user_id=f"u-{run_id()}", thread_id=f"thr-{run_id()}")
        researcher = user.agent("researcher", agent_group_id="analysis-crew")
        writer = user.agent("writer", agent_group_id="analysis-crew")
        fact_checker = researcher.agent("fact-checker")  # a child run of the researcher

        await researcher.remember(NOTE, visibility="RUN")

        async def sees(scope, text: str) -> bool:
            return any(text in (i.text or "") for i in await scope.search(text, kinds=["memory"]))

        visible = {
            "researcher (owner)": await sees(researcher, NOTE),
            "fact-checker (child run)": await sees(fact_checker, NOTE),
            "writer (sibling)": await sees(writer, NOTE),
            "the user": await sees(user, NOTE),
        }
        for who, seen in visible.items():
            print(f"RUN note seen by {who:<26} {seen}")
        assert list(visible.values()) == [True, True, False, False]

        await researcher.remember(SHARED, memory_type="SHARED", visibility="AGENT_GROUP")
        assert await sees(writer, SHARED), "an AGENT_GROUP memory reaches the crew"
        print("AGENT_GROUP memory seen by the writer: True")
        print(
            "run lineage:",
            fact_checker.scope.parent_agent_run_id,
            "->",
            fact_checker.scope.agent_run_id,
        )


if __name__ == "__main__":
    asyncio.run(main())
