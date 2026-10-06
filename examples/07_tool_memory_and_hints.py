"""07 · Tool memory: publish the catalog, record what ran, and get a plan back.

The service never runs a tool. It keeps a catalog, records each call a run made, and learns
from runs labelled successful: the procedure (which tool after which) and where each
argument came from. ``tool_hints`` reads it back: the next step and its arguments, bound
from an earlier step's output.

    uv run python examples/07_tool_memory_and_hints.py
"""

from __future__ import annotations

import asyncio

from _support import run_id, service

TASK = "update quote Q-1183 with EMEA price for SKU-22"
TOOLS = [{"name": "pricing.lookup_price"}, {"name": "crm.update_quote"}]
NAMES = [t["name"] for t in TOOLS]


async def main() -> None:
    async with service() as svc, svc.client() as memory:
        user = memory.bind(user_id=f"u-{run_id()}", thread_id=f"thr-{run_id()}")
        await user.agent("ops").advanced.tools.put_catalog(TOOLS)

        for i in range(3):
            run = user.agent("ops", agent_run_id=f"run-{run_id()}")
            price = {"price": 1200, "currency": "EUR", "quote_id": f"Q-{i}"}
            await run.record_tool(
                "pricing.lookup_price",
                {"sku": f"SKU-{i}", "region": "EMEA"},
                output=price,
                task=TASK,
                step=0,
            )
            await run.record_tool(
                "crm.update_quote",
                {"quote_id": f"Q-{i}", "amount": 1200},
                output={"ok": True},
                task=TASK,
                step=1,
            )
            # only a run labelled successful teaches a procedure
            await run.feedback("run", run.scope.agent_run_id, "confirm", source="system")

        hints = await user.agent("ops").tool_hints(TASK, available=NAMES)
        assert hints.plan is not None and hints.next is not None
        print(
            "plan:",
            " -> ".join(hints.plan.steps),
            f"({hints.plan.runs} runs, {hints.plan.success_rate:.0%})",
        )
        print("first step:", hints.next.name)
        assert hints.plan.steps == NAMES

        # a new run: after step 0, the hint is step 1 with its argument bound from step 0's output
        run = user.agent("ops", agent_run_id=f"run-{run_id()}")
        await run.record_tool(
            "pricing.lookup_price",
            {"sku": "SKU-22", "region": "EMEA"},
            output={"price": 1300, "currency": "EUR", "quote_id": "Q-77"},
            task=TASK,
            step=0,
        )
        after = await run.tool_hints(TASK, available=NAMES)
        assert after.next is not None
        print("next step:", after.next.name, after.next.args)
        assert after.next.name == "crm.update_quote" and after.next.args["quote_id"] == "Q-77"


if __name__ == "__main__":
    asyncio.run(main())
