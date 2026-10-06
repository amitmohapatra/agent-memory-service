"""01 · Quickstart: record a turn, ask for the context of the next one.

The smallest useful program: bind a scope once, add what the user said, and ask the service
for everything relevant to the next question. The bundle's ``rendered`` text is what you put
in front of your model; ``evidence_status`` says whether the service found what an answer
needs. Then the same turn as an agent run, ending with the run's outcome as feedback. These
are the README's two snippets, run.

    uv run python examples/01_quickstart_context.py
"""

from __future__ import annotations

import asyncio

from _support import run_id, service


async def main() -> None:
    async with service() as svc, svc.client() as memory:
        # the development key acts in the tenant "default"; the thread is created on first use
        ctx = memory.bind(user_id="u1", thread_id=f"chat-{run_id()}")

        await ctx.history.add([("USER", "I'm in Berlin and I prefer short answers.")])
        bundle = await ctx.context("draft a reply about the Q3 numbers")
        answer = f"(your agent answers here, from {bundle.token_estimate} tokens of context)"
        await ctx.history.add([("ASSISTANT", answer)])

        print("bundle_id:      ", bundle.bundle_id)
        print("evidence_status:", bundle.evidence_status)
        print("rendered:\n" + bundle.rendered)
        assert bundle.bundle_id and bundle.rendered
        assert "Berlin" in bundle.rendered, "the user's own turn is in the next context"

        said = [(m.role, m.content) for m in await ctx.history()]
        assert said[-1] == ("ASSISTANT", answer)
        print(f"\nthe thread holds {len(said)} visible messages")

        # The same turn as an agent run whose framework keeps its own history (window=False),
        # then the run's outcome: feedback is what the service learns from.
        run = memory.bind(user_id="u1", thread_id=f"chat-{run_id()}").agent("support")
        question = "What changed in EBITDA?"
        pushed = await run.context(question, window=False)
        await run.history.add([("USER", question), ("ASSISTANT", "EBITDA rose 21%.")])
        verdict = await run.feedback("run", run.scope.agent_run_id, "confirm")
        print(
            "run",
            verdict.target_id,
            "->",
            verdict.verdict,
            f"({pushed.token_estimate} tokens pushed)",
        )


if __name__ == "__main__":
    asyncio.run(main())
