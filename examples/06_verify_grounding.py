"""06 · Verify: did the answer follow from the context it was given?

``verify`` splits an answer into claims and checks each one against the bundle it was built
from (``bundle_id``): ``supported``, ``unsupported``, ``contradicted`` or ``borderline``. With
an agent run in the scope the report is also recorded as the judge's feedback on that run.

Offline, the NLI is the lexical stand-in, so the verdicts show the mechanism, not the
quality of a real entailment model.

    uv run python examples/06_verify_grounding.py
"""

from __future__ import annotations

import asyncio

from _support import FIXTURES, run_id, service

QUESTION = "Why did Adjusted EBITDA increase despite lower revenue?"
GOOD = "Adjusted EBITDA increased to EUR 98 million from EUR 81 million, despite lower revenue."
BAD = "Adjusted EBITDA increased to EUR 150 million from EUR 81 million."


async def main() -> None:
    async with service() as svc, svc.client() as memory:
        user = memory.bind(user_id=f"u-{run_id()}", thread_id=f"thr-{run_id()}")
        handle = await user.advanced.documents.add(FIXTURES / "acme_fy26_annual_report.md")
        await user.advanced.documents.wait_ready(handle.document_id)

        run = user.agent("analyst")
        bundle = await run.context(QUESTION)
        report = await run.verify(f"{GOOD} {BAD}", bundle_id=bundle.bundle_id)
        for claim in report.claims:
            print(f"{claim.verdict:<13} {claim.claim}")
        print("per-claim hallucination rate:", report.per_claim_hallucination_rate)
        assert [c.verdict for c in report.claims] == ["supported", "contradicted"]

        # the run was judged: the report is the judge's RUN feedback, applied without review
        assert report.feedback_id
        verdict = await run.feedback.get(report.feedback_id)
        print("recorded as feedback:", verdict.target_kind, verdict.target_id, verdict.verdict)


if __name__ == "__main__":
    asyncio.run(main())
