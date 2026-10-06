"""10 · Background jobs: what a write queues, how to follow it, and the rolling summary.

A write is acknowledged once it and its jobs are committed together (the transactional
outbox); extraction, indexing, archiving and summaries run afterwards in the worker (inline
here). Every acknowledgement names its jobs, and ``advanced.job(id)`` follows one. Every 20
messages the thread's durable summary rolls forward, so a long conversation is compacted into
something a context can carry.

    uv run python examples/10_background_jobs_and_summary.py
"""

from __future__ import annotations

import asyncio

from _support import run_id, service


async def main() -> None:
    async with service() as svc, svc.client() as memory:
        ctx = memory.bind(user_id=f"u-{run_id()}", thread_id=f"thr-{run_id()}")

        [ack] = await ctx.history.add([("USER", "My team is moving the launch to November.")])
        print("acknowledged:", ack.message_id, "queued:", ack.job_ids)
        for job_id in ack.job_ids:
            job = await ctx.advanced.job(job_id)
            print(f"  {job.job_id}: {job.status} after {job.attempts} attempt(s)")
        assert ack.job_ids

        turns = [
            (
                "USER" if i % 2 == 0 else "ASSISTANT",
                f"Launch planning note {i}: item {i} is on track.",
            )
            for i in range(19)
        ]
        await ctx.history.add(turns)  # message 20 crosses the summary mark

        thread = await ctx.history.thread()
        assert thread.summary is not None, "the 20th message rolls the durable summary forward"
        summary = thread.summary
        print(f"summary v{summary.version}, covers messages 1-{summary.covers_to_sequence}:")
        print("  " + summary.text[:150].replace("\n", "\n  "), "...")
        assert summary.covers_to_sequence == 20


if __name__ == "__main__":
    asyncio.run(main())
