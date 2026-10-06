"""04 · Documents: upload a report, wait until it is citable, retrieve from it.

An upload is acknowledged once the bytes are durable; parsing, chunking and indexing run in
the background (here inline). Once the document is ``READY`` its passages come back from
``search`` with their page, and ``context`` packs them with an evidence report: ``COMPLETE``
when everything an answer needs was found, ``INSUFFICIENT`` when it was not.

    uv run python examples/04_documents.py
"""

from __future__ import annotations

import asyncio

from _support import FIXTURES, run_id, service

QUESTION = "Why did Adjusted EBITDA increase despite lower revenue?"


async def main() -> None:
    async with service() as svc, svc.client() as memory:
        ctx = memory.bind(user_id=f"u-{run_id()}", thread_id=f"thr-{run_id()}")

        # A document lands in the thread it was uploaded to, so the thread must exist first:
        # here the user's message creates it (``history.update`` would too).
        await ctx.history.add([("USER", "Here is the ACME FY26 report.")])
        report = FIXTURES / "acme_fy26_annual_report.md"
        handle = await ctx.advanced.documents.add(report, title="ACME FY26 Annual Report")
        doc = await ctx.advanced.documents.wait_ready(handle.document_id)
        print("document:", doc.document_id, doc.status)
        assert doc.status == "READY", doc.status

        again = await ctx.advanced.documents.add(report, title="the same report, uploaded again")
        assert again.document_id == handle.document_id and again.deduplicated
        print("the same bytes again: deduplicated")

        passages = await ctx.search(QUESTION, kinds=["chunk"], limit=5)
        for item in passages:
            print(f"  p.{item.page}: {item.text[:70]!r}")
        assert passages and all(i.document_id == handle.document_id for i in passages)

        bundle = await ctx.context(QUESTION, format="full")
        print("evidence:", bundle.evidence_status, "-", len(bundle.knowledge), "passages")
        assert bundle.evidence_status == "COMPLETE" and bundle.knowledge

        unrelated = await ctx.context("Who won the 1998 football championship?", format="full")
        print("unrelated question:", unrelated.evidence_status)
        assert unrelated.evidence_status == "INSUFFICIENT", "say you do not know"


if __name__ == "__main__":
    asyncio.run(main())
