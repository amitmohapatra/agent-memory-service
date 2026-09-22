"""Critical retrieval gates over real-world PDFs parsed by Docling (``public_pdfs.json``):
Recall@20 = 1.00 and Evidence-Group Recall = 1.00 with all four PDFs indexed in one scope.
Needs the Docling parser and its models, so it carries the ``models`` marker; it writes
``benchmark/results/retrieval_gate_pdfs.json`` for the release gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmark.common import RESULTS, provenance
from benchmark.evaluation import CRITICAL_RECALL_K
from benchmark.evaluation.golden import (
    GoldenSet,
    RetrievedChunk,
    evaluate_question,
    summarize,
)

from memory_service.domain.context import MemoryExecutionContext
from memory_service.modules.jobs.registry import register_handlers

pytestmark = [pytest.mark.eval, pytest.mark.models]

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
GOLDEN = Path(__file__).resolve().parent / "golden" / "public_pdfs.json"
CTX = MemoryExecutionContext(tenant_id="acme", user_id="u1", workspace_id="ws1")


async def test_pdf_critical_recall_and_evidence_group_gates(container, uow_factory) -> None:
    pytest.importorskip("docling")
    if container.settings.documents.parser != "docling":
        pytest.skip("documents.parser=docling required (MEMORY_TEST_PROVIDERS=env)")
    golden = GoldenSet.load(GOLDEN)
    register_handlers(container)
    ingestion = container.services["ingestion"]
    aliases: dict[str, str] = {}
    for alias, filename in golden.documents.items():
        async with uow_factory() as uow:
            ack = await ingestion.accept_file(
                uow,
                CTX,
                filename=filename,
                media_type="application/pdf",
                data=(FIXTURES / filename).read_bytes(),
                title=alias,
            )
            await uow.commit()
        aliases[ack.document_id] = alias
    await container.tasks.drain()
    await container.tasks.drain()
    async with uow_factory() as uow:
        for document_id, alias in aliases.items():
            doc = await uow.documents.get(CTX.tenant_id, document_id)
            assert doc is not None and doc.status.value == "READY", (alias, doc)
    engine = container.services["retrieval"]
    k = CRITICAL_RECALL_K
    results = []
    evidence_status: dict[str, str] = {}
    for q in golden.questions:
        res = await engine.retrieve(CTX, q.query, limit=k)
        evidence_status[q.id] = str((res.diagnostics.get("evidence") or {}).get("status"))
        retrieved = [
            RetrievedChunk(
                document_alias=aliases.get(str(c.payload.get("document_id"))),
                page=c.payload.get("page"),
                text=c.text,
                record_id=c.record_id,
            )
            for c in res.candidates
        ]
        results.append(
            evaluate_question(q, retrieved, k=k, observed_type=res.routed.query_type.value)
        )
    summary = summarize(results, k=k)
    critical_ids = [q.id for q in golden.questions if q.critical]
    complete = sum(1 for i in critical_ids if evidence_status[i] == "COMPLETE")
    summary["critical_evidence_complete_rate"] = round(complete / len(critical_ids), 4)
    summary["evidence_status"] = evidence_status
    indexer = container.services["indexer"]
    report = {
        "gate": "retrieval_pdfs",
        "golden_set": golden.name,
        "parser": type(container.document_parser).__name__,
        "embedding": indexer.embedding.fingerprint(),
        "sparse": indexer.sparse.fingerprint(),
        "reranker": engine.reranker.fingerprint() if engine.reranker else None,
        "representative": not indexer.embedding.fingerprint().startswith("hash-"),
        **summary,
        "provenance": provenance(),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "retrieval_gate_pdfs.json").write_text(json.dumps(report, indent=2) + "\n")
    failing = [r for r in results if r.critical and not r.complete]
    assert not failing, "\n".join(
        f"{r.id}: missing {r.missing} ranks={r.first_rank}" for r in failing
    )
    assert summary["critical_recall_at_k"] == 1.0
    assert summary["critical_evidence_group_recall"] == 1.0
    assert summary["critical_evidence_complete_rate"] == 1.0, evidence_status
