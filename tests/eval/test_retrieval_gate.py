"""Critical retrieval gates (M6 baseline): Recall@20 = 1.00 and Evidence-Group Recall = 1.00
over the golden set, with a same-scope distractor document indexed. Writes the evidence file
``benchmark/results/retrieval_gate.json`` consumed by ``memory_service.tools.release_gate``.

Provenance matters: in the sandbox this runs with the deterministic hash embedding, which is
NOT representative of Granite quality. The result file says so (``representative: false``);
``make validate`` on a machine with the real models overwrites it.
"""

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

pytestmark = pytest.mark.eval

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
GOLDEN = Path(__file__).resolve().parent / "golden" / "acme_fy26.json"
CTX = MemoryExecutionContext(tenant_id="acme", user_id="u1", workspace_id="ws1")


async def _ingest_all(container, uow_factory, golden: GoldenSet) -> dict[str, str]:
    register_handlers(container)
    ingestion = container.services["ingestion"]
    aliases: dict[str, str] = {}
    for alias, filename in golden.documents.items():
        async with uow_factory() as uow:
            ack = await ingestion.accept_file(
                uow,
                CTX,
                filename=filename,
                media_type="text/markdown",
                data=(FIXTURES / filename).read_bytes(),
                title=alias,
            )
            await uow.commit()
        aliases[ack.document_id] = alias
    await container.tasks.drain()  # parse
    await container.tasks.drain()  # index
    return aliases


async def test_critical_recall_and_evidence_group_gates(container, uow_factory) -> None:
    golden = GoldenSet.load(GOLDEN)
    aliases = await _ingest_all(container, uow_factory, golden)
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
        "gate": "retrieval",
        "golden_set": golden.name,
        "embedding": indexer.embedding.fingerprint(),
        "sparse": indexer.sparse.fingerprint(),
        "reranker": engine.reranker.fingerprint() if engine.reranker else None,
        "representative": not indexer.embedding.fingerprint().startswith("hash-"),
        "note": (
            "hash embedding is a deterministic stand-in; quality numbers are not representative"
            if indexer.embedding.fingerprint().startswith("hash-")
            else None
        ),
        **summary,
        "provenance": provenance(),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "retrieval_gate.json").write_text(json.dumps(report, indent=2) + "\n")
    failing = [r for r in results if r.critical and not r.complete]
    assert not failing, "\n".join(
        f"{r.id}: missing {r.missing} ranks={r.first_rank}" for r in failing
    )
    assert summary["critical_recall_at_k"] == 1.0
    assert summary["critical_evidence_group_recall"] == 1.0
    assert summary["k"] == k
    # M9: every critical question ends with a COMPLETE evidence report (no abstention, no gaps)
    assert summary["critical_evidence_complete_rate"] == 1.0, evidence_status
    assert summary["routing_accuracy"] == 1.0, [
        p["query_type"]
        for p in summary["per_question"]
        if p["query_type"]["expected"] != p["query_type"]["observed"]
    ]
