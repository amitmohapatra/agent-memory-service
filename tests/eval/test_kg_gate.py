"""Knowledge-graph quality gate: every expected typed entity, alias and factual relation in
``golden/kg_facts.json`` is present after ingestion (fact recall = 1.00), no forbidden fact
or noise entity exists (false facts = 0, noise = 0), and each golden question resolves to
its expected fact through ``/v1/graph/query`` semantics. Writes
``benchmark/results/kg_gate.json`` for the release gate.

The extractor is rule-based (no LLM), so this gate measures a deterministic pipeline: it is
representative of what the service does with these document styles, and it is the file to
extend when a new document type is onboarded.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from benchmark.common import RESULTS, provenance

from memory_service.domain.context import MemoryExecutionContext
from memory_service.modules.jobs.registry import register_handlers

pytestmark = pytest.mark.eval

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
GOLDEN = Path(__file__).resolve().parent / "golden" / "kg_facts.json"
CTX = MemoryExecutionContext(tenant_id="acme", user_id="u1", workspace_id="ws1")


def _attrs_match(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    return all(actual.get(k) == v for k, v in expected.items())


async def _ingest(container, uow_factory, filename: str, title: str) -> str:
    async with uow_factory() as uow:
        ack = await container.services["ingestion"].accept_file(
            uow,
            CTX,
            filename=filename,
            media_type="text/markdown",
            data=(FIXTURES / filename).read_bytes(),
            title=title,
        )
        await uow.commit()
    await container.tasks.drain()
    await container.tasks.drain()
    return ack.document_id


async def test_kg_entities_facts_and_queries(container, uow_factory) -> None:
    register_handlers(container)
    golden = json.loads(GOLDEN.read_text())
    graph = container.services["graph"]
    report: dict[str, Any] = {
        "gate": "knowledge_graph",
        "golden_set": golden["name"],
        "provider": "native",
        "documents": {},
    }
    total_expected = total_found = false_facts = noise_found = queries_ok = queries_total = 0
    failures: list[str] = []
    for alias, filename in golden["documents"].items():
        doc_id = await _ingest(container, uow_factory, filename, alias)
        # every entity / relation of this document, through the store (scope-filtered)
        async with uow_factory() as uow:
            vis = await container.services["authz"].visibility(CTX, revisions=uow.revisions)
        keys = sorted(vis.keys)
        store = container.graph_store
        relations = await store.relations_for_document("acme", doc_id, scope_keys=keys)
        ids = {r.subject_id for r in relations} | {r.object_id for r in relations}
        entities = {
            e.entity_id: e for e in await store.get_entities("acme", sorted(ids), scope_keys=keys)
        }
        by_name = {e.canonical_name: e for e in entities.values()}
        doc_entity = next(e for e in entities.values() if e.entity_type == "DOCUMENT")
        doc_report: dict[str, Any] = {"entities": len(entities), "relations": len(relations)}
        # 1. expected entities with types and aliases
        missing_entities = []
        for canon, spec in golden["entities"][alias].items():
            e = by_name.get(canon)
            if e is None:
                missing_entities.append(f"{canon} (absent)")
                continue
            if e.entity_type != spec["type"]:
                missing_entities.append(f"{canon} typed {e.entity_type}, expected {spec['type']}")
            for a in spec.get("aliases", []):
                if a not in e.aliases:
                    missing_entities.append(f"{canon} lacks alias {a!r}")
        # 2. noise entities must not exist
        noise = [
            n
            for n in golden["noise"][alias]
            if n in by_name and by_name[n].entity_type not in ("SECTION", "PERIOD", "DATE")
        ]

        # 3. facts
        def matches(spec: dict[str, Any], r, entities=entities, doc_entity=doc_entity) -> bool:
            s = entities.get(r.subject_id)
            o = entities.get(r.object_id)
            if s is None or o is None or r.predicate != spec["predicate"]:
                return False
            if s.canonical_name != spec["subject"]:
                return False
            if spec["object"] == "doc":
                if o.entity_id != doc_entity.entity_id:
                    return False
            elif o.canonical_name != spec["object"]:
                return False
            if spec.get("page") is not None and r.evidence[0].page != spec["page"]:
                return False
            return _attrs_match(spec.get("attributes", {}), r.attributes)

        found_ids: dict[str, str] = {}
        missing_facts, forbidden_found = [], []
        for spec in golden["facts"][alias]:
            hits = [r for r in relations if matches(spec, r)]
            if spec.get("forbidden"):
                if hits:
                    forbidden_found.append(spec["id"])
                continue
            total_expected += 1
            if hits:
                total_found += 1
                found_ids[spec["id"]] = hits[0].relation_id
            else:
                missing_facts.append(spec["id"])
        # 4. questions resolve to the expected fact through the graph query
        query_failures = []
        for q in golden["queries"][alias]:
            queries_total += 1
            answer = await graph.query(CTX, query=q["question"], hops=q.get("hops", 1))
            expected_rel = found_ids.get(q["expect_fact"])
            if expected_rel and any(r.relation_id == expected_rel for r in answer.relations):
                queries_ok += 1
            else:
                query_failures.append(q["question"])
        false_facts += len(forbidden_found)
        noise_found += len(noise)
        doc_report.update(
            {
                "missing_entities": missing_entities,
                "noise_entities": noise,
                "missing_facts": missing_facts,
                "forbidden_facts_found": forbidden_found,
                "query_failures": query_failures,
                "entity_types": sorted({e.entity_type for e in entities.values()}),
                "predicates": sorted({r.predicate for r in relations}),
            }
        )
        failures += [
            f"{alias}: {m}"
            for m in missing_entities + missing_facts + forbidden_found + noise + query_failures
        ]
        report["documents"][alias] = doc_report
    report.update(
        {
            "fact_recall": round(total_found / total_expected, 4) if total_expected else 0.0,
            "expected_facts": total_expected,
            "false_facts": false_facts,
            "noise_entities": noise_found,
            "query_hit_rate": round(queries_ok / queries_total, 4) if queries_total else 0.0,
            "failures": failures,
            "provenance": provenance(),
        }
    )
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "kg_gate.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"
    )
    assert not failures, failures
    assert (
        report["fact_recall"] == 1.0
        and report["false_facts"] == 0
        and report["noise_entities"] == 0
    )
    assert report["query_hit_rate"] == 1.0
