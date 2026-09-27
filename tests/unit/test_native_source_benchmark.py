"""Evidence has to occur inside the reported depth, without flattening duplicates."""

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from benchmark.native_source_retrieval import (
    SourceLineage,
    coverage,
    observation_sources,
    summarize,
)

from memory_service.domain.evidence import EvidenceRef

pytestmark = pytest.mark.unit


def test_companions_below_the_cut_do_not_inflate_fixed_depth_completeness():
    candidates = [["one"]] * 50 + [["two"]]
    result = coverage(candidates, ["one", "two"])
    assert result["50"] == {"recall": 0.5, "complete": False}
    assert result["100"] == {"recall": 1.0, "complete": True}


def test_a_single_memory_can_cite_multiple_sources_without_spending_extra_positions():
    assert coverage([["one", "two"]], ["one", "two"])["10"]["complete"]
    assert coverage([], ["one"])["10"]["recall"] == 0.0
    assert coverage([], []) == {}


def test_adversarial_and_unannotated_questions_do_not_enter_retrieval_denominators():
    rows = [
        {"category": "multi_hop", "coverage": coverage([["one"]], ["one", "two"])},
        {"category": "single_hop", "coverage": {}},
        {"category": "adversarial", "coverage": coverage([], ["missing"])},
    ]
    summary = summarize(rows)
    assert summary["all_answerable"]["questions"] == 1
    assert summary["all_answerable"]["at"]["50"]["recall"] == 0.5


def ref(kind, source_id):
    return EvidenceRef(source_type=kind, source_id=source_id, observed_at=datetime.now(UTC))


async def test_derived_ancestry_is_separate_from_direct_evidence_and_cycles_terminate():
    class Store:
        async def __aenter__(self):
            self.memories = self
            return self

        async def __aexit__(self, *args):
            pass

        async def get_many(self, tenant, ids):
            assert tenant == "evaluation"
            graph = {
                "summary": [ref("memory", "belief"), ref("message", "obs_a")],
                "belief": [ref("memory", "summary"), ref("message", "obs_b")],
            }
            return [SimpleNamespace(memory_id=k, evidence=graph[k]) for k in ids if k in graph]

    mapping = {"obs_a": "D1:1", "obs_b": "D2:2"}
    evidence = [[ref("memory", "summary")], [ref("memory", "missing")], [ref("message", "obs_b")]]
    assert observation_sources(evidence[0], mapping) == set()
    assert observation_sources(evidence[2], mapping) == {"D2:2"}
    lineage = await SourceLineage(Store, "evaluation", mapping).resolve(evidence)
    assert lineage == [["D1:1", "D2:2"], [], ["D2:2"]]
    assert len(lineage) == len(evidence)  # no new ranking slots from expanded citations


async def test_unbounded_lineage_fails_explicitly_instead_of_inflating_results():
    class Store:
        async def __aenter__(self):
            self.memories = self
            return self

        async def __aexit__(self, *args):
            pass

        async def get_many(self, tenant, ids):
            return [
                SimpleNamespace(memory_id=k, evidence=[ref("memory", str(int(k) + 1))]) for k in ids
            ]

    with pytest.raises(ValueError, match="depth budget"):
        await SourceLineage(Store, "evaluation", {}).resolve([[ref("memory", "1")]])
