"""Grounding cascade: deterministic decomposition and citation validation, the lexical NLI
stand-in, verdicts and the hallucination rate, the borderline-band judge through the mocked
gateway (with token accounting) and the contradiction scan over unused evidence."""

from __future__ import annotations

import pytest

from memory_service.adapters.models.nli import LexicalNLI
from memory_service.config.settings import NLISettings
from memory_service.domain.context_bundle import (
    ContextBundle,
    ContextItem,
    ConversationWindow,
    EvidenceReport,
    UnusedEvidence,
)
from memory_service.domain.enums import EvidenceStatus, QueryType, Representation
from memory_service.modules.grounding.cascade import (
    Evidence,
    GroundingCascade,
    attach,
    bundle_evidence,
    decompose,
    resolve_citation,
)
from memory_service.modules.grounding.lexical import conflicts, coverage
from memory_service.modules.llm.cost import llm_tokens_used, start_llm_accounting
from tests.support_llm import mocked_gateway

pytestmark = pytest.mark.unit

E1 = Evidence(
    "chk_1",
    "Adjusted EBITDA increased to EUR 98 million in FY26 (FY25: EUR 90 million), driven by "
    "restructuring savings of EUR 12 million.",
    "chunk",
    "chunk_id:chk_1",
)
E2 = Evidence(
    "chk_2",
    "Revenue decreased by 4% to EUR 400 million in FY26 due to lower volumes.",
    "chunk",
    "chunk_id:chk_2",
)
E3 = Evidence("chk_3", "The company does not expect to pay a dividend for FY26.", "chunk")
SETTINGS = NLISettings(provider="lexical")


def cascade(**overrides):
    return GroundingCascade(LexicalNLI(), settings=SETTINGS, **overrides)


# --- decomposition ---------------------------------------------------------------------


def test_decompose_splits_sentences_and_clauses_and_lifts_citations() -> None:
    claims = decompose(
        "Adjusted EBITDA increased to EUR 98 million [1]. Revenue fell to EUR 400 million, "
        "and net debt was EUR 210 million at year end (source: chunk_id:chk_4).\n"
        "- Headcount was reduced by 350 employees; the Lyon plant will close in December."
    )
    assert [c.text for c in claims] == [
        "Adjusted EBITDA increased to EUR 98 million",
        "Revenue fell to EUR 400 million",
        "net debt was EUR 210 million at year end",
        "Headcount was reduced by 350 employees",
        "the Lyon plant will close in December",
    ]
    assert claims[0].citations == ("1",)
    assert claims[1].citations == ("chunk_id:chk_4",) and claims[2].citations == claims[1].citations


def test_decompose_drops_questions_short_hedges_and_discourse() -> None:
    claims = decompose(
        "Is EBITDA up? Maybe. I think so. In summary: Here is what I found. "
        "Perhaps the restructuring programme explains most of the EBITDA improvement in FY26. "
        "Let me know if you need more. Adjusted EBITDA increased to EUR 98 million."
    )
    assert [c.text for c in claims] == [
        "Perhaps the restructuring programme explains most of the EBITDA improvement in FY26",
        "Adjusted EBITDA increased to EUR 98 million",
    ]
    assert decompose("") == [] and decompose("Why? How?") == []
    assert len(decompose(". ".join(["Revenue was EUR 1 million"] * 60), max_claims=5)) == 5


def test_resolve_citation_by_ordinal_id_and_citation_key() -> None:
    evidence = [E1, E2]
    assert resolve_citation("1", evidence) is E1 and resolve_citation("2", evidence) is E2
    assert resolve_citation("3", evidence) is None and resolve_citation("0", evidence) is None
    assert resolve_citation("chk_2", evidence) is E2
    assert resolve_citation("chunk_id:chk_1", evidence) is E1
    assert resolve_citation("chunk_id:chk_9", evidence) is None


# --- lexical signals and the stand-in NLI -----------------------------------------------


def test_lexical_signals_numbers_negation_polarity() -> None:
    assert coverage("Adjusted EBITDA increased to EUR 98 million", E1.text) > 0.8
    assert conflicts("Adjusted EBITDA increased to EUR 97 million", E1.text) == ["number"]
    assert conflicts("Adjusted EBITDA increased to EUR 98 million in FY26", E1.text) == []
    assert conflicts("The company expects to pay a dividend for FY26", E3.text) == ["negation"]
    assert conflicts("Adjusted EBITDA decreased to EUR 98 million", E1.text) == ["polarity"]
    assert conflicts("Adjusted EBITDA increased to EUR 98 million", E1.text) == []


async def test_lexical_nli_scores_are_deterministic_and_representative_false() -> None:
    nli = LexicalNLI()
    assert nli.representative is False and nli.fingerprint() == "lexical-nli-v1"
    scores = await nli.entail([E1.text, E2.text, "unrelated text about weather"], "EBITDA rose")
    assert len(scores) == 3
    for s in scores:
        assert abs(s.entailment + s.neutral + s.contradiction - 1.0) < 1e-6
    supported = await nli.entail([E1.text], "Adjusted EBITDA increased to EUR 98 million in FY26")
    assert supported[0].entailment >= 0.9
    wrong = await nli.entail([E1.text], "Adjusted EBITDA increased to EUR 150 million in FY26")
    assert wrong[0].contradiction >= 0.8 and wrong[0].entailment < 0.1
    again = await nli.entail([E1.text], "Adjusted EBITDA increased to EUR 150 million in FY26")
    assert again == wrong
    assert (await nli.entail([E1.text], "?"))[0].neutral == 1.0
    assert await nli.entail([], "anything") == []


# --- cascade verdicts --------------------------------------------------------------------


async def test_cascade_verdicts_and_hallucination_rate() -> None:
    report = await cascade().verify(
        "Adjusted EBITDA increased to EUR 98 million in FY26. "
        "Revenue decreased to EUR 400 million. "
        "Adjusted EBITDA increased to EUR 150 million. "
        "The company expects to pay a dividend for FY26. "
        "The CFO resigned in June 2026.",
        [E1, E2, E3],
    )
    verdicts = [(c.verdict, c.method, c.evidence_ids) for c in report.claims]
    assert verdicts == [
        ("supported", "nli", ["chk_1"]),
        ("supported", "nli", ["chk_2"]),
        ("contradicted", "nli", ["chk_1"]),
        ("contradicted", "nli", ["chk_3"]),
        ("unsupported", "nli", []),
    ]
    assert report.supported == 2 and report.contradicted == 2 and report.unsupported == 1
    assert report.per_claim_hallucination_rate == 0.6 and report.grounded is False
    assert report.representative is False and report.nli_provider == "lexical-nli-v1"
    assert report.evidence_count == 3 and report.llm_tokens == 0 and report.judge_consulted == 0
    assert any("stand-in" in n for n in report.notes)
    empty = await cascade().verify("Really?", [E1])
    assert empty.claims == [] and empty.per_claim_hallucination_rate == 0.0
    assert "no assertive claims" in empty.notes[0]


async def test_citation_validation_is_deterministic() -> None:
    c = cascade()
    ok = await c.verify("Adjusted EBITDA increased to EUR 98 million [1].", [E1, E2])
    assert ok.claims[0].verdict == "supported" and ok.claims[0].citations == ["1"]
    assert ok.claims[0].evidence_ids == ["chk_1"]
    by_key = await c.verify("Revenue decreased to EUR 400 million [chunk_id:chk_2].", [E1, E2])
    assert by_key.claims[0].verdict == "supported" and by_key.claims[0].evidence_ids == ["chk_2"]
    # the cited item is about something else entirely
    wrong = await c.verify("Revenue decreased to EUR 400 million [1].", [E3, E2])
    assert wrong.claims[0].verdict == "unsupported" and wrong.claims[0].method == "citation"
    assert "does not mention" in wrong.claims[0].notes[0]
    # an id-like citation that resolves to nothing is a hard failure
    missing = await c.verify("Net debt was EUR 210 million [7].", [E1])
    assert missing.claims[0].verdict == "unsupported" and missing.claims[0].method == "citation"
    assert "does not resolve" in missing.claims[0].notes[0]
    # a free-text source that is not an id is ignored (the claim is judged uncited)
    free = await c.verify(
        "Adjusted EBITDA increased to EUR 98 million (source: annual report p. 11).", [E1]
    )
    assert free.claims[0].verdict == "supported" and "treated as uncited" in free.claims[0].notes[0]


async def test_contradiction_scan_over_unused_evidence() -> None:
    unused = [Evidence("chk_u", E2.text, "chunk")]
    report = await cascade().verify("Revenue was EUR 450 million in FY26.", [E3], unused=unused)
    claim = report.claims[0]
    assert claim.verdict == "contradicted" and claim.contradicted_by == ["chk_u"]
    assert "contradicted by unused evidence chk_u" in claim.notes[-1]
    assert report.unused_count == 1
    # a supported claim keeps its verdict; the collision is recorded, not promoted
    kept = await cascade().verify(
        "Adjusted EBITDA increased to EUR 98 million in FY26.",
        [E1],
        unused=[Evidence("chk_old", "Adjusted EBITDA increased to EUR 81 million in FY25.")],
    )
    assert kept.claims[0].verdict == "supported" and kept.claims[0].contradicted_by == ["chk_old"]


async def test_borderline_band_uses_the_judge_and_falls_back_to_borderline() -> None:
    # ~50% token coverage -> entailment inside the (0.3, 0.7) band
    claim = "Restructuring savings explain the EBITDA growth reported for the year."
    start_llm_accounting()
    with mocked_gateway([{"supported": True, "reason": "savings are stated as the driver"}]) as gw:
        report = await cascade(assist=gw.assist(uses=["grounding_judge"])).verify(claim, [E1])
        assert gw.route.call_count == 1
        prompt = gw.prompts()[0]["messages"][1]["content"]
    # decomposition strips terminal punctuation: a claim is a proposition, not a sentence
    assert prompt.startswith(f"Claim: {claim.rstrip('.')}") and "[1] Adjusted EBITDA" in prompt
    c = report.claims[0]
    assert c.verdict == "supported" and c.method == "judge" and "judge:" in c.notes[-1]
    assert 0.3 <= c.support <= 0.7
    assert report.judge_consulted == 1 and report.llm_tokens == 30 and llm_tokens_used() == 30
    with mocked_gateway([{"supported": False, "reason": "growth is not stated"}]) as gw:
        report = await cascade(assist=gw.assist(uses=["grounding_judge"])).verify(claim, [E1])
    assert report.claims[0].verdict == "unsupported" and report.claims[0].method == "judge"
    assert report.per_claim_hallucination_rate == 1.0
    with mocked_gateway(failing=True) as gw:
        report = await cascade(assist=gw.assist(uses=["grounding_judge"])).verify(claim, [E1])
    assert report.claims[0].verdict == "borderline" and report.claims[0].method == "nli"
    assert report.borderline == 1 and report.per_claim_hallucination_rate == 0.0
    # not enabled: never consulted, the claim stays borderline
    with mocked_gateway([{"supported": True, "reason": "x"}]) as gw:
        report = await cascade(assist=gw.assist(uses=[])).verify(claim, [E1])
        assert gw.route.call_count == 0
    assert report.claims[0].verdict == "borderline" and report.judge_consulted == 0


def test_bundle_evidence_order_and_attach() -> None:
    def item(item_id: str, rep: Representation, text: str) -> ContextItem:
        return ContextItem(
            item_id=item_id, representation=rep, text=text, citation=f"{rep.value}:{item_id}"
        )

    bundle = ContextBundle(
        query="q",
        query_type=QueryType.GENERAL_SEMANTIC,
        conversation=ConversationWindow(),
        memories=[item("mem_1", Representation.MEMORY, "My timezone is CET.")],
        knowledge=[item("chk_1", Representation.CHUNK, E1.text)],
        graph_facts=[item("rel_1", Representation.RELATION, "EBITDA driven_by savings")],
        summaries=[item("sum_1", Representation.SUMMARY, "Summary of section 3.")],
        evidence=EvidenceReport(
            status=EvidenceStatus.COMPLETE,
            unused=[UnusedEvidence(item_id="chk_9", kind="chunk", text=E2.text)],
        ),
        token_budget=100,
        token_estimate=10,
    )
    packed, unused = bundle_evidence(bundle)
    assert [e.item_id for e in packed] == ["mem_1", "rel_1", "sum_1", "chk_1"]
    assert [e.item_id for e in unused] == ["chk_9"] and unused[0].text == E2.text
    assert resolve_citation("4", packed).item_id == "chk_1"  # type: ignore[union-attr]
    report = GroundingReport_stub()
    out = attach(bundle, report)
    assert out.evidence.grounding is report and out.evidence.unused == bundle.evidence.unused


def GroundingReport_stub():  # noqa: N802 - tiny helper keeps the test readable
    from memory_service.domain.grounding import GroundingReport

    return GroundingReport(nli_provider="lexical-nli-v1")
