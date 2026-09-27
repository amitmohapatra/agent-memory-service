"""Language must not prevent claims or their cited sources reaching the NLI model.

The controlled classifier below tests routing, not multilingual model accuracy.
"""

# ruff: noqa: RUF001 — literal multilingual fixtures.

import pytest

from memory_service.adapters.models.nli import LexicalNLI
from memory_service.config.constants import NLISettings
from memory_service.modules.grounding.cascade import Evidence, GroundingCascade, decompose
from memory_service.modules.grounding.lexical import content_tokens
from memory_service.ports.models import NLIScore

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "text",
    [
        "我的办公室位于北京市。",
        "私の事務所は東京都にあります。",
        "Мой офис находится в Москве.",
        "मेरा कार्यालय दिल्ली में है।",
        "يقع مكتبي في مدينة القاهرة.",
        "میرا دفتر اسلام آباد میں ہے۔",
        "สำนักงานของฉันอยู่ในกรุงเทพ",
        "Το γραφείο μου βρίσκεται στην Αθήνα.",
        "제 사무실은 서울에 있습니다.",
    ],
)
def test_non_english_claims_are_not_silently_discarded(text):
    claims = decompose(text)
    assert len(claims) == 1
    assert content_tokens(claims[0].text)


def test_adjacent_cjk_sentences_and_citations_remain_separate():
    claims = decompose("我的办公室位于北京市[1]。我们的仓库位于上海市[2]。")
    assert len(claims) == 2
    assert [c.citations for c in claims] == [("1",), ("2",)]
    assert decompose("你的办公室位于北京市吗？") == []
    assert decompose("هل يقع مكتبك في القاهرة؟") == []


class RecordingNLI(LexicalNLI):
    representative = True

    def __init__(self, score: NLIScore):
        self.score_value = score
        self.seen = []

    async def entail_groups(self, groups):
        self.seen.extend(groups)
        return [[self.score_value for _ in premises] for premises, _ in groups]


@pytest.mark.parametrize("citation", ["", " [1]"])
@pytest.mark.parametrize(
    "score,verdict",
    [
        (NLIScore(entailment=0.98, neutral=0.01, contradiction=0.01), "supported"),
        (NLIScore(entailment=0.01, neutral=0.01, contradiction=0.98), "contradicted"),
        (NLIScore(entailment=0.01, neutral=0.98, contradiction=0.01), "unsupported"),
    ],
)
async def test_cross_language_evidence_is_decided_by_model(citation, score, verdict):
    nli = RecordingNLI(score)
    cascade = GroundingCascade(nli, settings=NLISettings())
    evidence = Evidence("source", "The office is located in Beijing.")
    report = await cascade.verify("我的办公室位于北京市" + citation + "。", [evidence])
    assert len(nli.seen) == 1
    assert nli.seen[0][0] == [evidence.text]
    assert report.claims[0].verdict == verdict


async def test_model_cannot_receive_blanked_generated_sources_or_invalid_citations():
    nli = RecordingNLI(NLIScore(entailment=0.99, neutral=0.01, contradiction=0.0))
    cascade = GroundingCascade(nli, settings=NLISettings())
    for citation in ("", " [1]", " [7]"):
        report = await cascade.verify(
            "我的办公室位于北京市" + citation + "。", [Evidence("generated", "")]
        )
        assert report.supported == 0
    assert nli.seen == []


async def test_uncited_cross_language_work_remains_bounded():
    nli = RecordingNLI(NLIScore(entailment=0.01, neutral=0.98, contradiction=0.01))
    cascade = GroundingCascade(nli, settings=NLISettings(premises_per_claim=3))
    await cascade.verify(
        "我的办公室位于北京市。",
        [Evidence(str(i), "An unrelated English passage.") for i in range(20)],
    )
    assert len(nli.seen[0][0]) == 3


@pytest.mark.parametrize(
    "claim", ["Evliyim.", "Je travaille.", "Estoy casado.", "我已婚。", "Office closed."]
)
async def test_short_assertions_reach_the_classifier(claim):
    nli = RecordingNLI(NLIScore(entailment=0.01, neutral=0.98, contradiction=0.01))
    cascade = GroundingCascade(nli, settings=NLISettings())
    report = await cascade.verify(claim, [Evidence("source", "An unrelated passage.")])
    assert len(nli.seen) == 1
    assert len(report.claims) == 1
    assert report.claims[0].verdict == "unsupported"


@pytest.mark.parametrize(
    "text", ["Thanks!", "OK.", "Got it.", "...", "هل أنت متزوج؟", "Evli misin?"]
)
def test_short_questions_and_known_acknowledgements_are_not_claims(text):
    assert decompose(text) == []
