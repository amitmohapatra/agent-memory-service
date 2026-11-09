"""A restatement in other words is the memory already held, not a second one.

The hard extraction eval's ``dup-1`` and ``dup-2`` ("My favourite cuisine is Thai." / "Thai
food is my favourite, by the way."; "I work at Shopify as a data engineer." / "As I said, I'm
a data engineer at Shopify.") were each stored twice: their wording shares too few words for
the lexical rule and the dense band never compared them. Two genuinely different facts about
the same thing must stay two memories. The scores scripted below are the ones the frozen
encoder and NLI head give each pair (cosine; entailment each way; contradiction).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import UTC, datetime

import pytest

from memory_service.config.constants import MemoryIntelligenceSettings
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import DedupDecision, Lifetime, MemoryType, Visibility
from memory_service.domain.evidence import EvidenceRef
from memory_service.modules.memory.native import NativeMemoryIntelligence
from memory_service.modules.memory.pipeline import build_memory
from memory_service.ports.intelligence import MemoryCandidate
from memory_service.ports.models import NLIScore

pytestmark = pytest.mark.unit

CTX = MemoryExecutionContext(tenant_id="acme", user_id="u1", workspace_id="ws1", thread_id="t1")

#: (held, new, cosine, entailment held->new, entailment new->held, contradiction)
SAME = [
    (
        "My favourite cuisine is Thai.",
        "Thai food is my favourite, by the way.",
        0.85,
        0.99,
        0.99,
        0.0,
    ),
    (
        "I work at Shopify as a data engineer.",
        "As I said, I'm a data engineer at Shopify.",
        0.87,
        0.96,
        1.0,
        0.0,
    ),
    # the same restatements in Hindi, Arabic, German and Spanish, as the multilingual encoder
    # and NLI head read them
    ("मेरा पसंदीदा खाना थाई है।", "थाई खाना मुझे सबसे ज़्यादा पसंद है।", 0.85, 0.99, 0.98, 0.0),
    ("أنا أعمل مهندس بيانات في شوبيفاي.", "وظيفتي مهندس بيانات لدى شوبيفاي.", 0.90, 0.99, 1.0, 0.0),
    (
        "Mein Lieblingsessen ist Thai.",
        "Thailändisches Essen mag ich am liebsten.",
        0.77,
        1.0,
        0.98,
        0.0,
    ),
    (
        "Trabajo como ingeniera de datos en Shopify.",
        "Soy ingeniera de datos en Shopify.",
        0.89,
        0.96,
        0.99,
        0.0,
    ),
]
APART = [
    ("My favourite cuisine is Thai.", "My favourite cuisine is Italian.", 0.74, 0.0, 0.0, 0.71),
    (
        "I work at Shopify as a data engineer.",
        "My sister works at Shopify as a data engineer.",
        0.81,
        0.0,
        0.0,
        0.73,
    ),
    ("I like Thai food.", "I don't like Thai food.", 0.89, 0.0, 0.03, 1.0),
    (
        "Ana moved to Lisbon.",
        "Ana is learning Portuguese since moving to Lisbon.",
        0.70,
        0.0,
        1.0,
        0.0,
    ),
    ("Gina loves dance.", "Gina loves contemporary dance.", 0.86, 0.0, 1.0, 0.0),
    ("Zoe started kindergarten.", "Zoe loves the art corner.", 0.52, 0.01, 0.03, 0.12),
    (
        "Our biggest client, Halvorsen Foods, is renewing.",
        "The Halvorsen Foods renewal is for three years.",
        0.75,
        0.0,
        0.0,
        0.0,
    ),
    # two facts about one sister, in Hindi, Arabic, German and Spanish
    ("मेरी बहन प्रिया डॉक्टर है।", "मेरी बहन प्रिया दुबई चली गई।", 0.55, 0.0, 0.0, 0.0),
    ("أختي سارة طبيبة.", "أختي سارة انتقلت إلى دبي.", 0.66, 0.0, 0.01, 0.01),
    (
        "Meine Schwester Anna ist Ärztin.",
        "Meine Schwester Anna ist nach Wien gezogen.",
        0.64,
        0.01,
        0.0,
        0.01,
    ),
    ("Mi hermana Lucía es abogada.", "Mi hermana Lucía trabaja en Madrid.", 0.70, 0.02, 0.0, 0.0),
    # entailed both ways but below the cosine bar: the head alone never merges
    ("I adopted a dog.", "A dog was adopted by me last spring.", 0.60, 0.95, 0.95, 0.0),
]


class Scripted:
    """The frozen encoder and NLI head, as the pairs above: two texts at their cosine, and
    entailment by (premise, hypothesis)."""

    representative = True

    def __init__(self, held: str, new: str, cos: float, there: float, back: float, c: float):
        self.held, self.new = held, new
        self.vectors = {held: [1.0, 0.0], new: [cos, math.sqrt(1 - cos * cos)]}
        self.entail = {(held, new): (there, c), (new, held): (back, c)}

    def fingerprint(self) -> str:
        return "scripted"

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.vectors.get(t, [0.0, 0.0]) for t in texts]

    async def entail_groups(
        self, groups: Sequence[tuple[Sequence[str], str]]
    ) -> list[list[NLIScore]]:
        out = []
        for premises, hypothesis in groups:
            row = []
            for premise in premises:
                e, c = self.entail.get((premise, hypothesis), (0.0, 0.0))
                row.append(NLIScore(entailment=e, neutral=max(0.0, 1 - e - c), contradiction=c))
            out.append(row)
        return out


def _reading(text: str) -> MemoryCandidate:
    return MemoryCandidate(
        content=text,
        memory_type=MemoryType.PREFERENCE if "favourite" in text else MemoryType.SEMANTIC,
        lifetime=Lifetime.LONG_TERM,
        visibility=Visibility.USER,
        category="source_fact",
        evidence=[
            EvidenceRef(source_type="message", source_id="m1", observed_at=datetime.now(UTC))
        ],
    )


async def _decide(pair: tuple) -> DedupDecision:
    held, new, *_ = pair
    model = Scripted(*pair)
    native = NativeMemoryIntelligence(MemoryIntelligenceSettings(), model, nli=model)  # type: ignore[arg-type]
    stored = build_memory(_reading(held), CTX, now=datetime.now(UTC))
    return (await native.consolidate(_reading(new), [stored], CTX)).decision


@pytest.mark.parametrize("pair", SAME, ids=["dup-1", "dup-2", "hi", "ar", "de", "es"])
async def test_a_restatement_in_other_words_is_the_memory_already_held(pair) -> None:
    assert await _decide(pair) in (DedupDecision.REINFORCE, DedupDecision.MERGE)


@pytest.mark.parametrize("pair", APART, ids=lambda p: p[1][:24] if isinstance(p, str) else None)
async def test_two_different_facts_about_one_thing_stay_two(pair) -> None:
    assert await _decide(pair) is DedupDecision.CREATE


async def test_without_a_trained_head_nothing_merges_on_meaning() -> None:
    model = Scripted(*SAME[0])
    model.representative = False
    native = NativeMemoryIntelligence(MemoryIntelligenceSettings(), model, nli=model)  # type: ignore[arg-type]
    stored = build_memory(_reading(SAME[0][0]), CTX, now=datetime.now(UTC))
    decision = (await native.consolidate(_reading(SAME[0][1]), [stored], CTX)).decision
    assert decision is DedupDecision.CREATE
