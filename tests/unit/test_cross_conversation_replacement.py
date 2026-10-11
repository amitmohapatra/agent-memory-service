"""A user's later statement replaces an earlier one of theirs across conversations (ADR 0038),
without a database: what a statement names, when two are about the same thing, the grades a
contradiction earns, the write path against scripted models, and how the read path ranks and
labels an earlier value."""

# ruff: noqa: RUF001 - the Hindi and Arabic fixtures are literal text in their own script

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.context_bundle import ContextItem, _memory_line
from memory_service.domain.enums import (
    MemoryType,
    Representation,
    StatementKind,
    TemporalStatus,
)
from memory_service.domain.memory import CanonicalMemory, earlier_value, replaced_by
from memory_service.modules.context.views import _memory as memory_view
from memory_service.modules.memory.replacement import (
    CrossConversationReplacement,
    Grade,
    PairScores,
    Relation,
    Thresholds,
    anchors,
    decide,
    reinstate,
    relate,
)
from memory_service.modules.retrieval.engine import Candidate, current_first
from memory_service.ports.models import NLIScore

pytestmark = pytest.mark.unit

T0 = datetime(2026, 3, 1, 10, 0, tzinfo=UTC)
T1 = T0 + timedelta(days=40)
CTX = MemoryExecutionContext(tenant_id="acme", user_id="u1", thread_id="thr_b")

ULINE = "Our primary supplier for heavy-duty pallets is Uline."
PACKAGINGCORP = (
    "Actually, we terminated our contract with Uline yesterday due to pricing. "
    "Our new exclusive pallet supplier is PackagingCorp."
)
ULINE_HI = "हेवी-ड्यूटी पैलेट के लिए हमारा मुख्य सप्लायर Uline है।"
PACKAGINGCORP_HI = (
    "असल में, हमने कल क़ीमतों की वजह से Uline के साथ अपना कॉन्ट्रैक्ट ख़त्म कर दिया। "
    "पैलेट का हमारा नया एक्सक्लूसिव सप्लायर PackagingCorp है।"
)


# --------------------------------------------------------------------------- anchors


def test_a_statement_names_its_names_and_identifiers() -> None:
    a = anchors("Service van #7 is assigned to Ana Silva.", "en")
    assert "ana silva" in a.names and ("van", "7") in a.ids
    # a sentence-initial capital is not a name; a label before a number is not one either
    assert "service" not in a.names and "van" not in a.names
    b = anchors("The reorder point for SKU MK-220 is 40 units.", "en")
    assert any(v == "mk220" for _, v in b.ids) and "sku" not in b.names


def test_latin_names_inside_another_script_are_names() -> None:
    a = anchors(ULINE_HI, "hi")
    assert "uline" in a.names
    forklift = anchors("फ़ोर्कलिफ़्ट #4 रखरखाव के लिए सेवा से बाहर है।", "hi")
    assert ("फ़ोर्कलिफ़्ट", "4") in forklift.ids


def test_german_nouns_are_not_names_but_names_are() -> None:
    a = anchors("Bei Chemikalienunfällen im Lager Dave Morris unter Durchwahl 210 anrufen.", "de")
    b = anchors("Dave Morris hat das Unternehmen verlassen. Leiter ist jetzt Kenji Ito.", "de")
    assert relate(b, a).shared_name  # "dave morris" inside "lager dave morris"
    assert not anchors("Kundenretouren laufen wie bisher über Tor 9.", "de").names


# --------------------------------------------------------------------------- relate


@pytest.mark.parametrize(
    ("new", "old", "lang"),
    [
        (
            "Forklift #4 has been repaired and is back on the floor.",
            "Forklift #3 is assigned to the cold room this week.",
            "en",
        ),
        (
            "फ़ोर्कलिफ़्ट #4 ठीक कर दिया गया है और वापस फ़्लोर पर है।",
            "फ़ोर्कलिफ़्ट #3 इस हफ़्ते कोल्ड रूम में लगा है।",
            "hi",
        ),
        (
            "Y los pañuelos faciales, SKU FT-1180, llevan el UPC 084213551029.",
            "Para que conste: el UPC de nuestro papel de cocina, SKU PT-2207, es 084213557106.",
            "es",
        ),
    ],
)
def test_two_things_of_one_kind_are_apart(new: str, old: str, lang: str) -> None:
    assert relate(anchors(new, lang), anchors(old, lang)).apart


def test_a_shared_name_keeps_a_changed_extension_together() -> None:
    new = "Dave Morris has left the company. The spill lead is now Kenji Ito, extension 233."
    old = "For chemical spills in the stockroom, call Dave Morris on extension 210."
    relation = relate(anchors(new, "en"), anchors(old, "en"), new, old)
    assert relation.shared_name and not relation.apart


def test_a_number_that_changed_on_the_same_thing() -> None:
    new = "Change that: the reorder point for MK-220 is 60 units now."
    old = "The reorder point for SKU MK-220 (whole milk, 1 gallon) is 40 units."
    relation = relate(anchors(new, "en"), anchors(old, "en"), new, old)
    assert relation.shared_id and relation.value_changed


# --------------------------------------------------------------------------- decide

NAMED = Relation(shared_name=True, shared_id=False, apart=False)
UNANCHORED = Relation(shared_name=False, shared_id=False, apart=False)


def test_the_grades_of_a_contradiction() -> None:
    assert decide(NAMED, PairScores(None, 0.95)).grade is Grade.SUPERSEDED
    assert decide(NAMED, PairScores(None, 0.87)).grade is Grade.EARLIER
    assert decide(NAMED, PairScores(None, 0.6)).grade is None


def test_a_change_needs_less_about_a_named_thing() -> None:
    lifecycle = [StatementKind.LIFECYCLE]
    assert decide(NAMED, PairScores(None, 0.5), lifecycle).grade is Grade.EARLIER
    # without a shared name or identifier a change says nothing about this statement
    assert decide(UNANCHORED, PairScores(0.6, 0.5), lifecycle).grade is None
    assert decide(NAMED, PairScores(None, 0.3), lifecycle).grade is None


def test_unrelated_or_apart_is_never_replaced() -> None:
    assert decide(UNANCHORED, PairScores(0.3, 1.0)).grade is None  # "Region South" vs BOGO
    assert decide(UNANCHORED, PairScores(None, 1.0)).grade is None  # no encoder, no anchor
    apart = Relation(shared_name=False, shared_id=False, apart=True)
    assert decide(apart, PairScores(0.9, 1.0)).grade is None
    assert decide(UNANCHORED, PairScores(0.6, 0.9)).grade is Grade.SUPERSEDED


def test_thresholds_are_the_measured_ones() -> None:
    t = Thresholds()
    assert (t.earlier, t.change, t.supersede) == (0.85, 0.4, 0.9)


# --------------------------------------------------------------------------- write path


class ScriptedNLI:
    """A trained head's interface: contradiction by (premise word, hypothesis word)."""

    representative = True

    def __init__(self, contradictions: dict[tuple[str, str], float]) -> None:
        self.contradictions = contradictions
        self.pairs: list[tuple[str, str]] = []

    async def entail_groups(
        self, groups: Sequence[tuple[Sequence[str], str]]
    ) -> list[list[NLIScore]]:
        out = []
        for premises, hypothesis in groups:
            row = []
            for premise in premises:
                self.pairs.append((premise, hypothesis))
                c = max(
                    (
                        v
                        for (p, h), v in self.contradictions.items()
                        if p in premise and h in hypothesis
                    ),
                    default=0.02,
                )
                row.append(NLIScore(entailment=0.0, neutral=1 - c, contradiction=c))
            out.append(row)
        return out


def _memory(content: str, *, at: datetime, source: str, **fields: Any) -> CanonicalMemory:
    base: dict[str, Any] = {
        "tenant_id": "acme",
        "content": content,
        "memory_type": MemoryType.OBSERVATION,
        "lifetime": "LONG_TERM",
        "visibility": "USER",
        "scope": {"level": "THREAD", "tenant_id": "acme", "thread_id": "thr"},
        "owner_principal": "user:u1",
        "normalized_hash": content,
        "temporal": {"observed_at": at},
        "evidence": [{"source_type": "message", "source_id": source, "observed_at": at}],
        "system_metadata": {"category": "verbatim_turn"},
        "created_at": at,
        "updated_at": at,
    }
    return CanonicalMemory.model_validate({**base, **fields})


class _Memories:
    def __init__(self, rows: list[CanonicalMemory]) -> None:
        self.rows = {m.memory_id: m for m in rows}
        self.updated: list[CanonicalMemory] = []

    async def statements_of(
        self, tenant_id: str, owner: str, *, before: datetime, limit: int
    ) -> list[CanonicalMemory]:
        rows = [
            m
            for m in self.rows.values()
            if m.owner_principal == owner
            and m.temporal.status is TemporalStatus.CURRENT
            and m.temporal.observed_at <= before
        ]
        return sorted(rows, key=lambda m: m.temporal.observed_at, reverse=True)[:limit]

    async def twins(self, tenant_id: str, memory: CanonicalMemory, **_: Any) -> list[Any]:
        return [
            m
            for m in self.rows.values()
            if m.normalized_hash == memory.normalized_hash
            and m.memory_id != memory.memory_id
            and m.owner_principal == memory.owner_principal
        ]

    async def update(self, memory: CanonicalMemory) -> None:
        self.rows[memory.memory_id] = memory
        self.updated.append(memory)

    async def replaced_by_any(self, tenant_id: str, ids: Sequence[str]) -> list[CanonicalMemory]:
        return [m for m in self.rows.values() if replaced_by(m.system_metadata) in ids]


def _uow(rows: list[CanonicalMemory]) -> Any:
    return SimpleNamespace(memories=_Memories(rows))


def _setup(old: str, new: str) -> tuple[Any, CanonicalMemory, CanonicalMemory, CanonicalMemory]:
    turn = _memory(old, at=T0, source="msg_a")
    fact = _memory(
        old,
        at=T0,
        source="msg_a",
        memory_type=MemoryType.SEMANTIC,
        system_metadata={"category": "fact"},
    )
    newer = _memory(new, at=T1, source="msg_b")
    return _uow([turn, fact, newer]), turn, fact, newer


@pytest.mark.parametrize(("old", "new"), [(ULINE, PACKAGINGCORP), (ULINE_HI, PACKAGINGCORP_HI)])
async def test_a_contradicted_statement_and_every_form_of_it_is_superseded(
    old: str, new: str
) -> None:
    uow, turn, fact, newer = _setup(old, new)
    nli = ScriptedNLI({("PackagingCorp", "Uline"): 0.97})
    outcome = await CrossConversationReplacement(nli).run(uow, CTX, [newer], now=T1)
    assert {m.memory_id for m in outcome.changed} == {turn.memory_id, fact.memory_id}
    for form in (turn, fact):
        stored = uow.memories.rows[form.memory_id]
        assert stored.temporal.status is TemporalStatus.SUPERSEDED
        assert stored.temporal.superseded_by == newer.memory_id
        assert replaced_by(stored.system_metadata) == newer.memory_id
    assert uow.memories.rows[newer.memory_id].temporal.status is TemporalStatus.CURRENT


async def test_a_weaker_contradiction_marks_an_earlier_value_that_stays_current() -> None:
    uow, turn, _, newer = _setup(ULINE, PACKAGINGCORP)
    nli = ScriptedNLI({("PackagingCorp", "Uline"): 0.87})
    await CrossConversationReplacement(nli).run(uow, CTX, [newer], now=T1)
    stored = uow.memories.rows[turn.memory_id]
    assert stored.temporal.status is TemporalStatus.CURRENT
    assert earlier_value(stored.system_metadata)


async def test_what_is_about_something_else_or_said_later_is_untouched() -> None:
    other = _memory("Golden Crust delivers bread on Tuesdays.", at=T0, source="msg_c")
    later = _memory(
        "Our primary supplier for heavy-duty pallets is Uline.",
        at=T1 + timedelta(1),
        source="msg_d",
    )
    newer = _memory(PACKAGINGCORP, at=T1, source="msg_b")
    uow = _uow([other, later, newer])
    nli = ScriptedNLI({("PackagingCorp", ""): 0.99})
    outcome = await CrossConversationReplacement(nli).run(uow, CTX, [newer], now=T1)
    assert outcome.changed == []  # no shared name with the bakery; the other is newer


async def test_without_a_trained_head_nothing_is_compared() -> None:
    uow, _, _, newer = _setup(ULINE, PACKAGINGCORP)
    stand_in = SimpleNamespace(representative=False)
    replacement = CrossConversationReplacement(stand_in)  # type: ignore[arg-type]
    assert not replacement.enabled
    assert (await replacement.run(uow, CTX, [newer], now=T1)).changed == []


async def test_another_persons_statement_is_never_replaced() -> None:
    uow, turn, _, newer = _setup(ULINE, PACKAGINGCORP)
    uow.memories.rows[turn.memory_id] = turn.model_copy(update={"owner_principal": "user:u2"})
    nli = ScriptedNLI({("PackagingCorp", "Uline"): 0.99})
    outcome = await CrossConversationReplacement(nli).run(uow, CTX, [newer], now=T1)
    assert turn.memory_id not in {m.memory_id for m in outcome.changed}


async def test_forgetting_the_newer_statement_reinstates_what_it_replaced() -> None:
    uow, turn, fact, newer = _setup(ULINE, PACKAGINGCORP)
    nli = ScriptedNLI({("PackagingCorp", "Uline"): 0.97})
    await CrossConversationReplacement(nli).run(uow, CTX, [newer], now=T1)
    back = await reinstate(uow, "acme", [newer.memory_id], now=T1)
    assert {m.memory_id for m in back} == {turn.memory_id, fact.memory_id}
    for form in (turn, fact):
        stored = uow.memories.rows[form.memory_id]
        assert stored.temporal.status is TemporalStatus.CURRENT
        assert stored.temporal.superseded_by is None and stored.temporal.valid_to is None
        assert replaced_by(stored.system_metadata) is None


# --------------------------------------------------------------------------- read path


def _candidate(record_id: str, **payload: Any) -> Candidate:
    return Candidate(record_id=record_id, kind="memory", text=record_id, score=1.0, payload=payload)


def test_an_earlier_value_ranks_after_every_current_item() -> None:
    ranked = current_first(
        [_candidate("old", earlier_value=True), _candidate("a"), _candidate("b")]
    )
    assert [c.record_id for c in ranked] == ["a", "b", "old"]


def test_an_earlier_value_is_labelled_wherever_it_is_served() -> None:
    item = ContextItem(
        item_id="m1",
        representation=Representation.MEMORY,
        text=ULINE,
        citation="memory_id:m1",
        attributes={"earlier_value": True, "observed_at": "2026-03-01T10:00:00+00:00"},
    )
    assert memory_view(item)["superseded"] is True
    assert "(earlier value, since replaced)" in _memory_line(item, "m1")
    current = item.model_copy(update={"attributes": {}})
    assert "superseded" not in memory_view(current)
    assert "earlier value" not in _memory_line(current, "m1")


# --------------------------------------------------------------------------- the model's say


class FakeAssist:
    """The tenant's model through ``LLMAssist``'s interface: a verdict per call, counted;
    ``None`` is what ``structured`` returns when the model fails or is not allowed."""

    def __init__(self, verdict: str | None, *, allowed: bool = True) -> None:
        self.verdict, self.allowed, self.calls = verdict, allowed, 0

    def wants(self, use: str) -> bool:
        return self.allowed and use == "conflict_adjudication"

    async def structured(self, use: str, **_: Any) -> dict[str, Any] | None:
        self.calls += 1
        return {"verdict": self.verdict} if self.verdict else None


class FixedEncoder:
    """Unit vectors whose pairwise cosine is ``cosine`` between any two different texts."""

    def __init__(self, cosine: float) -> None:
        self.cosine = cosine

    def fingerprint(self) -> str:
        return "fixed-test-encoder"

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        import math

        # every text shares the first axis at `cosine` and has an axis of its own
        out = []
        rest = math.sqrt(1 - self.cosine)
        for n, _ in enumerate(texts):
            v = [0.0] * (len(texts) + 1)
            v[0] = math.sqrt(self.cosine)
            v[n + 1] = rest
            out.append(v)
        return out


SEATTLE = "I live in Seattle, in the Fremont neighbourhood."
AUSTIN = "Big news: I moved to Austin last month, I'm in an apartment near Zilker Park now."


def _moved() -> tuple[Any, CanonicalMemory, CanonicalMemory]:
    old = _memory(SEATTLE, at=T0, source="msg_a")
    new = _memory(AUSTIN, at=T1, source="msg_b")
    return _uow([old, new]), old, new


async def test_an_unanchored_far_pair_is_never_replaced_or_asked() -> None:
    """ "I live in Seattle" / "I moved to Austin": no shared name, cosine 0.4. Neither the head
    nor the model decides it (with Granite such pairs were marked wrongly: ADR 0038)."""
    uow, _, new = _moved()
    nli = ScriptedNLI({("Austin", "Seattle"): 0.99})
    assist = FakeAssist("update")
    out = await CrossConversationReplacement(nli, FixedEncoder(0.4), assist=assist).run(  # type: ignore[arg-type]
        uow, CTX, [new], now=T1
    )
    assert out.changed == [] and assist.calls == 0


#: no change cue (the lexicon reads it as a plain fact): the head's 0.6 alone marks nothing
PLAIN_NEW = "Our heavy-duty pallets come from PackagingCorp and Uline."


def _band() -> tuple[Any, CanonicalMemory, CanonicalMemory]:
    old = _memory(ULINE, at=T0, source="msg_a")
    new = _memory(PLAIN_NEW, at=T1, source="msg_b")
    return _uow([old, new]), old, new


@pytest.mark.parametrize(
    ("verdict", "replaced"),
    [("update", True), ("contradict", True), ("same", False), ("different", False), (None, False)],
)
async def test_the_model_settles_the_heads_unsure_band(verdict: str | None, replaced: bool) -> None:
    """Shared name, contradiction 0.6 (marks nothing on the head's word): the model's
    'update' or 'contradict' marks an earlier value, anything else (or a failed call) none."""
    uow, old, new = _band()
    assist = FakeAssist(verdict)
    nli = ScriptedNLI({("PackagingCorp", "Uline"): 0.6})
    out = await CrossConversationReplacement(nli, assist=assist).run(  # type: ignore[arg-type]
        uow, CTX, [new], now=T1
    )
    assert assist.calls == 1
    assert bool(out.changed) is replaced
    if replaced:
        stored = uow.memories.rows[old.memory_id]
        assert earlier_value(stored.system_metadata), "the model's word is an earlier value only"
        assert stored.temporal.status is TemporalStatus.CURRENT


async def test_no_contradiction_is_never_put_to_the_model() -> None:
    """Same named thing, the head reads no contradiction ("is back from leave", a multi-hop
    bridge): not asked."""
    old = _memory("Maria Lopez is on parental leave until further notice.", at=T0, source="msg_a")
    new = _memory("Maria Lopez is back from parental leave as of Monday.", at=T1, source="msg_b")
    assist = FakeAssist("update")
    out = await CrossConversationReplacement(ScriptedNLI({}), assist=assist).run(  # type: ignore[arg-type]
        _uow([old, new]), CTX, [new], now=T1
    )
    assert assist.calls == 0 and out.changed == []


async def test_the_model_is_asked_about_three_pairs_at_most() -> None:
    olds = [
        _memory(
            f"Uline note {n}: our pallets come from Uline.", at=T0 + timedelta(n), source=f"msg_{n}"
        )
        for n in range(6)
    ]
    new = _memory(PLAIN_NEW, at=T1, source="msg_b")
    assist = FakeAssist("update")
    nli = ScriptedNLI({("PackagingCorp", "Uline"): 0.6})
    await CrossConversationReplacement(nli, assist=assist).run(  # type: ignore[arg-type]
        _uow([*olds, new]), CTX, [new], now=T1
    )
    assert assist.calls == 3


async def test_a_failing_head_raises_to_the_pipeline_which_keeps_the_message() -> None:
    """The class lets a head failure through; the pipeline catches it
    (``ObservationPipeline._replace_earlier``) and stores the statement unreplaced."""

    class Broken(ScriptedNLI):
        async def entail_groups(self, groups):  # type: ignore[no-untyped-def]
            raise RuntimeError("onnx session failed")

    uow, _, _, newer = _setup(ULINE, PACKAGINGCORP)
    with pytest.raises(RuntimeError):
        await CrossConversationReplacement(Broken({})).run(uow, CTX, [newer], now=T1)
