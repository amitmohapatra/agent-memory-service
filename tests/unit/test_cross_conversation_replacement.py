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
from memory_service.domain.context_bundle import (
    ContextBundle,
    ContextItem,
    ConversationWindow,
    EvidenceReport,
    _memory_line,
)
from memory_service.domain.enums import (
    EvidenceStatus,
    MemoryType,
    QueryType,
    Representation,
    StatementKind,
    TemporalStatus,
)
from memory_service.domain.evidence import EvidenceRef, EvidenceSource
from memory_service.domain.memory import CanonicalMemory, earlier_value, replaced_by
from memory_service.modules.context.handles import record_of
from memory_service.modules.context.views import _memory as memory_view
from memory_service.modules.context.views import full_view, prompt_view
from memory_service.modules.memory.replacement import (
    RELATED_COSINE,
    CrossConversationReplacement,
    Grade,
    PairScores,
    Relation,
    Thresholds,
    anchors,
    decide,
    finished_after_ongoing,
    misread_rule,
    reinstate,
    relate,
    slot_changed,
    slots_of,
    spoken_slots,
    statement_of,
)
from memory_service.modules.memory.statements import Lexicon
from memory_service.modules.retrieval.engine import Candidate, RetrievalEngine, current_first
from memory_service.ports.models import NLIScore

pytestmark = pytest.mark.unit

T0 = datetime(2026, 3, 1, 10, 0, tzinfo=UTC)
T1 = T0 + timedelta(days=40)
CTX = MemoryExecutionContext(tenant_id="acme", user_id="u1", thread_id="thr_b")

OLD = "Our main supplier of shipping crates is Brightway."
NEW = (
    "We ended the Brightway agreement last week over delivery times. "
    "Shipping crates now come from Corrimax only."
)
OLD_HI = "शिपिंग क्रेट का हमारा मुख्य सप्लायर Brightway है।"
NEW_HI = (
    "पिछले हफ़्ते डिलीवरी में देरी के कारण हमने Brightway के साथ समझौता ख़त्म कर दिया। "
    "अब शिपिंग क्रेट सिर्फ़ Corrimax से आते हैं।"
)


# --------------------------------------------------------------------------- anchors


def test_a_statement_names_its_names_and_identifiers() -> None:
    a = anchors("Delivery bike #7 is assigned to Lior Adler.", "en")
    assert "lior adler" in a.names and ("bike", "7") in a.ids
    # a sentence-initial capital is not a name; a label before a number is not one either
    assert "delivery" not in a.names and "bike" not in a.names
    b = anchors("The minimum stock for SKU QX-410 is 25 cases.", "en")
    assert any(v == "qx410" for _, v in b.ids) and "sku" not in b.names


def test_a_possessive_names_the_same_person() -> None:
    """"Jordan Lee's temp assignment has ended" is about the Jordan Lee of the shift logs: the
    possessive is no part of the name, or the two share no name and the ended assignment
    leaves every log of it current."""
    new = "Jordan Lee's temp assignment has ended; his badge is deactivated."
    old = "Shift log 2026-10-16: Jordan Lee (temp) worked receiving, unloaded 18 pallets."
    assert anchors(new, "en").names == {"jordan lee"}
    assert anchors("We asked Priya’s Team Lead.", "en").names == {"priya", "team lead"}
    assert relate(anchors(new, "en"), anchors(old, "en"), new, old).shared_name


def test_latin_names_inside_another_script_are_names() -> None:
    a = anchors(OLD_HI, "hi")
    assert "brightway" in a.names
    trolley = anchors("ट्रॉली #4 आज मरम्मत में है।", "hi")
    assert ("ट्रॉली", "4") in trolley.ids


@pytest.mark.parametrize(
    ("new", "old", "shared"),
    [
        # a title both use bare: the same thing
        ("Ich habe Dune gestern Abend zu Ende gelesen.", "Ich lese gerade Dune.", True),
        # a noun after a determiner, a label before a number, a unit after one: never
        ("Die Halle ist voll.", "Wir räumen die Halle morgen.", False),
        ("Retouren gehen über Tor 9.", "Tor 14 ist reserviert.", False),
        ("Die Nachtschicht endet um 3 Uhr.", "Meine Schicht beginnt um 22 Uhr.", False),
    ],
)
def test_a_german_word_both_use_bare_is_a_name(new: str, old: str, shared: bool) -> None:
    """German capitalises every noun, so one capitalised word is no name on its own; used bare
    (no determiner, no number beside it) in both statements, it names one thing."""
    assert relate(anchors(new, "de"), anchors(old, "de")).shared_name is shared


def test_german_nouns_are_not_names_but_names_are() -> None:
    a = anchors("Bei Stromausfällen im Lager Owen Pike unter Durchwahl 318 anrufen.", "de")
    b = anchors("Owen Pike ist nicht mehr bei uns. Zuständig ist jetzt Rhea Sood.", "de")
    assert relate(b, a).shared_name  # "owen pike" inside "lager owen pike"
    assert not anchors("Leergut geht wie immer über Halle 2.", "de").names


# --------------------------------------------------------------------------- relate


@pytest.mark.parametrize(
    ("new", "old", "lang"),
    [
        (
            "Delivery bike #5 got new brakes and is out on routes again.",
            "Delivery bike #2 stays in the yard for the week.",
            "en",
        ),
        (
            "ट्रॉली #5 की मरम्मत हो गई है और वह फिर से काम पर है।",
            "ट्रॉली #2 इस हफ़्ते गोदाम में रहेगी।",
            "hi",
        ),
        (
            "Y las bolsas de basura, SKU KV-3301, llevan el UPC 071234560018.",
            "Para que conste: el UPC del film transparente, SKU LM-8812, es 071234569912.",
            "es",
        ),
    ],
)
def test_two_things_of_one_kind_are_apart(new: str, old: str, lang: str) -> None:
    assert relate(anchors(new, lang), anchors(old, lang)).apart


def test_a_shared_name_keeps_a_changed_extension_together() -> None:
    new = "Owen Pike has left the company. The outage contact is now Rhea Sood, extension 412."
    old = "For power outages in the warehouse, call Owen Pike on extension 318."
    relation = relate(anchors(new, "en"), anchors(old, "en"), new, old)
    assert relation.shared_name and not relation.apart


def test_a_number_that_changed_on_the_same_thing() -> None:
    new = "Update: the minimum stock for QX-410 is 40 cases now."
    old = "The minimum stock for SKU QX-410 (sparkling water, 1 litre) is 25 cases."
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
    # "Depot 7 stays in the coastal zone" / "the two-for-one runs only inland"
    assert decide(UNANCHORED, PairScores(0.3, 1.0)).grade is None
    assert decide(UNANCHORED, PairScores(None, 1.0)).grade is None  # no encoder, no anchor
    apart = Relation(shared_name=False, shared_id=False, apart=True)
    assert decide(apart, PairScores(0.9, 1.0)).grade is None
    assert decide(UNANCHORED, PairScores(0.6, 0.9)).grade is Grade.SUPERSEDED


def test_thresholds_are_the_measured_ones() -> None:
    t = Thresholds()
    assert (t.earlier, t.change, t.supersede) == (0.85, 0.4, 0.9)
    assert RELATED_COSINE == 0.5
    # below the encoder's bar a change cue relates nothing
    lifecycle = [StatementKind.LIFECYCLE]
    assert decide(UNANCHORED, PairScores(0.35, 0.99), lifecycle).grade is None


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
        self,
        tenant_id: str,
        owner: str,
        *,
        before: datetime,
        limit: int,
        after: datetime | None = None,
    ) -> list[CanonicalMemory]:
        rows = [
            m
            for m in self.rows.values()
            if m.owner_principal == owner
            and m.temporal.status is TemporalStatus.CURRENT
            and m.temporal.observed_at <= before
            and (after is None or m.temporal.observed_at > after)
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


@pytest.mark.parametrize(("old", "new"), [(OLD, NEW), (OLD_HI, NEW_HI)])
async def test_a_contradicted_statement_and_every_form_of_it_is_superseded(
    old: str, new: str
) -> None:
    uow, turn, fact, newer = _setup(old, new)
    nli = ScriptedNLI({("Corrimax", "Brightway"): 0.97})
    outcome = await CrossConversationReplacement(nli).run(uow, CTX, [newer], now=T1)
    assert {m.memory_id for m in outcome.changed} == {turn.memory_id, fact.memory_id}
    for form in (turn, fact):
        stored = uow.memories.rows[form.memory_id]
        assert stored.temporal.status is TemporalStatus.SUPERSEDED
        assert stored.temporal.superseded_by == newer.memory_id
        assert replaced_by(stored.system_metadata) == newer.memory_id
    assert uow.memories.rows[newer.memory_id].temporal.status is TemporalStatus.CURRENT


async def test_a_weaker_contradiction_marks_an_earlier_value_that_stays_current() -> None:
    uow, turn, _, newer = _setup(OLD, NEW)
    nli = ScriptedNLI({("Corrimax", "Brightway"): 0.87})
    await CrossConversationReplacement(nli).run(uow, CTX, [newer], now=T1)
    stored = uow.memories.rows[turn.memory_id]
    assert stored.temporal.status is TemporalStatus.CURRENT
    assert earlier_value(stored.system_metadata)


async def test_what_is_about_something_else_or_said_later_is_untouched() -> None:
    other = _memory("Fernhill Bakery delivers bread on Tuesdays.", at=T0, source="msg_c")
    later = _memory(OLD, at=T1 + timedelta(1), source="msg_d")
    newer = _memory(NEW, at=T1, source="msg_b")
    uow = _uow([other, later, newer])
    nli = ScriptedNLI({("Corrimax", ""): 0.99})
    outcome = await CrossConversationReplacement(nli).run(uow, CTX, [newer], now=T1)
    # no shared name with the bakery; the statement said after this one is the newer, and
    # it is this one that it replaces (event time decides, not the order of storing)
    assert [m.memory_id for m in outcome.changed] == [newer.memory_id]
    assert replaced_by(uow.memories.rows[newer.memory_id].system_metadata) == later.memory_id
    assert uow.memories.rows[later.memory_id].temporal.status is TemporalStatus.CURRENT


async def test_without_a_trained_head_nothing_is_compared() -> None:
    uow, _, _, newer = _setup(OLD, NEW)
    stand_in = SimpleNamespace(representative=False)
    replacement = CrossConversationReplacement(stand_in)  # type: ignore[arg-type]
    assert not replacement.enabled
    assert (await replacement.run(uow, CTX, [newer], now=T1)).changed == []


async def test_another_persons_statement_is_never_replaced() -> None:
    uow, turn, _, newer = _setup(OLD, NEW)
    uow.memories.rows[turn.memory_id] = turn.model_copy(update={"owner_principal": "user:u2"})
    nli = ScriptedNLI({("Corrimax", "Brightway"): 0.99})
    outcome = await CrossConversationReplacement(nli).run(uow, CTX, [newer], now=T1)
    assert turn.memory_id not in {m.memory_id for m in outcome.changed}


async def test_forgetting_the_newer_statement_reinstates_what_it_replaced() -> None:
    uow, turn, fact, newer = _setup(OLD, NEW)
    nli = ScriptedNLI({("Corrimax", "Brightway"): 0.97})
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
        text=OLD,
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


LEEDS = "I live in Leeds, near Headingley."
BRISTOL = "Quick update: I moved to Bristol in spring and rent a flat near Clifton now."


def _moved() -> tuple[Any, CanonicalMemory, CanonicalMemory]:
    old = _memory(LEEDS, at=T0, source="msg_a")
    new = _memory(BRISTOL, at=T1, source="msg_b")
    return _uow([old, new]), old, new


async def test_an_unanchored_far_pair_is_never_replaced_or_asked() -> None:
    """ "I live in Leeds" / "I moved to Bristol": no shared name, no slot read, cosine 0.4.
    Neither the head nor the model decides it, even beside a change cue and a near-certain
    contradiction: two different named things that both changed read the same way."""
    uow, _, new = _moved()
    assist = FakeAssist("update")
    nli = ScriptedNLI({("Bristol", "Leeds"): 0.99})
    out = await CrossConversationReplacement(nli, FixedEncoder(0.4), assist=assist).run(  # type: ignore[arg-type]
        uow, CTX, [new], now=T1
    )
    assert out.changed == [] and assist.calls == 0


#: no change cue (the lexicon reads it as a plain fact): the head's 0.6 alone marks nothing
PLAIN_NEW = "Our shipping crates come from Corrimax and Brightway."


def _band() -> tuple[Any, CanonicalMemory, CanonicalMemory]:
    old = _memory(OLD, at=T0, source="msg_a")
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
    nli = ScriptedNLI({("Corrimax", "Brightway"): 0.6})
    out = await CrossConversationReplacement(nli, assist=assist).run(  # type: ignore[arg-type]
        uow, CTX, [new], now=T1
    )
    assert assist.calls == 1
    assert bool(out.changed) is replaced
    if replaced:
        stored = uow.memories.rows[old.memory_id]
        assert earlier_value(stored.system_metadata), "the model's word is an earlier value only"
        assert stored.temporal.status is TemporalStatus.CURRENT


@pytest.mark.parametrize(
    ("old", "new", "asked"),
    [
        # a stated change about the thing both name: asked, whatever the head read
        (
            "Ana Petrova is on sabbatical.",
            "Ana Petrova is back from sabbatical as of Monday.",
            True,
        ),
        (
            "Forklift #4 is in the repair shop until Friday.",
            "Forklift #4 is back in service.",
            True,
        ),
        # the change is about someone else, tied by a shared place: never asked
        (
            "Staff of the Lyon office follow the French holiday calendar.",
            "Ines Moreau joined the Lyon office this month.",
            False,
        ),
        # no change stated, no contradiction read: never asked
        (
            "Ana Petrova leads the Leeds depot.",
            "Ana Petrova likes the new rota.",
            False,
        ),
    ],
)
async def test_a_stated_change_about_the_shared_thing_is_put_to_the_model(
    old: str, new: str, asked: bool
) -> None:
    """The head reads "is back from sabbatical" against "is on sabbatical" as no
    contradiction (in Arabic and Hindi it reads most such pairs that way): the model is asked
    when the new statement states a change and the first thing it names is one the old
    statement names; a bridge between two facts ("joined the Lyon office") is not."""
    old_m = _memory(old, at=T0, source="msg_a")
    new_m = _memory(new, at=T1, source="msg_b")
    assist = FakeAssist("update")
    uow = _uow([old_m, new_m])
    out = await CrossConversationReplacement(ScriptedNLI({}), assist=assist).run(  # type: ignore[arg-type]
        uow, CTX, [new_m], now=T1
    )
    assert assist.calls == (1 if asked else 0)
    assert bool(out.changed) is asked
    if asked:
        stored = uow.memories.rows[old_m.memory_id]
        assert earlier_value(stored.system_metadata), "the model's word is an earlier value only"


@pytest.mark.parametrize(
    ("old", "new", "cosine", "marked"),
    [
        # the head reads these at 0.02 in Arabic and Hindi: the stated change about the thing
        # the older statement is about, close by the encoder, marks it
        (
            "Ana Petrova is on sabbatical.",
            "Ana Petrova is back from sabbatical as of Monday.",
            0.6,
            True,
        ),
        # the same pair the encoder puts apart: about another aspect of Ana, left alone
        (
            "Ana Petrova is on sabbatical.",
            "Ana Petrova is back from sabbatical as of Monday.",
            0.4,
            False,
        ),
        # a bridge: leads with someone the older statement does not name
        (
            "Staff of the Lyon office follow the French holiday calendar.",
            "Ines Moreau joined the Lyon office this month.",
            0.9,
            False,
        ),
    ],
)
async def test_a_stated_change_about_the_same_thing_marks_an_earlier_value(
    old: str, new: str, cosine: float, marked: bool
) -> None:
    old_m = _memory(old, at=T0, source="msg_a")
    new_m = _memory(new, at=T1, source="msg_b")
    uow = _uow([old_m, new_m])
    out = await CrossConversationReplacement(ScriptedNLI({}), FixedEncoder(cosine)).run(
        uow, CTX, [new_m], now=T1
    )
    assert bool(out.changed) is marked
    if marked:
        stored = uow.memories.rows[old_m.memory_id]
        assert earlier_value(stored.system_metadata), "an earlier value, never superseded"
        assert stored.temporal.status is TemporalStatus.CURRENT


async def test_a_stated_change_is_not_put_to_the_model_without_one() -> None:
    old = _memory("Ana Petrova is on sabbatical.", at=T0, source="msg_a")
    new = _memory("Ana Petrova is back from sabbatical as of Monday.", at=T1, source="msg_b")
    out = await CrossConversationReplacement(ScriptedNLI({})).run(
        _uow([old, new]), CTX, [new], now=T1
    )
    assert out.changed == [], "without the tenant's model the head's word stands"


@pytest.mark.parametrize(
    "said",
    [
        # a status cue and a shared identifier: marked "whatever the head read" before
        "For employee e-401: list the trips, the booked flights and the pending expense claims.",
        "Stores S12 and S40 both carry style ST-101. Which should get stock first?",
        OLD_HI,
    ],
)
async def test_the_same_words_said_again_replace_nothing(said: str) -> None:
    """S5: a request a user repeated in four conversations marked three of its copies "since
    replaced", whatever the head, the encoder or the model read. The copies are not merged
    either: each stays current in its own conversation."""

    def said_in(thread: str, at: datetime, source: str) -> CanonicalMemory:
        scope = {"level": "THREAD", "tenant_id": "acme", "thread_id": thread}
        return _memory(said, at=at, source=source, scope=scope)

    old, new = said_in("thr_a", T0, "msg_a"), said_in("thr_b", T1, "msg_b")
    uow = _uow([old, new])
    replacement = CrossConversationReplacement(
        ScriptedNLI({("", ""): 0.99}),
        FixedEncoder(0.99),
        assist=FakeAssist("update"),  # type: ignore[arg-type]
    )
    assert (await replacement.run(uow, CTX, [new], now=T1)).changed == []
    # and stored out of order, the later copy does not replace the earlier one either
    assert (await replacement.run(uow, CTX, [old], now=T1)).changed == []
    for memory in (old, new):
        stored = uow.memories.rows[memory.memory_id]
        assert stored.temporal.status is TemporalStatus.CURRENT
        assert replaced_by(stored.system_metadata) is None
    assert {m.scope.thread_id for m in uow.memories.rows.values()} == {"thr_a", "thr_b"}


async def test_the_model_is_asked_about_three_pairs_at_most() -> None:
    olds = [
        _memory(
            f"Brightway note {n}: our crates come from Brightway.",
            at=T0 + timedelta(n),
            source=f"msg_{n}",
        )
        for n in range(6)
    ]
    new = _memory(PLAIN_NEW, at=T1, source="msg_b")
    assist = FakeAssist("update")
    nli = ScriptedNLI({("Corrimax", "Brightway"): 0.6})
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

    uow, _, _, newer = _setup(OLD, NEW)
    with pytest.raises(RuntimeError):
        await CrossConversationReplacement(Broken({})).run(uow, CTX, [newer], now=T1)


# --------------------------------------------------------------------------- change cues


@pytest.mark.parametrize(
    ("sentence", "kind"),
    [
        ("Priya handed the Leeds depot over to Tom on Friday.", StatementKind.LIFECYCLE),
        ("Jonas hat die Filiale an Mia übergeben.", StatementKind.LIFECYCLE),
        ("राहुल ने गोदाम 4 सुनीता को सौंप दिया।", StatementKind.LIFECYCLE),
        ("Ana is back from sabbatical.", StatementKind.STATUS),
        ("عادت ليلى من الإجازة أمس.", StatementKind.STATUS),
    ],
)
def test_a_handover_and_a_return_are_suspected_changes(sentence: str, kind: StatementKind) -> None:
    """Weak cues: the lexicon suspects the change, the head (or nothing) confirms it."""
    label = Lexicon.default().label(sentence)
    assert label.maybe is kind and not label.decided


# --------------------------------------------------------------------------- slots


def _reading(
    content: str, at: datetime, source: str, predicate: str, value: str
) -> CanonicalMemory:
    return _memory(
        content,
        at=at,
        source=source,
        memory_type=MemoryType.USER,
        subject="user:u1",
        predicate=predicate,
        object=value,
        system_metadata={"category": "attribute"},
    )


def test_a_single_valued_slot_with_another_value_is_a_change() -> None:
    leeds = slots_of([_reading("I live in Leeds.", T0, "m1", "lives_in", "Leeds")])
    bristol = slots_of([_reading("I moved to Bristol.", T1, "m2", "lives_in", "Bristol")])
    assert slot_changed(bristol, leeds)
    assert not slot_changed(leeds, leeds)  # the same value again is no change
    # a slot that accumulates is never a change ("likes" one thing, then another)
    likes = slots_of([_reading("I like jazz.", T0, "m1", "likes", "jazz")])
    assert likes == {}


async def test_a_slot_relates_two_statements_with_nothing_else_in_common() -> None:
    """The extraction read one slot of one subject from both messages: the turns and the
    readings are compared without any shared name and without an encoder."""
    old_turn = _memory("I live in Leeds, near Headingley.", at=T0, source="msg_a")
    old_fact = _reading("I live in Leeds, near Headingley.", T0, "msg_a", "lives_in", "Leeds")
    new_turn = _memory(BRISTOL, at=T1, source="msg_b")
    new_fact = _reading(BRISTOL, T1, "msg_b", "lives_in", "Bristol")
    uow = _uow([old_turn, old_fact, new_turn, new_fact])
    nli = ScriptedNLI({("Bristol", "Leeds"): 0.6})
    out = await CrossConversationReplacement(nli).run(uow, CTX, [new_turn, new_fact], now=T1)
    assert {m.memory_id for m in out.changed} == {old_turn.memory_id, old_fact.memory_id}
    assert all(earlier_value(uow.memories.rows[m.memory_id].system_metadata) for m in out.changed)


async def test_a_slot_of_another_subject_relates_nothing() -> None:
    old = _reading("Mum lives in Leeds.", T0, "msg_a", "lives_in", "Leeds")
    old = old.model_copy(update={"subject": "person:mum"})
    new_turn = _memory(BRISTOL, at=T1, source="msg_b")
    new_fact = _reading(BRISTOL, T1, "msg_b", "lives_in", "Bristol")
    nli = ScriptedNLI({("Bristol", "Leeds"): 0.6})
    out = await CrossConversationReplacement(nli).run(
        _uow([old, new_turn, new_fact]), CTX, [new_turn, new_fact], now=T1
    )
    assert out.changed == []


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("Ich wohne in Leeds, nahe am Park.", "Ich bin letzten Monat nach Bristol gezogen."),
        ("Vivo en Leeds, cerca del parque.", "El mes pasado me mudé a Bristol."),
        ("أعيش في Leeds، قرب الحديقة.", "انتقلت إلى Bristol الشهر الماضي."),
        ("मैं Leeds में, पार्क के पास रहता हूँ।", "अब मैं Bristol में एक फ़्लैट में रहता हूँ।"),
    ],
)
def test_where_the_speaker_lives_is_read_in_every_pack_language(old: str, new: str) -> None:
    """Without a model nothing reads these: no shared name, the encoder below 0.5. The
    speaker's own residence, two places: one slot changed."""
    assert slot_changed(spoken_slots(new), spoken_slots(old))
    assert not slot_changed(spoken_slots(old), spoken_slots(old))


@pytest.mark.parametrize(
    "text",
    [
        "Mein Bruder wohnt in Leeds.",  # not the speaker
        "انتقلت Maria إلى Lyon.",  # she moved
        "Ich lebe vegan.",
        "Vivo con mi hermana.",
        "We moved the pallets to Dock 4.",
    ],
)
def test_only_the_speakers_own_residence_is_a_slot(text: str) -> None:
    assert spoken_slots(text) == {}


async def test_a_move_relates_two_statements_with_nothing_else_in_common() -> None:
    old = _memory("Ich wohne in Leeds, nahe am Park.", at=T0, source="msg_a", lang="de")
    new = _memory("Ich bin letzten Monat nach Bristol gezogen.", at=T1, source="msg_b", lang="de")
    nli = ScriptedNLI({("Bristol", "Leeds"): 0.95})
    uow = _uow([old, new])
    out = await CrossConversationReplacement(nli).run(uow, CTX, [new], now=T1)
    assert [m.memory_id for m in out.changed] == [old.memory_id]
    assert uow.memories.rows[old.memory_id].temporal.status is TemporalStatus.SUPERSEDED


def test_a_hindi_cue_matches_with_or_without_the_nukta() -> None:
    lexicon = Lexicon.default()
    for sentence in ("कल रात मैंने किताब ख़त्म की।", "कल रात मैंने किताब खत्म की।"):
        assert lexicon.label(sentence).maybe is StatementKind.LIFECYCLE


# --------------------------------------------------------------------------- misread rules


def _rule(content: str, at: datetime, source: str) -> CanonicalMemory:
    return _memory(
        content,
        at=at,
        source=source,
        memory_type=MemoryType.SEMANTIC,
        system_metadata={"category": "rule", "statement_kind": "RULE"},
    )


def test_a_fact_stored_as_a_rule_is_misread_and_a_rule_is_not() -> None:
    assert misread_rule(_rule(OLD, T0, "msg_a"))
    assert misread_rule(_rule(OLD_HI, T0, "msg_a"))
    assert not misread_rule(_rule("Always pack fragile items in double boxes.", T0, "msg_a"))
    assert not misread_rule(_rule("हमेशा नाज़ुक सामान को दो डिब्बों में पैक करो।", T0, "msg_a"))


@pytest.mark.parametrize(("old", "new"), [(OLD, NEW), (OLD_HI, NEW_HI)])
async def test_a_fact_misread_as_a_standing_rule_is_replaced(old: str, new: str) -> None:
    misread = _rule(old, T0, "msg_a")
    newer = _memory(new, at=T1, source="msg_b")
    uow = _uow([misread, newer])
    nli = ScriptedNLI({("Corrimax", "Brightway"): 0.97})
    out = await CrossConversationReplacement(nli).run(uow, CTX, [newer], now=T1)
    assert [m.memory_id for m in out.changed] == [misread.memory_id]


async def test_a_standing_rule_is_left_to_the_rule_that_revises_it() -> None:
    rule = _rule("Always order shipping crates from Brightway.", T0, "msg_a")
    newer = _memory(NEW, at=T1, source="msg_b")
    nli = ScriptedNLI({("Corrimax", "Brightway"): 0.97})
    out = await CrossConversationReplacement(nli).run(_uow([rule, newer]), CTX, [newer], now=T1)
    assert out.changed == []


async def test_a_misread_rule_is_never_compared_on_the_encoder_alone() -> None:
    misread = _rule("Our crates are stacked six high in the yard.", T0, "msg_a")
    newer = _memory("We now stack them four high, per the new safety audit.", at=T1, source="msg_b")
    nli = ScriptedNLI({("four", "six"): 0.99})
    out = await CrossConversationReplacement(nli, FixedEncoder(0.8)).run(
        _uow([misread, newer]), CTX, [newer], now=T1
    )
    assert out.changed == [] and nli.pairs == []


# --------------------------------------------------------------------------- history


def _item(item_id: str, text: str, *, earlier: bool = False) -> ContextItem:
    return ContextItem(
        item_id=item_id,
        representation=Representation.MEMORY,
        text=text,
        citation=f"memory_id:{item_id}",
        attributes={
            "observed_at": "2026-03-01T10:00:00+00:00",
            **({"earlier_value": True} if earlier else {}),
        },
    )


def _bundle(**sections: Any) -> ContextBundle:
    return ContextBundle(
        query="q",
        query_type=QueryType.USER_MEMORY,
        conversation=ConversationWindow(thread_id="t"),
        evidence=EvidenceReport(status=EvidenceStatus.COMPLETE),
        token_budget=100,
        token_estimate=10,
        **sections,
    )


@pytest.mark.parametrize(("old", "new"), [(OLD, NEW), (OLD_HI, NEW_HI)])
def test_an_earlier_value_is_rendered_apart_under_its_own_heading(old: str, new: str) -> None:
    bundle = _bundle(memories=[_item("cur", new)], history=[_item("was", old, earlier=True)])
    text = bundle.render()
    current, _, history = text.partition("## Earlier values (since replaced)")
    assert new in current and old not in current
    assert old in history and "[h1]" in history
    assert bundle.handles() == {"m1": "cur", "h1": "was"}
    view = full_view(bundle)
    assert [m["text"] for m in view["memories"]] == [new]
    assert [(m["text"], m["superseded"]) for m in view["history"]] == [(old, True)]


def test_an_earlier_value_is_no_evidence_for_a_claim() -> None:
    bundle = _bundle(memories=[_item("cur", NEW)], history=[_item("was", OLD, earlier=True)])
    record = record_of(bundle)
    assert record.handles["h1"] == "was"
    assert [e.item_id for e in record.evidence] == ["cur"]


# --------------------------------------------------------------------------- follow


class _Visible:
    def __init__(self, hidden: frozenset[str] = frozenset()) -> None:
        self.hidden = hidden

    def allows(self, tenant_id: str, keys: Sequence[str]) -> bool:
        return not (set(keys) & self.hidden)


def _engine(rows: list[CanonicalMemory]) -> Any:
    by_id = {m.memory_id: m for m in rows}

    class _Uow:
        memories = SimpleNamespace(
            get_many=lambda tenant, ids: _async([by_id[i] for i in ids if i in by_id])
        )

        async def __aenter__(self) -> Any:
            return self

        async def __aexit__(self, *_: Any) -> None:
            return None

    return SimpleNamespace(uow_factory=_Uow)


async def _async(value: Any) -> Any:
    return value


def _earlier_candidate(memory: CanonicalMemory, newer: str) -> Candidate:
    return _candidate(memory.memory_id, earlier_value=True, replaced_by=newer)


@pytest.mark.parametrize(("old", "new"), [(OLD, NEW), (OLD_HI, NEW_HI)])
async def test_an_earlier_value_brings_what_replaced_it(old: str, new: str) -> None:
    """A search that found only the old statement serves the new one too, before it."""
    was = _memory(old, at=T0, source="msg_a")
    now = _memory(new, at=T1, source="msg_b")
    found = [_candidate("other"), _earlier_candidate(was, now.memory_id)]
    diagnostics: dict[str, Any] = {}
    out = await RetrievalEngine._follow_replacements(
        _engine([was, now]), CTX, found, _Visible(), diagnostics
    )
    assert [c.record_id for c in out] == ["other", now.memory_id, was.memory_id]
    assert out[1].expansion_edge == "REPLACEMENT" and diagnostics["replacements_followed"] == 1


async def test_a_chain_is_followed_to_its_current_end_and_no_further() -> None:
    a = _memory("A", at=T0, source="m_a")
    b = _memory("B", at=T0 + timedelta(1), source="m_b")
    c = _memory("C", at=T1, source="m_c")
    b = b.model_copy(
        update={"system_metadata": {"replaced_by": {"memory_id": c.memory_id, "grade": "EARLIER"}}}
    )
    out = await RetrievalEngine._follow_replacements(
        _engine([a, b, c]), CTX, [_earlier_candidate(a, b.memory_id)], _Visible(), {}
    )
    assert [x.record_id for x in out] == [c.memory_id, a.memory_id, b.memory_id]
    assert [x.payload.get("earlier_value", False) for x in out] == [False, True, True]


async def test_a_replacement_the_reader_may_not_see_is_not_added() -> None:
    was = _memory(OLD, at=T0, source="msg_a")
    now = _memory(NEW, at=T1, source="msg_b")
    now = now.model_copy(
        update={"system_metadata": {**now.system_metadata, "visibility_keys": ["user:acme/u2"]}}
    )
    found = [_earlier_candidate(was, now.memory_id)]
    out = await RetrievalEngine._follow_replacements(
        _engine([was, now]), CTX, found, _Visible(frozenset({"user:acme/u2"})), {}
    )
    assert [c.record_id for c in out] == [was.memory_id]


def _said_in(item: ContextItem, message: str) -> ContextItem:
    return item.model_copy(
        update={
            "evidence": [
                EvidenceRef(
                    source_type=EvidenceSource.MESSAGE,
                    source_id=message,
                    message_id=message,
                    observed_at=T0,
                )
            ]
        }
    )


def test_a_sentence_shown_whole_in_its_turn_is_not_shown_again() -> None:
    turn = _said_in(_item("turn", "Owen Pike has left. Rhea Sood leads outages now."), "msg_1")
    reading = _said_in(_item("fact", "Owen Pike has left."), "msg_1")
    elsewhere = _said_in(_item("other", "Owen Pike has left."), "msg_2")
    bundle = _bundle(memories=[turn, reading, elsewhere])
    assert bundle.redundant() == {"fact"}, "the same words from another message stay"
    assert bundle.render().count("Owen Pike has left.") == 2
    # the prompt form's ids are the memories it shows: a verdict cites what the model saw
    assert prompt_view(bundle)["memory_ids"] == ["turn", "other"]


def test_a_mention_the_history_names_is_not_a_current_fact() -> None:
    mention = ContextItem(
        item_id="f",
        representation=Representation.RELATION,
        text="user mentions Leeds",
        citation="fact:f",
        attributes={"predicate": "mentions", "object": "Leeds"},
    )
    bundle = _bundle(
        memories=[_item("cur", BRISTOL)],
        history=[_item("was", LEEDS, earlier=True)],
        graph_facts=[mention],
    )
    assert "f" in bundle.redundant()
    assert "mentions Leeds" not in bundle.render()


@pytest.mark.parametrize(
    ("sentence", "kind"),
    [
        ("आज से गोदाम 3 की नई किराया दर ₹40,000 है।", StatementKind.FACT),  # "from today": a date
        ("From today the new rent for depot 3 is $500.", StatementKind.FACT),
        ("अब से गोदाम 3 सुबह 6 बजे खुलता है।", StatementKind.RULE),  # "from now on": standing
        ("From now on depot 3 opens at 6.", StatementKind.RULE),
    ],
)
def test_from_today_dates_a_fact_and_from_now_on_makes_a_rule(
    sentence: str, kind: StatementKind
) -> None:
    """The same in every language: "from today" says when a value took effect, "from now
    on" makes an arrangement standing."""
    label = Lexicon.default().label(sentence)
    assert label.kind is kind and label.decided


async def test_a_misread_rule_stored_as_a_preference_is_replaced_too() -> None:
    misread = _rule(OLD, T0, "msg_a").model_copy(update={"memory_type": MemoryType.PREFERENCE})
    newer = _memory(NEW, at=T1, source="msg_b")
    nli = ScriptedNLI({("Corrimax", "Brightway"): 0.97})
    out = await CrossConversationReplacement(nli).run(_uow([misread, newer]), CTX, [newer], now=T1)
    assert [m.memory_id for m in out.changed] == [misread.memory_id]
    plain = _memory("I prefer tables.", at=T0, source="msg_c", memory_type=MemoryType.PREFERENCE)
    assert not statement_of(plain, "user:u1"), "a preference keeps its own slot rules"


async def test_a_replacement_the_search_found_itself_is_kept_with_the_old_value() -> None:
    """Found by the search, but weakly: it travels with the earlier value it replaced, so the
    relevance floor never leaves the old value alone."""
    was = _memory(OLD, at=T0, source="msg_a")
    now = _memory(NEW, at=T1, source="msg_b")
    found = [_candidate(now.memory_id), _earlier_candidate(was, now.memory_id)]
    out = await RetrievalEngine._follow_replacements(
        _engine([was, now]), CTX, found, _Visible(), {}
    )
    assert [c.record_id for c in out] == [now.memory_id, was.memory_id]
    assert out[0].expanded_from == was.memory_id and out[0].expansion_edge == "REPLACEMENT"


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (
            "I'm reading Dune at the moment, about a third of the way through.",
            "I finished Dune last night.",
        ),
        (
            "Ich lese gerade Dune – ungefähr ein Drittel habe ich durch.",
            "Ich habe Dune gestern Abend zu Ende gelesen.",
        ),
        ("Ahora mismo estoy leyendo Dune, voy por un tercio.", "Anoche terminé Dune."),
        ("أقرأ رواية Dune حاليًا — أنهيت نحو ثلثها.", "أنهيت Dune الليلة الماضية."),
        ("मैं इन दिनों Dune पढ़ रहा हूँ — लगभग एक तिहाई हो गई है।", "कल रात मैंने Dune ख़त्म की।"),
        (
            "Maria Lopez من قسم الحسابات الدائنة في إجازة والدية حتى إشعار آخر.",
            "عادت Maria Lopez من الإجازة الوالدية منذ الاثنين.",
        ),
    ],
)
async def test_a_finished_state_after_an_ongoing_one_marks_an_earlier_value(
    old: str, new: str
) -> None:
    """The head reads these at 0.02-0.3 in German, Arabic and Hindi; the words say it."""
    assert finished_after_ongoing(new, old)
    old_m = _memory(old, at=T0, source="msg_a")
    new_m = _memory(new, at=T1, source="msg_b")
    uow = _uow([old_m, new_m])
    out = await CrossConversationReplacement(ScriptedNLI({})).run(uow, CTX, [new_m], now=T1)
    assert [m.memory_id for m in out.changed] == [old_m.memory_id]
    stored = uow.memories.rows[old_m.memory_id]
    assert earlier_value(stored.system_metadata), "an earlier value, never superseded"
    assert stored.temporal.status is TemporalStatus.CURRENT


@pytest.mark.parametrize(
    ("old", "new"),
    [
        # nothing in common: the finished thing is another one
        ("I'm reading Dune at the moment.", "I finished Neuromancer last night."),
        # the older statement says nothing was going on
        ("Dune is my favourite novel.", "I finished Dune last night."),
        # the newer one says nothing finished
        ("I'm reading Dune at the moment.", "Dune has a great ending, I hear."),
        # two things of one kind
        ("Forklift #3 is still out of service.", "Forklift #4 is repaired."),
    ],
)
async def test_a_finished_state_needs_an_ongoing_one_about_the_same_thing(
    old: str, new: str
) -> None:
    old_m = _memory(old, at=T0, source="msg_a")
    new_m = _memory(new, at=T1, source="msg_b")
    out = await CrossConversationReplacement(ScriptedNLI({})).run(
        _uow([old_m, new_m]), CTX, [new_m], now=T1
    )
    assert out.changed == []


async def test_a_later_statement_stored_first_replaces_the_older_one_stored_after_it() -> None:
    """With a model, messages are read in parallel: the later one can be stored first. Event
    time decides which statement is the newer, never the order of storing."""
    newer = _memory("Ana Petrova is back from sabbatical as of Monday.", at=T1, source="msg_b")
    older = _memory("Ana Petrova is on sabbatical until further notice.", at=T0, source="msg_a")
    uow = _uow([newer, older])
    out = await CrossConversationReplacement(ScriptedNLI({})).run(uow, CTX, [older], now=T1)
    assert [m.memory_id for m in out.changed] == [older.memory_id]
    stored = uow.memories.rows[older.memory_id]
    assert earlier_value(stored.system_metadata)
    assert replaced_by(stored.system_metadata) == newer.memory_id
    assert stored.temporal.status is TemporalStatus.CURRENT


async def test_a_later_statement_about_something_else_leaves_the_older_one() -> None:
    newer = _memory("I finished Neuromancer last night.", at=T1, source="msg_b")
    older = _memory("I'm reading Dune at the moment.", at=T0, source="msg_a")
    out = await CrossConversationReplacement(ScriptedNLI({})).run(
        _uow([newer, older]), CTX, [older], now=T1
    )
    assert out.changed == []


@pytest.mark.parametrize(
    ("new", "old", "same"),
    [
        # links of a chain: what they share is all they have in common
        (
            "The dispatcher for zone ZN-V59 is Esme Iwasaki.",
            "Site SITE-7786 is in zone ZN-V59.",
            False,
        ),
        (
            "El despachador de la zona ZN-V59 es Esme Iwasaki.",
            "El sitio SITE-7786 está en la zona ZN-V59.",
            False,
        ),
        (
            "Der Einkäufer für Region K41 ist Lena Ortiz.",
            "Region K41 wird aus DC-Harbor-T318 beliefert.",
            False,
        ),
        ("يتم تزويد المنطقة K41 من DC-Harbor-T318.", "المتجر 5128 يتبع المنطقة K41.", False),  # noqa: RUF001
        ("रीजन K41 को DC-Harbor-T318 से माल मिलता है।", "स्टोर 5128 रीजन K41 में आता है।", False),
        # one thing said again with another value, or a change of it
        ("Site SITE-7786 is in zone ZN-V59.", "Site SITE-7786 is in zone ZN-U97.", True),
        (
            "El sitio SITE-7786 está en la zona ZN-V59.",
            "El sitio SITE-7786 está en la zona ZN-U97.",
            True,
        ),
        ("Filiale 5128 gehört zur Region K41.", "Filiale 5128 gehört zur Region Z12.", True),
        ("المتجر 5128 يتبع المنطقة K41.", "المتجر 5128 يتبع المنطقة Z12.", True),  # noqa: RUF001
        ("स्टोर 5128 रीजन K41 में आता है।", "स्टोर 5128 रीजन Z12 में आता है।", True),
        (
            "Update: Amara Uchenna has replaced Thea Cervantes as the buyer for Region L60.",
            "The buyer for Region L60 is Thea Cervantes.",
            True,
        ),
    ],
)
def test_two_statements_say_the_same_of_what_they_share(new, old, same):
    from memory_service.modules.memory.replacement import says_the_same_of_it

    assert says_the_same_of_it(new, old) is same


@pytest.mark.parametrize(
    "old",
    ["Store 6474 belongs to Region G10.", "The buyer for Region G10 is Catalina Mbeki."],
)
def test_a_label_before_a_code_ties_no_two_statements(old):
    """ "Region" labels L60 and G10, it names nothing both are about: an update about Region
    L60 is apart from statements about Region G10."""
    from memory_service.modules.memory.replacement import anchors, relate

    new = "Update: Amara Uchenna has replaced Thea Cervantes as the buyer for Region L60."
    assert relate(anchors(new, "en"), anchors(old, "en"), new, old).apart


def test_the_same_labelled_code_still_ties_them():
    from memory_service.modules.memory.replacement import anchors, relate

    new = "Update: Amara Uchenna has replaced Thea Cervantes as the buyer for Region L60."
    old = "The buyer for Region L60 is Thea Cervantes."
    relation = relate(anchors(new, "en"), anchors(old, "en"), new, old)
    assert relation.anchored and not relation.apart


@pytest.mark.parametrize(
    ("old", "new", "lang"),
    [
        (
            "Users log IT tickets in Zendesk.",
            "We moved the help desk from Zendesk to Freshservice last month.",
            "en",
        ),
        (
            "يسجّل المستخدمون تذاكر تقنية المعلومات في Zendesk.",
            "نقلنا مكتب الدعم من Zendesk إلى Freshservice الشهر الماضي.",
            "ar",
        ),
    ],
)
def test_a_statement_that_names_nothing_of_its_own_is_no_link_of_a_chain(old, new, lang):
    """The older names nothing the newer does not: the newer says what became of it, so the
    pair goes to the head, it is not the next link of a chain."""
    from memory_service.modules.memory.replacement import _links

    facts = {"category": "fact", "statement_kind": "FACT"}
    older = _memory(old, at=T0, source="msg_a", lang=lang, system_metadata=facts)
    newer = _memory(new, at=T1, source="msg_b", lang=lang, system_metadata=facts)
    assert not _links(newer, older)


@pytest.mark.parametrize(
    ("old", "new", "lang"),
    [
        (
            "Site SITE-7786 is in zone ZN-V59.",
            "The dispatcher for zone ZN-V59 is Esme Iwasaki.",
            "en",
        ),
        ("المتجر 5128 يتبع المنطقة K41.", "يتم تزويد المنطقة K41 من DC-Harbor-T318.", "ar"),  # noqa: RUF001
    ],
)
def test_an_older_link_that_names_a_thing_of_its_own_stays_a_link(old, new, lang):
    from memory_service.modules.memory.replacement import _links

    facts = {"category": "fact", "statement_kind": "FACT"}
    older = _memory(old, at=T0, source="msg_a", lang=lang, system_metadata=facts)
    newer = _memory(new, at=T1, source="msg_b", lang=lang, system_metadata=facts)
    assert _links(newer, older)
