"""ADR 0041: a question about now reads the newest value of its slot, and evidence that names
what the question names but not what it asks is a bridge to one more hop."""

from __future__ import annotations

import pytest

from memory_service.modules.retrieval import engine as engine_module
from memory_service.modules.retrieval.engine import Candidate, RetrievalEngine
from memory_service.modules.retrieval.slots import (
    bridge_query,
    latest_values,
    names_any,
    question_slot,
)
from memory_service.ports.search import Retriever, SearchHit, SparseVector

pytestmark = pytest.mark.unit


def _mem(rid: str, text: str, day: str, **extra) -> Candidate:
    return Candidate(
        record_id=rid,
        kind="memory",
        text=text,
        score=1.0,
        retrievers=[],
        payload={"observed_at": f"{day}T09:00:00+00:00", "text": text, **extra},
    )


def _said(c: Candidate) -> str:
    return c.payload["observed_at"]


def test_the_newest_value_of_the_asked_slot_is_current_and_the_rest_are_earlier() -> None:
    held = [
        _mem("a", "Reorder point for MK-220 is 40.", "2026-09-01"),
        _mem("c", "We set the MK-220 reorder point to 75 today.", "2026-10-09"),
        _mem("b", "MK-220 reorder point is now 60.", "2026-09-20"),
        _mem("d", "MK-220 ships in cases of 12.", "2026-10-10"),  # another slot
        _mem("e", "MK-110 reorder point is 30.", "2026-10-10"),  # another thing
    ]
    got = {
        c.record_id: c
        for c in latest_values("What's the reorder point for MK-220?", held, newest_of=_said)
    }
    assert [r for r, c in got.items() if c.payload.get("earlier_value")] == ["a", "b"]
    assert got["a"].payload["replaced_by"] == "c"
    assert not got["d"].payload.get("earlier_value") and not got["e"].payload.get("earlier_value")


@pytest.mark.parametrize(
    "question",
    [
        "What was the reorder point for MK-220 in May?",  # the past: an event, not the state
        "What's the weather like?",  # names nothing
    ],
)
def test_past_questions_and_questions_naming_nothing_are_left_alone(question: str) -> None:
    held = [
        _mem("a", "Reorder point for MK-220 is 40.", "2026-09-01"),
        _mem("b", "MK-220 reorder point is now 60.", "2026-09-20"),
    ]
    assert not any(
        c.payload.get("earlier_value") for c in latest_values(question, held, newest_of=_said)
    )


def test_the_same_value_said_again_is_no_earlier_value() -> None:
    held = [
        _mem("a", "Van #7 is driven by Peter Brandt.", "2026-09-01"),
        _mem("b", "Peter Brandt drives van #7 now.", "2026-10-01"),
    ]
    assert not any(
        c.payload.get("earlier_value")
        for c in latest_values("Who drives van #7?", held, newest_of=_said)
    )


@pytest.mark.parametrize(
    ("question", "evidence", "expected"),
    [
        ("Who is the buyer for store 4779?", "Store 4779 is in Region X29.", "Region X29 buyer"),
        (
            "Wer ist der Einkäufer für Filiale 4779?",
            "Filiale 4779 gehört zur Region X29.",
            "Region X29 einkäufer",
        ),
        # the evidence answers: no hop
        ("Who is the buyer for store 4779?", "The buyer for store 4779 is Ana Ruiz.", None),
        # the evidence is about something else: no hop
        ("Who is the buyer for store 4779?", "Region X29 had a good quarter.", None),
    ],
)
def test_a_bridge_names_the_next_thing_to_ask_about(question, evidence, expected) -> None:
    got = bridge_query(question, evidence)
    assert (got and got.split()[-2:]) == (expected and expected.split()[-2:])


def test_identifiers_match_however_they_are_spelled() -> None:
    assert names_any("The buyer for region X-29 is Priya", frozenset({"x29"}))
    assert not names_any("Region X30", frozenset({"x29"}))
    assert question_slot("Who is the buyer for store 4779?").asked == (frozenset({"buyer"}),)


class _Store:
    def __init__(self, hits: list[SearchHit]) -> None:
        self.hits, self.asked = hits, []

    async def search_sparse(self, collection, vector, flt, *, limit):
        self.asked.append((vector, limit))
        return self.hits[:limit]


class _Sparse:
    def encode_query(self, text: str) -> SparseVector:
        return SparseVector(indices=[1], values=[1.0])


class _Indexer:
    sparse = _Sparse()

    def collection(self, name: str) -> str:
        return name


class _Visibility:
    def search_filter(self, **must):
        from memory_service.ports.search import SearchFilter

        return SearchFilter(tenant_id="t", must=dict(must))


def _hit(rid: str, text: str) -> SearchHit:
    return SearchHit(
        record_id=rid, score=1.0, retriever=Retriever.BM25, payload={"text": text, "kind": "memory"}
    )


@pytest.mark.asyncio
async def test_the_second_hop_adds_what_the_bridged_thing_holds_after_the_bridge() -> None:
    engine = RetrievalEngine.__new__(RetrievalEngine)
    engine.store = _Store(
        [
            _hit("m1", "Store 4779 is in Region X29."),  # in hand already
            _hit("m3", "Region X30's buyer is Lee."),  # not the bridged thing
            _hit("m2", "The buyer for Region X29 is Priya Shah."),
        ]
    )
    engine.indexer = _Indexer()
    routed = type("R", (), {"query": "Who is the buyer for store 4779?"})()
    bridge = _mem("m1", "Store 4779 is in Region X29.", "2026-10-01")
    other = _mem("m9", "Store 4779 opens at 7 am.", "2026-10-02")
    diagnostics: dict = {}
    got = await engine._second_hop(routed, [bridge, other], _Visibility(), diagnostics)
    assert [c.record_id for c in got] == ["m1", "m2", "m9"]
    assert got[1].expansion_edge == "SECOND_HOP" and got[1].expanded_from == "m1"
    assert diagnostics["second_hop"]["added"] == 1
    assert engine.store.asked[0][1] == engine_module.SECOND_HOP_K


@pytest.mark.asyncio
async def test_no_bridge_no_search() -> None:
    engine = RetrievalEngine.__new__(RetrievalEngine)
    engine.store = _Store([])
    engine.indexer = _Indexer()
    routed = type("R", (), {"query": "Who is the buyer for store 4779?"})()
    held = [_mem("m1", "The buyer for store 4779 is Ana Ruiz.", "2026-10-01")]
    assert await engine._second_hop(routed, held, _Visibility(), {}) == held
    assert engine.store.asked == []


@pytest.mark.parametrize(
    ("question", "older", "newer"),
    [
        (
            "Who is my regional merchandise manager?",
            "My regional merchandise manager is Ana Ruiz.",
            "My regional merchandise manager is Lee Chen.",
        ),
        (
            "Wer ist mein regionaler Merchandise-Manager?",
            "Mein regionaler Merchandise-Manager ist Ana Ruiz.",
            "Mein regionaler Merchandise-Manager ist Lee Chen.",
        ),
        (
            "What's the current count for aisle B-42?",
            "Cycle count for aisle B-42: 340 units.",
            "Recount for aisle B-42 came back at 410 units.",
        ),
        (
            "What was the last PO I raised?",
            "I just raised PO-123456 for SKU 654321.",
            "I just raised PO-222222 for SKU 777777.",
        ),
        (
            "Who is the buyer for Region K42?",
            "The buyer for Region K42 is Omar Haddad.",
            "Update: Lena Vogel has replaced Omar Haddad as the buyer for Region K42.",
        ),
    ],
)
def test_a_slot_of_the_users_own_and_a_replacement_naming_the_old_value(question, older, newer):
    held = [_mem("a", older, "2026-10-01"), _mem("b", newer, "2026-10-08")]
    got = latest_values(question, held, newest_of=_said)
    assert [c.record_id for c in got if c.payload.get("earlier_value")] == ["a"]


def test_a_question_about_many_things_keeps_every_current_value() -> None:
    held = [
        _mem("a", "Forklift #2 is out of service at the loading dock.", "2026-10-01"),
        _mem("b", "Forklift #4 is available again at the loading dock.", "2026-10-08"),
    ]
    got = latest_values(
        "Which forklifts are available for the loading dock?", held, newest_of=_said
    )
    assert not any(c.payload.get("earlier_value") for c in got)


# ------------------------------------------------------------------ a list below the floor

LISTS = {
    "en": ("Which are the pilot branches?", "The pilot branches are {}."),
    "de": ("Welche sind die Pilotfilialen?", "Die Pilotfilialen sind {}."),
    "es": ("¿Cuáles son las sucursales piloto?", "Las sucursales piloto son {}."),
    "ar": ("ما هي الفروع التجريبية؟", "الفروع التجريبية هي {}."),  # noqa: RUF001
    "hi": ("पायलट शाखाएँ कौन सी हैं?", "पायलट शाखाएँ {} हैं।"),
}


@pytest.mark.parametrize("size", [3, 6, 10, 20, 40])
@pytest.mark.parametrize("lang", sorted(LISTS))
def test_a_list_that_answers_the_question_is_one(lang, size):
    from memory_service.modules.retrieval import slots

    asked, said = LISTS[lang]
    text = said.format(", ".join(f"BR-{700 + n}" for n in range(size)))
    assert slots.answers_as_list(slots.question_slot(asked), text)


@pytest.mark.parametrize(
    ("asked", "text"),
    [
        ("Which are the pilot branches?", "The pilot branches are BR-700, BR-701."),  # two
        ("Which are the pilot branches?", "The closed branches are BR-700, BR-701, BR-702."),
        ("Which stores stock RT335327?", "Stores stock XN494339 at 1641, 4307 and 4610."),
        ("Which are the pilot branches?", "Pilot week starts on 3, 4 or 5 May."),  # no branches
    ],
)
def test_what_is_no_list_answering_the_question(asked, text):
    from memory_service.modules.retrieval import slots

    assert not slots.answers_as_list(slots.question_slot(asked), text)


# ------------------------------------------------------------------ a rule restated


@pytest.mark.parametrize(
    ("older", "newer"),
    [
        (
            "Always put the work order number in the email subject, prefixed with WO-E3.",
            "Always put the work order number in the email subject, prefixed with WO-K3.",
        ),
        (
            "فضّل دائمًا EJ159063 لمسودات النقص. لا تعتمد أي مسودة نهائيًا أبدًا.",  # noqa: RUF001
            "فضّل دائمًا LE219684 لمسودات النقص. لا تعتمد أي مسودة نهائيًا أبدًا.",  # noqa: RUF001
        ),
        (
            "Pon siempre el número de orden de trabajo en el asunto, con el prefijo WO-E3.",
            "Pon siempre el número de orden de trabajo en el asunto, con el prefijo WO-K3.",
        ),
    ],
)
def test_a_rule_restated_with_new_values_leaves_the_older_an_earlier_value(older, newer):
    from memory_service.modules.retrieval.engine import restated_rules_first

    held = [
        _mem("old", older, "2026-10-10", category="verbatim_turn", standing_instruction="yes"),
        _mem("new", newer, "2026-10-17", category="verbatim_turn", statement_kind="RULE"),
        _mem("fact", "Our hero style this month is SL641114.", "2026-10-09"),
    ]
    out = {c.record_id: bool(c.payload.get("earlier_value")) for c in restated_rules_first(held)}
    assert out == {"old": True, "new": False, "fact": False}


def test_facts_restated_are_left_to_the_slot_rules():
    from memory_service.modules.retrieval.engine import restated_rules_first

    held = [
        _mem("a", "Our hero style this month is XP932108.", "2026-10-10", statement_kind="FACT"),
        _mem("b", "Our hero style this month is SL641114.", "2026-10-17", statement_kind="FACT"),
    ]
    assert not any(c.payload.get("earlier_value") for c in restated_rules_first(held))


PO = "I raised PO-77812 for 2,000 corrugated boxes; it's late."


@pytest.mark.parametrize(
    ("question", "evidence", "words"),
    [
        ("Who should I chase about PO-77812?", PO, {"corrugated", "boxes"}),
        (
            "Wen soll ich wegen PO-77812 anmahnen?",
            "Ich habe PO-77812 über 2.000 Wellpappkartons bestellt; sie ist überfällig.",
            {"wellpappkartons"},
        ),
        (
            "¿A quién reclamo por la PO-77812?",
            "Hice la PO-77812 de 2.000 cajas de cartón ondulado; va con retraso.",
            {"cajas", "cartón", "ondulado"},
        ),
    ],
)
def test_who_to_ask_about_a_record_that_names_only_its_object_hops_on_the_object(
    question, evidence, words
) -> None:
    """A record that names nothing but its number and its object ("PO-77812 for corrugated
    boxes") leads to whoever the object's own record names (capability eval,
    multihop-retail-po-supplier: the supplier was never reached)."""
    got = bridge_query(question, evidence)
    assert got and words <= set(got.lower().split()), got


@pytest.mark.parametrize(
    "question",
    [
        "When is PO-77812 due?",  # asks for no one
        "Who raised PO-77812?",  # the record says what is asked
    ],
)
def test_no_object_hop_unless_someone_is_asked_for_and_not_answered(question) -> None:
    assert bridge_query(question, PO) is None


@pytest.mark.asyncio
async def test_the_object_hop_adds_only_a_record_of_the_object_that_names_someone() -> None:
    engine = RetrievalEngine.__new__(RetrievalEngine)
    engine.store = _Store(
        [
            _hit("m2", "PO-77790 for shrink wrap arrived complete."),  # another thing
            _hit("m4", "Corrugated boxes were raised late again."),  # the object, no one named
            _hit("m3", "Our new supplier for corrugated boxes is PackRight; the rep is Mia Chen."),
        ]
    )
    engine.indexer = _Indexer()
    routed = type("R", (), {"query": "Who should I chase about PO-77812?"})()
    po = _mem("m1", PO, "2026-10-20")
    diagnostics: dict = {}
    got = await engine._second_hop(routed, [po], _Visibility(), diagnostics)
    assert [c.record_id for c in got] == ["m1", "m3"], diagnostics


@pytest.mark.asyncio
async def test_a_record_the_hop_reaches_that_was_already_found_is_kept_as_reached() -> None:
    """The supplier's record was found by the search too, far from the question, and the
    context dropped it under the relevance floor: the hop skipped it as in hand. What the hop
    reaches is judged by the hop, wherever it already was."""
    engine = RetrievalEngine.__new__(RetrievalEngine)
    supplier = "Our new supplier for corrugated boxes is PackRight; the rep is Mia Chen."
    engine.store = _Store([_hit("m3", supplier)])
    engine.indexer = _Indexer()
    routed = type("R", (), {"query": "Who should I chase about PO-77812?"})()
    po = _mem("m1", PO, "2026-10-20")
    other = _mem("m2", "PO-77790 for shrink wrap arrived complete.", "2026-10-20")
    found = _mem("m3", supplier, "2026-09-27")
    got = await engine._second_hop(routed, [po, other, found], _Visibility(), {})
    assert [c.record_id for c in got] == ["m1", "m3", "m2"]
    assert got[1].expanded_from == "m1" and got[1].expansion_edge == "SECOND_HOP"
