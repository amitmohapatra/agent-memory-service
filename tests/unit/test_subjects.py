"""Same-subject matching (``domain.subjects``, ``modules.memory.subjects``): normalisation,
identifier blocks, the vocabulary packs, learned abbreviations, and the write path's use of
them in consolidation."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from importlib import resources

import pytest

from memory_service.config.constants import MemoryIntelligenceSettings
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import DedupDecision, Lifetime, MemoryType, ObservationKind
from memory_service.domain.evidence import EvidenceRef, EvidenceSource
from memory_service.domain.glossary import RETAIL
from memory_service.domain.ids import content_hash
from memory_service.domain.observation import Observation
from memory_service.domain.subjects import (
    PACK_NAMES,
    SubjectVerdict,
    abbreviates,
    compare,
    defined_aliases,
    pack_aliases,
    parse,
    spellings,
    vocabulary,
)
from memory_service.modules.memory.native import NativeMemoryIntelligence
from memory_service.modules.memory.pipeline import build_memory
from memory_service.modules.memory.subjects import SubjectMatcher
from memory_service.ports.intelligence import MemoryCandidate
from memory_service.ports.models import ProviderInfo
from tests.support_llm import mocked_gateway

pytestmark = pytest.mark.unit

SAME, POSSIBLE, DIFFERENT = SubjectVerdict.SAME, SubjectVerdict.POSSIBLE, SubjectVerdict.DIFFERENT
CTX = MemoryExecutionContext(tenant_id="acme", user_id="u1", workspace_id="ws1", thread_id="thr_1")


def verdict(a: str, b: str, **kw) -> SubjectVerdict:
    return compare(parse(a), parse(b), **kw).verdict


# --------------------------------------------------------------------------- normalisation


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("Forklift 4", "forklift #4"),
        ("Forklift No. 4", "FORKLIFT-4"),
        ("forklift4", "Forklift #4"),
        ("SKU-1001", "sku 1001"),
        ("Bin A-12", "bin A12"),
        ("Gabelstapler Nr. 4", "Gabelstapler #4"),
        ("montacargas núm. 4", "Montacargas 4"),
        ("المستودع رقم ٣", "مستودع 3"),  # Arabic-Indic digit, article, number marker
        ("गोदाम नंबर ३", "गोदाम 3"),  # Devanagari digit and marker
        ("the Berlin office", "Berlin office"),
        ("Forklift 4's battery", "battery of forklift 4"),
        ("Dr. Priya Sharma", "Priya Sharma"),
        ("Acme Logistics GmbH", "ACME logistics"),
        ("U.S. stores", "US stores"),
        ("Wal-Mart", "Walmart"),
        ("on-boarding checklist", "onboarding checklist"),
        ("Berlin store", "Berlin stores"),
        ("شركة النور", "شركه النور"),  # ta marbuta written as heh
    ],
)
def test_spelling_that_does_not_change_the_subject_is_the_same_subject(a, b) -> None:
    assert verdict(a, b) is SAME


def test_a_parse_keeps_identifiers_apart_from_words() -> None:
    s = parse("Forklift No. 4's battery")
    assert s.words == ("forklift", "battery") and s.ids == {"4"}
    assert parse("Q3 2026 forecast").ids == {"q3", "2026"}
    assert parse("5kg rice").ids == parse("5 kilogram rice").ids == {"5kg"}
    assert parse("$5 coupon").ids == {"5usd"}
    assert parse("October 3 delivery").ids == {"m10", "3"}
    assert parse("user:u1").identity == "user:u1"


# --------------------------------------------------------------------------- hard blocks


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("Forklift #3", "Forklift #4"),
        ("Warehouse 3", "Warehouse 13"),
        ("SKU-1001", "SKU-1010"),
        ("PO 4471", "PO 4417"),
        ("Bin A12", "Bin B12"),
        ("Aisle 3", "Aisle 3B"),
        ("Q3 2026 forecast", "Q4 2026 forecast"),
        ("2026-10-01 delivery", "2026-10-02 delivery"),
        ("October 3 delivery", "October 4 delivery"),
        ("Lieferung vom 3. Oktober", "Lieferung vom 4. Oktober"),
        ("5 kg rice bag", "5 lb rice bag"),
        ("$5 coupon", "€5 coupon"),
        ("12-pack cola", "24-pack cola"),
        ("Store 12", "Store 012"),  # a leading zero is not dropped: it may be significant
        ("المستودع 3", "المستودع 13"),
        ("गोदाम 3", "गोदाम 13"),
        ("Acme Inc", "Acme Ltd"),
    ],
)
def test_differing_identifiers_units_dates_or_legal_forms_block_a_merge(a, b) -> None:
    assert verdict(a, b) is DIFFERENT
    # nothing lifts a block: not a vector, not a known name
    assert verdict(a, b, cosine=0.99) is DIFFERENT


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("John Smith", "John Miller"),
        ("Acme Logistics", "Acme Foods"),
        ("Global Freight Solutions", "Global Freight Services"),
        ("North Warehouse", "South Warehouse"),
        ("Jon Smith", "Joan Smith"),  # short names are never typos of each other
        ("Payments API", "Payments UI"),
    ],
)
def test_names_that_share_words_are_different(a, b) -> None:
    assert verdict(a, b) is DIFFERENT


def test_uncertain_pairs_are_possible_never_same() -> None:
    assert verdict("Jonathan Smith", "Jonathon Smith") is POSSIBLE  # one edit
    assert verdict("Tom Baker", "Tom Barker") is POSSIBLE  # ...which is why it is not SAME
    assert verdict("Global Freight Solutions", "GFS") is POSSIBLE  # initialism
    assert verdict("Acme", "Acme Logistics") is POSSIBLE  # short form
    assert verdict("Forklift", "Forklift 4") is POSSIBLE  # one side names no identifier
    assert verdict("Forklift 4", "Forklift 4 battery") is POSSIBLE  # part of it
    assert verdict("Forklift 4", "Gabelstapler 4") is DIFFERENT
    assert verdict("Forklift 4", "Gabelstapler 4", cosine=0.95) is POSSIBLE  # the encoder


def test_a_short_form_two_known_names_extend_is_no_evidence() -> None:
    names = [parse("John Smith"), parse("John Miller")]
    assert compare(parse("John"), parse("John Smith")).verdict is POSSIBLE
    assert compare(parse("John"), parse("John Smith"), names=names).verdict is DIFFERENT


# --------------------------------------------------------------------------- packs


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("hazmat storage", "hazardous materials storage"),
        ("OOS items", "out-of-stock items"),
        ("PO4471", "Purchase Order #4471"),
        ("DC 3", "Distribution Centre 3"),
        ("3PL partner", "third-party logistics partner"),
        ("RTV pallets", "return to supplier pallets"),  # an alias inside a canonical phrase
        ("Acme vendor", "Acme supplier"),
        ("Dept 12", "department 12"),
        ("Abt. 12", "Abteilung 12"),
    ],
)
def test_pack_aliases_name_the_same_subject(a, b) -> None:
    assert verdict(a, b) is SAME


def test_an_acronym_that_is_a_word_counts_only_in_capitals_or_before_a_number() -> None:
    assert parse("OH stock").words == ("hand", "stock")  # "on hand"; "on" is a connective
    assert parse("oh stock").words == ("oh", "stock")
    assert parse("po 4471").words == ("purchase", "order")  # before a number
    assert parse("Oct 3").ids == {"m10", "3"} and "oct" in parse("oct team").words


def test_the_packs_are_data_and_the_glossary_reads_the_same_table() -> None:
    for name in PACK_NAMES:
        raw = resources.files("memory_service.domain.vocabulary").joinpath(f"{name}.json")
        pack = json.loads(raw.read_text(encoding="utf-8"))
        assert pack["name"] == name and pack["description"]
    forms: dict[str, str] = {}
    for name in PACK_NAMES:
        for canonical, aliases in pack_aliases(name).items():
            assert canonical and all(aliases), canonical
            for form in aliases:
                assert forms.setdefault(form.casefold(), canonical) == canonical, form
    # one table, two readers: the retail glossary's acronyms are the pack's
    retail = pack_aliases("retail")
    assert RETAIL["OOS"] == "out of stock" and RETAIL["WOS"] == "weeks of supply"
    assert all(RETAIL[f] == c for c, fs in retail.items() for f in fs if f.isupper())


# --------------------------------------------------------------------------- learned


def test_abbreviations_defined_in_text_are_learned() -> None:
    text = (
        "Store hazardous materials (hazmat) in cage 2. OOS (out of stock) items go to the "
        "back. WOS stands for weeks of supply. Das Zentrallager (ZL) liefert täglich. "
        "We met in Berlin (Germany) on Monday (holiday)."
    )
    assert defined_aliases(text) == [
        ("hazmat", "hazardous materials"),
        ("ZL", "Zentrallager"),
        ("OOS", "out of stock"),
        ("WOS", "weeks of supply"),
    ]
    assert abbreviates("SKU", ["stock", "keeping", "unit"])
    assert abbreviates("TPL", ["third-party", "logistics"])
    assert not abbreviates("Germany", ["Berlin"])


def test_a_learned_abbreviation_makes_two_spellings_one_subject() -> None:
    plain = vocabulary()
    learned = plain.with_aliases(
        defined_aliases("Open a ticket with the Store Support Center (SSC).")
    )
    assert (
        compare(parse("SSC tickets", plain), parse("store support center tickets", plain)).verdict
        is DIFFERENT
    )
    assert (
        compare(
            parse("SSC tickets", learned), parse("store support center tickets", learned)
        ).verdict
        is SAME
    )
    assert plain.with_aliases([]) is plain and learned.learned == (("SSC", "Store Support Center"),)


# --------------------------------------------------------------------------- spellings


def test_spellings_cover_identifier_joins_and_aliases() -> None:
    out = spellings("SKU-1001 hazmat")
    assert out[0] == "sku-1001 hazmat"
    assert {"sku 1001 hazmat", "sku1001 hazmat", "sku-1001 hazardous material"} <= set(out)
    assert len(out) <= 12
    assert spellings("user:u1") == ["user:u1"] and spellings("") == []


# --------------------------------------------------------------------------- the matcher


def _cand(content: str, subject: str | None, predicate: str, obj: str | None = None):
    return MemoryCandidate(
        content=content,
        memory_type=MemoryType.SEMANTIC,
        lifetime=Lifetime.LONG_TERM,
        subject=subject,
        predicate=predicate,
        object=obj,
        evidence=[
            EvidenceRef(
                source_type=EvidenceSource.MESSAGE, source_id="m1", observed_at=datetime.now(UTC)
            )
        ],
    )


def _stored(content: str, subject: str | None, predicate: str, obj: str | None = None):
    return build_memory(_cand(content, subject, predicate, obj), CTX, now=datetime.now(UTC))


def test_an_identity_is_the_same_subject_only_in_the_same_slot_or_topic() -> None:
    matcher = SubjectMatcher()
    city = _stored("I live in Berlin.", "user:u1", "lives_in", "berlin")
    tabs = _stored("I prefer tabs over spaces.", "user:u1", "prefers", "tabs over spaces")
    tea = _stored("I like tea in the morning.", "user:u1", "prefers", "tea in the morning")
    other = _stored("I live in Paris.", "user:u2", "lives_in", "paris")
    pairs = matcher.pairs(
        _cand("I moved to Munich.", "user:u1", "lives_in", "munich"), [city, other]
    )
    assert pairs[city.memory_id].statement.verdict is SAME
    assert pairs[other.memory_id].subject.verdict is DIFFERENT
    pairs = matcher.pairs(
        _cand("I prefer spaces instead of tabs.", "user:u1", "prefers", "spaces instead of tabs"),
        [tabs, tea],
    )
    assert pairs[tabs.memory_id].subject.verdict is SAME  # the same user...
    assert pairs[tabs.memory_id].statement.verdict is POSSIBLE  # ...and topic
    assert pairs[tea.memory_id].statement.verdict is DIFFERENT  # ...but not this topic


def test_the_memories_at_hand_teach_vocabulary_and_names() -> None:
    matcher = SubjectMatcher()
    defining = _stored("Inbound goes to the cross-dock facility (CDF) first.", "user:u1", "said")
    door = _stored("Cross-dock facility door 3 is blocked.", "cross-dock facility door 3", "is")
    smith = _stored("John Smith is the store manager.", "john smith", "is")
    miller = _stored("John Miller is the night lead.", "john miller", "is")
    pairs = matcher.pairs(
        _cand("CDF door 3 is open.", "cdf door 3", "is"), [defining, door, smith, miller]
    )
    assert pairs[door.memory_id].subject.verdict is SAME
    pairs = matcher.pairs(_cand("John is on leave.", "john", "is"), [smith, miller])
    assert {p.subject.verdict for p in pairs.values()} == {DIFFERENT}


class _Encoder:
    """Two-dimensional vectors: 'forklift'-like subjects point one way, the rest another."""

    info = ProviderInfo(name="fake", license="-", origin="test", locality="local")
    dimension = 2
    relevance_floor = 0.0

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed_documents(self, texts):
        self.calls.append(list(texts))
        return [
            [1.0, 0.0] if ("forklift" in t.lower() or "stapler" in t.lower()) else [0.0, 1.0]
            for t in texts
        ]

    async def embed_query(self, text):
        return (await self.embed_documents([text]))[0]

    def fingerprint(self) -> str:
        return "fake-encoder"


async def test_the_encoder_only_reaches_pairs_the_words_left_open_and_caches() -> None:
    encoder = _Encoder()
    matcher = SubjectMatcher(encoder)
    german = _stored("Gabelstapler 4 ist in Gang 3.", "gabelstapler 4", "is")
    other = _stored("Forklift 3 is in aisle 3.", "forklift 3", "is")
    cand = _cand("Forklift 4 is in aisle 5.", "forklift 4", "is")
    pairs = matcher.pairs(cand, [german, other])
    assert pairs[german.memory_id].subject.verdict is DIFFERENT
    refined = await matcher.with_vectors(cand, [german, other], pairs)
    assert refined[german.memory_id].subject.verdict is POSSIBLE
    assert refined[other.memory_id].subject.verdict is DIFFERENT  # blocked, never re-scored
    assert encoder.calls == [["forklift 4", "gabelstapler 4"]]
    await matcher.with_vectors(cand, [german, other], pairs)
    assert len(encoder.calls) == 1  # both subjects were cached


async def test_the_hash_stand_in_is_never_asked() -> None:
    from memory_service.adapters.models.embeddings import HashEmbedding

    assert SubjectMatcher(HashEmbedding()).embedding is None


# --------------------------------------------------------------------------- consolidation


def _obs(text: str) -> Observation:
    return Observation(
        tenant_id=CTX.tenant_id,
        kind=ObservationKind.MESSAGE,
        content=text,
        content_hash=content_hash(text),
        user_id=CTX.user_id,
        workspace_id=CTX.workspace_id,
        thread_id=CTX.thread_id,
        principal_id=CTX.principal_id,
    )


async def _fact(provider: NativeMemoryIntelligence, text: str) -> MemoryCandidate:
    cands = await provider.extract(_obs(text), CTX)
    return await provider.classify(next(c for c in cands if c.category == "fact"), CTX)


async def _consolidate(existing_text: str, incoming_text: str, assist=None):
    plain = NativeMemoryIntelligence(MemoryIntelligenceSettings())
    existing = [build_memory(await _fact(plain, existing_text), CTX, now=datetime.now(UTC))]
    provider = NativeMemoryIntelligence(MemoryIntelligenceSettings(), assist=assist)
    return await provider.consolidate(await _fact(plain, incoming_text), existing, CTX)


async def test_a_respelled_subject_reinforces_the_same_fact() -> None:
    out = await _consolidate("FORKLIFT-4 uses 48V batteries.", "Forklift 4 uses 48V batteries.")
    assert out.decision is DedupDecision.REINFORCE, out.reason
    out = await _consolidate("PO-4471 is approved.", "Purchase order 4471 is approved.")
    assert out.decision is DedupDecision.REINFORCE, out.reason


async def test_another_identifier_never_merges_however_alike_the_sentences() -> None:
    out = await _consolidate("Forklift 3 uses 48V batteries.", "Forklift 4 uses 48V batteries.")
    assert out.decision is DedupDecision.CREATE
    out = await _consolidate("Warehouse 3 is closed.", "Warehouse 13 is closed.")
    assert out.decision is DedupDecision.CREATE


async def test_the_adjudicator_is_asked_about_the_same_subject_not_about_shared_words() -> None:
    # the same subject in other words (word overlap 0.43): asked now, never asked before
    with mocked_gateway([{"verdict": "update"}]) as gw:
        out = await _consolidate(
            "The billing service runs on Cloud Run.",
            "Billing Service runs on Kubernetes in Frankfurt.",
            assist=gw.assist(uses=["conflict_adjudication"]),
        )
    assert gw.route.call_count == 1 and out.decision is DedupDecision.SUPERSEDE
    assert out.reason.startswith("model: update")
    # another subject with most words in common (overlap 0.67): never asked now
    with mocked_gateway([{"verdict": "update"}]) as gw:
        out = await _consolidate(
            "The billing service runs on Cloud Run.",
            "The shipping service runs on Cloud Run.",
            assist=gw.assist(uses=["conflict_adjudication"]),
        )
    assert gw.route.call_count == 0 and out.decision is DedupDecision.CREATE
