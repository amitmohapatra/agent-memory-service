# ruff: noqa: RUF001 - literal multilingual fixtures.
"""Multilingual ingestion: the language stored at write, the model reading what the English
rules cannot (typed facts, KG relations, chunk context), and the router never letting an
English cue pattern decide a question in another language.

The model is scripted (``tests.support_llm``): what is under test is what the service asks,
and what it keeps of the answer."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from memory_service.config.constants import MemoryIntelligenceSettings
from memory_service.domain.documents import Chunk, DocumentNode
from memory_service.domain.enums import MemoryType, QueryType, Representation
from memory_service.domain.ids import content_hash
from memory_service.domain.language import detect_language, english_evidence, is_english
from memory_service.domain.observation import Observation
from memory_service.modules.graph.native import NativeGraphEnrichment
from memory_service.modules.ingestion.chunking import situated_candidates
from memory_service.modules.llm.assist import SOURCE_LANGUAGE_RULE
from memory_service.modules.memory.native import NativeMemoryIntelligence
from memory_service.modules.memory.pipeline import build_memory
from memory_service.modules.memory.source_facts import select_facts
from memory_service.modules.retrieval.router import QueryRouter
from tests.support_llm import mocked_gateway
from tests.unit.test_memory_native import CTX, _obs

pytestmark = pytest.mark.unit

NOW = datetime(2026, 9, 30, tzinfo=UTC)

#: One message per language: (message, the fact a model returns for it, slot, value).
FIXTURES = {
    "ja": (
        "私は東京に住んでいます。コーヒーが大好きです。",
        "ユーザーは東京に住んでいる",
        "lives_in",
        "東京",
    ),
    "hi": ("मैं दिल्ली में रहता हूँ। मुझे चाय पसंद है।", "उपयोगकर्ता दिल्ली में रहता है", "lives_in", "दिल्ली"),
    "de": (
        "Ich wohne jetzt in Köln. Ich arbeite bei Siemens.",
        "Der Nutzer wohnt in Köln",
        "lives_in",
        "Köln",
    ),
    "es": (
        "Vivo en Madrid desde el año pasado. Me encanta el café.",
        "El usuario vive en Madrid",
        "lives_in",
        "Madrid",
    ),
    "zh": ("我住在上海。我在阿里巴巴工作。", "用户住在上海", "lives_in", "上海"),
}


@pytest.mark.parametrize(
    ("text", "lang"),
    [
        ("I moved to Berlin last year and I work at Acme.", "en"),
        ("Ich bin letztes Jahr nach Berlin gezogen und arbeite bei Acme.", "de"),
        ("Me mudé a Madrid el año pasado y trabajo en Acme.", "es"),
        ("Je suis allé à Paris et je travaille chez Acme.", "fr"),
        ("Eu moro em São Paulo e trabalho na Acme.", "pt"),
        ("私はAcme Corporationで働いています。", "ja"),
        ("去年我搬到了上海，在Acme工作。", "zh"),
        ("मैं पिछले साल दिल्ली चला गया।", "hi"),
        ("저는 서울에 살아요.", "ko"),
        ("Я живу в Москве.", "ru"),
        ("Я живу в Києві.", "uk"),
        ("أعيش في القاهرة", "ar"),
        ("¿Dónde vive Ana?", "es"),
        ("Wo wohnt Anna?", "de"),
        ("Acme Q3", "en"),
        ("12345 !!", "und"),
    ],
)
def test_language_is_decided_without_a_model(text: str, lang: str) -> None:
    assert detect_language(text) == lang
    assert detect_language(text) == detect_language(text), "deterministic"


def test_every_stored_record_carries_its_language() -> None:
    assert _obs(FIXTURES["ja"][0]).lang == "ja"
    assert _obs("I prefer dark mode.").lang == "en"
    assert _obs(FIXTURES["de"][0]).lang == "de"
    stored = Observation.model_validate({**_obs("Hola").model_dump(), "lang": "es"})
    assert stored.lang == "es", "a stored value is kept, not re-derived"
    chunk = _chunk("Der Umsatz stieg im dritten Quartal um zwölf Prozent.")
    assert chunk.lang == "de"


def _reply(fact: str, slot: str, value: str, *, kind: str = "profile") -> dict:
    return {
        "facts": [
            {"text": fact, "kind": kind, "slot": slot, "value": value, "sentences": [0]},
        ]
    }


@pytest.mark.parametrize("lang", sorted(FIXTURES))
async def test_a_message_the_rules_cannot_read_becomes_typed_facts_in_its_language(
    lang: str,
) -> None:
    message, fact, slot, value = FIXTURES[lang]
    observation = _obs(message)
    assert observation.lang == lang
    with mocked_gateway([_reply(fact, slot, value)]) as gateway:
        native = NativeMemoryIntelligence(
            MemoryIntelligenceSettings(), assist=gateway.assist(uses=["contextual_extraction"])
        )
        candidates = await native.extract(observation, CTX)
    facts = [c for c in candidates if c.category == "source_fact"]
    assert len(facts) == 1, candidates
    (extracted,) = facts
    assert extracted.content == fact and detect_language(extracted.content) in (lang, "en")
    assert extracted.memory_type is MemoryType.USER
    assert (extracted.predicate, extracted.object) == ("lives_in", value)
    assert extracted.subject == "user:u1" and extracted.provider == "llm"
    assert extracted.evidence[0].source_id == observation.message_id
    assert any(c.category == "verbatim_turn" for c in candidates), "the turn is kept too"
    (prompt,) = gateway.prompts()
    assert prompt["model"] == "gemini/gemini-3.8-flash", "contextual_extraction is a fast-tier use"
    system, user = (m["content"] for m in prompt["messages"])
    assert SOURCE_LANGUAGE_RULE in system
    sent = json.loads(user)["sentences"]
    assert sent[0]["index"] == 0 and sent[0]["text"] in message


async def test_an_english_message_stays_on_the_rules_and_narrative_units() -> None:
    with mocked_gateway([{"units": []}]) as gateway:
        native = NativeMemoryIntelligence(
            MemoryIntelligenceSettings(), assist=gateway.assist(uses=["contextual_extraction"])
        )
        candidates = await native.extract(_obs("I live in Berlin."), CTX)
    assert any(c.predicate == "lives_in" for c in candidates)
    assert not any(c.category == "source_fact" for c in candidates)
    assert gateway.route.call_count == 0, "one rule-parsed sentence needs no model"


async def test_without_a_key_a_foreign_message_is_kept_verbatim() -> None:
    native = NativeMemoryIntelligence(MemoryIntelligenceSettings())
    candidates = await native.extract(_obs(FIXTURES["ja"][0]), CTX)
    assert [c.category for c in candidates] == ["verbatim_turn"]


def test_model_output_that_is_translated_ungrounded_or_malformed_is_dropped() -> None:
    sentences = ["Ich wohne jetzt in Köln.", "Ich arbeite bei Siemens."]
    kept = select_facts(
        {
            "facts": [
                # translated into English: dropped
                {
                    "text": "The user lives in Cologne",
                    "kind": "profile",
                    "slot": "lives_in",
                    "value": "Cologne",
                    "sentences": [0],
                },
                # a slot value not in the cited text: kept as a plain fact, no slot
                {
                    "text": "Der Nutzer arbeitet bei Siemens AG",
                    "kind": "fact",
                    "slot": "works_at",
                    "value": "Siemens AG",
                    "sentences": [1],
                },
                # cites nothing real
                {
                    "text": "Der Nutzer mag Hunde",
                    "kind": "preference",
                    "slot": "none",
                    "value": "",
                    "sentences": [7],
                },
                # an unknown kind
                {
                    "text": "Der Nutzer",
                    "kind": "opinion",
                    "slot": "none",
                    "value": "",
                    "sentences": [0],
                },
                "not an object",
            ]
        },
        sentences,
        {0, 1},
    )
    assert [(f.text, f.memory_type, f.predicate) for f in kept] == [
        ("Der Nutzer arbeitet bei Siemens AG", MemoryType.SEMANTIC, None)
    ]
    japanese = select_facts(
        _reply("The user lives in Tokyo", "lives_in", "Tokyo"), ["私は東京に住んでいます。"], {0}
    )
    assert japanese == [], "another script than the source is a translation"
    assert select_facts({"facts": "nope"}, sentences, {0}) == []
    assert english_evidence("the user lives in the city") >= 2


@pytest.mark.parametrize(
    ("query", "lang"),
    [
        ("Wann war das Treffen in 2024 mit Acme?", "de"),  # "in 2024" is an English cue
        ("¿Por qué decidimos cambiar de proveedor?", "es"),
        ("田中さんはどこに住んでいますか？", "ja"),
        ("मेरी पसंदीदा चाय कौन सी है?", "hi"),
        ("上个月我们和谁开会了？", "zh"),
    ],
)
def test_the_router_never_routes_another_language_by_english_cues(query: str, lang: str) -> None:
    routed = QueryRouter().route(query)
    assert routed.lang == lang
    assert routed.query_type is QueryType.GENERAL_SEMANTIC
    assert routed.needs_graph and routed.needs_memories and routed.needs_knowledge
    assert not any(routed.signals.values()), "so query expansion may run when the read allows"


def test_an_identifier_is_exact_in_any_language_and_english_still_routes() -> None:
    assert QueryRouter().route("Was steht in doc_01J8ZK7Q9V3W?").query_type is (
        QueryType.EXACT_IDENTIFIER
    )
    assert QueryRouter().route("When did we ship the exporter?").query_type is QueryType.TEMPORAL


async def _memory(text: str):
    native = NativeMemoryIntelligence(MemoryIntelligenceSettings())
    observation = _obs(text)
    candidate = await native.classify((await native.extract(observation, CTX))[0], CTX)
    memory = build_memory(candidate, CTX, now=NOW)
    memory.system_metadata["visibility_keys"] = ["tenant:acme"]
    return memory


@pytest.mark.parametrize(
    ("text", "subject", "obj"),
    [
        ("田中さんはトヨタで働いています。", "田中", "トヨタ"),
        ("अनीता इन्फोसिस में काम करती है।", "अनीता", "इन्फोसिस"),
        ("Anna arbeitet bei Siemens in München.", "Anna", "Siemens"),
        ("Lucía trabaja en Telefónica.", "Lucía", "Telefónica"),
        ("王伟在华为工作。", "王伟", "华为"),
    ],
)
async def test_the_graph_reads_relations_out_of_text_the_entity_rules_cannot(
    text: str, subject: str, obj: str
) -> None:
    memory = await _memory(text)
    assert not is_english(memory.lang)
    reply = {
        "relations": [
            {"subject": subject, "predicate": "works at", "object": obj, "confidence": 0.9},
            # a translated name is not in the text: never an entity
            {"subject": subject, "predicate": "lives_in", "object": "Tokyo", "confidence": 0.9},
        ]
    }
    with mocked_gateway([reply]) as gateway:
        graph = NativeGraphEnrichment(assist=gateway.assist(uses=["relation_extraction"]))
        entities, relations = await graph.enrich_memory(memory, CTX)
    names = {e.canonical_name for e in entities}
    assert {subject.casefold(), obj.casefold()} <= names and "tokyo" not in names
    typed = [r for r in relations if r.attributes.get("extraction") == "llm"]
    assert [r.predicate for r in typed] == ["works_at"]
    assert gateway.route.call_count == 1
    assert SOURCE_LANGUAGE_RULE in gateway.prompts()[0]["messages"][0]["content"]


def _chunk(text: str, node: str = "nod_1", ordinal: int = 0) -> Chunk:
    return Chunk(
        node_id=node,
        document_id="doc_1",
        document_version_id="dv_1",
        tenant_id="acme",
        ordinal=ordinal,
        text=text,
        text_hash=content_hash(text),
        contextual_text=text,
    )


def test_chunks_not_in_english_are_given_a_situating_context() -> None:
    chunks = [
        _chunk("Revenue rose twelve percent in the third quarter.", "nod_1"),
        _chunk("売上高は第3四半期に12%増加した。", "nod_2"),
        _chunk("Der Umsatz stieg im dritten Quartal um zwölf Prozent.", "nod_3"),
    ]
    nodes = [
        DocumentNode(
            node_id=f"nod_{i}",
            document_id="doc_1",
            document_version_id="dv_1",
            tenant_id="acme",
            representation=Representation.PARAGRAPH,
            ordinal=i,
            depth=1,
            text=chunks[i - 1].text,
            text_hash=chunks[i - 1].text_hash,
        )
        for i in (1, 2, 3)
    ]
    assert situated_candidates(chunks, nodes, max_chunks=8) == [1, 2]
    assert situated_candidates(chunks, nodes, max_chunks=1) == [1]
