"""Graph lookup terms and explicitly quoted names survive script/normalization changes."""

# ruff: noqa: RUF001 — literal multilingual fixtures.

import unicodedata

import pytest

from memory_service.modules.graph.service import query_terms
from memory_service.modules.ingestion.context_graph import canonical_entity, extract_entities

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "query,name",
    [
        ("北京办公室在哪里？", "北京"),
        ("北京市的办公室在哪里？", "北京市"),
        ("東京大学の研究室はどこですか？", "東京大学"),
        ("Где находится Москва?", "москва"),
        ("दिल्ली कार्यालय कहाँ है?", "दिल्ली"),
        ("أين مكتب القاهرة؟", "القاهرة"),
        ("Wo liegt München?", "münchen"),
        ("谁负责Acme的供应商？", "acme"),
        ("Acmeの取引先を率いているのは誰ですか？", "acme"),
    ],
)
def test_entity_lookup_has_original_script_candidates(query, name):
    assert name in query_terms(query)


@pytest.mark.parametrize("name", ["München", "Αθήνα", "दिल्ली", "القَاهِرَة", "café"])
def test_entity_identity_is_stable_under_unicode_decomposition(name):
    assert canonical_entity(name) == canonical_entity(unicodedata.normalize("NFD", name))


@pytest.mark.parametrize(
    "text,name",
    [
        ('The office is called "القاهرة".', "القاهرة"),
        ("部署名は「東京研究室」です。", "東京研究室"),
        ("公司名称是“北京科技”。", "北京科技"),
        ("Le laboratoire «Étoile» est ouvert.", "Étoile"),
    ],
)
def test_explicit_quoted_entity_names_do_not_require_latin_letters(text, name):
    assert name in extract_entities(text)


def test_query_term_work_and_output_are_bounded():
    assert len(query_terms("北京研究中心" * 500, max_terms=2)) <= 6
    assert query_terms("北京", max_terms=0) == []
