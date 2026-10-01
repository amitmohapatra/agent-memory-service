"""Script coverage, canonical Unicode and bounded lexical output without model calls."""

import pytest

from memory_service.adapters.models.sparse import Bm25SparseEncoder, tokenize

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "text",
    [
        "मुझे दिल्ली में रहना पसंद है।",
        "أعيش في القاهرة",
        "Я живу в Москве",
        "我住在北京",
        "私は東京に住んでいます",
        "ฉันอาศัยอยู่ในกรุงเทพ",
        "Αθήνα είναι η πρωτεύουσα",
        "Tôi sống ở Hà Nội",
        "İstanbul Türkiye",
    ],
)
def test_non_latin_text_has_stable_nonempty_sparse_signal(text):
    encoder = Bm25SparseEncoder()
    (document,) = encoder.encode_documents([text])
    query = encoder.encode_query(text)
    assert query.indices and set(query.indices) == set(document.indices)
    assert query == encoder.encode_query(text)


def test_canonical_unicode_and_combining_marks_are_preserved():
    assert tokenize("café") == tokenize("cafe\u0301") == ["café"]
    assert "दिल्ली" in tokenize("दिल्ली में")
    assert "北京" in tokenize("我住在北京")
    assert "東京" in tokenize("東京に住む")


def test_ascii_terms_are_snowball_stemmed_and_other_scripts_are_not_mangled():
    assert tokenize("The servers are running in Berlin.") == ["server", "run", "berlin"]
    assert "κόσμος".casefold() in tokenize("κόσμος")
    assert tokenize(" ") == [] and tokenize("🙂 !!!") == []
    assert tokenize("Привет servers") == ["привет", "server"]


@pytest.mark.parametrize(
    ("asked", "said"),
    [
        ("pets", "pet"),
        ("cars", "car"),
        ("dancing", "dance"),
        ("loved", "love"),
        ("stressed", "stress"),
        ("running", "run"),
        ("hiking", "hike"),
        ("adopted", "adoption"),
    ],
)
def test_a_question_and_its_answer_share_a_term_across_inflections(asked: str, said: str):
    """Every pair here shared no term under the suffix list Snowball replaced."""
    assert tokenize(asked) == tokenize(said)


def test_output_is_linear_and_the_vector_space_version_changes():
    text = "北京" * 5000
    assert len(tokenize(text)) <= 2 * len(text)
    assert len(tokenize("क" + "\u093e" * 10000)) == 1
    assert Bm25SparseEncoder().fingerprint().startswith("bm25-v3-snowball-")
