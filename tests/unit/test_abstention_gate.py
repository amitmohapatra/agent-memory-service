"""The abstention rule, at the level where it is decided.

Found by the degenerate-input benchmark, not by review: an empty query came back COMPLETE
with ten memories attached. Two separate defects combined to cause it, and fixing either one
alone would have made the other worse, so both are pinned here.
"""

from __future__ import annotations

import pytest

from memory_service.modules.context.evidence import content_terms, overlaps


class _C:
    """Enough of a Candidate for ``overlaps``."""

    def __init__(self, text: str, kind: str = "memory") -> None:
        self.text, self.kind = text, kind


CORPUS = [_C("The FOMC held the federal funds rate at 5.25 to 5.50 percent in January 2024")]


@pytest.mark.parametrize(
    "query",
    ["", "   ", "?", "????????", "🙂🙃", "the the the and of", "..."],
    ids=["empty", "spaces", "one-char", "punctuation", "emoji", "stopwords", "dots"],
)
def test_a_query_with_no_content_cannot_be_grounded(query: str) -> None:
    # Previously ``bool(candidates)`` — vacuously true — so every one of these was COMPLETE.
    assert overlaps(query, CORPUS) is False


def test_a_matching_query_still_overlaps() -> None:
    assert overlaps("what did the FOMC do with the rate", CORPUS) is True


def test_an_off_topic_query_does_not_overlap() -> None:
    assert overlaps("what is the population of Ulaanbaatar", CORPUS) is False


def test_non_latin_queries_produce_terms() -> None:
    """The gate is ASCII-blind no more.

    ``_WORD`` used to be ``[a-z][a-z0-9-]+``, so Cyrillic, Greek, Hebrew, Arabic and CJK all
    yielded zero terms — and under the old vacuous-true rule that meant those languages could
    never abstain. Under the new rule it would mean they *always* abstain, which is worse.
    Both halves had to change together.
    """
    assert content_terms("Какова численность населения")
    assert content_terms("Ποιος είναι ο πληθυσμός")  # noqa: RUF001 - non-Latin is the point
    # CJK has no spaces: bigrams are the unit, as in Lucene's CJKAnalyzer
    assert "日本" in content_terms("日本語のテキスト")


def test_a_cjk_query_matches_cjk_evidence() -> None:
    corpus = [_C("東京の人口は約1400万人です")]
    assert overlaps("東京の人口", corpus) is True
    assert overlaps("大阪の天気", corpus) is False


def test_the_query_cap_default_is_generous_enough_for_real_questions() -> None:
    """2048 characters is far past any real question and well past the models' 512 tokens.

    The cap exists to bound a pathological query, not to clip a long one: measured on the
    degenerate-input benchmark, 2,000 characters of noise cost 20 seconds against 1 second
    for an ordinary question, because a cross-encoder pair costs what its longest side costs.
    """
    from memory_service.config.settings import RetrievalSettings

    cfg = RetrievalSettings()
    assert cfg.max_query_chars >= 2048
    long_but_real = "Summarise everything the committee said about inflation. " * 8
    assert len(long_but_real) < cfg.max_query_chars
