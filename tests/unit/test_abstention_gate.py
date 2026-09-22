"""The abstention rule, at the level where it is decided.

Found by the degenerate-input benchmark, not by review: an empty query came back COMPLETE
with ten memories attached. Two separate defects combined to cause it, and fixing either one
alone would have made the other worse, so both are pinned here.
"""

from __future__ import annotations

import itertools

import pytest

from memory_service.modules.context.evidence import content_terms, overlaps, unsupported_subject


class _C:
    """Enough of a Candidate for ``overlaps`` and ``unsupported_subject``.

    ``record_id`` included: the rules memoise a record's content terms by it, because one
    bundle is walked several times over and the text does not change between walks.
    """

    _ids = itertools.count()

    def __init__(self, text: str, kind: str = "memory", subject: str | None = None) -> None:
        self.text, self.kind = text, kind
        self.record_id = f"mem_{next(self._ids)}"
        self.payload = {"subject": subject} if subject else {}


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
    from memory_service.config.constants import RetrievalSettings

    cfg = RetrievalSettings()
    assert cfg.max_query_chars >= 2048
    long_but_real = "Summarise everything the committee said about inflation. " * 8
    assert len(long_but_real) < cfg.max_query_chars


# --- the subject rule: a two-person conversation shares terms with any question about
# --- either person, so ``overlaps`` alone can never see a wrong-person premise.

CONVERSATION = [
    _C(
        "[2023-05-20] Caroline: my grandma gave me her necklace, I treasure it",
        subject="user:caroline",
    ),
    _C("[2023-05-21] Melanie: we went camping at the beach with the kids", subject="user:melanie"),
    _C("[2023-05-22] Caroline: I'm single and honestly enjoying it", subject="user:caroline"),
]


def test_a_question_about_the_wrong_person_is_flagged() -> None:
    note = unsupported_subject("What was grandma's gift to Melanie?", CONVERSATION)
    assert note == "no retrieved memory about Melanie mentions gift, grandma"
    # ...and the plain overlap rule is exactly why this is needed: it says yes
    assert overlaps("What was grandma's gift to Melanie?", CONVERSATION)


def test_a_question_the_right_person_answers_is_not_flagged() -> None:
    assert unsupported_subject("Where did Melanie go camping?", CONVERSATION) is None
    assert unsupported_subject("What did Caroline's grandma give her?", CONVERSATION) is None


def test_a_person_nobody_mentioned_is_flagged_by_name() -> None:
    assert unsupported_subject("What does Jonathan do for work?", CONVERSATION) == (
        "no retrieved memory is about Jonathan"
    )


def test_sentence_openers_and_questions_without_names_are_left_alone() -> None:
    assert unsupported_subject("What happened at the beach?", CONVERSATION) is None
    assert unsupported_subject("When was the camping trip?", CONVERSATION) is None


def test_a_name_that_opens_the_question_still_counts_when_the_evidence_knows_it() -> None:
    assert unsupported_subject("Melanie's grandma gave her what?", CONVERSATION) == (
        "no retrieved memory about Melanie mentions gave, grandma"
    )


def test_document_bundles_are_not_subject_checked() -> None:
    docs = [_C("Caroline Herschel catalogued nebulae in 1783", kind="chunk")]
    assert unsupported_subject("What did Melanie catalogue?", docs) is None


def test_a_question_word_is_never_a_name_even_when_the_evidence_capitalises_it() -> None:
    """Verbatim turns put a capitalised "What" into the evidence; that must not make "What
    was grandma's gift?" a question about someone called What."""
    evidence = [
        *CONVERSATION,
        _C("[2023-05-23] Melanie: What a week! Camping was great.", subject="user:melanie"),
    ]
    assert unsupported_subject("What was the camping trip like?", evidence) is None
    assert unsupported_subject("When did Melanie go camping?", evidence) is None
