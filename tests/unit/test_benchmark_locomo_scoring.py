"""The LoCoMo harness's own scoring, tested directly.

This harness has now reported three numbers that described itself rather than the service:
0.52% (matching verbatim dialogue turns against memories the pipeline had rewritten), 0.0
abstention (reading ``evidence_status`` off the wrong object), and 4.35% recall (requiring the
whole gold answer as a contiguous substring). Every one of them was believable enough to act
on. The scoring functions are pure, so there is no excuse for not pinning them.
"""

from __future__ import annotations

from benchmark.locomo import (
    ANSWER_OVERLAP_HIT,
    _answer_present,
    _content_tokens,
    _evidence_ids,
    _overlap,
    _stratified,
)


def test_stopwords_do_not_count_towards_overlap() -> None:
    # "The sunday before 25 May 2023" against a bundle that shares only filler words must not
    # score: with stopwords kept, "the"/"before" alone would put this above the threshold.
    gold = _content_tokens("The sunday before 25 May 2023")
    unrelated = _content_tokens("The report was published before the committee met")
    assert _overlap(gold, unrelated) < ANSWER_OVERLAP_HIT


def test_a_differently_phrased_date_still_counts() -> None:
    gold = _content_tokens("7 May 2023")
    bundle = _content_tokens("Caroline attended the support group on May 7, 2023, in the evening")
    assert _overlap(gold, bundle) >= ANSWER_OVERLAP_HIT


def test_overlap_is_zero_when_there_is_nothing_to_find() -> None:
    assert _overlap(set(), _content_tokens("anything")) == 0.0
    assert _overlap(_content_tokens("clarinet violin"), set()) == 0.0


def test_strict_substring_is_a_floor_not_the_metric() -> None:
    """The old rule, pinned so its behaviour stays understood rather than re-adopted."""
    assert _answer_present("she plays the clarinet and violin", "clarinet and violin")
    # the exact failure that produced 4.35%: correct evidence, different wording
    assert not _answer_present("Melanie ran the race last Sunday", "The sunday before 25 May 2023")


def test_short_answers_match_whole_tokens_only() -> None:
    assert _answer_present("the meeting lasted 7 hours", "7")
    assert not _answer_present("the meeting lasted 17 hours", "7")


def test_evidence_ids_are_parsed_from_locomos_string_encoded_list() -> None:
    # LoCoMo stores this as a *string* that looks like a list, not as a list.
    assert _evidence_ids({"evidence": "['D1:3', 'D1:5']"}) == ["D1:3", "D1:5"]
    assert _evidence_ids({"evidence": ["D2:8"]}) == ["D2:8"]
    assert _evidence_ids({}) == []


def test_sample_is_deterministic_and_keeps_every_category() -> None:
    questions = [
        {"category": str(c), "question": f"q{c}-{i}"} for c in (1, 2, 5) for i in range(20)
    ]
    first = _stratified(questions, 9)
    assert [q["question"] for q in first] == [q["question"] for q in _stratified(questions, 9)]
    # an ablation compares two runs; a sample that drifts between them compares nothing
    assert {q["category"] for q in first} == {"1", "2", "5"}


def test_sampling_more_than_exists_returns_everything() -> None:
    questions = [{"category": "1", "question": f"q{i}"} for i in range(5)]
    assert _stratified(questions, 50) == questions
    assert _stratified(questions, 0) == questions
