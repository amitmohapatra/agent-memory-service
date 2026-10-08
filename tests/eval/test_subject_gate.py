"""Same-subject gate: the subject matcher never merges two different subjects.

Writes ``benchmark/results/subject_gate.json`` (the words-only matcher, as every write path
without the conflict adjudicator runs it). The encoder's share and the timings are
``benchmark/subjects.py``'s, under real weights."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmark.common import RESULTS, provenance
from benchmark.evaluation.subject_pairs import evaluate_subject_pairs, load_subject_pairs

from memory_service.modules.memory.subjects import SubjectMatcher

pytestmark = pytest.mark.eval

PAIRS = Path(__file__).resolve().parent / "golden" / "subject_pairs.json"
#: the measured F1 of the words-only matcher on this set, less a small margin: a change that
#: costs recall below it must say so here
F1_FLOOR = 0.80


async def test_no_subject_false_merges_and_f1_holds() -> None:
    cases = load_subject_pairs(PAIRS)
    assert len(cases) >= 400
    assert {c.lang for c in cases} >= {"en", "de", "es", "ar", "hi"}
    report = await evaluate_subject_pairs(SubjectMatcher(), cases)
    out = {
        "gate": "subjects",
        "golden_set": "subject_pairs",
        "matcher": "words (no encoder)",
        **report,
        "provenance": provenance(),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "subject_gate.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    merges = [e for e in report["errors"] if e["label"] == "different"]
    assert report["false_merges"] == 0, merges
    assert report["hard_negative_false_merges"] == 0, merges
    assert report["f1"] >= F1_FLOOR, report["errors"]
