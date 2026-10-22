"""The memory-vs-frameworks benchmark's own logic: workloads, budgets, intervals, scoring.

No library under test is imported here (each runs in its own interpreter); what is tested is
what decides a number in the report.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmark.memvs import llm, score
from benchmark.memvs.meter import _json_chunks
from benchmark.memvs.model import (
    DatasetName,
    Hit,
    Query,
    Role,
    Session,
    Turn,
    Unit,
    Workload,
    turn_session,
)
from benchmark.memvs.systems.base import Attribution, render, spoken, turn_text

pytestmark = pytest.mark.unit

LME = Path("benchmark/data/longmemeval/longmemeval_s_cleaned.json")
LOCOMO = Path("benchmark/data/locomo10.json")


def _unit(queries: tuple[Query, ...] = ()) -> Unit:
    turns = (
        Turn("s1:0", Role.USER, "My locker code is 4821."),
        Turn("s1:1", Role.ASSISTANT, "Noted."),
    )
    return Unit("u1", "t1", "user-1", (Session("s1", "2026-01-02T09:00:00+00:00", turns),), queries)


class TestModel:
    def test_workload_round_trips_through_json(self, tmp_path: Path) -> None:
        q = Query(
            "q1", "What is my locker code?", evidence_turns=("s1:0",), evidence_sessions=("s1",)
        )
        w = Workload(DatasetName.ISOLATION, (_unit((q,)),), {"k": 1})
        w.dump(tmp_path / "w.json")
        assert Workload.load(tmp_path / "w.json") == w

    def test_turn_session_is_the_prefix_before_the_last_colon(self) -> None:
        assert turn_session("D12:3") == "D12"
        assert turn_session("answer_4be1b6b4_2:7") == "answer_4be1b6b4_2"

    @pytest.mark.parametrize(
        ("speaker", "expected"),
        [
            (None, "hola, me llamo Ana"),
            ("event", "hola, me llamo Ana"),
            ("Caroline", "Caroline: hola, me llamo Ana"),
        ],
    )
    def test_spoken_keeps_a_named_speaker_in_the_text(
        self, speaker: str | None, expected: str
    ) -> None:
        assert spoken(Turn("s:0", Role.USER, "hola, me llamo Ana", speaker)) == expected

    def test_turn_text_names_the_role_when_no_speaker(self) -> None:
        assert (
            turn_text(Turn("s:0", Role.ASSISTANT, "मैं मदद कर सकता हूँ"))
            == "assistant: मैं मदद कर सकता हूँ"
        )


class TestBudget:
    def test_lines_stop_at_the_budget_but_the_first_always_fits(self) -> None:
        long = "x" * (llm.MEMORY_TOKEN_BUDGET * llm.CHARS_PER_TOKEN * 2)
        assert llm.within_budget([long, "second"]) == long

    def test_lines_in_order_within_budget(self) -> None:
        lines = [f"line {i} " + "y" * 396 for i in range(40)]
        kept = llm.within_budget(lines, 1000).split("\n")
        assert kept == lines[: len(kept)]
        assert 9 <= len(kept) <= 10

    def test_render_dates_each_hit(self) -> None:
        hits = [Hit("Caroline went camping", at="2023-05-08"), Hit("no date")]
        assert render(hits) == "- [2023-05-08] Caroline went camping\n- no date"

    def test_answer_prompt_says_nothing_on_file_for_an_empty_memory(self) -> None:
        system = llm.answer_messages("", "Wer ist mein Lieferant?", None)[0]["content"]
        assert llm.NOTHING_ON_FILE in system and "Current date: unknown" in system


class TestJudge:
    @pytest.mark.parametrize(
        ("category", "marker"),
        [
            ("temporal-reasoning", "off-by-one"),
            ("knowledge-update", "updated answer"),
            ("single-session-preference", "Rubric"),
            ("single_hop", "Correct Answer"),
        ],
    )
    def test_template_by_question_type(
        self, category: str, marker: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sent: list[str] = []

        def complete(
            self: llm.Gateway, messages: list[dict[str, str]], max_tokens: int
        ) -> llm.Completion:
            sent.append(messages[0]["content"])
            return llm.Completion("Yes.", 10, 1, 1.0)

        monkeypatch.setattr(llm.Gateway, "complete", complete)
        assert llm.judge(llm.Gateway("u", "k", "m"), category, "q", "gold", "resp") is True
        assert marker in sent[0]

    def test_a_no_is_wrong(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            llm.Gateway, "complete", lambda *a, **k: llm.Completion(" no", 1, 1, 1.0)
        )
        assert llm.judge(llm.Gateway("u", "k", "m"), "multi-session", "q", "g", "r") is False

    def test_gateway_from_env_names_what_is_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in (llm.ENV_BASE_URL, llm.ENV_KEY, llm.ENV_MODEL):
            monkeypatch.delenv(name, raising=False)
        with pytest.raises(llm.LLMError, match=llm.ENV_BASE_URL):
            llm.Gateway.from_env()


class TestIntervals:
    def test_constant_values_have_a_point_interval(self) -> None:
        est = score.bootstrap([1.0] * 20)
        assert est is not None and (est.mean, est.low, est.high) == (1.0, 1.0, 1.0)

    def test_empty_values_have_no_estimate(self) -> None:
        assert score.bootstrap([]) is None
        assert score.paired({}, {"a": 1.0}) is None

    def test_a_clear_paired_gain_excludes_zero(self) -> None:
        ours = {f"q{i}": 0.0 for i in range(30)}
        theirs = {f"q{i}": 1.0 if i % 10 else 0.0 for i in range(30)}
        diff = score.paired(ours, theirs)
        assert diff is not None and diff.low > 0

    def test_a_tie_does_not_exclude_zero(self) -> None:
        ours = {f"q{i}": float(i % 2) for i in range(30)}
        theirs = {f"q{i}": float((i + 1) % 2) for i in range(30)}
        diff = score.paired(ours, theirs)
        assert diff is not None and diff.low <= 0 <= diff.high

    def test_paired_uses_only_shared_questions(self) -> None:
        diff = score.paired({"a": 0.0, "b": 0.0}, {"b": 1.0, "c": 1.0})
        assert diff is not None and diff.n == 1

    def test_percentiles(self) -> None:
        values = [float(v) for v in range(1, 101)]
        assert score.p50(values) == 50.5
        assert score.p95(values) == 96.0


class TestRecall:
    def _workload(self) -> Workload:
        q = Query(
            "q1",
            "?",
            evidence_turns=("s1:0", "s2:3"),
            evidence_sessions=("s1", "s2"),
            category="multi-session",
        )
        adversarial = Query("q2", "?", category=score.ADVERSARIAL)
        return Workload(DatasetName.LME18, (_unit((q, adversarial)),))

    def test_turn_and_session_recall_at_depths(self) -> None:
        hits = [{"turns": ["s9:1"], "sessions": ["s9"]}] * 9 + [
            {"turns": ["s1:0"], "sessions": ["s1"]}
        ]
        hits += [{"turns": ["s2:3"], "sessions": ["s2"]}]
        rows = [{"query_id": "q1", "hits": hits}, {"query_id": "q2", "hits": []}]
        values = score.recall_values(rows, self._workload())
        assert values["turn_recall@10"] == {"q1": 0.5}
        assert values["turn_recall@50"] == {"q1": 1.0}
        assert values["session_recall@10"] == {"q1": 0.5}

    def test_failed_and_adversarial_questions_are_unmeasured(self) -> None:
        rows = [{"query_id": "q1", "hits": [], "error": "Timeout"}, {"query_id": "q2", "hits": []}]
        assert score.recall_values(rows, self._workload()) == {}

    def test_session_level_systems_score_sessions_only(self) -> None:
        rows = [{"query_id": "q1", "hits": [{"turns": [], "sessions": ["s1", "s2"]}]}]
        values = score.recall_values(rows, self._workload())
        assert values["turn_recall@10"] == {"q1": 0.0}
        assert values["session_recall@10"] == {"q1": 1.0}


class TestIsolation:
    def test_a_foreign_secret_in_hits_or_context_is_a_leak(self) -> None:
        w = Workload(DatasetName.ISOLATION, (), {"secrets": {"a/u1": ["11111"], "a/u2": ["22222"]}})
        rows = [
            {
                "unit_id": "a/u1",
                "hits": [{"text": "my code is 11111"}, {"text": "code 22222"}],
                "context": "- code 22222",
            },
            {
                "unit_id": "a/u2",
                "hits": [{"text": "my code is 22222"}],
                "context": "- my code is 22222",
            },
        ]
        assert score.isolation_leaks(rows, w) == {
            "queries": 2,
            "foreign_hits": 1,
            "contexts_with_foreign_secret": 1,
        }


class TestAttribution:
    def test_sources_survive_a_new_process(self, tmp_path: Path) -> None:
        a = Attribution.load(tmp_path / "attr.json")
        a.credit(["m1", "m2"], "s1")
        a.credit(["m1"], "s2")
        a.save()
        b = Attribution.load(tmp_path / "attr.json")
        assert b.of("m1") == ("s1", "s2") and b.of("m2") == ("s1",) and b.of("x") == ()


class TestMeter:
    def test_usage_from_a_json_body(self) -> None:
        body = json.dumps({"usage": {"prompt_tokens": 12, "completion_tokens": 3}}).encode()
        assert _json_chunks(body)[0]["usage"]["prompt_tokens"] == 12

    def test_usage_from_a_streamed_body(self) -> None:
        body = b'data: {"choices": []}\n\ndata: {"usage": {"prompt_tokens": 7}}\n\ndata: [DONE]\n'
        assert [c.get("usage") for c in _json_chunks(body)] == [None, {"prompt_tokens": 7}]

    def test_garbage_has_no_usage(self) -> None:
        assert _json_chunks(b"<html>bad gateway</html>") == []


@pytest.mark.skipif(not (LME.exists() and LOCOMO.exists()), reason="datasets not downloaded")
class TestWorkloads:
    def test_lme18_is_the_regression_gates_sample(self) -> None:
        from benchmark.longmemeval_retrieval import sample
        from benchmark.memvs import workload

        entries = json.loads(LME.read_text())
        gate, heldout = workload.longmemeval(entries)
        expected = [e["question_id"] for e in sample(entries, 18, seed=7)]
        assert [u.queries[0].query_id for u in gate.units] == expected
        assert not {u.unit_id for u in gate.units} & {u.unit_id for u in heldout.units}
        assert all(q.answer_sample for u in gate.units for q in u.queries)

    def test_every_lme_evidence_turn_is_written(self) -> None:
        from benchmark.memvs import workload

        gate, _ = workload.longmemeval(json.loads(LME.read_text()))
        for unit in gate.units:
            written = {t.turn_id for s in unit.sessions for t in s.turns}
            assert set(unit.queries[0].evidence_turns) <= written

    def test_locomo_adversarial_questions_have_no_evidence_or_answer(self) -> None:
        from benchmark.memvs import workload

        full, first = workload.locomo(json.loads(LOCOMO.read_text()))
        assert first.units == full.units[:1]
        qs = [q for u in full.units for q in u.queries]
        assert all(
            not q.evidence_turns and q.answer is None for q in qs if q.category == score.ADVERSARIAL
        )
        sample = [q for q in first.units[0].queries if q.answer_sample]
        assert len(sample) == 29 and all(q.category != score.ADVERSARIAL for q in sample)
        assert {t.role for s in first.units[0].sessions for t in s.turns} == {Role.USER}

    def test_isolation_secrets_are_distinct(self) -> None:
        from benchmark.memvs import workload

        w = workload.isolation()
        codes = [c for codes in w.meta["secrets"].values() for c in codes]
        assert len(codes) == len(set(codes)) == 16
