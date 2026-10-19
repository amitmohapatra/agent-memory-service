"""The accumulated-memory stress set (benchmark/capabilities/stress): its generator is
deterministic and its values never collide, and its judge scores stale values as wrong when
they come first or are stated as current. Pure: no service."""

from __future__ import annotations

from benchmark.capabilities.scoring import Item, normalize
from benchmark.capabilities.stress import build as sb
from benchmark.capabilities.stress.build import Expect
from benchmark.capabilities.stress.judge import judge


def test_the_final_run_is_deterministic_and_independent_of_the_number_of_earlier_runs() -> None:
    a, b, c = sb.build(6), sb.build(6), sb.build(2)
    assert [x.runs for x in a] == [x.runs for x in b]
    assert [x.final for x in a] == [x.final for x in c]
    assert [x.runs[1] for x in a] == [x.runs[1] for x in c]
    assert sb.build(6, seed="other")[0].final != a[0].final


def test_every_script_covers_every_category_and_language_with_distinct_values() -> None:
    built = sb.build(sb.MAX_PRIOR_RUNS)
    for b in built:
        assert {q["category"] for q in b.script["queries"]} == set(sb.CATEGORIES)
        for t in b.script["turns"]:
            assert len(t) == len(b.languages)
        for li in range(len(b.languages)):
            values = {
                normalize(sb.text_of(run[s], li))
                for run in b.runs
                for s, spec in b.script["slots"].items()
                if "fixed" not in spec and not spec.get("once")
            }
            # a slot never repeats a value across runs, nor shares one with another slot
            n = sum(
                1
                for run in b.runs
                for s, spec in b.script["slots"].items()
                if "fixed" not in spec and not spec.get("once")
            )
            assert len(values) == n
        s = sb.scenario(b)
        assert [x.id for x in s.sessions][-1] == "run-final"
        assert len(s.sessions) == sb.MAX_PRIOR_RUNS + 1


def test_expectations_take_the_final_values_and_every_other_value_as_stale() -> None:
    b = next(x for x in sb.build(6) if x.id == "retail-replenishment")
    e = sb.expect(b, "replace", "en")
    assert e.contains == (b.final["new_buyer"],)
    assert b.final["old_buyer"] in e.stale and b.runs[3]["new_buyer"] in e.stale
    assert b.final["new_buyer"] not in e.stale
    hop = sb.expect(b, "hop2", "en")
    assert b.final["store"] not in hop.stale and b.runs[2]["store"] in hop.stale
    assert b.split["replace"] in ("dev", "heldout")


def _items(*texts: str, historical: tuple[int, ...] = ()) -> list[Item]:
    return [Item("memory", t, i, i in historical) for i, t in enumerate(texts)]


E = Expect(
    "update_replacement", False, ("Delia Tanaka",), ("Liora Moreau", "Ana Ruiz"), ("G47",), ()
)


def test_the_current_value_first_and_alone_passes() -> None:
    j = judge(
        E,
        _items("Delia Tanaka has replaced Liora Moreau as buyer for G47", "weather"),
        "SUFFICIENT",
    )
    assert j.passed and not j.stale_leak


def test_a_stale_value_first_is_a_leak_even_with_the_current_value_present() -> None:
    j = judge(
        E,
        _items("The buyer for Region K11 is Ana Ruiz", "The buyer for Region G47 is Delia Tanaka"),
        None,
    )
    assert j.stale_first and not j.stale_current and j.reason == "stale_leak"


def test_a_stale_value_stated_as_current_about_the_anchor_is_a_leak() -> None:
    j = judge(
        E, _items("Delia Tanaka buys for G47", "The buyer for Region G47 is Liora Moreau"), None
    )
    assert j.stale_current and not j.stale_first and not j.passed


def test_a_historical_stale_item_is_not_a_leak() -> None:
    j = judge(
        E,
        _items(
            "The buyer for Region G47 is Liora Moreau", "Delia Tanaka buys for G47", historical=(0,)
        ),
        None,
    )
    assert j.passed


def test_insufficient_on_an_answerable_question_is_a_false_refusal() -> None:
    j = judge(E, _items("Delia Tanaka buys for G47"), "INSUFFICIENT")
    assert j.reason == "false_refusal" and j.abstained


def test_a_never_stated_question_passes_only_on_insufficient() -> None:
    e = Expect("abstention", True, (), (), None, ("Delia Tanaka",))
    assert judge(e, _items("Delia Tanaka buys for G47"), "INSUFFICIENT").passed
    j = judge(e, _items("Delia Tanaka buys for G47"), "SUFFICIENT")
    assert j.reason == "no_abstain" and j.near_served
