"""The capability eval suite's fast, deterministic slice (benchmark/capabilities).

What it holds: the dataset is well-formed and every assertion in it can be met by what its
scenario writes, in every language; the split is deterministic and balanced; the scorer and
the corpus generator do what they say; the gate fails on any drop; and the runner drives a
real in-process service through the SDK end to end. The full runs - the live service, the
model, the harness - are ``make eval-capabilities`` targets, not tests.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
from collections import Counter
from datetime import date
from pathlib import Path

import httpx
import pytest
from benchmark.capabilities import gate, generate, scoring
from benchmark.capabilities.dataset import (
    DOMAINS,
    LANGUAGES,
    PHRASING_KINDS,
    Selection,
    load_dataset,
    load_json,
    select,
)
from benchmark.capabilities.report import build, markdown
from benchmark.capabilities.validate import problems

pytestmark = pytest.mark.eval

DATASET = load_dataset()


# --------------------------------------------------------------------------- the dataset


def test_every_assertion_is_satisfiable_in_every_language() -> None:
    assert problems(DATASET) == []


def test_the_dataset_covers_eleven_capabilities_five_domains_and_five_languages() -> None:
    assert [c.number for c in DATASET.capabilities] == list(range(1, 12))
    for cap in DATASET.capabilities:
        assert {s.domain for s in cap.scenarios} == set(DOMAINS), cap.name
        assert any(s.origin for s in cap.scenarios), f"{cap.name}: no InventorySmart case"
        assert any(s.code_switched for s in cap.scenarios), f"{cap.name}: no code-switched case"
        for s in cap.scenarios:
            for q in s.queries:
                assert sorted(p.kind for p in q.phrasings) == sorted(PHRASING_KINDS)
                for p in q.phrasings:
                    assert set(LANGUAGES) <= set(p.text)
    counts = DATASET.counts()
    assert counts["scenarios"] >= 100 and counts["phrasings_per_language"] >= 1100


def test_the_split_is_deterministic_and_balanced() -> None:
    again = load_dataset()
    assert [q.split for s in DATASET.scenarios for q in s.queries] == [
        q.split for s in again.scenarios for q in s.queries
    ]
    for cap in DATASET.capabilities:
        halves = Counter(q.split for s in cap.scenarios for q in s.queries)
        assert abs(halves["dev"] - halves["heldout"]) <= 1, (cap.name, halves)


def test_selection_is_stable_and_samples_put_the_origin_cases_first() -> None:
    sample = load_json("_modes.json")["samples"]["harness"]
    sel = Selection(
        languages=tuple(sample["languages"]),
        scenarios_per_domain=sample["scenarios_per_domain"],
        kinds=tuple(sample["kinds"]),
    )
    picked = select(DATASET, sel)
    assert picked == select(load_dataset(), sel)
    origins = {s.origin for s in picked if s.origin}
    assert len(origins) == 11, "every InventorySmart case is in the harness sample"
    assert all(len(p.phrasings) == 1 for s in picked for p in s.queries)


# --------------------------------------------------------------------------- the scorer


def _items(*texts: str, historical: tuple[int, ...] = ()) -> list[scoring.Item]:
    return [scoring.Item("memory", t, i, i in historical) for i, t in enumerate(texts)]


def test_contains_groups_alternatives_and_language_maps() -> None:
    expect = {"contains": ["8492", [{"en": "door 14", "de": "Tor 14"}, "fourteen"]]}
    ok = scoring.score(expect, _items("Code 8492.", "Use Tor 14."), "de", surface="context")
    assert ok.passed and ok.groups_met == 2
    miss = scoring.score(expect, _items("Code 18492."), "en", surface="context")
    assert not miss.passed and miss.reason == "missing_evidence" and miss.groups_met == 0


def test_normalisation_folds_digits_separators_and_arabic_marks() -> None:
    assert scoring.normalize("رمز ٨٤٩٢ و 5,000") == scoring.normalize("رمز 8492 و 5000")
    assert scoring.resolve("5000", "de").matches(scoring.normalize("5.000 Bäume"))
    assert scoring.resolve("الرصيف", "ar").matches(scoring.normalize("الرَّصِيف"))
    assert not scoring.resolve("8492", "en").matches(scoring.normalize("ref 84920"))


def test_a_stale_value_leaks_unless_marked_or_superseded_in_the_same_item() -> None:
    expect = {
        "contains": ["PackagingCorp"],
        "absent": [{"text": "Uline", "unless": ["PackagingCorp"]}],
    }
    corrected = _items("We dropped Uline; PackagingCorp supplies pallets now.")
    assert scoring.score(expect, corrected, "en", surface="context").passed
    stale = _items("Our pallet supplier is Uline.", "PackagingCorp supplies pallets now.")
    v = scoring.score(expect, stale, "en", surface="context")
    assert not v.passed and v.reason == "stale_leak" and v.leaks == ["Uline"]
    marked = _items("Our pallet supplier is Uline.", "PackagingCorp now.", historical=(0,))
    assert scoring.score(expect, marked, "en", surface="context").passed


def test_abstention_on_each_surface() -> None:
    expect = {"abstain": True, "answer": {"abstain": True, "absent": [{"re": r"\d{3}-\d{4}"}]}}
    said = scoring.score(expect, [], "en", surface="context", evidence_status="INSUFFICIENT")
    assert said.passed and said.abstained
    guessed = scoring.score(
        expect, _items("x"), "en", surface="context", evidence_status="COMPLETE"
    )
    assert guessed.reason == "no_abstain"
    markers = load_json("_languages.json")["refusal"]["en"]
    refused = scoring.items_from_answer("I don't have his number on file.")
    assert scoring.score(expect, refused, "en", surface="answer", refusal_markers=markers).passed
    invented = scoring.items_from_answer("His cell is 555-0142.")
    assert (
        scoring.score(expect, invented, "en", surface="answer", refusal_markers=markers).reason
        == "hallucinated"
    )
    # an answerable question the service refuses is a false abstention
    v = scoring.score(
        {"contains": ["8492"]}, [], "en", surface="context", evidence_status="INSUFFICIENT"
    )
    assert v.reason == "false_abstain"
    assert not scoring.scorable(expect, "search") and scoring.scorable(
        {"contains": ["x"]}, "search"
    )


def test_table_columns_on_an_answer() -> None:
    expect = {"answer": {"table_columns": ["SKU", {"en": "Action Required", "de": "Maßnahme"}]}}
    table = scoring.items_from_answer(
        "| SKU | Item | Action Required |\n|---|---|---|\n| A | b | c |"
    )
    assert scoring.score(expect, table, "en", surface="answer").passed
    prose = scoring.items_from_answer("SKU A needs action required.")
    assert scoring.score(expect, prose, "en", surface="answer").reason == "format"


def test_bundle_items_keep_render_order_and_mark_ended_facts() -> None:
    bundle = {
        "profile": [{"block": "user", "text": "Prefers tables"}],
        "memories": [{"id": "m1", "text": "Code 8492", "relevance": 0.5}],
        "graph_facts": [
            {
                "id": "f",
                "subject": "A",
                "predicate": "is",
                "object": "B",
                "relevance": 0.4,
                "valid_to": "2026-01-01",
            }
        ],
    }
    items = scoring.items_from_bundle(bundle)
    assert [i.kind for i in items] == ["profile", "memory", "fact"]
    assert items[2].historical and not items[1].historical


# --------------------------------------------------------------------------- generation


def test_the_corpus_and_its_ground_truth_are_deterministic() -> None:
    spec = load_json("scale_personal.json")
    today = date(2026, 10, 8)
    a = generate.build_corpus(spec, "en", today, "standard")
    b = generate.build_corpus(spec, "en", today, "standard")
    assert [t.text for ts in a.sessions.values() for t in ts] == [
        t.text for ts in b.sessions.values() for t in ts
    ]
    q = {
        "record": "log",
        "period": {"year_offset": -1, "quarter": 3},
        "group_by": "category",
        "value": {"sum": "amount"},
        "top": 3,
    }
    truth = generate.aggregate(a, q, today)
    # recomputed by hand from the generated records
    totals: Counter[str] = Counter()
    for rec in a.records:
        if generate.in_period(rec.period, q["period"], today):
            for v in rec.values:
                totals[v["category"]] += v["amount"]
    assert truth.answer == [
        k for k, _ in sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
    ]
    assert len(truth.evidence) == 13, "a quarter of weekly reports"
    expect = generate.aggregate_expect(truth, q)
    assert len(expect["contains"]) == 13 and len(expect["answer"]["absent"]) == 7


def test_period_arithmetic() -> None:
    today = date(2026, 10, 8)
    assert generate.in_period(date(2025, 8, 1), {"year_offset": -1, "quarter": 3}, today)
    assert not generate.in_period(date(2025, 10, 1), {"year_offset": -1, "quarter": 3}, today)
    assert generate.in_period(date(2026, 9, 3), {"months_ago": 1}, today)
    assert generate.period_label(date(2025, 7, 14), "week") == "2025-W29"


# --------------------------------------------------------------------------- the gate


def _run(
    acc: float,
    p95: float,
    *,
    trustworthy: bool = True,
    sha: str | None = None,
    load: float = 0.2,
    sample: str | None = None,
) -> dict:
    records = [
        {
            "key": f"s/q/{k}/en",
            "capability": "single_fact_recall",
            "scenario": "s",
            "origin": None,
            "domain": "retail",
            "query": "q",
            "split": "dev",
            "kind": k,
            "language": "en",
            "question": "?",
            "placement": "new_session",
            "surface": "context",
            "latency_ms": p95,
            "passed": i < acc * 10,
            "reason": None if i < acc * 10 else "missing_evidence",
            "groups_met": 1,
            "groups_required": 1,
            "missing": [],
            "leaks": [],
            "abstained": False,
            "items": 1,
            "preview": [],
        }
        for i, k in enumerate(PHRASING_KINDS)
    ]
    meta = {
        "mode": "nollm",
        "selection": {"languages": ["en"]},
        "host_load_during_asks": {"latency_trustworthy": trustworthy, "max_load_per_core": load},
        "sample": sample,
    }
    out = build(DATASET, records, meta)
    if sha:
        out["dataset"]["sha256"] = sha
    return out


def test_the_gate_fails_on_any_drop_and_on_slower_context(tmp_path: Path) -> None:
    base = tmp_path / "base.json"
    base.write_text(json.dumps(_run(0.7, 100.0)))
    recorded = gate.record([base], tmp_path / "gate.json")
    assert gate.check(_run(0.7, 105.0), recorded) == []
    assert gate.check(_run(0.8, 100.0), recorded) == [], "an improvement passes"
    dropped = gate.check(_run(0.6, 100.0), recorded)
    assert dropped and "context/new_session/single_fact_recall/en" in dropped[0]
    slower = gate.check(_run(0.7, 111.0), recorded)
    assert slower and "context p95" in slower[0]
    loaded = gate.check(_run(0.7, 100.0, trustworthy=False), recorded)
    assert loaded and "load" in loaded[0]
    assert "dataset changed" in gate.check(_run(0.7, 100.0, sha="0" * 64), recorded)[0]
    assert markdown(_run(0.7, 100.0)).startswith("# Capability eval")
    # repeats recorded together: each cell's lowest, and the range they spanned is its noise
    repeat = tmp_path / "repeat.json"
    repeat.write_text(json.dumps(_run(0.6, 90.0)))
    merged = gate.record([base, repeat], tmp_path / "gate2.json")
    assert merged["baselines"]["nollm"]["repeat_runs"] == 2
    assert gate.check(_run(0.6, 100.0), merged) == []
    # the threshold is already the lower repeat: below it is below, noisy or not
    assert gate.check(_run(0.5, 100.0), merged)
    # a latency-only re-measure on a quieter machine replaces the latency, not the cells
    quiet = tmp_path / "quiet.json"
    quiet.write_text(json.dumps(_run(0.7, 80.0, load=0.05)))
    remeasured = gate.record([base, repeat, quiet], tmp_path / "gate4.json")
    assert remeasured["baselines"]["nollm"]["latency"]["context_p95_ms"] == 80.0
    assert gate.check(_run(0.6, 100.0), remeasured), "100 ms is over 80 ms + 10 %"
    # a run whose model calls failed measured the outage, not the service: never a baseline
    broken = _run(0.7, 100.0)
    broken["meta"]["answer_errors"] = 3
    failed = tmp_path / "harness.failed.json"
    failed.write_text(json.dumps(broken))
    with pytest.raises(SystemExit, match="not a baseline"):
        gate.record([base, failed], tmp_path / "gate3.json")


def test_gate_sets_core_has_its_own_baseline_and_scale_is_held_to_the_full_one(
    tmp_path: Path,
) -> None:
    full, core = tmp_path / "nollm.json", tmp_path / "nollm_core.json"
    full.write_text(json.dumps(_run(0.7, 100.0)))
    core.write_text(json.dumps(_run(0.5, 40.0, sample="core")))
    recorded = gate.record([full, core], tmp_path / "gate.json")
    assert set(recorded["baselines"]) == {"nollm", "nollm/core"}
    assert recorded["baselines"]["nollm/core"]["latency"]["context_p95_ms"] == 40.0
    assert gate.check(_run(0.5, 42.0, sample="core"), recorded) == []
    assert gate.check(_run(0.4, 40.0, sample="core"), recorded), "a core cell dropped"
    # the scale set: the full baseline's matching cells, its own latency not compared
    assert gate.check(_run(0.7, 900.0, sample="scale"), recorded) == []
    assert gate.check(_run(0.6, 100.0, sample="scale"), recorded), "a scale cell dropped"
    # a set whose read path runs the model on the measuring host: latency reported, not gated
    llm_core = tmp_path / "llm_core.json"
    run = _run(0.5, 300.0, sample="llm_core")
    run["mode"] = run["meta"]["mode"] = "llm"
    llm_core.write_text(json.dumps(run))
    with_llm = gate.record([llm_core], tmp_path / "gate.json")
    loaded = _run(0.5, 900.0, trustworthy=False, sample="llm_core")
    loaded["mode"] = loaded["meta"]["mode"] = "llm"
    assert gate.check(loaded, with_llm) == []
    scale = tmp_path / "nollm_scale.json"
    scale.write_text(json.dumps(_run(0.7, 100.0, sample="scale")))
    with pytest.raises(SystemExit, match="gated against"):
        gate.record([full, scale], tmp_path / "gate2.json")


def test_gate_rerun_rule_empty_runs_and_new_cells(tmp_path: Path) -> None:
    import copy

    first, second = tmp_path / "a.json", tmp_path / "b.json"
    first.write_text(json.dumps(_run(0.7, 100.0)))
    second.write_text(json.dumps(_run(0.6, 100.0)))
    recorded = gate.record([first, second], tmp_path / "gate.json")
    cell = "context/new_session/single_fact_recall/en"
    assert recorded["baselines"]["nollm"]["noisy"][cell] == [0.6, 0.7]
    # below the threshold (the lower repeat), a noisy cell asks for one rerun ...
    dropped = gate.check(_run(0.5, 100.0), recorded)
    assert dropped and all(f.startswith(gate.RERUN) for f in dropped)
    # ... and the rerun's own value decides
    assert gate.check(_run(0.5, 100.0), recorded, rerun=_run(0.6, 100.0)) == []
    assert not gate.check(_run(0.5, 100.0), recorded, rerun=_run(0.5, 100.0))[0].startswith(
        gate.RERUN
    )
    # a run that measured nothing is neither a baseline nor a pass
    empty = _run(0.7, 100.0)
    empty["cells"], empty["records"] = [], []
    (tmp_path / "empty.json").write_text(json.dumps(empty))
    with pytest.raises(SystemExit, match="measured nothing"):
        gate.record([tmp_path / "empty.json"], tmp_path / "gate2.json")
    assert gate.check(empty, recorded) == ["the run measured no cells"]
    # a cell the baseline does not hold yet is reported, not gated
    wider = _run(0.7, 100.0)
    extra = copy.deepcopy(wider["cells"][0])
    extra["language"] = "de"
    wider["cells"].append(extra)
    assert gate.new_cells(wider, recorded) == ["context/new_session/single_fact_recall/de"]


def test_the_gate_is_a_ratchet_with_report_only_targets(tmp_path: Path) -> None:
    cell = "context/new_session/single_fact_recall/en"
    g = tmp_path / "gate.json"

    def record(acc: float) -> dict:
        run = tmp_path / f"run_{acc}.json"
        run.write_text(json.dumps(_run(acc, 100.0)))
        return gate.record([run], g)

    assert record(0.6)["baselines"]["nollm"]["cells"][cell]["accuracy"] == 0.6
    # a better run raises the threshold; a worse one is refused: accuracy may only go up
    raised = record(0.8)
    assert raised["baselines"]["nollm"]["cells"][cell]["accuracy"] == 0.8
    with pytest.raises(SystemExit, match="would record a regression"):
        record(0.7)
    assert gate.check(_run(0.7, 100.0), raised), "below the best recorded value"
    # a target beside the threshold is reported, never failed, and kept by a re-record
    with_target = {**raised, "targets": {"nollm": {"cells": {cell: 0.9}}}}
    g.write_text(json.dumps(with_target))
    assert gate.check(_run(0.8, 100.0), with_target) == []
    assert gate.below_targets(_run(0.8, 100.0), with_target) == [f"{cell}: 0.8 below target 0.9"]
    assert record(0.9)["targets"] == with_target["targets"]
    assert gate.below_targets(_run(0.9, 100.0), with_target) == []


def test_thresholds_do_not_carry_across_identity_schemes(tmp_path: Path) -> None:
    """Runs that named their users another way indexed other text: their values are another
    measurement. A run is refused against them; a re-record starts the thresholds afresh."""
    cell = "context/new_session/single_fact_recall/en"
    g = tmp_path / "gate.json"

    def named(acc: float, scheme: str | None) -> Path:
        run = _run(acc, 100.0)
        run["meta"]["identities"] = scheme
        path = tmp_path / f"run_{acc}_{(scheme or 'old').replace('/', '_')}.json"
        path.write_text(json.dumps(run))
        return path

    old = gate.record([named(0.8, None)], g)
    current = _run(0.6, 100.0)
    current["meta"]["identities"] = "scenario-language/v1"
    refused = gate.check(current, old)
    assert refused and "named its users another way" in refused[0]
    # not "a regression" from 0.8: the 0.8 was measured on other index text
    fresh = gate.record([named(0.6, "scenario-language/v1")], g)
    assert fresh["baselines"]["nollm"]["cells"][cell]["accuracy"] == 0.6
    assert fresh["baselines"]["nollm"]["identities"] == "scenario-language/v1"
    assert gate.check(current, fresh) == []
    # and from then on it ratchets as before
    with pytest.raises(SystemExit, match="would record a regression"):
        gate.record([named(0.4, "scenario-language/v1")], g)
    with pytest.raises(SystemExit, match="named their users differently"):
        gate.record([named(0.6, "scenario-language/v1"), named(0.6, None)], tmp_path / "g2.json")


def test_eval_identities_are_a_function_of_scenario_and_language() -> None:
    """Two runs write the same users and threads: nothing drawn per run reaches the store."""
    from datetime import UTC, datetime

    from benchmark.capabilities.runner import identity, parse, run_day, thread_name

    a = identity("recall-retail-hazmat-code", "en")
    assert a == identity("recall-retail-hazmat-code", "en")
    assert a != identity("recall-retail-hazmat-code", "de")
    assert a != identity("recall-retail-dock-door", "en")
    # opaque, like a production user id: the subject is indexed with every memory's text
    assert a.startswith("ce") and "hazmat" not in a and len(a) == 14
    assert thread_name(a, "s1") == f"{a}-s1"
    users = {identity(s.id, lang) for s in DATASET.scenarios for lang in LANGUAGES}
    assert len(users) == len(DATASET.scenarios) * len(LANGUAGES)
    settings, _ = parse(["--mode", "nollm", "--run-date", "2026-10-12"])
    assert run_day(settings) == datetime(2026, 10, 12, tzinfo=UTC)
    settings.run_date = None
    assert run_day(settings).date() == datetime.now(UTC).date()


def test_a_run_counts_the_model_calls_its_uses_made() -> None:
    from benchmark.capabilities.runner import llm_counts

    text = "\n".join(
        [
            "# HELP memory_llm_assist_total Module-level LLM assistance",
            'memory_llm_assist_total{outcome="used",use="relation_extraction"} 40.0',
            'memory_llm_assist_total{outcome="fallback",use="relation_extraction"} 2.0',
            'memory_llm_assist_total{outcome="used",use="summaries"} 5.0',
            'memory_llm_assist_total_created{outcome="used",use="summaries"} 1.7e9',
            'memory_llm_requests_total{outcome="ok",use="summaries"} 5.0',
            'memory_llm_requests_total{outcome="circuit_open",use="relation_extraction"} 1.0',
        ]
    )
    counts = llm_counts(text)
    assert counts["assist"] == {"used": 45.0, "fallback": 2.0}
    assert counts["assist_by_use"]["relation_extraction"] == {"used": 40.0, "fallback": 2.0}
    assert counts["requests"] == {"ok": 5.0, "circuit_open": 1.0}
    assert llm_counts("") == {"assist": {}, "assist_by_use": {}, "requests": {}}


def test_an_llm_run_whose_model_was_unreachable_is_not_a_baseline(tmp_path: Path) -> None:
    """The eval's model server killed mid-run (out of memory) turned every later use into its
    fallback: the run measured the model-free path under the llm label."""

    def llm(requests: dict[str, float] | None) -> dict:
        run = _run(0.7, 100.0)
        run["mode"] = run["meta"]["mode"] = "llm"
        if requests is not None:
            run["meta"]["llm_calls"] = {
                "api": {"requests": {k: v for k, v in requests.items() if k == "circuit_open"}},
                "worker": {"requests": {k: v for k, v in requests.items() if k != "circuit_open"}},
            }
        return run

    healthy = tmp_path / "healthy.json"
    # the model's own bad answers are measured, and one transient failure in 100 is tolerated
    healthy.write_text(json.dumps(llm({"ok": 98.0, "invalid_output": 1.0, "exhausted": 1.0})))
    recorded = gate.record([healthy], tmp_path / "gate.json")
    assert gate.model_unreached(llm({"ok": 98.0, "invalid_output": 1.0, "exhausted": 1.0})) is None
    assert gate.check(llm({"ok": 100.0}), recorded) == []
    down = llm({"ok": 1499.0, "exhausted": 14.0, "circuit_open": 1120.0})
    why = gate.model_unreached(down)
    assert why and "1134 of 2633 model calls never reached the model" in why
    assert gate.check(down, recorded) == [why]
    assert "made no model call" in gate.check(llm({}), recorded)[0]
    broken = tmp_path / "down.json"
    broken.write_text(json.dumps(down))
    with pytest.raises(SystemExit, match="never reached the model.*not a baseline"):
        gate.record([healthy, broken], tmp_path / "gate2.json")
    # not judged: another mode, or a run that predates the count
    assert gate.model_unreached(_run(0.7, 100.0)) is None
    assert gate.model_unreached(llm(None)) is None


async def test_the_llm_stack_starts_the_service_with_the_eval_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the model call's timeout and the query expansion's deadline differ from the
    shipped service, and only when set."""
    import types

    from benchmark.capabilities import stack_service

    from memory_service.config.constants import LLM, RETRIEVAL

    monkeypatch.delenv("CAPEVAL_LLM_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("CAPEVAL_QUERY_EXPANSION_TIMEOUT_MS", raising=False)
    assert stack_service.overrides() is None
    monkeypatch.setenv("CAPEVAL_LLM_TIMEOUT_SECONDS", "600")
    assert stack_service.overrides().retrieval is None, "the shipped retrieval tuning"
    monkeypatch.setenv("CAPEVAL_QUERY_EXPANSION_TIMEOUT_MS", "60000")
    tuned = stack_service.overrides()
    assert tuned.llm.timeout_seconds == 600.0
    assert dataclasses.replace(tuned.llm, timeout_seconds=LLM.timeout_seconds) == LLM
    # past the setting's production bound (5 s): a CPU model needs seconds per expansion
    assert tuned.retrieval.query_expansion_timeout_ms == 60000
    assert tuned.retrieval.model_copy(update={"query_expansion_timeout_ms": 250}) == RETRIEVAL

    seen: list[object] = []

    async def build_container(
        settings: object, version: str, *, overrides: object = None, **kw: object
    ) -> str:
        seen.append(overrides)
        return "container"

    module = types.SimpleNamespace(build_container=build_container)
    stack_service._with_overrides(module, tuned)  # type: ignore[arg-type]
    assert await module.build_container(None, "v", role="worker") == "container"
    assert seen == [tuned]


def test_noise_is_allowed_once(tmp_path: Path) -> None:
    """llm_core implicit_personalization/hi measured 0.5 and 0.667: the threshold is the lower,
    0.5, and its noise is already spent - 0.333 must not pass."""
    cell = "context/new_session/single_fact_recall/en"

    def with_cell(acc: float) -> dict:
        run = _run(0.7, 100.0)
        for c in run["cells"]:
            if (c["surface"], c["placement"], c["capability"], c["language"]) == tuple(
                cell.split("/")
            ):
                c["accuracy"], c["n"] = acc, 3
        return run

    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(with_cell(0.5)))
    b.write_text(json.dumps(with_cell(0.6667)))
    recorded = gate.record([a, b], tmp_path / "gate.json")
    assert recorded["baselines"]["nollm"]["cells"][cell]["accuracy"] == 0.5
    low = gate.check(with_cell(0.3333), recorded)
    assert low and all(f.startswith(gate.RERUN) for f in low), "below the lower repeat"
    # the rerun's own value decides, not the better of the two
    assert gate.check(with_cell(0.3333), recorded, rerun=with_cell(0.3333))
    assert gate.check(with_cell(0.3333), recorded, rerun=with_cell(0.5)) == []
    # a rerun of a different run (another code sha) is refused
    other = with_cell(0.5)
    other["provenance"] = {**other.get("provenance", {}), "git_commit": "0" * 40}
    refused_rerun = gate.check(with_cell(0.3333), recorded, rerun=other)
    assert refused_rerun and "not a rerun of this run" in refused_rerun[0]


def test_an_empty_answer_is_not_a_refusal() -> None:
    v = scoring.score({"abstain": True}, [], "en", surface="answer")
    assert (v.abstained, v.reason) == (False, "no_abstain")


def test_reasons_are_given_in_precedence_order() -> None:
    # unknowable, answered, and with an invented value: the first applicable reason wins
    v = scoring.score(
        {"abstain": True, "absent": [{"text": "555-0101"}]},
        _items("Call him at 555-0101."),
        "en",
        surface="answer",
    )
    assert v.reason == scoring.REASONS[0] == "hallucinated"


# --------------------------------------------------------------------------- isolated stack

STACK = Path(__file__).resolve().parents[2] / "benchmark/capabilities/isolated_stack.sh"


def _stack(tmp_path: Path, **env: str) -> subprocess.CompletedProcess[str]:
    """Run the isolated stack with a command that must never run: every case here is refused
    before anything is created."""
    import os
    import uuid

    base = {
        "CAPEVAL_WORK": f"/var/tmp/capeval_test_{uuid.uuid4().hex[:8]}",
        "CAPEVAL_DB": "capeval_test",
        "CAPEVAL_REDIS_DB": "15",
        # ports nothing listens on
        "CAPEVAL_API_PORT": "18191",
        "CAPEVAL_METRICS_PORT": "19481",
        "CAPEVAL_QDRANT_HTTP": "16533",
        "CAPEVAL_QDRANT_GRPC": "16534",
        "CAPEVAL_FGA_HTTP": "18281",
    }
    marker = tmp_path / "ran"
    return subprocess.run(
        ["bash", str(STACK), "touch", str(marker)],
        env={**os.environ, **base, **env},
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.mark.parametrize(
    ("env", "why"),
    [
        ({"CAPEVAL_DB": "live_prod"}, "does not match ^capeval"),
        ({"CAPEVAL_QDRANT_HTTP": "6333"}, "live instance"),
        ({"CAPEVAL_QDRANT_GRPC": "6334"}, "live instance"),
        ({"CAPEVAL_WORK": "/tmp/elsewhere"}, "not under /var/tmp/capeval"),
        ({"CAPEVAL_WORK": "/var/tmp/capeval_x/../live"}, "not under /var/tmp/capeval"),
        ({"CAPEVAL_REDIS_DB": "0"}, "reserved"),
        ({"CAPEVAL_REDIS_DB": "13"}, "reserved"),
        ({"CAPEVAL_REDIS_DB": "00"}, "reserved"),
        ({"CAPEVAL_QDRANT_HTTP": "06333"}, "live instance"),
        ({"CAPEVAL_LOGS": "/var/tmp/capeval_x/../wt"}, "not under /var/tmp/capeval"),
        ({"CAPEVAL_LOGS": "/tmp/capeval_logs"}, "not under /var/tmp/capeval"),
        ({"CAPEVAL_API_PORT": "80a"}, "is not a number"),
        ({"PGHOST": "db.example.com"}, "is not local"),
        ({"CAPEVAL_GATEWAY": "1", "BIFROST_URL": "", "BIFROST_VIRTUAL_KEY": ""}, "needs BIFROST"),
    ],
)
def test_the_isolated_stack_refuses_a_live_resource(tmp_path: Path, env: dict, why: str) -> None:
    done = _stack(tmp_path, **env)
    assert done.returncode == 2 and why in done.stderr, done.stderr
    assert not (tmp_path / "ran").exists()


def test_the_isolated_stack_refuses_logs_inside_another_runs_work(tmp_path: Path) -> None:
    import uuid

    other = Path(f"/var/tmp/capeval_test_{uuid.uuid4().hex[:8]}")
    lock = Path(f"{other}.lock")
    lock.touch()
    try:
        done = _stack(tmp_path, CAPEVAL_LOGS=f"{other}/logs")
        assert done.returncode == 2 and "inside the WORK of an isolated run" in done.stderr
        assert not other.exists()
        own = _stack(tmp_path, CAPEVAL_WORK=f"{other}x", CAPEVAL_LOGS=f"{other}x/logs")
        assert own.returncode == 2 and "inside this run's WORK" in own.stderr, own.stderr
    finally:
        lock.unlink(missing_ok=True)


def test_the_isolated_stack_waits_for_room_in_its_memory_cgroup(tmp_path: Path) -> None:
    """The host had 6 GB available while the container's 14 GB cgroup had 2 GB left: the
    encoders were started on the host's number and the cgroup's OOM killer took the API (and
    the model server beside it). The room is the smaller of the two."""
    root = tmp_path / "cgroup"
    mib = 1 << 20
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        hid, controllers, rel = line.split(":", 2)
        if controllers == "memory" or hid == "0":
            d = root / ("memory" if controllers == "memory" else "") / rel.lstrip("/")
            d.mkdir(parents=True, exist_ok=True)
            # v1 and v2 names side by side: the script reads whichever the level has
            (d / "memory.limit_in_bytes").write_text(f"{14_000 * mib}\n")
            # held: anon + shmem + mapped files; the rest of the page cache is reclaimable
            (d / "memory.stat").write_text(
                f"total_cache {3_000 * mib}\ntotal_rss {11_880 * mib}\n"
                f"total_shmem {500 * mib}\ntotal_mapped_file {1_500 * mib}\n"
                f"total_inactive_file {900 * mib}\n"
            )
    done = _stack(
        tmp_path,
        CAPEVAL_CGROUP_ROOT=str(root),
        CAPEVAL_MIN_MEM_MB="1000",
        CAPEVAL_MEM_WAITS="0",
        CAPEVAL_REDIS_DB="14",  # not the default 15, which a real run may be using
    )
    assert done.returncode == 2 and "only 120 MB of memory available" in done.stderr, done.stderr
    assert not (tmp_path / "ran").exists()


def test_the_isolated_stack_refuses_a_port_that_already_answers(tmp_path: Path) -> None:
    import http.server
    import threading

    server = http.server.HTTPServer(("127.0.0.1", 0), http.server.SimpleHTTPRequestHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        done = _stack(tmp_path, CAPEVAL_QDRANT_HTTP=str(server.server_address[1]))
    finally:
        server.shutdown()
    assert done.returncode == 2 and "already answers" in done.stderr, done.stderr


def test_the_isolated_stack_removes_only_what_its_manifest_names(tmp_path: Path) -> None:
    """A manifest left by an earlier run naming a pid that is now someone else's: the process
    is left alone, and a WORK the manifest does not name is not removed."""
    import shutil
    import uuid

    work = Path(f"/var/tmp/capeval_test_{uuid.uuid4().hex[:8]}")
    stranger = subprocess.Popen(["sleep", "60"])
    try:
        work.mkdir()
        Path(f"{work}.manifest").write_text(
            f"qdrant_pid={stranger.pid}\nqdrant_marker=QDRANT__STORAGE__STORAGE_PATH={work}/qdrant\n"
            "qdrant_http=16533\n"
        )
        done = _stack(tmp_path, CAPEVAL_WORK=str(work))
        assert done.returncode == 2 and "no manifest names it" in done.stderr, done.stderr
        assert stranger.poll() is None, "a process that is not ours was signalled"
        assert work.is_dir()
    finally:
        stranger.kill()
        shutil.rmtree(work, ignore_errors=True)
        for leftover in (f"{work}.manifest", f"{work}.lock"):
            Path(leftover).unlink(missing_ok=True)
        shutil.rmtree(f"{work}.logs", ignore_errors=True)


def test_the_isolated_stack_kills_a_worker_that_ignores_sigterm(tmp_path: Path) -> None:
    """A worker left draining a job ignores SIGTERM and would keep the stack's port: teardown
    escalates to SIGKILL for a process that is still the one it started."""
    import shutil
    import sys
    import uuid

    work = Path(f"/var/tmp/capeval_test_{uuid.uuid4().hex[:8]}")
    marker = "capeval-worker-19481"
    stubborn = subprocess.Popen(
        [
            sys.executable,
            "-c",
            f"import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)  # {marker}",
        ]
    )
    try:
        work.mkdir()
        Path(f"{work}.manifest").write_text(f"worker_pid={stubborn.pid}\nworker_marker={marker}\n")
        done = _stack(tmp_path, CAPEVAL_WORK=str(work), CAPEVAL_KILL_WAIT="1")
        assert done.returncode == 2 and "no manifest names it" in done.stderr, done.stderr
        assert stubborn.wait(timeout=10) == -9
    finally:
        stubborn.kill()
        shutil.rmtree(work, ignore_errors=True)
        for leftover in (f"{work}.manifest", f"{work}.lock"):
            Path(leftover).unlink(missing_ok=True)
        shutil.rmtree(f"{work}.logs", ignore_errors=True)


# --------------------------------------------------------------------------- harness mode


async def test_harness_mode_waits_for_the_model_and_scores_a_failed_call_as_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    from benchmark.capabilities import harness_mode
    from benchmark.capabilities.dataset import cases
    from benchmark.capabilities.runner import Written, parse

    # the model's readiness signal: a one-token call that fails until the model has loaded
    calls: list[int] = []

    class Loading:
        async def complete(self, *_a: object, **_k: object) -> dict:
            calls.append(1)
            if len(calls) < 3:
                raise RuntimeError("gateway returned 502")
            return {}

    async def no_wait(_s: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", no_wait)
    await harness_mode.wait_for_model(Loading(), "m", deadline_s=60)
    assert len(calls) == 3

    class Down:
        async def complete(self, *_a: object, **_k: object) -> dict:
            raise RuntimeError("gateway circuit open")

    with pytest.raises(RuntimeError, match="did not answer"):
        await harness_mode.wait_for_model(Down(), "m", deadline_s=0.01)

    # a run whose model call failed is an error, never the agent declining to answer
    scenario = next(s for s in DATASET.scenarios if s.capability == "single_fact_recall")
    case = next(iter(cases(scenario, "en")))
    settings, _ = parse(["--mode", "nollm"])

    class Failing:
        async def run(self, *_a: object, **_k: object) -> object:
            raise RuntimeError("gateway returned 502")

    record = await harness_mode._ask(Failing(), Written(scenario, "en", "u"), case, settings)
    assert (record["passed"], record["reason"]) == (False, "error")
    assert record["abstained"] is None, "an empty answer from a failed run is not a refusal"
    assert "502" in record["error"]


# --------------------------------------------------------------------------- end to end


@pytest.fixture
def app(make_settings):
    from memory_service.api.app import create_app
    from tests.conftest import PG_AVAILABLE, _test_overrides

    if not PG_AVAILABLE:
        pytest.fail("PostgreSQL not reachable: the eval suite needs it (tests/eval/conftest.py)")
    settings = make_settings(authentication={"bootstrap_admin_key": "capeval-boot"})
    return create_app(settings, overrides=_test_overrides(tasks="inline"))


@pytest.fixture
def running(app):
    from fastapi.testclient import TestClient
    from sqlalchemy import text

    from tests.e2e.conftest import TABLES
    from tests.support_real import reset_real_backends

    with TestClient(app, raise_server_exceptions=False) as c:
        container = app.state.container

        async def _truncate() -> None:
            async with container.database.engine.begin() as conn:
                await conn.execute(text("SET LOCAL statement_timeout = 0"))
                await conn.execute(
                    text("TRUNCATE " + ", ".join(TABLES) + " RESTART IDENTITY CASCADE")
                )
            await reset_real_backends(container)

        assert c.portal is not None
        c.portal.call(_truncate)
        c.portal.call(container.services["read_audit"].close)
        c.portal.call(container.services["tenant_registry"].close)
        yield c


@pytest.mark.timeout(600)
async def test_the_runner_drives_the_service_through_the_sdk(app, running) -> None:
    """A slice of every capability, English, two phrasings each, written, settled, asked and
    scored against the in-process service - the same code path as the live run."""
    from benchmark.capabilities.runner import Settings, run

    from trellis.memory import MemoryClient

    def factory(key: str) -> MemoryClient:
        http = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://memory.test"
        )
        return MemoryClient("http://memory.test", api_key=key, http_client=http, timeout=60)

    settings = Settings(
        url="http://memory.test",
        mode="nollm",
        tenant="capeval-test",
        admin_key="capeval-boot",
        api_key=None,
        worker_metrics_url=None,
        selection=Selection(
            languages=("en",),
            scenarios_per_capability=1,
            kinds=("direct", "terse"),
            capabilities=tuple(c.name for c in DATASET.capabilities if c.name != "scale"),
        ),
        write_concurrency=2,
        probe_seconds=5.0,
    )
    payload = await run(settings, client_factory=factory)
    records = payload["records"]
    assert records, "nothing was asked"
    # every capability but scale, two phrasings, two placements on context and one search
    caps = {r["capability"] for r in records}
    assert len(caps) == 10
    assert {(r["surface"], r["placement"]) for r in records} == {
        ("context", "new_session"),
        ("context", "same_session"),
        ("search", "new_session"),
    }
    assert payload["meta"]["ingestion"]["failed_jobs"] == 0
    assert payload["meta"]["ingestion"]["turns"] > 0
    assert all(r["latency_ms"] > 0 for r in records)
    assert payload["summary"]["latency_ms"]["context_p95"] is not None
    # the single-fact question is answerable from its own thread: the stand-in encoder still
    # matches the code lexically, so this pins that write -> settle -> ask -> score is wired
    recall = [
        r for r in records if r["capability"] == "single_fact_recall" and r["surface"] == "search"
    ]
    assert recall and all(r["passed"] for r in recall), recall
    assert set(payload["by_capability"]) == caps
    # the run leaves the tenant as it found it: every memory it wrote is forgotten and every
    # thread deleted, through the service's API
    cleanup = payload["meta"]["cleanup"]
    assert cleanup["purged"] and cleanup["error_count"] == 0, cleanup
    assert cleanup["memories_forgotten"] > 0 and cleanup["threads_deleted"] > 0
    from sqlalchemy import text

    async with app.state.container.database.engine.connect() as conn:
        live = (
            await conn.execute(
                text(
                    "SELECT (SELECT count(*) FROM memories WHERE tenant_id = :t AND deleted_at IS NULL),"
                    " (SELECT count(*) FROM threads WHERE tenant_id = :t AND deleted_at IS NULL)"
                ),
                {"t": "capeval-test"},
            )
        ).one()
    assert tuple(live) == (0, 0), f"left behind: {live[0]} memories, {live[1]} threads"
    # users are named from scenario and language, so a second run must not share the store
    from benchmark.capabilities.runner import IDENTITIES

    assert payload["meta"]["identities"] == IDENTITIES
    with pytest.raises(SystemExit, match="must not share a store"):
        await run(settings, client_factory=factory)


def test_a_run_purges_unless_kept_or_reused() -> None:
    from benchmark.capabilities.cleanup import should_purge
    from benchmark.capabilities.runner import parse

    assert should_purge(parse(["--mode", "nollm"])[0])
    assert not should_purge(parse(["--mode", "nollm", "--keep"])[0])
    assert not should_purge(parse(["--mode", "nollm", "--reuse", "x.raw.json.gz"])[0])
