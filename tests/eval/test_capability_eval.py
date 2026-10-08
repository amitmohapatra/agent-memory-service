"""The capability eval suite's fast, deterministic slice (benchmark/capabilities).

What it holds: the dataset is well-formed and every assertion in it can be met by what its
scenario writes, in every language; the split is deterministic and balanced; the scorer and
the corpus generator do what they say; the gate fails on any drop; and the runner drives a
real in-process service through the SDK end to end. The full runs - the live service, the
model, the harness - are ``make eval-capabilities`` targets, not tests.
"""

from __future__ import annotations

import json
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
    acc: float, p95: float, *, trustworthy: bool = True, sha: str | None = None, load: float = 0.2
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
    # a repeat run that reached less in a cell lowers that cell's threshold, nothing else
    repeat = tmp_path / "repeat.json"
    repeat.write_text(json.dumps(_run(0.6, 90.0)))
    merged = gate.record([base, repeat], tmp_path / "gate2.json")
    assert merged["baselines"]["nollm"]["repeat_runs"] == 2
    assert gate.check(_run(0.6, 100.0), merged) == []
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
