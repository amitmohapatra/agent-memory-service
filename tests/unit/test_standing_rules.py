"""ADR 0041: the user's standing rules and preferences that apply to a request come back as
profile sections, in any conversation; a later rule on the same subject supersedes."""

from __future__ import annotations

import pytest

from memory_service.domain.context_bundle import ProfileBlockView
from memory_service.modules.context import standing
from memory_service.modules.context.sections import Pinned, _profile_that_fits, within_budget
from memory_service.ports.search import Retriever, SearchFilter, SearchHit, SparseVector, VectorName

pytestmark = pytest.mark.unit


def _hit(rid: str, text: str, category: str, day: str = "2026-10-01", **extra) -> SearchHit:
    extra.setdefault("source_refs", [{"source_type": "message", "source_id": f"obs_{rid}"}])
    return SearchHit(
        record_id=rid,
        score=extra.pop("score", 0.1),
        retriever=Retriever.BM25,
        payload={
            "text": text,
            "category": category,
            "observed_at": f"{day}T09:00:00+00:00",
            **extra,
        },
    )


def _admits(flt: SearchFilter, payload: dict) -> bool:
    must = all(payload.get(k) in v for k, v in flt.must_any.items() if k != "visibility_keys")
    should = not flt.should_any or any(payload.get(k) in v for k, v in flt.should_any.items())
    return must and should


class _Space:
    name = VectorName.DENSE_ML


class _Spaces:
    primary_space = _Space()


class _Indexer:
    spaces = _Spaces()


class _Store:
    def __init__(self, hits: list[SearchHit]) -> None:
        self.hits, self.filters = hits, []

    async def search_dense(self, collection, name, vector, flt, *, limit):
        self.filters.append(flt)
        return [h for h in self.hits if _admits(flt, h.payload)]

    async def search_sparse(self, collection, vector, flt, *, limit):
        return []


class _Engine:
    indexer = _Indexer()

    def __init__(self, hits: list[SearchHit]) -> None:
        self.store = _Store(hits)

    def memory_scope(self, visibility):
        return "memories", SearchFilter(tenant_id="t", must_any={"visibility_keys": ["user:t/u"]})


class _Vectors:
    dense = {VectorName.DENSE_ML: [1.0, 0.0]}
    sparse = SparseVector(indices=[1], values=[1.0])


RULES = [
    _hit("r1", "Never create a transfer when ATA is 0.", "rule"),
    _hit(
        "r2",
        "Always answer stockout audits as a markdown table; no prose.",
        "rule",
        day="2026-09-01",
    ),
    _hit("r3", "Answer stockout audits as a CSV file from now on.", "rule", day="2026-10-05"),
    _hit("r4", "Never book deliveries on Sundays.", "rule"),
    _hit("p1", "I care about high-ROS styles with short WOS.", "preference"),
    _hit("p2", "I manage store 4779.", "preference", score=0.45),
]


@pytest.mark.asyncio
async def test_the_rules_that_apply_and_the_newest_of_one_subject() -> None:
    engine = _Engine(RULES)
    blocks = await standing.standing_blocks(
        engine, "Run the stockout audit for region 5", None, _Vectors()
    )
    rules = next(b for b in blocks if b.block == standing.RULES_BLOCK).text
    assert "CSV" in rules and "markdown" not in rules  # the later rule on the subject
    assert "transfer" not in rules and "Sundays" not in rules  # they do not apply
    # the store searched those categories only, within the reader's own scope
    assert engine.store.filters[0].should_any == {
        "category": list(standing.RULE_CATEGORIES),
        "statement_kind": list(standing.RULE_KINDS),
        "standing_instruction": ["yes"],
    }
    assert engine.store.filters[0].must_any["visibility_keys"] == ["user:t/u"]
    prefs = next(b for b in blocks if b.block == standing.PREFERENCES_BLOCK).text
    assert prefs == "- I manage store 4779."  # related by the encoder, not by its words


@pytest.mark.asyncio
async def test_what_was_said_with_an_applying_preference_comes_with_it() -> None:
    turn = [{"source_type": "message", "source_id": "obs_t"}]
    hits = [
        _hit("p1", "I manage store 4779 at Northgate.", "preference", source_refs=turn),
        _hit("p2", "I care about high-ROS styles with short WOS.", "preference", source_refs=turn),
        _hit("p3", "I like hiking.", "preference"),
    ]
    blocks = await standing.standing_blocks(
        _Engine(hits), "Which styles should I look at this week?", None, _Vectors()
    )
    prefs = blocks[0].text.splitlines()
    assert prefs == [
        "- I care about high-ROS styles with short WOS.",
        "- I manage store 4779 at Northgate.",
    ]


@pytest.mark.asyncio
async def test_a_rule_applies_by_shared_words_in_another_language() -> None:
    hits = [_hit("r1", "Nie einen Transfer anlegen, wenn ATA 0 ist.", "rule")]
    blocks = await standing.standing_blocks(
        _Engine(hits), "Leg einen Transfer an, ATA ist 0", None, _Vectors()
    )
    assert blocks and "Transfer" in blocks[0].text


@pytest.mark.asyncio
async def test_an_earlier_value_never_applies() -> None:
    hits = [_hit("r1", "Never create a transfer when ATA is 0.", "rule", earlier_value=True)]
    assert not await standing.standing_blocks(
        _Engine(hits), "create a transfer, ATA is 0", None, _Vectors()
    )


def test_the_applying_sections_are_kept_when_the_whole_profile_does_not_fit() -> None:
    big = ProfileBlockView(block="user", text="- x " * 4000, version=1)
    rules = ProfileBlockView(block="standing_rules", text="- Never book on Sundays.", version=0)
    kept, cost = _profile_that_fits([rules, big], 200)
    assert kept == [rules] and cost > 0
    pinned, _ = within_budget(Pinned(profile=[rules, big]), 400)
    assert pinned.profile == [rules]


@pytest.mark.asyncio
async def test_a_restated_rule_or_preference_with_new_values_keeps_only_the_newest() -> None:
    hits = [
        _hit("a", "Never schedule anything before 06:50 on weekdays.", "rule", day="2026-09-01"),
        _hit("b", "Never schedule anything before 06:20 on weekdays.", "rule", day="2026-10-01"),
        _hit("p", "I manage store JN4786 Lakeshore Arcade.", "preference", day="2026-09-01"),
        _hit("q", "I manage store ZB1871 Willow Creek Mall.", "preference", day="2026-10-01"),
    ]
    blocks = {
        b.block: b.text
        for b in await standing.standing_blocks(
            _Engine(hits), "Schedule my store visit on weekdays", None, _Vectors()
        )
    }
    assert "06:20" in blocks["standing_rules"] and "06:50" not in blocks["standing_rules"]
    assert "ZB1871" in blocks["preferences"] and "JN4786" not in blocks["preferences"]


STOCK_RULE_TASK = "Create a stock rule for ST-101 at S40, min 1 max 3."


@pytest.mark.parametrize(
    ("rule", "forbidden"),
    [
        ("Never create a stock rule at store S40", True),
        ("Never create stock rules at store S40.", True),
        ("Never create stock rules", True),
        ("Never create a stock rule at store S12", False),
        ("Never create a stock rule for ST-202 at store S40", False),
        ("Always create a stock rule at store S40", False),
        ("Never delete a stock rule at store S40", False),
        ("Nunca llames a create_stock_rule para la tienda S40", True),
        ("Nie create_stock_rule für Filiale S40 aufrufen.", True),
        ("لا تستدعِ create_stock_rule أبدًا للمتجر S40", True),
        ("Nunca llames a create_stock_rule para la tienda S12", False),
    ],
)
def test_a_never_rule_forbids_the_action_it_says_on_the_values_it_names(
    rule: str, forbidden: bool
) -> None:
    assert standing.forbids(rule, "create_stock_rule", [STOCK_RULE_TASK]) is forbidden


@pytest.mark.asyncio
async def test_a_replaced_never_rule_is_not_listed_and_forbids_nothing() -> None:
    """The hints read the rules the section lists: a "never" rule a later one on the same
    subject superseded, or one a later statement replaced (an earlier value), is not one."""
    newer = "Create stock rules at store S40 only with my approval from now on."
    hits = [
        _hit("a", "Never create a stock rule at store S40.", "rule", day="2026-09-01"),
        _hit("b", newer, "rule", day="2026-10-01"),
    ]
    blocks = await standing.standing_blocks(_Engine(hits), STOCK_RULE_TASK, None, _Vectors())
    assert standing.listed_rules(blocks) == [newer]
    assert not standing.forbids(newer, "create_stock_rule", [STOCK_RULE_TASK])
    earlier = [_hit("a", "Never create a stock rule at store S40.", "rule", earlier_value=True)]
    blocks = await standing.standing_blocks(_Engine(earlier), STOCK_RULE_TASK, None, _Vectors())
    assert standing.listed_rules(blocks) == []
