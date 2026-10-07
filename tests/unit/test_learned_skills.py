"""Learned skills, piece by piece: where a call is learned (per agent, across its users), who
may read the result, how a skill is named and shown (in any script, with what fixed a failing
step, as an addition to the agent's own skill), the toolbox size that earns tool hints, and
dismissing one."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from memory_service.domain.context_bundle import skills_section
from memory_service.domain.errors import NotFound
from memory_service.domain.revisions import RevisionKind
from memory_service.domain.tools import (
    SkillView,
    StoredProcedure,
    ToolInvocation,
    agent_audience,
)
from memory_service.modules.context.sections import TOOL_HINTS_MIN, ToolsRequest
from memory_service.modules.tools.learning import audience_of, contributors, merge
from memory_service.modules.tools.procedures import Procedure, ProcedureStep
from memory_service.modules.tools.skills import (
    LearnedSkills,
    fixes,
    learned_skill,
    skill_name,
    skill_view,
    task_steps,
    with_skill,
)

pytestmark = pytest.mark.unit


def _procedure(**kw) -> StoredProcedure:
    base = {
        "procedure_id": "prc_1",
        "tenant_id": "acme",
        "scope_key": agent_audience("acme", "support"),
        "pattern": "update quote {id} with {entity} price for {id}",
        "title": "Update a quote's regional price",
        "steps": [
            {"ordinal": 0, "tool": "lookup_price", "expected_output": ["price", "quote"]},
            {
                "ordinal": 1,
                "tool": "update_quote",
                "failure_modes": {"Locked": "ask the owner to unlock it", "Other": ""},
            },
        ],
        "support": 4,
        "success_rate": 0.75,
        "status": "active",
        "steps_hash": "h1",
        "agent_id": "support",
        "users": 2,
    }
    return StoredProcedure(**{**base, **kw})


def _with_skill_steps() -> dict:
    return {
        "steps": [
            {"ordinal": 0, "tool": "load_skill"},
            {"ordinal": 1, "tool": "find_order"},
            {"ordinal": 2, "tool": "read_skill_file"},
            {"ordinal": 3, "tool": "refund"},
        ],
        "bindings": [
            {"step": 0, "argument": "name", "literal": "refund-policy"},
            {"step": 3, "argument": "payment", "source_step": 1, "source_field": "payment"},
        ],
    }


def _call(**kw) -> ToolInvocation:
    base = {
        "tenant_id": "acme",
        "tool_id": "tool_1",
        "tool_name": "refund",
        "agent_id": "support",
        "user_id": "ann",
        "visibility_keys": ["principal:acme/agent:ann/support"],
        "task_pattern": "refund order {id}",
    }
    return ToolInvocation(**{**base, **kw})


# --- where a call is learned -------------------------------------------------------------


def test_an_agents_own_calls_are_learned_together_whichever_user_it_ran_for() -> None:
    ann = _call()
    bob = _call(user_id="bob", visibility_keys=["principal:acme/agent:bob/support"])
    assert audience_of(ann) == audience_of(bob) == (agent_audience("acme", "support"), "support")


def test_an_agent_with_no_user_is_learned_with_the_agent() -> None:
    unattended = _call(user_id=None, visibility_keys=["principal:acme/agent:support"])
    assert audience_of(unattended) == (agent_audience("acme", "support"), "support")


def test_calls_shared_wider_and_calls_of_no_agent_keep_their_own_audience() -> None:
    shared = _call(visibility_keys=["workspace:acme/ops", "principal:acme/agent:ann/support"])
    assert audience_of(shared) == ("workspace:acme/ops", None)
    by_user = _call(agent_id=None, visibility_keys=["principal:acme/user:ann"])
    assert audience_of(by_user) == ("principal:acme/user:ann", None)


def test_another_agents_name_in_the_key_is_not_the_agents_own_record() -> None:
    # a record written as "support" but keyed to another principal stays where it was keyed
    forged = _call(visibility_keys=["principal:acme/agent:ann/other"])
    assert audience_of(forged) == ("principal:acme/agent:ann/other", None)


def test_contributors_count_distinct_users_and_name_the_only_one() -> None:
    assert contributors([_call(), _call()]) == (1, "ann")
    assert contributors([_call(), _call(user_id="bob")]) == (2, None)
    assert contributors([_call(user_id=None)]) == (1, None)
    assert contributors([_call(), _call(user_id=None)]) == (2, None)


def _mined(support: int = 3, rate: float = 1.0) -> Procedure:
    return Procedure(
        task_pattern="refund order {id}",
        steps=[
            ProcedureStep(ordinal=0, tool="find_order"),
            ProcedureStep(ordinal=1, tool="refund"),
        ],
        support=support,
        success_rate=rate,
    )


def test_a_merge_records_the_agent_and_who_produced_it() -> None:
    merged = merge(
        None,
        _mined(),
        tenant_id="acme",
        audience=agent_audience("acme", "support"),
        owner=_call(),
        agent_id="support",
        users=1,
        sole_user="ann",
    )
    assert merged is not None
    assert (merged.agent_id, merged.users, merged.sole_user) == ("support", 1, "ann")
    assert merged.status == "active"


def test_a_dismissed_skill_stays_dismissed_until_its_steps_change() -> None:
    first = merge(None, _mined(), tenant_id="acme", audience="a", owner=_call(), agent_id="support")
    assert first is not None
    dismissed = first.model_copy(update={"status": "rejected"})
    again = merge(
        dismissed, _mined(), tenant_id="acme", audience="a", owner=_call(), agent_id="support"
    )
    assert again is not None and again.status == "rejected"
    changed = Procedure(
        task_pattern="refund order {id}",
        steps=[ProcedureStep(ordinal=0, tool="refund")],
        support=3,
        success_rate=1.0,
    )
    offered = merge(
        dismissed, changed, tenant_id="acme", audience="a", owner=_call(), agent_id="support"
    )
    assert offered is not None and offered.status == "active"


# --- how a skill is named and shown ------------------------------------------------------


def test_a_skill_is_named_from_its_title_in_lowercase_words() -> None:
    assert skill_name(_procedure()) == "update-a-quote-s-regional-price"
    assert skill_name(_procedure(title="")) == "update-quote-with-price-for"


@pytest.mark.parametrize(
    ("title", "name"),
    [
        ("Rückerstattung einer Bestellung", "rückerstattung-einer-bestellung"),
        ("استرداد الطلب", "استرداد-الطلب"),
        ("ऑर्डर का रिफंड", "ऑर्डर-का-रिफंड"),
        ("退款", "退款"),
    ],
)
def test_a_skill_is_named_in_the_script_its_task_is_in(title: str, name: str) -> None:
    assert skill_name(_procedure(title=title)) == name


def test_a_name_is_cut_at_a_word_and_never_empty() -> None:
    long = skill_name(_procedure(title=" ".join(["refund"] * 30)))
    assert len(long) <= 64 and not long.endswith("-")
    assert skill_name(_procedure(title="", pattern="{id} {id}")) == "procedure"


def test_the_steps_shown_leave_out_the_skill_machinery() -> None:
    procedure = _procedure(**_with_skill_steps())
    assert task_steps(procedure) == ["find_order", "refund"]


def test_a_skill_opened_by_every_run_is_what_the_learned_one_adds_to() -> None:
    assert with_skill(_procedure(**_with_skill_steps())) == "refund-policy"
    assert with_skill(_procedure()) is None
    varied = _with_skill_steps()
    varied["bindings"][0] = {"step": 0, "argument": "name", "literal": None}
    assert with_skill(_procedure(**varied)) is None, "runs opened different skills"


def test_fixes_are_what_worked_when_a_step_failed() -> None:
    assert fixes(_procedure()) == ["update_quote on Locked: ask the owner to unlock it"]


def test_the_view_and_the_listing_carry_the_same_skill() -> None:
    procedure = _procedure(**_with_skill_steps())
    view = skill_view(procedure)
    assert (view.name, view.steps, view.with_skill) == (
        "update-a-quote-s-regional-price",
        ["find_order", "refund"],
        "refund-policy",
    )
    listed = learned_skill(procedure.model_copy(update={"status": "rejected"}))
    assert listed.status == "dismissed" and listed.steps == view.steps
    assert learned_skill(procedure.model_copy(update={"status": "retired"})).status == "retired"


def test_the_context_shows_each_skill_in_full_and_an_addition_as_one() -> None:
    plain = skill_view(_procedure())
    added = skill_view(_procedure(procedure_id="prc_2", **_with_skill_steps()))
    text = skills_section([plain, added])
    assert text is not None
    assert text.startswith("## Learned skills for this task\n")
    assert (
        "- update-a-quote-s-regional-price: lookup_price -> update_quote (worked 75% of 4 runs)"
        in text
    )
    assert "  - if update_quote on Locked: ask the owner to unlock it" in text
    assert "- adds to your skill refund-policy: find_order -> refund" in text
    assert skills_section([]) is None


# --- which toolbox earns tool hints ------------------------------------------------------


def test_tool_hints_need_a_toolbox_of_five_or_the_catalog_skills_need_any_tool() -> None:
    small = ToolsRequest(available=["a", "b"])
    assert small.any and not small.hinted
    assert ToolsRequest(available=[str(i) for i in range(TOOL_HINTS_MIN)]).hinted
    assert ToolsRequest(available=None).hinted
    none = ToolsRequest(available=[])
    assert not none.any and not none.hinted
    unhinted = ToolsRequest(available=[str(i) for i in range(8)], hints=False)
    assert unhinted.any and not unhinted.hinted, "learned skills only"
    assert (
        unhinted.fingerprint() != ToolsRequest(available=[str(i) for i in range(8)]).fingerprint()
    )


# --- dismissing --------------------------------------------------------------------------


class _Procedures:
    def __init__(self, found: StoredProcedure | None) -> None:
        self.found = found
        self.rejected: list[str] = []

    async def get(self, tenant_id: str, procedure_id: str) -> StoredProcedure | None:
        return self.found

    async def reject(self, tenant_id: str, procedure_id: str) -> bool:
        self.rejected.append(procedure_id)
        assert self.found is not None
        self.found = self.found.model_copy(update={"status": "rejected"})
        return True

    async def learned(self, tenant_id: str, *, agent_id: str | None, limit: int) -> list:
        return [self.found] if self.found and (agent_id in (None, self.found.agent_id)) else []


class _Revisions:
    def __init__(self) -> None:
        self.bumped: list[tuple] = []

    async def bump(self, tenant_id: str, kind: RevisionKind, identifier: str) -> None:
        self.bumped.append((tenant_id, kind, identifier))


class _Uow:
    def __init__(self, found: StoredProcedure | None) -> None:
        self.procedures = _Procedures(found)
        self.revisions = _Revisions()
        self.commits = 0

    async def __aenter__(self) -> _Uow:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def commit(self) -> None:
        self.commits += 1


async def test_dismissing_rejects_it_once_and_rereads_the_contexts_that_offered_it() -> None:
    uow = _Uow(_procedure())
    skills = LearnedSkills(lambda: uow)  # type: ignore[arg-type,return-value]
    dismissed = await skills.dismiss("acme", "prc_1")
    assert dismissed.status == "dismissed"
    assert uow.procedures.rejected == ["prc_1"]
    assert uow.revisions.bumped == [("acme", RevisionKind.TENANT, "")]
    again = await skills.dismiss("acme", "prc_1")
    assert again.status == "dismissed" and uow.procedures.rejected == ["prc_1"], "no-op"


@pytest.mark.parametrize("found", [None, _procedure(status="candidate")])
async def test_there_is_nothing_to_dismiss_before_it_was_learned(found) -> None:
    with pytest.raises(NotFound):
        await LearnedSkills(lambda: _Uow(found)).dismiss("acme", "prc_1")  # type: ignore[arg-type,return-value]


async def test_the_listing_is_one_agents_when_asked() -> None:
    uow = _Uow(_procedure(updated_at=datetime(2026, 10, 1, tzinfo=UTC)))
    skills = LearnedSkills(lambda: uow)  # type: ignore[arg-type,return-value]
    assert [s.name for s in await skills.list("acme", agent_id="support")] == [
        "update-a-quote-s-regional-price"
    ]
    assert await skills.list("acme", agent_id="billing") == []


def test_a_skill_view_is_frozen() -> None:
    view = SkillView(id="prc_1", name="x")
    with pytest.raises(Exception):  # noqa: B017 - pydantic's frozen error
        view.name = "y"  # type: ignore[misc]
