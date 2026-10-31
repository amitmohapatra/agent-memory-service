"""The capability gate's learned-skill scenario sees a regression: with a five-tool toolbox
the Tools section names every tool, so naming the steps proved nothing. Pure: no service."""

from __future__ import annotations

from types import SimpleNamespace

from benchmark.capabilities.dataset import load_dataset
from benchmark.capabilities.scoring import items_from_bundle, score


def _mfa_expect() -> dict:
    for cap in load_dataset().capabilities:
        for s in cap.scenarios:
            if s.id == "procedure-it-mfa-skill":
                return dict(s.queries[0].expect)
    raise AssertionError("procedure-it-mfa-skill is missing")


TOOLS = ["iam-lookup_user", "iam-reset_mfa", "mail-send_notice", "crm-read_ticket", "wiki-search"]


def test_the_mfa_scenario_fails_when_no_skill_was_learned() -> None:
    """B14: after the runner's system labels stopped teaching, 0 procedures were learned and
    the scenario still passed 10/10: its tool names came from the hints alone."""
    hints_only = {"tools": [{"name": n} for n in TOOLS], "skills": []}
    assert not score(_mfa_expect(), items_from_bundle(hints_only), "en", surface="context").passed
    skill = {
        "name": "reset-multi-factor-authentication-for-a-locked-out-user",
        "steps": ["iam-lookup_user", "iam-reset_mfa", "mail-send_notice"],
    }
    learned = {**hints_only, "skills": [skill]}
    assert score(_mfa_expect(), items_from_bundle(learned), "en", surface="context").passed


class _Ctx:
    """An SDK context that records what it was asked to do, and by which credential."""

    def __init__(self, key: str, sent: list[tuple], **scope: str) -> None:
        self.key, self.sent, self.scope_fields = key, sent, scope

    def bind(self, **scope: str) -> _Ctx:
        return _Ctx(self.key, self.sent, **{**self.scope_fields, **scope})

    def agent(self, agent_id: str, *, agent_run_id: str) -> _Ctx:
        return self.bind(agent_id=agent_id, agent_run_id=agent_run_id)

    @property
    def scope(self) -> SimpleNamespace:
        return SimpleNamespace(agent_run_id=self.scope_fields.get("agent_run_id"))

    async def record_tool(self, tool: str, args: dict, **_: object) -> None:
        self.sent.append(("record_tool", self.key, tool))

    async def feedback(self, kind: str, target: str, verdict: str, *, source: str) -> None:
        self.sent.append(("feedback", self.key, self.scope_fields, kind, target, verdict, source))


async def test_the_runner_labels_its_tool_runs_as_a_person_through_the_admin_key() -> None:
    """B14: a run's own ``system`` report of success teaches no procedure, so the runner's
    tool runs are labelled by a person through the tenant's administrator credential: applied
    as it arrives, and learned from. The calls are still recorded as the agent."""
    from benchmark.capabilities.runner import Service, Written, _tool_runs

    scenario = next(
        s
        for cap in load_dataset().capabilities
        for s in cap.scenarios
        if s.id == "procedure-it-mfa-skill"
    )
    session = next(s for s in scenario.sessions if s.tool_runs)
    sent: list[tuple] = []
    service = Service(_Ctx("service", sent), None, _Ctx("admin", sent))
    w = Written(scenario, "en", "u-1")
    await _tool_runs(service, w, session, "t-1", "capeval-nollm")
    labels = [x for x in sent if x[0] == "feedback"]
    assert labels and all(key == "admin" for _, key, *_ in labels)
    assert all(x[-1] == "human" and x[-2] == "confirm" for x in labels)
    assert all(x[2]["tenant_id"] == "capeval-nollm" and x[2]["user_id"] == "u-1" for x in labels)
    assert {x[4] for x in labels} == {f"run-u-1-{session.id}-{n}" for n in range(len(labels))}
    assert {key for kind, key, *_ in sent if kind == "record_tool"} == {"service"}
