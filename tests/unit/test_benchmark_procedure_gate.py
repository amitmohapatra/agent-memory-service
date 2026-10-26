"""The capability gate's learned-skill scenario sees a regression: with a five-tool toolbox
the Tools section names every tool, so naming the steps proved nothing. Pure: no service."""

from __future__ import annotations

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
