"""A benchmark's pinned settings must survive the environment that configures its gateway.

``_settings`` used to discover which values came from the environment with
``Settings().model_dump(exclude_unset=True)``. On a pydantic-settings object that does not
mean what it says: every field counts as set because a settings *source* provided it, so the
dump came back complete - every section, every default inlined - and overlaying it onto the
benchmark's defaults overwrote them all.

The judged LoCoMo configuration intends ``max_tokens=16384, timeout_seconds=120,
max_retries=0`` and was getting ``1024, 30.0, 2`` on every run ever taken. A reasoning model
given 1024 tokens spends them thinking and returns an empty string, which is the "output
budget was exhausted" failure that lost 19 of v6's 233 answerable rows; and retries at 2 sends
up to three wire requests per logical call, which is why the pacer's rate was never the
actual request rate.
"""

from __future__ import annotations

import os

import pytest
from benchmark import env as bench_env
from benchmark.retrieval import _env_paths, _overlay, _settings

pytestmark = pytest.mark.unit

#: What the Makefile exports for a judged run: the gateway's address and model, nothing about
#: budgets or retries - those are the benchmark's to pin.
JUDGED_ENV = {
    "MEMORY__MODELS__LLM__ENABLED": "true",
    "MEMORY__MODELS__LLM__MODEL": "deepseek/deepseek-flash",
    "MEMORY__MODELS__LLM__FAST_MODEL": "deepseek/deepseek-flash",
    "MEMORY__MODELS__LLM__USES": '["grounding_judge"]',
    "MEMORY__MODELS__LLM__FAST_USES": '["grounding_judge"]',
}


@pytest.fixture
def judged(monkeypatch: pytest.MonkeyPatch) -> None:
    """A judged run's environment.

    ``benchmark.env.BENCH`` is built once at import from the environment, so setting
    ``BENCH_DEPTH`` here is too late on its own - the singleton has to be rebuilt after the
    variable exists. That is a fact about the module, not about the thing under test.
    """
    for name in [n for n in list(os.environ) if n.startswith("MEMORY__")]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("BENCH_DEPTH", "judged")
    for name, value in JUDGED_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(bench_env, "BENCH", bench_env.BenchEnv.from_environ())


def test_the_judged_budget_survives_the_gateway_variables(judged: None) -> None:
    """The defect, pinned. Setting the model must not reset the output ceiling."""
    llm = _settings().models.llm
    assert llm.max_tokens == 16384, "a reasoning model at 1024 returns an empty string"
    assert llm.timeout_seconds == 120
    assert llm.max_retries == 0, "retries make the pacer's rate not the request rate"


def test_the_environment_still_wins_for_what_it_names(judged: None) -> None:
    """The other half: this must not become a merge that ignores the environment."""
    llm = _settings().models.llm
    assert llm.enabled is True
    assert llm.model == "deepseek/deepseek-flash"
    assert llm.uses == ["grounding_judge"]


def test_defaults_outside_the_models_section_survive_too(judged: None) -> None:
    """``service`` was being reset by the same mechanism, unnoticed."""
    settings = _settings()
    assert settings.service.environment == "test"
    assert settings.service.log_level == "WARNING"


def test_env_paths_reads_names_not_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMORY__MODELS__LLM__MODEL", "x")
    monkeypatch.setenv("MEMORY__DATABASE__URL", "y")
    paths = _env_paths()
    assert ("models", "llm", "model") in paths
    assert ("database", "url") in paths


def test_an_unknown_variable_does_not_invent_a_setting() -> None:
    """A typo must stay a typo, not become a validation error somewhere unrelated."""
    target: dict = {"models": {"llm": {"model": "kept"}}}
    _overlay(target, {"models": {"llm": {"model": "env"}}}, ("models", "llm", "nonesuch"))
    assert target == {"models": {"llm": {"model": "kept"}}}


def test_a_branch_the_defaults_lack_is_created() -> None:
    target: dict = {}
    _overlay(target, {"models": {"llm": {"model": "env"}}}, ("models", "llm", "model"))
    assert target == {"models": {"llm": {"model": "env"}}}
