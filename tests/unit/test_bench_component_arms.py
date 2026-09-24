"""A benchmark can be pointed at the real components it has always stood in for.

Every harness hard-coded `cache="memory"`, `authorization="memory"` and `nli="lexical"`, so no
benchmark in this repository has ever entered OpenFGA, Dragonfly or the DeBERTa head. That is
not a small simplification: the scope-cache invalidation defect - content writes discarding
the authorized scope 730 times over 369 ingested turns, each miss costing five sequential
ListObjects - survived every benchmark and first appeared as 503s in an HTTP load test,
because nothing before that had put OpenFGA on the path.

`None` in an override means "use the component the settings configure", which is the same
convention `search` already used for `BENCH_SEARCH=qdrant`.
"""

from __future__ import annotations

import pytest
from benchmark.env import BenchEnv

pytestmark = pytest.mark.unit


def test_the_default_still_stands_everything_in() -> None:
    """Unchanged defaults: every result already recorded stays comparable."""
    overrides = BenchEnv().overrides()
    assert overrides.authorization == "memory"
    assert overrides.cache == "memory"
    assert overrides.nli == "lexical"


@pytest.mark.parametrize(
    ("field", "value", "attribute"),
    [
        ("authorization", "openfga", "authorization"),
        ("cache", "dragonfly", "cache"),
        ("nli", "deberta", "nli"),
    ],
)
def test_asking_for_the_real_component_stops_standing_it_in(
    field: str, value: str, attribute: str
) -> None:
    overrides = BenchEnv(**{field: value}).overrides()  # type: ignore[arg-type]
    assert getattr(overrides, attribute) is None, (
        f"{field}={value} should defer to settings, not force a stand-in"
    )


def test_an_unknown_arm_is_refused_rather_than_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    """A typo must not silently leave the stand-in in place and label the run representative."""
    monkeypatch.setenv("BENCH_AUTHZ", "openfgaa")
    with pytest.raises(SystemExit):
        BenchEnv.from_environ()


def test_the_arms_are_read_from_their_own_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BENCH_AUTHZ", "openfga")
    monkeypatch.setenv("BENCH_CACHE", "dragonfly")
    monkeypatch.setenv("BENCH_NLI", "deberta")
    env = BenchEnv.from_environ()
    assert (env.authorization, env.cache, env.nli) == ("openfga", "dragonfly", "deberta")
