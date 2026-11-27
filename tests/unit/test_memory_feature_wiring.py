"""Write features can be measured independently; disabled modes build no SDK client."""

import pytest

pytestmark = pytest.mark.unit


def test_defaults_keep_unmeasured_features_off(make_settings):
    settings = make_settings()
    assert not settings.llm.wants("contextual_extraction")
