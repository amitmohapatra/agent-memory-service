"""Cross-script quantity guards must agree without dropping decimal separators."""

# ruff: noqa: RUF001 — literal multilingual fixtures.

import pytest

from memory_service.domain.text import normalise_number
from memory_service.modules.grounding.lexical import number_conflict
from memory_service.modules.grounding.lexical import numbers as grounding_numbers
from memory_service.modules.memory.native import numbers as memory_numbers

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("value", ["1,234.50", "١٬٢٣٤٫٥٠", "۱٬۲۳۴٫۵۰", "１，２３４．５０"])
def test_decimal_quantity_is_stable_across_scripts(value):
    assert normalise_number(value) == "1234.50"
    assert memory_numbers("Revenue: " + value) == {"1234.50"}
    assert grounding_numbers("Revenue: " + value) == {"1234.5"}
    assert not number_conflict("Revenue: " + value, "Revenue: 1234.50")
    assert number_conflict("Revenue: " + value, "Revenue: 1234.51")


def test_normalization_preserves_existing_leading_zero_and_comma_policy():
    assert normalise_number("001,234.50") == "001234.50"
    assert normalise_number("1,5") == "15"
    assert memory_numbers("Build 00123") == {"00123"}
