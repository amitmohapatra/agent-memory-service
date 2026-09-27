"""OCR scoring must retain non-Latin errors and report numbers independently."""

import pytest
from benchmark.multilingual_ocr import character_errors

pytestmark = pytest.mark.unit


def test_ocr_score_ignores_layout_but_does_not_erase_non_latin_content():
    assert character_errors("北京 2022", "北京\n2022") == {
        "edits": 0,
        "expected_characters": 6,
        "character_error_rate": 0,
        "all_numbers_preserved": True,
    }
    changed = character_errors("北京 2022", "上海 2023")
    assert changed["edits"] == 3 and not changed["all_numbers_preserved"]
    assert character_errors("مكتب", "")["character_error_rate"] == 1


def test_insertions_and_unicode_normalization_are_scored():
    assert character_errors("café", "cafe\u0301")["edits"] == 0
    assert character_errors("abc", "abcde")["edits"] == 2
    with pytest.raises(ValueError, match="character budget"):
        character_errors("a", "b" * 12001)
