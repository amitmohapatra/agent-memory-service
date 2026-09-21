"""Control-character stripping, and that every path that stores agent text uses it."""

from __future__ import annotations

from memory_service.domain.text import sanitise


def test_nul_bytes_are_removed() -> None:
    # PostgreSQL rejects these outright; one of them used to fail the whole INSERT
    assert sanitise("user said\x00\x00 hello") == "user said hello"


def test_whitespace_is_preserved() -> None:
    assert sanitise("line one\nline\ttwo\r\n") == "line one\nline\ttwo\r\n"


def test_c1_block_and_delete_are_removed() -> None:
    assert sanitise("a\x7fb\x9fc") == "abc"


def test_ordinary_text_is_untouched() -> None:
    text = "The FOMC held the rate at 5.25–5.50% — 日本語 also fine 🙂"  # noqa: RUF001
    assert sanitise(text) == text
