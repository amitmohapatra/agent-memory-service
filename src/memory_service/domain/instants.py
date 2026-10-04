"""Instants a caller sends: ISO 8601, and UTC when they name no offset.

The store compares against ``timestamptz`` and the domain against aware datetimes, so a
naive value (``2026-09-14T10:00:00``) used to be either refused by one model, compared
against an aware one and raised (a 500), or read in whatever zone the process ran in. Every
request instant goes through :data:`UtcDateTime` instead: an offset is kept, its absence
means UTC (ADR 0030).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Final

from pydantic import AfterValidator

#: The phrase every request instant's description ends with.
UTC_RULE: Final = "ISO 8601; a value without an offset is read as UTC."


def as_utc(instant: datetime) -> datetime:
    """``instant`` with its offset, or as UTC when it has none."""
    return instant.replace(tzinfo=UTC) if instant.tzinfo is None else instant


UtcDateTime = Annotated[datetime, AfterValidator(as_utc)]
