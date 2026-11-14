"""SEC-R-10 for the rows stored before redaction at intake (migration 0039).

A decision on a tool call was stored with the raw arguments it was sent and with any
``metadata.args_hash`` the client put there, which is now read as the call's hash. The
migration rewrites each such row as intake writes one, in batches, and is idempotent.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text

from memory_service.domain.tools import stable_hash
from tests.conftest import DB_URL

pytestmark = pytest.mark.integration

TOKEN = "tok-live-9f31c2a7"
RAW = {"quote": "Q-1", "auth": {"scheme": "bearer", "token": TOKEN}}
KEPT = {"quote": "Q-1", "auth": {"scheme": "bearer", "token": "[redacted]"}}
PLAIN = {"quote": "Q-2"}

#: (feedback_id, verdict, tool, metadata beyond the tool, correction)
OLD_ROWS: list[tuple[str, str, str, dict[str, Any], Any]] = [
    ("fb_raw", "approve", "crm-update_quote", {"args": RAW}, None),
    ("fb_forged", "approve", "crm-update_quote", {"args": RAW, "args_hash": "f" * 64}, None),
    ("fb_edit", "edit", "crm-update_quote", {"args": RAW}, RAW),
    ("fb_unknown", "reject", "uncatalogued", {"args": PLAIN, "args_hash": "f" * 64}, None),
    ("fb_no_args", "approve", "crm-update_quote", {"args_hash": "f" * 64}, None),
    # what intake writes today: left as it is
    (
        "fb_intake",
        "approve",
        "crm-update_quote",
        {"args": KEPT, "args_hash": stable_hash(RAW)},
        None,
    ),
]


async def _seed(engine) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO tools (tool_id, tenant_id, name, schema_hash, redact)"
                " VALUES ('tol_m', 'acme', 'crm-update_quote', 'h', '[\"auth.token\"]')"
            )
        )
        for feedback_id, verdict, tool, metadata, correction in OLD_ROWS:
            await conn.execute(
                text(
                    "INSERT INTO feedback (tenant_id, feedback_id, target_kind, target_id,"
                    " verdict, source, metadata, correction, created_at) VALUES ('acme', :id,"
                    " 'tool_call', 'call_1', :verdict, 'interrupt', CAST(:metadata AS jsonb),"
                    " CAST(:correction AS jsonb), now())"
                ),
                {
                    "id": feedback_id,
                    "verdict": verdict,
                    "metadata": json.dumps({"tool": tool, **metadata}),
                    "correction": None if correction is None else json.dumps(correction),
                },
            )


#: more rows than one batch of the migration (``BATCH`` = 500): read by the keyset
MANY = 600


async def _seed_many(engine) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO feedback (tenant_id, feedback_id, target_kind, target_id, verdict,"
                " source, metadata, created_at) SELECT 'bulk', 'fb_' || lpad(g::text, 4, '0'),"
                " 'tool_call', 'call_' || g, 'reject', 'interrupt',"
                " jsonb_build_object('tool', 'crm-update_quote', 'args',"
                " jsonb_build_object('n', g), 'args_hash', 'forged'), now()"
                " FROM generate_series(1, :n) AS g"
            ),
            {"n": MANY},
        )


async def _forged_left(engine) -> int:
    async with engine.begin() as conn:
        found = await conn.execute(
            text("SELECT count(*) FROM feedback WHERE metadata->>'args_hash' = 'forged'")
        )
        return int(found.scalar_one())


async def _stored(engine) -> dict[str, tuple[dict[str, Any], Any]]:
    async with engine.begin() as conn:
        rows = await conn.execute(
            text("SELECT feedback_id, metadata, correction FROM feedback WHERE tenant_id = 'acme'")
        )
        return {r.feedback_id: (r.metadata, r.correction) for r in rows}


async def _migrate_again() -> None:
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", DB_URL)
    await asyncio.to_thread(command.downgrade, cfg, "0038_attachment_document_index")
    await asyncio.to_thread(command.upgrade, cfg, "head")


async def test_old_decisions_are_rewritten_as_intake_writes_them(container) -> None:
    engine = container.database.engine
    await _seed(engine)
    await _seed_many(engine)
    before = await _stored(engine)
    await _migrate_again()
    after = await _stored(engine)

    assert after["fb_raw"][0]["args"] == KEPT
    assert after["fb_raw"][0]["args_hash"] == stable_hash(RAW)
    assert after["fb_forged"][0]["args_hash"] == stable_hash(RAW), "the client's hash is gone"
    assert after["fb_edit"][1] == KEPT
    # the catalog does not hold the tool: kept as sent, the hash of what is stored
    assert after["fb_unknown"][0]["args"] == PLAIN
    assert after["fb_unknown"][0]["args_hash"] == stable_hash(PLAIN)
    assert "args_hash" not in after["fb_no_args"][0]
    assert after["fb_intake"] == before["fb_intake"]
    assert await _forged_left(engine) == 0, "every batch is read"
    assert TOKEN not in json.dumps({k: v for k, v in after.items() if k != "fb_unknown"})

    await _migrate_again()  # run again, or resumed: nothing is rewritten twice
    assert await _stored(engine) == after
