"""Decisions on tool calls stored before redaction at intake keep only what their tool allows.

Until this version a decision (``POST /v1/feedback`` on a ``tool_call``) was stored with the
raw arguments it was sent (``metadata.args``, and an edit's ``correction``), a value the
tool's ``redact`` paths keep out of its recorded calls included, and with any
``metadata.args_hash`` the client put there, which is now read as the call's hash. Each such
row is rewritten as intake writes one today (``FeedbackService._as_its_tool_redacts``):

* ``metadata.args_hash`` is the hash of the arguments as stored (``stable_hash``: raw, on
  these rows), whatever the client sent; a row with no arguments keeps no hash;
* ``metadata.args`` and ``correction`` are redacted by the tool's catalog entry (the
  workspace's, else the tenant's) when the catalog holds the tool. A tool it does not hold
  declared nothing to redact: the row keeps its arguments, and is counted in the log.

A row intake already wrote (its hash of the raw arguments, its arguments carrying the
``[redacted]`` marker at a path its tool redacts) is left as it is, so the migration is
idempotent: run again, or resumed after a failure, it rewrites nothing twice. Rows are read
in batches of ``BATCH`` by ``(tenant_id, feedback_id)``, each row's update committed on its
own, so no lock outlives one row's update. Downgrade keeps the rows: redacted values cannot be restored.
"""

import json
import logging
from typing import Any

import sqlalchemy as sa
from alembic import op

from memory_service.domain.tools import REDACTED, ToolDescriptor, stable_hash

revision = "0039_redact_stored_decisions"
down_revision = "0038_attachment_document_index"
branch_labels = None
depends_on = None

BATCH = 500
log = logging.getLogger("alembic.runtime.migration")

_ROWS = sa.text(
    """
    SELECT tenant_id, feedback_id, workspace_id, metadata, correction FROM feedback
    WHERE target_kind = 'tool_call' AND (tenant_id, feedback_id) > (:tenant, :feedback)
    ORDER BY tenant_id, feedback_id
    LIMIT :batch
    """
)
_TOOL = sa.text(
    """
    SELECT redact FROM tools
    WHERE tenant_id = :tenant AND name = :name AND workspace_id IN ('', :workspace)
    ORDER BY workspace_id DESC
    LIMIT 1
    """
)
_WRITE = sa.text(
    "UPDATE feedback SET metadata = CAST(:metadata AS jsonb), correction = CAST(:correction AS"
    " jsonb) WHERE tenant_id = :tenant AND feedback_id = :feedback"
)


def _kept_by_intake(descriptor: ToolDescriptor, args: dict[str, Any], kept: Any) -> bool:
    """Whether intake already wrote this row: its arguments are its tool's redacted copy,
    carrying the marker, and its hash is of other (the raw) arguments."""
    return (
        isinstance(kept, str)
        and descriptor.redacted_args(args) == args
        and kept != stable_hash(args)
        and json.dumps(REDACTED) in json.dumps(args)
    )


def rewritten(
    metadata: dict[str, Any], correction: Any, descriptor: ToolDescriptor | None
) -> tuple[dict[str, Any], Any]:
    """One decision's ``metadata`` and ``correction`` as intake writes them today."""
    out = dict(metadata)
    args = out.get("args")
    if not isinstance(args, dict):
        out.pop("args_hash", None)
    elif descriptor is None:
        out["args_hash"] = stable_hash(args)
    elif not _kept_by_intake(descriptor, args, out.get("args_hash")):
        out.update({"args": descriptor.redacted_args(args), "args_hash": stable_hash(args)})
    if descriptor is not None and isinstance(correction, dict):
        correction = descriptor.redacted_args(correction)
    return out, correction


def upgrade() -> None:
    counts = {"read": 0, "rewritten": 0, "unknown_tool": 0}
    after = ("", "")
    with op.get_context().autocommit_block():  # each update commits on its own
        bind = op.get_bind()
        while True:
            rows = bind.execute(
                _ROWS, {"tenant": after[0], "feedback": after[1], "batch": BATCH}
            ).all()
            for row in rows:
                counts["read"] += 1
                metadata = dict(row.metadata or {})
                tool = str(metadata.get("tool") or "").strip()
                redact = bind.execute(
                    _TOOL,
                    {"tenant": row.tenant_id, "name": tool, "workspace": row.workspace_id or ""},
                ).scalar_one_or_none()
                descriptor = (
                    ToolDescriptor(tenant_id=row.tenant_id, name=tool, redact=list(redact))
                    if redact is not None
                    else None
                )
                if descriptor is None and isinstance(metadata.get("args"), dict):
                    counts["unknown_tool"] += 1
                kept, correction = rewritten(metadata, row.correction, descriptor)
                if (kept, correction) != (metadata, row.correction):
                    counts["rewritten"] += 1
                    bind.execute(
                        _WRITE,
                        {
                            "metadata": json.dumps(kept),
                            "correction": None if correction is None else json.dumps(correction),
                            "tenant": row.tenant_id,
                            "feedback": row.feedback_id,
                        },
                    )
            if len(rows) < BATCH:
                break
            after = (rows[-1].tenant_id, rows[-1].feedback_id)
    log.info(
        "decisions on tool calls: %(read)d read, %(rewritten)d rewritten, %(unknown_tool)d"
        " kept as sent (tool not in the catalog)",
        counts,
    )


def downgrade() -> None:
    """Redacted values cannot be restored; the rows stay as they are."""
