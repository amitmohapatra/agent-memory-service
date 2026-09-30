"""A visibility the context cannot express must be refused when the write is made.

Audience keys are built from the context's anchors, so AGENT_GROUP without an agent group
(or WORKSPACE without a workspace, and so on) cannot be expressed. Accepting such a memory
and failing later loses the write silently: the caller has its acknowledgement and no memory
ever appears. ``remember`` (and the ``memory_remember`` agent tool) check it first.
"""

from __future__ import annotations

import pytest

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import Visibility
from memory_service.domain.errors import ValidationFailed
from memory_service.domain.observation import ProcessingHints
from memory_service.modules.authz.visibility import validate_requested_visibility


def context(**fields) -> MemoryExecutionContext:
    return MemoryExecutionContext(
        tenant_id="acme", request_id="req-1", correlation_id="corr-1", trace_id="t" * 32, **fields
    )


@pytest.mark.parametrize(
    ("visibility", "missing"),
    [
        (Visibility.AGENT_GROUP, "agent_group_id"),
        (Visibility.USER, "user_id"),
        (Visibility.THREAD, "thread_id"),
        (Visibility.RUN, "agent_run_id"),
        (Visibility.WORKSPACE, "workspace_id"),
    ],
)
def test_unsatisfiable_visibility_is_rejected(visibility, missing):
    with pytest.raises(ValidationFailed) as exc:
        validate_requested_visibility(context(user_id=None), ProcessingHints(visibility=visibility))
    assert missing in str(exc.value)


def test_a_satisfiable_visibility_is_accepted():
    validate_requested_visibility(
        context(user_id="u1"), ProcessingHints(visibility=Visibility.USER)
    )


def test_no_visibility_is_left_to_the_scope():
    validate_requested_visibility(context(), None)
    validate_requested_visibility(context(), ProcessingHints())
