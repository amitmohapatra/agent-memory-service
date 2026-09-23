"""One place decides what lineage a stored row carries.

Fourteen fields were copied verbatim off the execution context onto an ``Observation`` at
three call sites - the conversation, ingestion and memory services. Each site was individually
correct, so nothing failed; the defect was only visible in their disagreement. Adding a field
to ``MemoryExecutionContext`` meant finding all three, and missing one would have dropped
provenance on exactly one ingest path, silently and permanently.

The last test is the one that matters: it fails when a new lineage field is added to both
types and left out of ``PROVENANCE_FIELDS``, which is the mistake this refactor exists to
make impossible.
"""

from __future__ import annotations

import pytest

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.observation import Observation

pytestmark = pytest.mark.unit

#: Shared by an Observation and a context, and deliberately NOT provenance.
#: ``created_at`` is when the observation was made, not when the request arrived - copying
#: the request's would backdate every row written later in the same request. ``custom_metadata``
#: is per-call: the message site puts the role and kind in it, so a blanket copy would
#: overwrite what the caller set.
NOT_PROVENANCE = {"created_at", "custom_metadata"}

CTX = MemoryExecutionContext(
    tenant_id="acme",
    workspace_id="ws1",
    user_id="u1",
    thread_id="thr1",
    session_id="ses1",
    turn_id="turn1",
    work_id="work1",
    task_id="task1",
    agent_id="research",
    agent_group_id="grp1",
    agent_run_id="run1",
    parent_agent_run_id="run0",
)


def test_provenance_carries_every_field_it_declares() -> None:
    provenance = CTX.provenance()
    assert set(provenance) == set(MemoryExecutionContext.PROVENANCE_FIELDS)


def test_every_value_is_the_context_s_own() -> None:
    for name, value in CTX.provenance().items():
        assert value == getattr(CTX, name), f"{name} was not copied off the context"


def test_the_computed_principal_is_carried_not_just_the_stored_fields() -> None:
    """``principal_id`` is a property, so a model_fields-driven copy would have missed it."""
    assert CTX.provenance()["principal_id"] == CTX.principal_id
    assert CTX.principal_id == "agent:u1/research"


def test_an_observation_accepts_the_whole_spread() -> None:
    """The call sites spread this straight in, so it must not carry an unknown key."""
    observation = Observation(**CTX.provenance(), kind="MESSAGE", content="hi", content_hash="h")
    for name in MemoryExecutionContext.PROVENANCE_FIELDS:
        assert getattr(observation, name) == getattr(CTX, name)


def test_a_new_lineage_field_cannot_be_added_to_only_one_of_the_two() -> None:
    """The guard. Adding a field to both types and not to PROVENANCE_FIELDS fails here.

    Without it the old failure mode returns in a new shape: the field exists, rows have a
    column for it, and it is never populated - with every individual call site still correct.
    """
    shared = (set(MemoryExecutionContext.model_fields) | {"principal_id"}) & set(
        Observation.model_fields
    )
    missing = shared - set(MemoryExecutionContext.PROVENANCE_FIELDS) - NOT_PROVENANCE
    assert not missing, (
        f"{sorted(missing)} is on both MemoryExecutionContext and Observation but is not "
        "carried by provenance(); add it to PROVENANCE_FIELDS, or to NOT_PROVENANCE here "
        "with the reason it is excluded"
    )
