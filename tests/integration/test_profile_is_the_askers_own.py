"""The profile's rules and preferences are the asking person's own. A statement another person
made readable to the whole tenant is something they said - one of the memories when it bears
on the question - never the asker's rule: a benchmark conversation stored tenant-wide listed
one speaker's "Keep it up!" as the rule of whoever asked."""

from __future__ import annotations

import pytest

from memory_service.domain.enums import Visibility
from memory_service.domain.observation import ProcessingHints
from tests.integration.test_memory import U1, U2, _observe

pytestmark = pytest.mark.integration

RULE = "Never change a retail price without my sign-off."
ASKED = "Can you change the retail price of the tote bag?"


def _listed(bundle) -> str:
    return "\n".join(
        b.text for b in bundle.profile if b.block in ("standing_rules", "preferences", "about_user")
    )


async def test_a_rule_another_person_shared_is_not_the_askers(container, uow_factory) -> None:
    await _observe(
        container, uow_factory, U1, RULE, hints=ProcessingHints(visibility=Visibility.TENANT)
    )
    builder = container.services["context_builder"]
    own = await builder.build(U1, ASKED)
    assert RULE.rstrip(".") in _listed(own), "the person who said it: their rule"
    other = await builder.build(U2, ASKED)
    assert RULE.rstrip(".") not in _listed(other), "someone else's rule is not the asker's"
