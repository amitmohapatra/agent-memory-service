"""Learned skills, piece by piece: a procedure's draft, its name and body, the version rule,
and the two stores it is published to (a skills folder, the gateway's repository)."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from memory_service.adapters.skills import (
    BifrostSkills,
    FolderSkills,
    front_matter_metadata,
    skill_md,
)
from memory_service.domain.errors import Conflict, NotFound, ProviderNotConfigured
from memory_service.domain.tools import SkillDecision, StoredProcedure
from memory_service.modules.tools.skills import (
    SkillDrafts,
    draft,
    skill_body,
    skill_name,
)
from memory_service.ports.skills import SkillContent, next_version, valid_name

pytestmark = pytest.mark.unit

GATEWAY = "http://gateway.test"


def _procedure(**kw) -> StoredProcedure:
    base = {
        "tenant_id": "acme",
        "scope_key": "user:acme/u1",
        "pattern": "update quote {id} with {entity} price for {id}",
        "title": "Update a quote's regional price",
        "strategy": "Look the price up first.\nAvoid: guessing the region.",
        "steps": [
            {"ordinal": 0, "tool": "lookup_price", "expected_output": ["price", "quote"]},
            {
                "ordinal": 1,
                "tool": "update_quote",
                "failure_modes": {"Locked": "ask the owner to unlock it", "Other": ""},
            },
        ],
        "support": 4,
        "success_rate": 0.75,
        "status": "active",
        "steps_hash": "h1",
    }
    return StoredProcedure(**{**base, **kw})


def _content(name: str = "quote-price", tenant: str = "acme") -> SkillContent:
    return SkillContent(
        name=name,
        description='Update a quote\'s price: "EMEA" first.',
        body="# Body\n\nSteps.",
        metadata={"source": "trellis-memory", "trellis_tenant": tenant},
    )


# ------------------------------------------------------------------ the draft


def test_a_name_is_the_title_in_hyphenated_words_cut_at_a_word() -> None:
    assert skill_name(_procedure()) == "update-a-quote-s-regional-price"
    assert skill_name(_procedure(title="")) == "update-quote-with-price-for"
    long = _procedure(title=" ".join(["procurement"] * 10))
    assert len(skill_name(long)) <= 64 and not skill_name(long).endswith("-")
    assert skill_name(_procedure(title="", pattern="{id} {id}")) == "procedure"
    assert all(valid_name(skill_name(_procedure(title=t))) for t in ("Ünïcode & co", "a--b"))


def test_the_body_has_the_title_the_strategy_and_the_steps_in_order() -> None:
    body = skill_body(_procedure())
    assert body.startswith("# Update a quote's regional price\n")
    assert "Tasks like: `update quote {id} with {entity} price for {id}`" in body
    assert "Avoid: guessing the region." in body
    assert body.index("1. `lookup_price` - returns price, quote") < body.index("2. `update_quote`")
    assert "   - on Locked: ask the owner to unlock it" in body and "on Other" not in body
    assert "Look the price" not in skill_body(_procedure(strategy=""))


def test_only_an_active_undecided_procedure_is_a_draft_and_new_steps_bring_it_back() -> None:
    assert draft(_procedure(status="candidate")) is None
    new = draft(_procedure())
    assert new is not None and new.state == "new" and new.published is None
    assert new.description.startswith("Update a quote's regional price. Use for tasks like")
    assert new.description.endswith("Worked in 75% of 4 runs.")

    published = SkillDecision(
        state="published", steps_hash="h1", name="quote-price", version="1.0.0"
    )
    assert draft(_procedure(skill=published)) is None, "decided for these steps"
    changed = draft(_procedure(skill=published, steps_hash="h2"))
    assert changed is not None and changed.state == "changed"
    assert changed.name == "quote-price" and changed.published == published

    dismissed = SkillDecision(state="dismissed", steps_hash="h1")
    assert draft(_procedure(skill=dismissed)) is None
    again = draft(_procedure(skill=dismissed, steps_hash="h2"))
    assert again is not None and again.state == "new"
    # dismissing a changed draft keeps naming what is out there
    after = SkillDecision(state="dismissed", steps_hash="h2", name="quote-price", version="1.0.0")
    later = draft(_procedure(skill=after, steps_hash="h3"))
    assert later is not None and later.state == "changed" and later.name == "quote-price"


def test_versions_start_at_one_and_move_by_minor() -> None:
    assert next_version(None) == "1.0.0"
    assert next_version("1.0.0") == "1.1.0"
    assert next_version("2.9.4") == "2.10.0"
    for bad in ("1.0", "v1.0.0", "1.x.0"):
        with pytest.raises(Conflict):
            next_version(bad)
    assert valid_name("quote-price") and not valid_name("Quote") and not valid_name("a" * 65)
    assert not valid_name("../etc") and not valid_name("a--b") and not valid_name("-a")


# ------------------------------------------------------------------ the folder


def test_skill_md_is_front_matter_any_reader_parses_and_the_metadata_reads_back() -> None:
    text = skill_md(_content(), "1.2.0")
    head, body = text.split("\n---\n", 1)
    assert head.startswith("---\nname: quote-price\n")
    assert body.strip() == "# Body\n\nSteps."
    assert json.loads(head.splitlines()[2].removeprefix("description: ")) == (
        'Update a quote\'s price: "EMEA" first.'
    )
    meta = front_matter_metadata(text)
    assert meta == {"source": "trellis-memory", "trellis_tenant": "acme", "version": "1.2.0"}
    assert front_matter_metadata("no front matter") == {}
    assert front_matter_metadata("---\nname: x\nmetadata:\n  version: 2.0.0\n---\n") == {
        "version": "2.0.0"
    }
    assert front_matter_metadata('---\nmetadata:\n  v: "unclosed\n---\n') == {"v": '"unclosed'}


async def test_the_folder_gets_a_new_skill_then_its_next_version_and_keeps_others(
    tmp_path,
) -> None:
    store = FolderSkills(tmp_path)
    assert await store.publish(_content(), tenant_id="acme") == "1.0.0"
    assert await store.publish(_content(), tenant_id="acme") == "1.1.0"
    written = (tmp_path / "quote-price" / "SKILL.md").read_text()
    assert front_matter_metadata(written)["version"] == "1.1.0"
    assert [p.name for p in (tmp_path / "quote-price").iterdir()] == ["SKILL.md"]

    with pytest.raises(Conflict):
        await store.publish(_content(tenant="globex"), tenant_id="globex")
    (tmp_path / "house").mkdir()
    (tmp_path / "house" / "SKILL.md").write_text("---\nname: house\n---\nOurs.")
    with pytest.raises(Conflict):
        await store.publish(_content("house"), tenant_id="acme")
    with pytest.raises(Conflict):
        await store.publish(_content("../escape"), tenant_id="acme")
    assert not (tmp_path.parent / "escape").exists()


async def test_a_failed_write_leaves_no_half_file(tmp_path, monkeypatch) -> None:
    store = FolderSkills(tmp_path)

    def boom(*_: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("memory_service.adapters.skills.os.replace", boom)
    with pytest.raises(OSError):
        await store.publish(_content(), tenant_id="acme")
    assert list((tmp_path / "quote-price").iterdir()) == []


# ------------------------------------------------------------------ the gateway


def _listing(*skills: dict) -> dict:
    return {"skills": list(skills), "total_count": len(skills)}


def _skill(metadata: dict, highest: str = "1.0.0") -> dict:
    return {
        "id": "sk_1",
        "name": "quote-price",
        "description": "d",
        "latest_version": "1.0.0",
        "highest_version": highest,
        "metadata": metadata,
    }


@respx.mock
async def test_the_gateway_creates_a_new_skill_at_one_and_publishes_the_next_minor() -> None:
    respx.get(f"{GATEWAY}/api/skills").mock(return_value=httpx.Response(200, json=_listing()))
    created = respx.post(f"{GATEWAY}/api/skills").mock(
        return_value=httpx.Response(200, json={"skill": _skill({"trellis_tenant": "acme"})})
    )
    store = BifrostSkills(f"{GATEWAY}/v1", token="admin-token")
    assert await store.publish(_content(), tenant_id="acme") == "1.0.0"
    sent = json.loads(created.calls[0].request.content)
    assert sent["name"] == "quote-price" and sent["version"] == "1.0.0"
    assert sent["skill_md_body"] == "# Body\n\nSteps."
    assert sent["metadata"]["trellis_tenant"] == "acme"
    assert created.calls[0].request.headers["authorization"] == "Bearer admin-token"

    owned = _skill({"trellis_tenant": "acme"}, highest="1.3.0")
    respx.get(f"{GATEWAY}/api/skills").mock(return_value=httpx.Response(200, json=_listing(owned)))
    respx.get(f"{GATEWAY}/api/skills/sk_1").mock(
        return_value=httpx.Response(200, json={"skill": owned})
    )
    put = respx.put(f"{GATEWAY}/api/skills/sk_1").mock(
        return_value=httpx.Response(200, json={"skill": owned})
    )
    assert await store.publish(_content(), tenant_id="acme") == "1.4.0"
    assert json.loads(put.calls[0].request.content)["version"] == "1.4.0"


@respx.mock
async def test_the_gateway_refuses_a_skill_another_tenant_or_a_person_published() -> None:
    for metadata in ({"trellis_tenant": "globex"}, {}):
        foreign = _skill(metadata)
        respx.get(f"{GATEWAY}/api/skills").mock(
            return_value=httpx.Response(200, json=_listing(foreign))
        )
        respx.get(f"{GATEWAY}/api/skills/sk_1").mock(
            return_value=httpx.Response(200, json={"skill": foreign})
        )
        with pytest.raises(Conflict):
            await BifrostSkills(GATEWAY).publish(_content(), tenant_id="acme")


# ------------------------------------------------------------------ the service's refusals


class _NoUow:
    async def __aenter__(self) -> _NoUow:
        raise AssertionError("no store: nothing is read")

    async def __aexit__(self, *_: object) -> None:
        return None


async def test_publishing_without_a_store_or_without_a_draft_is_refused(tmp_path) -> None:
    with pytest.raises(ProviderNotConfigured):
        await SkillDrafts(_NoUow, None).publish("acme", "p1", by=None)  # type: ignore[arg-type,return-value]

    class _Procedures:
        def __init__(self, found: StoredProcedure | None) -> None:
            self.found = found

        async def get(self, tenant_id: str, procedure_id: str) -> StoredProcedure | None:
            return self.found

    class _Uow:
        def __init__(self, found: StoredProcedure | None) -> None:
            self.procedures = _Procedures(found)

        async def __aenter__(self) -> _Uow:
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

    store = FolderSkills(tmp_path)
    with pytest.raises(NotFound):
        await SkillDrafts(lambda: _Uow(None), store).publish("acme", "p1", by=None)  # type: ignore[arg-type,return-value]
    with pytest.raises(Conflict):
        await SkillDrafts(lambda: _Uow(_procedure(status="retired")), store).dismiss(  # type: ignore[arg-type,return-value]
            "acme", "p1", by=None
        )
    with pytest.raises(Conflict):
        await SkillDrafts(lambda: _Uow(_procedure()), store).publish(  # type: ignore[arg-type,return-value]
            "acme", "p1", by=None, name="Not A Name"
        )


def test_the_store_is_the_folder_when_set_else_the_gateway_else_none(tmp_path) -> None:
    from types import SimpleNamespace

    from memory_service.adapters.wiring import _skill_store
    from memory_service.config.settings import Settings

    def store(**env: str):  # type: ignore[no-untyped-def]
        return _skill_store(SimpleNamespace(settings=Settings(_env_file=None, **env)))  # type: ignore[arg-type]

    assert store() is None
    folder = store(skills_dir=str(tmp_path), bifrost_url=GATEWAY)
    assert isinstance(folder, FolderSkills) and folder.root == tmp_path
    gateway = store(bifrost_url=GATEWAY, bifrost_admin_token="t")
    assert isinstance(gateway, BifrostSkills) and gateway.token == "t"
    assert store(bifrost_url=GATEWAY, bifrost_admin_token="").token is None
    assert store(bifrost_url=GATEWAY).token is None
