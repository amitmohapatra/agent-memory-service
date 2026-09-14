"""The in-memory ReBAC model must be identical to deploy/openfga/model.fga."""

from pathlib import Path

from memory_service.adapters.authz.model import MODEL, parse_fga

FGA = Path(__file__).resolve().parents[2] / "deploy" / "openfga" / "model.fga"


def test_python_model_matches_fga_dsl() -> None:
    parsed = parse_fga(FGA.read_text())
    assert set(parsed) == set(MODEL), set(parsed) ^ set(MODEL)
    for type_name, type_def in MODEL.items():
        assert set(parsed[type_name].relations) == set(type_def.relations), type_name
        for rel_name, rel in type_def.relations.items():
            got = parsed[type_name].relations[rel_name]
            assert set(got.direct) == set(rel.direct), f"{type_name}.{rel_name} direct"
            assert set(got.computed) == set(rel.computed), f"{type_name}.{rel_name} computed"
            assert set(got.ttu) == set(rel.ttu), f"{type_name}.{rel_name} ttu"


def test_dsl_to_json_shape() -> None:
    from memory_service.adapters.authz.openfga_provider import _dsl_to_json

    body = _dsl_to_json(FGA.read_text())
    assert body["schema_version"] == "1.1"
    types = {t["type"]: t for t in body["type_definitions"]}
    assert "user" in types and types["user"]["relations"] == {}
    thread = types["thread"]
    assert "union" in thread["relations"]["can_read"]
    assert thread["metadata"]["relations"]["participant"]["directly_related_user_types"] == [
        {"type": "user"},
        {"type": "agent"},
        {"type": "group", "relation": "member"},
    ]
