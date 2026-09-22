"""/version must describe the service that is running, not the one that was requested."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from memory_service.api.app import create_app

pytestmark = pytest.mark.integration

HEADERS = {"X-API-Key": "test-key", "X-Memory-Tenant": "acme"}


def test_version_reports_the_parser_that_is_actually_running(make_settings) -> None:
    """``documents.parser`` defaults to docling, which is absent in an image built without
    the extra — the builtin does the work and the endpoint used to report "docling" anyway."""
    settings = make_settings(documents={"parser": "docling"})
    with TestClient(create_app(settings), raise_server_exceptions=False) as client:
        body = client.get("/version", headers=HEADERS).json()

    running = body["providers"]["document_parser"]
    if running != "docling":
        assert any("document_parser" in note for note in body["degraded"]), (
            f"running {running!r} instead of the configured parser, and /version does not say so"
        )


def test_a_service_that_is_what_it_was_asked_to_be_reports_nothing_degraded(
    make_settings,
) -> None:
    settings = make_settings(documents={"parser": "builtin"})
    with TestClient(create_app(settings), raise_server_exceptions=False) as client:
        body = client.get("/version", headers=HEADERS).json()
    assert body["providers"]["document_parser"] == "builtin"
    assert not [n for n in body["degraded"] if "document_parser" in n]


def test_a_stand_in_classifier_is_declared_degraded(make_settings) -> None:
    """The lexical NLI stand-in keeps grounding running, with verdicts that mean much less.
    Nothing else in the system says so."""
    settings = make_settings(models={"nli": {"provider": "lexical"}})
    with TestClient(create_app(settings), raise_server_exceptions=False) as client:
        body = client.get("/version", headers=HEADERS).json()
    assert any("nli" in note for note in body["degraded"]), body["degraded"]
