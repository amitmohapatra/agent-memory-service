"""Deliveries are signed the way receivers verify them, and only public https targets."""

from __future__ import annotations

import socket

import pytest

from memory_service.domain.errors import ValidationFailed
from memory_service.modules.webhooks import signing
from memory_service.modules.webhooks.targets import (
    TargetPolicy,
    TargetRefused,
    resolved_addresses,
    validate_url,
)
from trellis.memory import webhooks as sdk_webhooks

SECRET = "0123456789abcdef" * 4
BODY = b'{"event_id":"evt_1","type":"memory.created"}'


def test_the_service_signs_the_way_the_sdk_verifies() -> None:
    header = signing.sign(SECRET, 1_790_000_000, BODY)
    assert header.startswith("t=1790000000,v1=") and len(header.split("v1=")[1]) == 64
    assert header == sdk_webhooks.sign(SECRET, 1_790_000_000, BODY)
    assert sdk_webhooks.verify_signature(SECRET, header, BODY, now=1_790_000_010)
    assert not sdk_webhooks.verify_signature(SECRET, header, BODY + b" ", now=1_790_000_010)
    assert not sdk_webhooks.verify_signature("other", header, BODY, now=1_790_000_010)
    assert not sdk_webhooks.verify_signature(SECRET, header, BODY, now=1_790_000_000 + 301)
    assert not sdk_webhooks.verify_signature(SECRET, header, BODY, now=1_790_000_000 - 301)
    assert sdk_webhooks.verify_signature(
        SECRET, header, BODY, now=1_790_000_000 + 301, tolerance_seconds=400
    )


@pytest.mark.parametrize(
    "header",
    ["", "t=abc,v1=00", "v1=00", "t=1", "garbage", "t=1,v2=00", "t=1,v1=" + "\u00e9" * 64],
)
def test_a_malformed_signature_never_verifies(header: str) -> None:
    assert not sdk_webhooks.verify_signature(SECRET, header, BODY, now=1)
    assert not sdk_webhooks.verify_signature(SECRET, None, BODY)


STRICT = TargetPolicy()
DEV = TargetPolicy(allow_local_targets=True)


@pytest.mark.parametrize(
    "url",
    [
        "http://hooks.example.com/x",  # not https
        "ftp://hooks.example.com/x",
        "https://user:pw@hooks.example.com/x",  # credentials
        "https://hooks.example.com/x#frag",
        "https://localhost/x",
        "https://api.internal/x",
        "https://box.local/x",
        "https://127.0.0.1/x",
        "https://10.1.2.3/x",
        "https://169.254.169.254/latest/meta-data",  # cloud metadata
        "https://[::1]/x",
        "https://[::ffff:10.0.0.1]/x",  # v4-mapped private
        "https://100.64.0.1/x",  # shared address space (CGNAT, cloud metadata on some hosts)
        "https:///x",
        "",
    ],
)
def test_private_or_unsafe_targets_are_refused_before_any_lookup(url: str) -> None:
    with pytest.raises(TargetRefused) as exc:
        validate_url(url, STRICT)
    assert isinstance(exc.value, ValidationFailed) and exc.value.http_status == 422


def test_public_https_targets_are_normalised_and_dev_flags_relax_the_policy() -> None:
    assert validate_url(" https://Hooks.example.com ", STRICT) == "https://Hooks.example.com/"
    assert (
        validate_url("https://hooks.example.com/a?b=1", STRICT) == "https://hooks.example.com/a?b=1"
    )
    assert validate_url("http://localhost:8000/hook", DEV) == "http://localhost:8000/hook"
    assert validate_url("https://127.0.0.1:9/hook", DEV) == "https://127.0.0.1:9/hook"
    with pytest.raises(TargetRefused):
        validate_url("https://x.example/" + "a" * 3000, STRICT)


async def test_the_resolved_addresses_must_all_be_public(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    answers: dict[str, list[str]] = {
        "public.example": ["93.184.216.34", "2606:2800:220:1:248:1893:25c8:1946"],
        "rebind.example": ["93.184.216.34", "10.0.0.5"],  # one private answer poisons it
        "nowhere.example": [],
    }

    async def fake_getaddrinfo(host, port, *, type=0):  # noqa: A002
        if host not in answers:
            raise socket.gaierror("no such host")
        return [(socket.AF_INET, type, 0, "", (address, 0)) for address in answers[host]]

    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", fake_getaddrinfo)
    assert await resolved_addresses("https://public.example/x", STRICT) == sorted(
        answers["public.example"]
    )
    with pytest.raises(TargetRefused, match="private"):
        await resolved_addresses("https://rebind.example/x", STRICT)
    with pytest.raises(TargetRefused, match="resolve"):
        await resolved_addresses("https://nowhere.example/x", STRICT)
    with pytest.raises(TargetRefused, match="resolve"):
        await resolved_addresses("https://missing.example/x", STRICT)
    assert await resolved_addresses("https://rebind.example/x", DEV) == sorted(
        answers["rebind.example"]
    )
