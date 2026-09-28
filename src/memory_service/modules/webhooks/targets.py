"""Where a webhook may point. The service performs the request, so the target is checked
as a server-side request forgery would be: scheme, shape, and every address the host
resolves to, at subscription time and again at delivery time (DNS can change in between).
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

from memory_service.config.constants import WEBHOOK_LOCAL_HOSTS, WEBHOOK_LOCAL_SUFFIXES
from memory_service.domain.errors import ValidationFailed
from memory_service.domain.webhooks import URL_MAX_CHARS


@dataclass(frozen=True)
class TargetPolicy:
    """``allow_local_targets`` permits http and non-public addresses: development only."""

    allow_local_targets: bool = False


class TargetRefused(ValidationFailed):
    """The URL points somewhere a webhook may not go."""


def _is_public(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    return address.is_global and not (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    )


def validate_url(url: str, policy: TargetPolicy) -> str:
    """The normalised URL, or ``TargetRefused``. Pure: no name resolution here."""
    if len(url) > URL_MAX_CHARS:
        raise TargetRefused("webhook url is too long", details={"max_chars": URL_MAX_CHARS})
    parts = urlsplit(url.strip())
    if parts.scheme not in ("https", "http") or (
        parts.scheme == "http" and not policy.allow_local_targets
    ):
        raise TargetRefused("webhook url must use https", details={"scheme": parts.scheme})
    if not parts.hostname or parts.username or parts.password or parts.fragment:
        raise TargetRefused("webhook url needs a host and no credentials or fragment")
    host = parts.hostname.lower()
    if not policy.allow_local_targets:
        if host in WEBHOOK_LOCAL_HOSTS or host.endswith(WEBHOOK_LOCAL_SUFFIXES):
            raise TargetRefused("webhook url may not point at a local host", details={"host": host})
        try:
            literal = ipaddress.ip_address(host)
        except ValueError:
            literal = None
        if literal is not None and not _is_public(literal):
            raise TargetRefused(
                "webhook url may not point at a private address", details={"host": host}
            )
    return urlunsplit((parts.scheme, parts.netloc, parts.path or "/", parts.query, ""))


async def resolved_addresses(url: str, policy: TargetPolicy) -> list[str]:
    """Every address the host resolves to; refused unless all of them are public."""
    host = urlsplit(url).hostname or ""
    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise TargetRefused("webhook host does not resolve", details={"host": host}) from exc
    addresses = sorted({str(info[4][0]) for info in infos})
    if not addresses:
        raise TargetRefused("webhook host does not resolve", details={"host": host})
    if not policy.allow_local_targets:
        for address in addresses:
            if not _is_public(ipaddress.ip_address(address)):
                raise TargetRefused(
                    "webhook host resolves to a private address", details={"host": host}
                )
    return addresses
