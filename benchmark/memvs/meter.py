"""A pass-through proxy in front of the gateway that counts every model call and its tokens.

Each phase's process starts one (``Meter.start``) and points every client at it: a library's
own model calls (Mem0's extraction, LangMem's manager, Graphiti's pipeline, CrewAI's analysis),
the shared answerer and agent-harness's ReAct loop are then counted the same way, from the
gateway's own ``usage`` - including the calls made while writing, which no library reports.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx

HOST = "127.0.0.1"
TIMEOUT_S = 900.0
#: hop-by-hop headers a proxy must not forward
_HOP = {"host", "content-length", "connection", "transfer-encoding", "accept-encoding"}


@dataclass
class Counts:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    failed: int = 0


class TokenMeter:
    def __init__(self, upstream: str) -> None:
        self.upstream = upstream.rstrip("/")
        self.counts = Counts()
        self._lock = threading.Lock()
        meter = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - the stdlib's name
                self._forward("POST", counted=True)

            def do_GET(self) -> None:  # noqa: N802 - listings (models, tools) pass uncounted
                self._forward("GET", counted=False)

            def _forward(self, method: str, *, counted: bool) -> None:
                body = self.rfile.read(int(self.headers.get("content-length") or 0))
                headers = {k: v for k, v in self.headers.items() if k.lower() not in _HOP}
                try:
                    reply = httpx.request(
                        method,
                        meter.upstream + self.path.removeprefix("/v1"),
                        content=body or None,
                        headers=headers,
                        timeout=TIMEOUT_S,
                    )
                except httpx.HTTPError:
                    if counted:
                        meter._count(None, ok=False)
                    self.send_error(502)
                    return
                if counted:
                    meter._count(reply.content, ok=reply.status_code == httpx.codes.OK)
                self.send_response(reply.status_code)
                for k, v in reply.headers.items():
                    if k.lower() not in _HOP and k.lower() != "content-encoding":
                        self.send_header(k, v)
                self.send_header("content-length", str(len(reply.content)))
                self.end_headers()
                self.wfile.write(reply.content)

            def log_message(self, *args: object) -> None:
                return

        self._server = ThreadingHTTPServer((HOST, 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://{HOST}:{self._server.server_address[1]}/v1"

    def _count(self, content: bytes | None, *, ok: bool) -> None:
        usage: dict[str, int] = {}
        if content:
            for chunk in _json_chunks(content):
                usage = chunk.get("usage") or usage
        with self._lock:
            self.counts.calls += 1
            self.counts.failed += not ok
            self.counts.prompt_tokens += int(usage.get("prompt_tokens") or 0)
            self.counts.completion_tokens += int(usage.get("completion_tokens") or 0)

    def start(self) -> TokenMeter:
        self._thread.start()
        return self

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return asdict(self.counts)

    def stop(self) -> None:
        self._server.shutdown()


def _json_chunks(content: bytes) -> list[dict]:
    """A JSON body, or the ``data:`` events of a streamed one."""
    text = content.decode("utf-8", "replace")
    try:
        return [json.loads(text)]
    except ValueError:
        out = []
        for line in text.splitlines():
            if line.startswith("data:") and line[5:].strip() not in ("", "[DONE]"):
                try:
                    out.append(json.loads(line[5:]))
                except ValueError:
                    continue
        return out
