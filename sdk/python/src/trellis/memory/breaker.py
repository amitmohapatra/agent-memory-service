"""Fail fast while the service is down, instead of paying the timeout on every call.

Modelled on bifrost-sdk's breaker, and for the same reason: retries handle one bad call, the
breaker handles a bad service. Without it an agent turn during an outage pays the connect
timeout and every retry on each of its memory calls before it can degrade; with it, once
``threshold`` calls in a row have failed, the next ones raise :class:`CircuitOpenError` at
once for ``open_seconds``.

What counts as a failure is the service not being there: a request that got no response, or
a 5xx. A 4xx is the service answering, and a 429 is the service working and asking for less -
counting rate limits turns backpressure into an outage, which is the lesson bifrost-sdk's
breaker records. Neither opens the circuit.

One thing is added to bifrost's version: an explicit half-open state. When the open period
has passed, ONE call goes through as the probe while the others keep failing fast; its
outcome closes the circuit or opens it for another period. Without it every caller waiting
on the circuit stampedes the service the moment the period ends.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Final, Literal

from trellis.memory.errors import CircuitOpenError

#: Consecutive failed calls that open the circuit; 0 disables the breaker.
DEFAULT_FAILURE_THRESHOLD: Final = 5
#: How long an open circuit refuses calls before it lets one probe through.
DEFAULT_OPEN_SECONDS: Final = 30.0

CircuitState = Literal["closed", "open", "half_open"]


class CircuitBreaker:
    """Consecutive-failure breaker with a single half-open probe. One per client."""

    __slots__ = (
        "_clock",
        "_probing",
        "consecutive_failures",
        "open_seconds",
        "open_until",
        "threshold",
    )

    def __init__(
        self,
        threshold: int = DEFAULT_FAILURE_THRESHOLD,
        open_seconds: float = DEFAULT_OPEN_SECONDS,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.threshold = threshold
        self.open_seconds = open_seconds
        self.consecutive_failures = 0
        self.open_until = 0.0
        self._clock = clock
        self._probing = False

    @property
    def state(self) -> CircuitState:
        if not self.threshold or self.consecutive_failures < self.threshold:
            return "closed"
        return "open" if self._clock() < self.open_until else "half_open"

    def acquire(self) -> bool:
        """Let a call through, or raise :class:`CircuitOpenError`. True when the call is the
        half-open probe, whose outcome decides the circuit."""
        state = self.state
        if state == "closed":
            return False
        if state == "open" or self._probing:
            remaining = max(0.0, self.open_until - self._clock())
            raise CircuitOpenError(
                f"memory service circuit open after {self.consecutive_failures} failed calls",
                code="CIRCUIT_OPEN",
                status=0,
                retryable=True,
                retry_after=round(remaining, 1),
                details={"failures": self.consecutive_failures, "state": state},
            )
        self._probing = True
        return True

    def record_success(self) -> None:
        """The service answered: the circuit closes."""
        self.consecutive_failures = 0
        self._probing = False

    def record_failure(self) -> None:
        """The service was not there. Counted once per call, not per attempt: the breaker
        measures failed calls, and a threshold of 5 would otherwise open after two."""
        self._probing = False
        if not self.threshold:
            return
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.threshold:
            self.open_until = self._clock() + self.open_seconds

    def release(self) -> None:
        """A probe ended with no verdict (rate limited, cancelled): the next call probes."""
        self._probing = False
