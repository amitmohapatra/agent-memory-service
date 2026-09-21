"""Text that is safe to store.

PostgreSQL rejects NUL (0x00) in a ``text`` column outright — psycopg raises ``DataError``
and the whole transaction fails. That is not a theoretical concern: the document path hit it
first (one NUL in an uploaded file destroyed the entire document), and the observation path
hit it again from an agent writing back a tool result that contained a NUL byte, where it
surfaced as a 500 rather than as a stored observation.

Both paths now go through here. Sanitising at the two edges where external bytes become text
is deliberate — doing it once in the repository layer would hide the fact that the input was
malformed, and doing it in each caller is how it came to be missing from one of them.
"""

from __future__ import annotations

import re

#: C0 controls except tab (\x09), newline (\x0a) and carriage return (\x0d), plus DEL and the
#: C1 block. Whitespace is meaningful and kept; the rest is never intentional in text and is
#: either an encoding accident or a probe.
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def sanitise(text: str) -> str:
    """Strip control characters that cannot be stored, keeping tabs and newlines."""
    return _CONTROL.sub("", text)
