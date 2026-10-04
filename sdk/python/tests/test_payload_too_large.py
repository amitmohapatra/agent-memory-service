"""A 413 is PayloadTooLargeError, a ValidationError that is never retryable (the problem code
is the platform's PAYLOAD_TOO_LARGE, as agent-runs sends it)."""

from trellis.memory import PayloadTooLargeError, ValidationError
from trellis.memory.errors import error_from_problem


def test_the_problem_code_names_the_class() -> None:
    problem = {
        "type": "urn:trellis:problem:payload-too-large",
        "title": "Payload too large",
        "status": 413,
        "detail": "Body exceeds 26214400 bytes",
        "code": "PAYLOAD_TOO_LARGE",
        "retryable": False,
    }
    error = error_from_problem(413, problem)
    assert isinstance(error, PayloadTooLargeError) and isinstance(error, ValidationError)
    assert error.code == "PAYLOAD_TOO_LARGE" and error.retryable is False


def test_a_413_without_a_code_is_the_same_class() -> None:
    error = error_from_problem(413, {"title": "Request Entity Too Large", "detail": "too big"})
    assert isinstance(error, PayloadTooLargeError)
    assert error.code == "PAYLOAD_TOO_LARGE" and error.retryable is False
