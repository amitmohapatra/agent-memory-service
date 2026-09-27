"""Offline local-model screening cannot silently target a paid endpoint or execute tools."""

import json

import httpx
import pytest
from benchmark.local_narrative import assess, local_endpoint, payload, summary

pytestmark = pytest.mark.unit

CASE = {
    "sentences": [
        "Maya moved to Berlin.",
        "She joined a robotics firm.",
        "Omar studied chemistry.",
    ],
    "eligible": [0, 1, 2],
    "expected_spans": [[0, 1], [2, 2]],
}


@pytest.mark.parametrize(
    "url",
    [
        "https://api.openai.com",
        "http://example.com",
        "http://key@localhost:8000",
        "http://localhost:8000?key=secret",
    ],
)
def test_only_unauthenticated_loopback_can_receive_benchmark_requests(url):
    with pytest.raises(ValueError, match="local"):
        local_endpoint(url)
    assert local_endpoint("http://127.0.0.1:18189/") == "http://127.0.0.1:18189"


def test_span_selection_scores_grouping_not_just_retaining_whole_source():
    correct = assess({"units": [{"start": 0, "end": 1}, {"start": 2, "end": 2}]}, CASE)
    assert correct["exact_units"] and correct["expected_units_found"] == 2
    whole = assess({"units": [{"start": 0, "end": 2}]}, CASE)
    assert not whole["exact_units"]
    assert not assess({"units": []}, CASE)["exact_units"]
    assert assess({"units": [{"start": 1, "end": 1}]}, CASE)["selected"] == []


def test_model_payload_excludes_tools_and_uses_production_span_prompt():
    request = payload(CASE, "local-test")
    assert request["tools"] == [] and request["tool_choice"] == "none"
    assert request["chat_template_kwargs"]["enable_thinking"] is False
    assert (
        request["response_format"]["json_schema"]["schema"]["properties"]["units"]["maxItems"] == 6
    )
    assert "ZERO-BASED" in request["messages"][0]["content"]


def test_generation_schema_rejects_out_of_message_indices():
    from jsonschema import ValidationError, validate

    schema = payload(CASE, "local-test")["response_format"]["json_schema"]["schema"]
    validate({"units": [{"start": 0, "end": 2}]}, schema)
    # Observed Qwen output: valid under the old generic schema, no usable source span.
    for unit in ({"start": 0, "end": 10}, {"start": 10, "end": 10}, {"start": -1, "end": 0}):
        with pytest.raises(ValidationError):
            validate({"units": [unit]}, schema)


def test_failed_calls_stay_in_the_attempted_denominator():
    result = summary([{"error": "Timeout", "latency_ms": 1000}])
    assert result["attempted"] == 1 and result["parsed_outputs"] == 0
    assert result["exact_unit_cases"] == 0


async def test_runtime_probes_are_unconstrained_bounded_and_separate_from_extraction():
    from benchmark.local_narrative import runtime_probes

    replies = iter(["READY", "4.", "中国。"])

    def respond(request):
        body = json.loads(request.content)
        assert body["max_tokens"] == 32 and "response_format" not in body
        assert body["tools"] == [] and body["tool_choice"] == "none"
        return httpx.Response(200, json={"choices": [{"message": {"content": next(replies)}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        probes = await runtime_probes(client, "http://127.0.0.1:8080", "fixture-model")
    assert len(probes) == 3 and all(row["passed"] for row in probes)
    assert all("assessment" not in row for row in probes)


async def test_local_completion_rejects_tool_output_even_when_text_is_present():
    from benchmark.local_narrative import complete

    def respond(request):
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "{}", "tool_calls": [{"id": "x"}]}}]}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(ValueError, match="Tool calls are forbidden"):
            await complete(client, "http://127.0.0.1:8080", payload(CASE, "fixture-model"))
