"""Evaluate the service's narrative-span prompt on a local CPU llama-server.

Only loopback endpoints are allowed. No credentials, cloud providers, tools or MCP.
This measures source-span selection on synthetic fixtures, not answer accuracy.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx

from benchmark.common import file_sha256
from benchmark.harness import stats
from memory_service.modules.memory import narrative


def assess(output: object, case: dict) -> dict:
    sentences = case["sentences"]
    selected = narrative._select_units(output, sentences, set(case["eligible"]))
    expected = [" ".join(sentences[start : end + 1]) for start, end in case["expected_spans"]]
    # A standalone second sentence drops an antecedent or detaches a correction.
    detached = any(unit == sentences[1] for unit in selected)
    return {
        "selected": selected,
        "expected": expected,
        "exact_units": set(selected) == set(expected),
        "expected_units_found": sum(unit in selected for unit in expected),
        "expected_units": len(expected),
        "detached_dependency": detached,
    }


def completion_payload(model: str, messages: list[dict], *, max_tokens: int) -> dict:
    return {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "seed": 42,
        "max_tokens": max_tokens,
        "tools": [],
        "tool_choice": "none",
        "chat_template_kwargs": {"enable_thinking": False},
    }


def payload(case: dict, model: str) -> dict:
    request = completion_payload(
        model,
        [
            {"role": "system", "content": narrative._SYSTEM},
            {
                "role": "user",
                "content": narrative.source_payload(case["sentences"], case["eligible"]),
            },
        ],
        max_tokens=1536,
    )
    request["response_format"] = {
        "type": "json_schema",
        "json_schema": {
            "name": "narrative_units",
            "strict": True,
            "schema": narrative.span_schema(len(case["sentences"])),
        },
    }
    return request


async def complete(client: httpx.AsyncClient, endpoint: str, request: dict) -> tuple[str, dict]:
    response = await client.post(endpoint + "/v1/chat/completions", json=request)
    response.raise_for_status()
    result = response.json()
    message = result["choices"][0]["message"]
    if message.get("tool_calls"):
        raise ValueError("Tool calls are forbidden")
    if not isinstance(message.get("content"), str):
        raise ValueError("The completion did not contain text")
    return message["content"], result.get("usage", {})


async def runtime_probes(client: httpx.AsyncClient, endpoint: str, model: str) -> list[dict]:
    """Unconstrained sanity checks, after scoring so they cannot warm measured requests.

    These do not certify language/task accuracy. They distinguish a broken runtime or
    chat template from a failure specific to constrained narrative extraction.
    """
    rows = []
    for prompt, expected in (
        ("Reply with exactly the word READY and nothing else.", "READY"),
        ("What is two plus two? Reply with only the number.", "4"),
        ("北京是哪个国家的首都？请只回答国家名。", "中国"),  # noqa: RUF001
    ):
        row = {"prompt": prompt, "expected": expected, "passed": False}
        try:
            text, usage = await complete(
                client,
                endpoint,
                completion_payload(
                    model,
                    [{"role": "user", "content": prompt}],
                    max_tokens=32,
                ),
            )
            row.update(text=text, usage=usage, passed=text.strip().rstrip(".!。") == expected)
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
            row["error"] = type(exc).__name__
        rows.append(row)
    return rows


def summary(rows: list[dict]) -> dict:
    completed = [row for row in rows if "assessment" in row]
    return {
        "attempted": len(rows),
        "parsed_outputs": len(completed),
        "exact_unit_cases": sum(row["assessment"]["exact_units"] for row in completed),
        "detached_dependencies": sum(row["assessment"]["detached_dependency"] for row in completed),
        "latency_ms": stats([row["latency_ms"] for row in rows]),
    }


def local_endpoint(url: str) -> str:
    parsed = urlparse(url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Use an unauthenticated local HTTP model endpoint")
    return url.rstrip("/")


async def run(args) -> None:
    base_url = local_endpoint(args.base_url)
    raw = await asyncio.to_thread(args.data.read_bytes)
    data = json.loads(raw)
    output = {
        "model": args.model,
        "sampling": {"temperature": 0},
        "weights_sha256": await asyncio.to_thread(file_sha256, args.weights),
        "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        "dataset": data["name"],
        "dataset_provenance": data["provenance"],
        "paid_calls": 0,
        "prompt_sha256": hashlib.sha256(narrative._SYSTEM.encode()).hexdigest(),
        "selector_sha256": file_sha256(Path(narrative.__file__)),
        "image_digest": args.image_digest,
        "records": [],
        "complete": False,
        "limitations": [
            "Synthetic component screen, not LoCoMo/SciFact or independent language certification.",
            "Uses the production prompt and source-span validator; bypasses Bifrost for local benchmarking.",
            "Sentence boundaries are supplied; this does not validate production sentence extraction.",
        ],
    }
    async with httpx.AsyncClient(timeout=180, follow_redirects=False, trust_env=False) as client:
        health = await client.get(base_url + "/health")
        health.raise_for_status()
        for case in data["records"]:
            started = time.perf_counter()
            row = {"id": case["id"], "language": case["language"], "category": case["category"]}
            try:
                text, usage = await complete(
                    client,
                    base_url,
                    payload(case, args.model),
                )
                parsed = json.loads(text)
                row.update(output=parsed, assessment=assess(parsed, case), usage=usage)
            except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
                row["error"] = type(exc).__name__
            row["latency_ms"] = (time.perf_counter() - started) * 1000
            output["records"].append(row)
            output["summary"] = summary(output["records"])
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
            print(case["id"], row.get("assessment", row.get("error")), flush=True)
        output["runtime_probes"] = await runtime_probes(client, base_url, args.model)
        output["local_completion_calls"] = len(output["records"]) + len(output["runtime_probes"])
    output["complete"] = True
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument(
        "--data", type=Path, default=Path("tests/eval/golden/multilingual_narrative.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
