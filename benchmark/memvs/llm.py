"""The one answerer and the one judge every system is measured with (standard library + httpx).

Every system's memory reaches the same model through the same instruction, within the same
token budget (``MEMORY_TOKEN_BUDGET``, what agent-harness pushes for an 8,192-token window),
and the answer is graded by the same model with LongMemEval's own judge prompts
(``LongMemEval/src/evaluation/evaluate_qa.py``; the multi-session form for LoCoMo). The token
counts are the gateway's (``usage``), not an estimate.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import httpx

#: agent-harness's push budget: 5 % of the window, clamped to [2000, 8000] tokens
MEMORY_TOKEN_BUDGET = 2000
#: a rough characters-per-token rate, used only to stop rendering at the budget
CHARS_PER_TOKEN = 4
ANSWER_MAX_TOKENS = 160
JUDGE_MAX_TOKENS = 4
REQUEST_TIMEOUT_S = 900.0
ENV_BASE_URL = "MEMVS_LLM_URL"  # the gateway's OpenAI-compatible root, ".../v1"
ENV_KEY = "MEMVS_LLM_KEY"
ENV_MODEL = "MEMVS_LLM_MODEL"

ANSWER_SYSTEM = (
    "You are a helpful assistant with a memory of your past conversations with the user. "
    "Answer the user's question using only the memory below. Be brief. If the memory does not "
    "contain what the question needs, say that you do not have that information - never guess "
    "a code, number, name or date. Follow any standing instructions or preferences the memory "
    "shows.\n\nCurrent date: {now}\n\n# Memory\n{memory}"
)
NOTHING_ON_FILE = "(nothing on file)"

_SAME = (
    "I will give you a question, a correct answer, and a response from a model. Please answer "
    "yes if the response contains the correct answer. Otherwise, answer no. If the response is "
    "equivalent to the correct answer or contains all the intermediate steps to get the correct "
    "answer, you should also answer yes. If the response only contains a subset of the "
    "information required by the answer, answer no. "
)
#: LongMemEval's judge prompts, verbatim, by question type
JUDGE_TEMPLATES: dict[str, str] = {
    "default": _SAME
    + "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response "
    "correct? Answer yes or no only.",
    "temporal-reasoning": _SAME
    + "In addition, do not penalize off-by-one errors for the number of days. If the question "
    "asks for the number of days/weeks/months, etc., and the model makes off-by-one errors "
    "(e.g., predicting 19 days when the answer is 18), the model's response is still correct. "
    "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response "
    "correct? Answer yes or no only.",
    "knowledge-update": "I will give you a question, a correct answer, and a response from a "
    "model. Please answer yes if the response contains the correct answer. Otherwise, answer "
    "no. If the response contains some previous information along with an updated answer, the "
    "response should be considered as correct as long as the updated answer is the required "
    "answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model "
    "response correct? Answer yes or no only.",
    "single-session-preference": "I will give you a question, a rubric for desired "
    "personalized response, and a response from a model. Please answer yes if the response "
    "satisfies the desired response. Otherwise, answer no. The model does not need to reflect "
    "all the points in the rubric. The response is correct as long as it recalls and utilizes "
    "the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel "
    "Response: {}\n\nIs the model response correct? Answer yes or no only.",
}


class LLMError(RuntimeError):
    """The gateway did not return a usable completion."""


@dataclass(frozen=True)
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


@dataclass(frozen=True)
class Gateway:
    base_url: str
    key: str
    model: str

    @classmethod
    def from_env(cls) -> Gateway:
        try:
            return cls(os.environ[ENV_BASE_URL], os.environ[ENV_KEY], os.environ[ENV_MODEL])
        except KeyError as missing:
            raise LLMError(f"{missing} is not set") from None

    def openai_kwargs(self) -> dict[str, str]:
        """What an OpenAI-compatible client of a library needs to reach the same model."""
        return {"base_url": self.base_url, "api_key": self.key, "model": self.model}

    def complete(self, messages: list[dict[str, str]], max_tokens: int) -> Completion:
        started = time.perf_counter()
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.key}"},
            json={
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0,
            },
            timeout=REQUEST_TIMEOUT_S,
        )
        if response.status_code != httpx.codes.OK:
            raise LLMError(f"gateway answered {response.status_code}: {response.text[:200]}")
        body: dict[str, Any] = response.json()
        choices = body.get("choices") or [{}]
        usage = body.get("usage") or {}
        return Completion(
            text=str((choices[0].get("message") or {}).get("content") or "").strip(),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            latency_ms=(time.perf_counter() - started) * 1000,
        )


def answer_messages(memory: str, question: str, now: str | None) -> list[dict[str, str]]:
    system = ANSWER_SYSTEM.format(now=now or "unknown", memory=memory or NOTHING_ON_FILE)
    return [{"role": "system", "content": system}, {"role": "user", "content": question}]


def answer(gateway: Gateway, memory: str, question: str, now: str | None) -> Completion:
    return gateway.complete(answer_messages(memory, question, now), ANSWER_MAX_TOKENS)


def judge(gateway: Gateway, category: str, question: str, gold: str, response: str) -> bool:
    template = JUDGE_TEMPLATES.get(category, JUDGE_TEMPLATES["default"])
    prompt = template.format(question, gold, response)
    verdict = gateway.complete([{"role": "user", "content": prompt}], JUDGE_MAX_TOKENS)
    return verdict.text.strip().lower().startswith("yes")


def within_budget(lines: list[str], budget_tokens: int = MEMORY_TOKEN_BUDGET) -> str:
    """The lines, in order, while they fit the budget (the first always does)."""
    out: list[str] = []
    used = 0
    for line in lines:
        cost = len(line) // CHARS_PER_TOKEN + 1
        if out and used + cost > budget_tokens:
            break
        out.append(line)
        used += cost
    return "\n".join(out)
