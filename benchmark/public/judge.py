"""Answer generation and the benchmarks' own LLM graders, all through the Bifrost adapter.

The LongMemEval templates are verbatim from ``src/evaluation/evaluate_qa.py`` of the
LongMemEval repository (label = the judge's reply contains "yes", temperature 0,
``max_tokens`` 10); the abstention template applies to question ids containing ``_abs``.
"""

from __future__ import annotations

from typing import Any

from memory_service.ports.models import LLMMessage

ANSWER_USE = "benchmark_answer"
JUDGE_USE = "benchmark_judge"

_STANDARD = (
    "I will give you a question, a correct answer, and a response from a model. Please "
    "answer yes if the response contains the correct answer. Otherwise, answer no. If the "
    "response is equivalent to the correct answer or contains all the intermediate steps to "
    "get the correct answer, you should also answer yes. If the response only contains a "
    "subset of the information required by the answer, answer no. \n\nQuestion: {}\n\n"
    "Correct Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes "
    "or no only."
)
_TEMPORAL = (
    "I will give you a question, a correct answer, and a response from a model. Please "
    "answer yes if the response contains the correct answer. Otherwise, answer no. If the "
    "response is equivalent to the correct answer or contains all the intermediate steps to "
    "get the correct answer, you should also answer yes. If the response only contains a "
    "subset of the information required by the answer, answer no. In addition, do not "
    "penalize off-by-one errors for the number of days. If the question asks for the number "
    "of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 "
    "days when the answer is 18), the model's response is still correct. \n\nQuestion: {}"
    "\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer "
    "yes or no only."
)
_KNOWLEDGE_UPDATE = (
    "I will give you a question, a correct answer, and a response from a model. Please "
    "answer yes if the response contains the correct answer. Otherwise, answer no. If the "
    "response contains some previous information along with an updated answer, the response "
    "should be considered as correct as long as the updated answer is the required answer."
    "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response "
    "correct? Answer yes or no only."
)
_PREFERENCE = (
    "I will give you a question, a rubric for desired personalized response, and a response "
    "from a model. Please answer yes if the response satisfies the desired response. "
    "Otherwise, answer no. The model does not need to reflect all the points in the rubric. "
    "The response is correct as long as it recalls and utilizes the user's personal "
    "information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the "
    "model response correct? Answer yes or no only."
)
_ABSTENTION = (
    "I will give you an unanswerable question, an explanation, and a response from a model. "
    "Please answer yes if the model correctly identifies the question as unanswerable. The "
    "model could say that the information is incomplete, or some other information is given "
    "but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: "
    "{}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no "
    "only."
)

LONGMEMEVAL_TEMPLATES: dict[str, str] = {
    "single-session-user": _STANDARD,
    "single-session-assistant": _STANDARD,
    "multi-session": _STANDARD,
    "temporal-reasoning": _TEMPORAL,
    "knowledge-update": _KNOWLEDGE_UPDATE,
    "single-session-preference": _PREFERENCE,
}


def anscheck_prompt(
    task: str, question: str, answer: str, response: str, *, abstention: bool = False
) -> str:
    if abstention:
        return _ABSTENTION.format(question, answer, response)
    template = LONGMEMEVAL_TEMPLATES.get(task)
    if template is None:
        raise NotImplementedError(f"LongMemEval question type {task!r} has no grader template")
    return template.format(question, answer, response)


def judge_label(reply: str) -> bool:
    return "yes" in reply.lower()


ANSWER_SYSTEM = (
    "You answer a question about the user's past conversations using only the memory "
    "context below. Be concise: give the answer directly, with the specific facts, names, "
    "dates or numbers it needs. If the context does not contain the information required to "
    "answer, reply exactly: No information available."
)


def answer_prompt(context: str, question: str, date: str | None) -> str:
    today = f"Current date: {date}\n\n" if date else ""
    return f"{today}{context.strip() or '(no context)'}\n\nQuestion: {question}"


async def generate_answer(
    llm: Any, *, context: str, question: str, date: str | None, max_tokens: int = 2048
) -> str:
    completion = await llm.complete(
        [
            LLMMessage(role="system", content=ANSWER_SYSTEM),
            LLMMessage(role="user", content=answer_prompt(context, question, date)),
        ],
        max_tokens=max_tokens,
        temperature=0.0,
        use=ANSWER_USE,
    )
    return completion.text.strip()


async def judge_once(llm: Any, prompt: str) -> bool:
    completion = await llm.complete(
        [LLMMessage(role="user", content=prompt)],
        # Ten was enough for a model that answers "yes"/"no" directly. deepseek-flash
        # reasons before it writes and returns nothing when the budget runs out first -
        # measured three times today in three different callers. The verdict is still
        # one word; the budget is for the thinking that precedes it.
        max_tokens=1024,
        temperature=0.0,
        use=JUDGE_USE,
    )
    return judge_label(completion.text)
