"""Reflective approach: generate, check, critique, fix.

  1. Solver: one call answers the task, like the baseline.
  2. Check: the task's feedback check judges the answer (Game of 24: the verifier says whether
     the expression is legal and makes 24; HumanEval: docstring examples plus the model's own
     tests). The official tests are never used here.
  3. If the check fails, a critic call explains what is wrong (no new answer), then a reviser
     call writes a new answer from the old answer, the check result and the critique.
  At most `max_attempts` answers in total; stops early when an answer passes the check. The
  final answer is the passing one, or the last one when none passes.

HumanEval only: before the first answer, a tester call writes assert lines from the task
alone (it never sees code). The same tests are used for every attempt, so they cannot be
bent to fit a given answer.
"""

from collections.abc import Iterator
from typing import Any

from agentic.approaches.base import Caller, load_prompt, user
from agentic.events import Attempt, Body, Critique, FinalAnswer, TestResult
from agentic.llm.base import LLMClient
from agentic.tasks.base import Item, Task

DEFAULTS = {"max_attempts": 3}
FEEDBACK_SOURCE = {"game24": "verifier", "humaneval": "own_tests"}


def extract_asserts(text: str) -> str | None:
    """The assert lines of a reply, without indentation, or None when there are none.
    Asserts that span several lines are skipped; one line each is what the prompt asks for."""
    unfinished = ("(", "[", "{", ",", "\\")
    lines = [line.strip() for line in text.splitlines()]
    asserts = [line for line in lines
               if line.startswith("assert ") and not line.endswith(unfinished)]
    return "\n".join(asserts) if asserts else None


def task_fields(item: Item, task: Task) -> dict[str, str]:
    """Values the prompt templates of this task can use."""
    if task.name == "game24":
        return {"numbers": " ".join(str(n) for n in item.payload["numbers"])}
    if task.name == "humaneval":
        return {"prompt": item.payload["prompt"]}
    raise NotImplementedError(f"Reflective for {task.name} is not built yet")


class Reflective:
    name = "reflective"

    def run(self, item: Item, task: Task, llm: LLMClient, *, seed: int,
            config: dict[str, Any]) -> Iterator[Body]:
        settings = {**DEFAULTS, **config}
        fields = task_fields(item, task)
        caller = Caller(llm, seed)

        tests = None
        if task.name == "humaneval":
            call = caller.chat("tester", user(load_prompt("reflective_tests_humaneval")
                                              .format(**fields)))
            yield call
            tests = extract_asserts(call.response)

        call = caller.chat("solver", user(task.build_prompt(item)))
        yield call
        answer = task.extract_answer(item, call.response)
        attempt_no = 1
        while True:
            yield Attempt(attempt_no=attempt_no, answer=answer, call_id=call.call_id)
            feedback = task.feedback(item, answer, tests)
            yield TestResult(source=FEEDBACK_SOURCE[task.name], ok=feedback.ok,
                             message=feedback.message, details=feedback.details,
                             attempt_no=attempt_no)
            if feedback.ok or attempt_no == settings["max_attempts"]:
                break

            values = {**fields, "answer": answer, "feedback": feedback.message}
            call = caller.chat("critic", user(load_prompt(f"reflective_critic_{task.name}")
                                              .format(**values)))
            yield call
            yield Critique(attempt_no=attempt_no, text=call.response, call_id=call.call_id)

            values["critique"] = call.response
            call = caller.chat("reviser", user(load_prompt(f"reflective_revise_{task.name}")
                                               .format(**values)))
            yield call
            answer = task.extract_answer(item, call.response)
            attempt_no += 1

        yield FinalAnswer(answer=answer)
