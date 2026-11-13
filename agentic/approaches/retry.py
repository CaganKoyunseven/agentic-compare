"""Retry with the verifier, Game of 24 only.

The simplest use of an exact checker: ask the plain baseline prompt again and again (each call
with its own seed) and stop at the first answer the verifier accepts, at most `max_samples`
times. The final answer is the accepted one, or the last one. No critique, no search.

Part of the feedback study (separate from the main results): it gets the same verifier feedback
as Reflective and Multi-agent, and measures what plain retries with that feedback achieve and
cost.
"""

from collections.abc import Iterator
from typing import Any

from agentic.approaches.base import Caller, user
from agentic.events import Attempt, Body, FinalAnswer, TestResult
from agentic.llm.base import LLMClient
from agentic.tasks.base import Item, Task

DEFAULTS = {"max_samples": 10}


class Retry:
    name = "retry"

    def run(self, item: Item, task: Task, llm: LLMClient, *, seed: int,
            config: dict[str, Any]) -> Iterator[Body]:
        if task.name != "game24":
            raise NotImplementedError("Retry is built for Game of 24 only")
        settings = {**DEFAULTS, **config}
        caller = Caller(llm, seed)
        prompt = user(task.build_prompt(item))
        answer = ""
        for attempt_no in range(1, settings["max_samples"] + 1):
            call = caller.chat("solver", prompt)
            yield call
            answer = task.extract_answer(item, call.response)
            yield Attempt(attempt_no=attempt_no, answer=answer, call_id=call.call_id)
            feedback = task.feedback(item, answer)
            yield TestResult(source="verifier", ok=feedback.ok, message=feedback.message,
                             details=feedback.details, attempt_no=attempt_no)
            if feedback.ok:
                break
        yield FinalAnswer(answer=answer)
